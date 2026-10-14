"""Rebuild report figures and evidence snapshot from saved models and notebooks.

Run from repository root with the environment used for modeling. No training occurs.
Compile afterward: cd output/pdf && pdflatex -interaction=nonstopmode freight_rate_report.tex
"""
from pathlib import Path
from io import StringIO
import hashlib
import json
import pickle
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from utils import (audit_city_boundaries, predict_market_curves, here_route_geometry,
                   example_route_maps, city_boundary_map, prepare_december_observations,
                   fill_linear_quote_signal, fill_shifted_market_index)
OUT = ROOT / 'output/pdf'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
train = pd.read_csv(ROOT / 'data/train-test.csv', parse_dates=['date'])
val = pd.read_csv(ROOT / 'data/validation.csv', parse_dates=['date'])
meta = json.loads((ROOT / 'outputs/final-model/metadata.json').read_text())
bundle = pickle.loads((ROOT / 'outputs/final-model/preprocessing.pkl').read_bytes())
curve = bundle['feature_state']['market_encoder'].curve_model
colors = {'train': '#2463a6', 'validation': '#218c65'}

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout='constrained')
for ax, col, label in zip(axes.flat, ['distance', 'weight', 'market_index', 'quote_signal'],
                          ['Distance (miles)', 'Weight (lb)', 'Market index', 'Quote signal']):
    all_values = pd.concat([train[col], val[col]]).dropna()
    bins = np.linspace(all_values.min(), all_values.max(), 40)
    for name, frame in [('train', train), ('validation', val)]:
        x = frame[col].dropna()
        ax.hist(x, bins=bins, weights=np.ones(len(x))/len(x)*100, histtype='step',
                lw=1.4, label=name, color=colors[name])
    ax.set(xlabel=label, ylabel='Nonmissing rows (%)')
    ax.grid(alpha=.15)
axes[0, 0].legend()
save(fig, 'eda_distributions')

known = val.pickup.isin(train.pickup) & val.delivery.isin(train.delivery)
new_cities = sorted((set(val.pickup) | set(val.delivery)) - (set(train.pickup) | set(train.delivery)))
# Match the temporal panels in eda.ipynb: daily means globally, load-weighted
# trailing 7-calendar-day means for trips from and to Detroit, within each split.
fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5), sharex=True)
right_axes = [ax.twinx() for ax in axes]
styles = {
    ('market_index', 'train'): ('#2463a6', '-', 'Market index: train'),
    ('market_index', 'validation'): ('#238b63', '-', 'Market index: validation'),
    ('quote_signal', 'train'): ('#db7b2b', '--', 'Quote signal: train'),
    ('quote_signal', 'validation'): ('#8b5cf6', '--', 'Quote signal: validation'),
    ('posted_rate', 'train'): ('#c43b45', '-', 'Posted rate: train'),
}
left_values, right_values = [], []
for row, city_column in enumerate([None, 'pickup', 'delivery']):
    ax, right = axes[row], right_axes[row]
    for name, frame in [('train', train), ('validation', val.loc[known])]:
        trips = frame if city_column is None else frame.loc[frame[city_column].eq('Detroit')]
        calendar = pd.date_range(frame.date.min(), frame.date.max())
        for metric in ['market_index', 'quote_signal', 'posted_rate']:
            if metric not in trips:
                continue
            daily_stats = trips.groupby('date')[metric].agg(['sum', 'count']).reindex(calendar)
            if row == 0:
                values = daily_stats['sum'] / daily_stats['count'].replace(0, np.nan)
            else:
                values = (daily_stats['sum'].fillna(0).rolling(7, min_periods=1).sum()
                          / daily_stats['count'].fillna(0).rolling(7, min_periods=1).sum().replace(0, np.nan))
            color, dash, label = styles[(metric, name)]
            target_ax = right if metric == 'posted_rate' else ax
            target_ax.plot(calendar, values, color=color, linestyle=dash, lw=1.05, label=label)
            (right_values if metric == 'posted_rate' else left_values).extend(values.dropna().tolist())
        if row > 0:
            ax.scatter(trips.date, trips.market_index, s=3, alpha=.10,
                       color=styles[('market_index', name)][0], edgecolors='none')
    ax.axvline(val.date.min(), color='#777777', lw=.8, linestyle=':')
    ax.axvspan(val.date.min(), val.date.max(), color='#238b63', alpha=.035)
    ax.set_ylabel('Market / quote', fontsize=9)
    right.set_ylabel('Posted rate (USD)', fontsize=9)
    right.spines['right'].set_visible(True)
    ax.grid(alpha=.15)
for ax, right, title in zip(axes, right_axes, ['All loads: daily means',
                                            'From Detroit: 7-day load-weighted means',
                                            'To Detroit: 7-day load-weighted means']):
    ax.set_title(title, fontsize=10)
    ax.set_ylim(min(left_values) * .9, max(left_values) * 1.06)
    right.set_ylim(0, max(right_values) * 1.1)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].set_xlabel('2025; shaded period = known-city validation', fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
rate_handles, rate_labels = right_axes[0].get_legend_handles_labels()
fig.legend(handles + rate_handles, labels + rate_labels, loc='lower center', ncol=2,
           fontsize=8, frameon=False, bbox_to_anchor=(.5, .005))
fig.subplots_adjust(left=.1, right=.88, top=.95, bottom=.16, hspace=.35)
save(fig, 'temporal_signals')

# Static version of the city-share / growth map in EDA; no web tiles required.
cities = sorted(set(train.pickup) | set(train.delivery))
train_days = len(pd.date_range(train.date.min(), train.date.max()))
val_days = len(pd.date_range(val.date.min(), val.date.max()))
city_train = pd.DataFrame({role: train[role].value_counts().reindex(cities, fill_value=0) / train_days
                           for role in ['pickup', 'delivery']})
city_val = pd.DataFrame({role: val.loc[known, role].value_counts().reindex(cities, fill_value=0) / val_days
                         for role in ['pickup', 'delivery']})
city_growth = (city_val / city_train.replace(0, np.nan) - 1) * 100
coordinates = pd.read_csv(ROOT / 'data/city-coordinates.csv').set_index('city')
map_data = coordinates.reindex(cities).copy()
map_data['share'] = city_train.sum(axis=1) * train_days / (2 * len(train)) * 100
map_data['growth'] = city_growth.sum(axis=1, min_count=2)
if map_data[['latitude', 'longitude', 'growth']].isna().any().any():
    raise ValueError('Missing city reference or growth component')
projector = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
state_lines = []
for feature in json.loads((ROOT / 'data/us-state-outlines-2025.geojson').read_text())['features']:
    geometry = transform(projector.transform, shape(feature['geometry'])).simplify(1500)
    polygons = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
    state_lines.extend(np.asarray(polygon.exterior.coords) for polygon in polygons)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.add_collection(LineCollection(state_lines, colors='#bfc6cd', linewidths=.5, zorder=1))
x, y = projector.transform(map_data.longitude.to_numpy(), map_data.latitude.to_numpy())
limit = float(map_data.growth.abs().max())
points = ax.scatter(x, y, s=map_data.share * 32, c=map_data.growth,
                    cmap='RdBu_r', vmin=-limit, vmax=limit, alpha=.9,
                    edgecolors='white', linewidths=.5, zorder=3)
new_points = coordinates.reindex(new_cities)
nx, ny = projector.transform(new_points.longitude.to_numpy(), new_points.latitude.to_numpy())
ax.scatter(nx, ny, marker='*', s=105, color='#c026d3', edgecolors='white', linewidths=.5, zorder=4)
for city, px, py in zip(new_cities, nx, ny):
    offset = {'Allentown': (4, 9), 'Norfolk': (5, -10), 'Charlotte': (5, -12),
              'Knoxville': (-40, -10), 'Jackson': (-30, -12)}.get(city, (5, 5))
    ax.annotate(city, (px, py), xytext=offset, textcoords='offset points', fontsize=7,
                color='#86198f', bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=.5), zorder=5)
all_x, all_y = np.r_[x, nx], np.r_[y, ny]
ax.set_xlim(all_x.min()-170000, all_x.max()+170000)
ax.set_ylim(all_y.min()-200000, all_y.max()+200000)
ax.set_aspect('equal'); ax.set_axis_off()
ax.set_title('City composition: training share and validation growth', fontsize=11)
bar = fig.colorbar(points, ax=ax, fraction=.035, pad=.025, shrink=.75)
bar.set_label('Pickup growth + delivery growth (%)', fontsize=8)
bar.ax.tick_params(labelsize=8)
handles = [Line2D([], [], marker='o', linestyle='none', markersize=np.sqrt(share*32),
                  markerfacecolor='#b95b59', markeredgecolor='white', label=f'{share:g}% train share')
           for share in [.5, 2, 5]]
handles.append(Line2D([], [], marker='*', linestyle='none', markersize=11,
                      markerfacecolor='#c026d3', markeredgecolor='white', label='New validation city'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.02), ncol=2,
          fontsize=8, frameon=False)
fig.subplots_adjust(left=.01, right=.91, top=.93, bottom=.17)
save(fig, 'city_composition')

# Current boundary audit and cached route geometry, rather than stale plot images.
audit, boundaries = audit_city_boundaries({'train': train, 'validation': val}, ROOT / 'data/city-boundaries-2025.geojson')
fig = city_boundary_map(audit, boundaries, ROOT / 'data/us-state-outlines-2025.geojson')
for text in fig.findobj(matplotlib.text.Text):
    if text.get_text(): text.set_fontsize(max(text.get_fontsize(), 14))
for text in fig.findobj(matplotlib.text.Annotation):
    if text.get_text() == 'Washington': text.set_position((-45, -12))
    if text.get_text() == 'Providence': text.set_position((7, 12))
for text in fig.texts:
    if text.get_text().startswith('US Census'): text.set_text('US Census TIGER/Line 2025; original coordinates unchanged.')
save(fig, 'city_boundaries')
lookup = bundle['city_centers'].set_index('city')
routes = []
for pickup, delivery in [('Dallas', 'Houston'), ('Detroit', 'Atlanta'), ('Los Angeles', 'New York')]:
    geometry = here_route_geometry(tuple(lookup.loc[pickup, ['lat', 'lon']]),
        tuple(lookup.loc[delivery, ['lat', 'lon']]), ROOT / 'data/route_cache', meta['here_snapshot'])
    routes.append(dict(pickup=pickup, delivery=delivery, **geometry))
fig = example_route_maps(routes, ROOT / 'data/us-state-outlines-2025.geojson')
for text in fig.findobj(matplotlib.text.Text):
    if text.get_text(): text.set_fontsize(max(text.get_fontsize(), 16))
save(fig, 'routes')

fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.7), sharex=True, layout='constrained')
all_market = pd.concat([train[['date', 'market_index']], val[['date', 'market_index']]], ignore_index=True)
daily = all_market.groupby('date').market_index.mean().dropna()
axes[0].plot(daily.index, daily, color='#999999', lw=.7, label='Observed daily mean')
segment_colors = ['#2463a6', '#218c65', '#b76b14', '#9557a5']
for number, segment in enumerate(curve['segments'], 1):
    dates = pd.date_range(segment['start'], segment['end'])
    trend = predict_market_curves(curve, dates, weekly=False)
    fit = predict_market_curves(curve, dates)
    for ax, values in zip(axes, [fit, trend, fit-trend]):
        ax.plot(dates, values, color=segment_colors[number-1], lw=1.4, label=f'Regime {number}')
        ax.axvspan(dates[0], dates[-1], color=segment_colors[number-1], alpha=.045)
for ax, title in zip(axes, ['Observed mean and quadratic + weekly fit', 'Quadratic trend', 'Weekly harmonic']):
    ax.set(title=title, ylabel='Index')
    ax.axvline(pd.Timestamp('2025-11-01'), color='#333333', linestyle=':', lw=.8)
    ax.grid(alpha=.15)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[0].legend(fontsize=8, ncol=3)
save(fig, 'market_decomposition')

borders = pd.read_csv(ROOT / 'outputs/golden-features-cv.csv')
border_scores = borders.groupby('border_count').apply(lambda rows: np.array([.2,.3,.5]) @ rows.sort_values('fold').MAPE_pct, include_groups=False)
fig, ax = plt.subplots(figsize=(7.2, 2.3), layout='constrained')
bars = ax.bar(border_scores.index.astype(str), border_scores.values, color=['#218c65', '#2463a6', '#2463a6', '#2463a6'])
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.set(xlabel='Distance feature border count', ylabel='Weighted CV MAPE (%)', ylim=(0, border_scores.max()*1.18))
ax.grid(axis='y', alpha=.15)
save(fig, 'distance_borders')

# Extract tables from executed notebook outputs, preserving the current model lineage.
nb = nbformat.read(ROOT / 'notebooks/modeling.ipynb', as_version=4)
def tables(index):
    result = []
    for output in nb.cells[index].get('outputs', []):
        if 'data' in output and 'text/html' in output['data']:
            result.extend(pd.read_html(StringIO(output['data']['text/html'])))
    return result
cv = tables(11)[0].drop(columns=['Unnamed: 0'], errors='ignore')
baseline = tables(12)[0].drop(columns=['Unnamed: 0'], errors='ignore')
summary = tables(17)[0]
# JSON evidence is machine readable, linked to immutable artifact hashes.
missing = {name: {'weight_missing': int(frame.weight.isna().sum()),
                  'weight_nonpositive': int(frame.weight.le(0).sum()),
                  'market_missing': int(frame.market_index.isna().sum()),
                  'date_min': str(frame.date.min().date()), 'date_max': str(frame.date.max().date())}
           for name, frame in [('train', train), ('validation', val)]}
dec_inputs = pd.read_csv(ROOT / 'data/december-chart-inputs.csv', parse_dates=['date'])
dec_observed = prepare_december_observations(dec_inputs, val)
dec_filled, quote_fit = fill_linear_quote_signal(dec_observed)
dec_filled, market_shift = fill_shifted_market_index(dec_filled, curve)
dec_predictions = pd.read_csv(ROOT / 'outputs/december-predictions.csv')
market_rmse = float(np.sqrt(np.mean((daily.to_numpy()-predict_market_curves(curve,daily.index))**2)))
snapshot = dict(model_metadata=meta, missing=missing, new_cities=new_cities,
    daily_counts=dict(train=len(train)/train.date.nunique(), validation=len(val)/val.date.nunique(),
                      validation_known=int(known.sum())/val.date.nunique(), known_validation_rows=int(known.sum())),
    boundary_audit=dict(cities=len(audit), inside=int(audit.inside_named_city.sum()),
                        outside_gt100=int(audit.distance_outside_city_km.gt(100).sum())),
    market_rmse=market_rmse, cv_metrics=cv.to_dict('records'), baseline_metrics=baseline.to_dict('records'),
    distance_border_scores=border_scores.to_dict(),
    quote_fit={**quote_fit, 'origin': str(quote_fit['origin'].date())}, market_shift=market_shift,
    december_rates=dict(min=float(dec_predictions.predicted_rate.min()), max=float(dec_predictions.predicted_rate.max()),
                        mean=float(dec_predictions.predicted_rate.mean())),
    artifact_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/'outputs/final-model/model.cbm', ROOT/'outputs/final-model/preprocessing.pkl',
        ROOT/'outputs/validation-predictions.csv', ROOT/'outputs/december-predictions.csv']})
(OUT/'evidence.json').write_text(json.dumps(snapshot, indent=2))
print(json.dumps({k:snapshot[k] for k in ['missing','new_cities','daily_counts','boundary_audit','december_rates']},indent=2))
print('Generated vector report figures and evidence snapshot in', OUT)

# Freeze all included notebook graphics next to the LaTeX source for portable builds.
for name in ['time-series-cv.png', 'catboost-cv-metrics.png', 'test-quote-product-baseline.png',
             'test-residuals.png', 'test-outlier-comparison.png', 'shap-validation.png',
             'test-worst-five-shap.png', 'december-quote-signal-fit.png',
             'december-market-index-fit.png', 'december-predictions-interval.png',
             'shap-fold_1.png', 'shap-fold_2.png', 'shap-fold_3.png', 'inference-predictions.png']:
    shutil.copy2(ROOT / 'outputs' / name, FIG / name)
shutil.copy2(ROOT / 'outputs/scorer_results/candidate_december.png', FIG / 'candidate_december.png')
