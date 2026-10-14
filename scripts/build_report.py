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
import numpy as np
import pandas as pd
import nbformat
from catboost import CatBoostRegressor

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
model = CatBoostRegressor().load_model(str(ROOT / 'outputs/final-model/model.cbm'))
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
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), layout='constrained')
for name, frame, color in [('Train', train, '#2463a6'), ('Validation: all', val, '#218c65'),
                            ('Validation: known cities', val.loc[known], '#d46b25')]:
    counts = frame.groupby('date').size().reindex(pd.date_range(frame.date.min(), frame.date.max()), fill_value=0)
    axes[0].plot(counts.index, counts, color=color, alpha=.25, lw=.7)
    axes[0].plot(counts.index, counts.rolling(7, min_periods=1).mean(), color=color, lw=1.4, label=name)
axes[0].set(title='Daily sample counts; bold lines are trailing 7-day means', ylabel='Rows / day')
axes[0].legend(fontsize=8, ncol=1)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
shares = pd.DataFrame({name: frame.equipment.value_counts(normalize=True)*100 for name, frame in [('Train', train), ('Validation', val)]})
shares.plot.bar(ax=axes[1], color=['#2463a6', '#218c65'], rot=0)
axes[1].set(title='Equipment composition', ylabel='Rows (%)', xlabel='')
for ax in axes: ax.grid(alpha=.15)
save(fig, 'sampling_shift')

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

importance = pd.Series(model.get_feature_importance(type='PredictionValuesChange'), index=meta['feature_columns'])
fig, ax = plt.subplots(figsize=(7.2, 4), layout='constrained')
importance.nlargest(15).sort_values().plot.barh(ax=ax, color='#2463a6')
ax.set(title='Full-train model: PredictionValuesChange importance', xlabel='Normalized importance (all features sum to 100)')
ax.grid(axis='x', alpha=.15)
save(fig, 'feature_importance')

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
    distance_border_scores=border_scores.to_dict(), feature_importance=importance.sort_values(ascending=False).to_dict(),
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
