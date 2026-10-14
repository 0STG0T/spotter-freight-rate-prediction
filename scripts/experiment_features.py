"""Ablate feature families on the existing temporal CV without changing saved models."""
from pathlib import Path
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
from templates import TimeSeriesCV
from utils import (load_data, fit_imputation_values, prepare_data, add_time_features,
                   add_city_route_features, MarketIndexFeatures, PastCityAggregates,
                   add_holiday_features)

OUT = ROOT / 'outputs/feature-experiments'
OUT.mkdir(exist_ok=True)
meta = json.loads((ROOT / 'outputs/final-model/metadata.json').read_text())
bundle = pickle.loads((ROOT / 'outputs/final-model/preprocessing.pkl').read_bytes())
raw, _ = load_data()
source = raw['train'].sort_values(['date', 'load_id'])
cutoff = source.date.max() - pd.Timedelta(days=13)
dev, test = source.loc[source.date < cutoff], source.loc[source.date >= cutoff]
curve = bundle['feature_state']['market_encoder'].curve_model
base_columns = meta['feature_columns']
params = {k: meta['model_params'][k] for k in [
    'depth', 'learning_rate', 'l2_leaf_reg', 'random_strength', 'rsm', 'border_count',
    'grow_policy', 'leaf_estimation_method', 'leaf_estimation_iterations',
    'leaf_estimation_backtracking', 'random_seed']}
params.update(iterations=1500, loss_function='RMSE', eval_metric='RMSE',
              thread_count=4, allow_writing_files=False, verbose=False,
              per_float_feature_quantization=['6:border_count=254'])


def prepare(train, valid):
    fill = fit_imputation_values(train)
    market = MarketIndexFeatures.fit(train, curve_model=curve)
    city = PastCityAggregates(strength=50)
    def base(frame):
        x = add_time_features(prepare_data(frame, fill))
        x = add_city_route_features(x, bundle['city_centers'], bundle['route_table'])
        return add_holiday_features(market.transform(x, frame.market_index))
    a, b = base(train), base(valid)
    a = a.join(city.fit_transform(a))
    b = b.join(city.transform(b))
    return a[base_columns], b[base_columns]


def physical(x):
    f = pd.DataFrame(index=x.index)
    f['quote_total'] = x.quote_signal * x.distance
    f['market_quote_total'] = x.market_index * f.quote_total
    f['quote_over_market'] = x.quote_signal / x.market_index.clip(lower=.1)
    f['quote_road_total'] = x.quote_signal * x.road_distance_miles
    g = pd.DataFrame(index=x.index)
    g['weight_per_mile'] = x.weight / x.distance.clip(lower=1)
    g['ton_miles'] = x.weight / 2000 * x.distance
    g['road_speed'] = x.road_distance_miles / x.driving_time_hours.clip(lower=.1)
    g['extra_road_miles'] = x.road_distance_miles - x.distance
    g['weight_capacity_fraction'] = x.weight / 45000
    h = pd.DataFrame(index=x.index)
    h['latitude_delta'] = x.delivery_lat - x.pickup_lat
    h['longitude_delta'] = x.delivery_lon - x.pickup_lon
    h['latitude_midpoint'] = (x.delivery_lat + x.pickup_lat) / 2
    h['longitude_midpoint'] = (x.delivery_lon + x.pickup_lon) / 2
    return {'quote_interactions': f, 'route_load': g, 'direction': h,
            **{column: g[[column]] for column in g}, 'quote_total_only': f[['quote_total']]}


def historical(train, valid, groups):
    """Smoothed log price/mile, using only labels from strictly earlier dates.

    Fixed prior log(2 USD/mile); validation mappings fit on its training split.
    No same-day labels, validation labels or final-model predictions are used.
    """
    a, b = pd.DataFrame(index=train.index), pd.DataFrame(index=valid.index)
    z = np.log(train.posted_rate / train.distance)
    for keys in groups:
        name = 'past_rpm_' + '_'.join(keys)
        work = train[keys + ['date']].assign(z=z)
        daily = work.groupby(keys + ['date'], observed=True).z.agg(['sum', 'count']).reset_index()
        daily = daily.sort_values('date')
        for col in ['sum', 'count']:
            daily['past_' + col] = daily.groupby(keys, observed=True)[col].cumsum() - daily[col]
        daily[name] = (daily.past_sum + 30 * np.log(2)) / (daily.past_count + 30)
        a[name] = train[keys + ['date']].merge(daily[keys + ['date', name]], on=keys+['date'],
                                             how='left', validate='many_to_one')[name].to_numpy()
        total = work.groupby(keys, observed=True).z.agg(['sum', 'count'])
        total[name] = (total['sum'] + 30*np.log(2)) / (total['count'] + 30)
        b[name] = valid[keys].merge(total[[name]], left_on=keys, right_index=True,
                                  how='left', validate='many_to_one')[name].fillna(np.log(2)).to_numpy()
    return a, b


def features(train, valid):
    a, b = prepare(train, valid)
    fa, fb = physical(a), physical(b)
    for name, groups in {'city_pricing': [['pickup'], ['delivery']],
                         'lane_pricing': [['pickup', 'delivery'], ['pickup', 'delivery', 'equipment']]}.items():
        fa[name], fb[name] = historical(train, valid, groups)
    return a, b, fa, fb


def fit_score(a, b, train, valid, iterations=None):
    config = dict(params)
    if iterations is not None:
        config['iterations'] = iterations
    model = CatBoostRegressor(**config)
    fit_args = dict(cat_features=meta['cat_features'])
    if iterations is None:
        fit_args.update(eval_set=(b, np.log(valid.posted_rate)), early_stopping_rounds=100)
    model.fit(a, np.log(train.posted_rate), **fit_args)
    pred = np.exp(model.predict(b))
    ape = np.abs(pred / valid.posted_rate.to_numpy() - 1) * 100
    return dict(MAPE_pct=float(ape.mean()), median_APE_pct=float(np.median(ape)),
                trees=model.tree_count_, features=a.shape[1])


if __name__ == '__main__':
    families = ['quote_interactions', 'route_load', 'direction', 'city_pricing', 'lane_pricing']
    variants = {'baseline': [], **{name: [name] for name in families}, 'all': families,
                **{name: [name] for name in ['weight_per_mile', 'ton_miles', 'road_speed',
                    'extra_road_miles', 'weight_capacity_fraction', 'quote_total_only']}}
    prepared = []
    for i, (ti, vi) in enumerate(TimeSeriesCV(3).split(dev), 1):
        train, valid = dev.iloc[ti], dev.iloc[vi]
        prepared.append((train, valid, *features(train, valid)))
        print(f'Prepared fold {i}: {len(train)} / {len(valid)}', flush=True)
    results = []
    for name, chosen in variants.items():
        for i, (train, valid, a, b, fa, fb) in enumerate(prepared, 1):
            xa = pd.concat([a] + [fa[k] for k in chosen], axis=1)
            xb = pd.concat([b] + [fb[k] for k in chosen], axis=1)
            assert np.isfinite(xa.select_dtypes('number')).all().all()
            result = dict(variant=name, fold=i, **fit_score(xa, xb, train, valid))
            results.append(result)
            pd.DataFrame(results).to_csv(OUT / 'fold-results.csv', index=False)
            print(result, flush=True)
    table = pd.DataFrame(results)
    summary = table.pivot(index='variant', columns='fold', values='MAPE_pct')
    summary['weighted_MAPE_pct'] = summary[[1, 2, 3]].to_numpy() @ [.2, .3, .5]
    summary = summary.sort_values('weighted_MAPE_pct')
    summary.to_csv(OUT / 'summary.csv')
    winner = summary.index[0]
    # Only baseline and the CV-selected winner are evaluated on the reserved test.
    a, b, fa, fb = features(dev, test)
    holdout = []
    for name in dict.fromkeys(['baseline', winner]):
        chosen = variants[name]
        iterations = int(table.loc[table.variant.eq(name), 'trees'].median())
        holdout.append(dict(variant=name, **fit_score(pd.concat([a]+[fa[k] for k in chosen], axis=1),
            pd.concat([b]+[fb[k] for k in chosen], axis=1), dev, test, iterations)))
    pd.DataFrame(holdout).to_csv(OUT / 'holdout-results.csv', index=False)
    manifest = dict(selected_by_cv=winner, weights=[.2,.3,.5], params=params,
                    market_fit_scope=meta['market_fit_scope'], holdout_start=str(cutoff.date()),
                    feature_groups=variants, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Fixed incumbent parameters; feature ablation, not retuning. Production artifacts unchanged.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(), flush=True)
    print(holdout, flush=True)
