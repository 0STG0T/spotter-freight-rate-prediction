"""Post-hoc error diagnostics. Test results must not be used for feature selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import experiment_features as exp

OUT = exp.OUT / 'error-analysis'
OUT.mkdir(exist_ok=True)
a, b, fa, fb = exp.features(exp.dev, exp.test)
fold_results = pd.read_csv(exp.OUT / 'fold-results.csv')
result = exp.test.copy()
for name in ['baseline', 'direction', 'route_load', 'lane_pricing', 'quote_interactions']:
    train = a if name == 'baseline' else a.join(fa[name])
    valid = b if name == 'baseline' else b.join(fb[name])
    params = dict(exp.params, iterations=int(fold_results.loc[fold_results.variant.eq(name), 'trees'].median()))
    model = CatBoostRegressor(**params)
    model.fit(train, np.log(exp.dev.posted_rate), cat_features=exp.meta['cat_features'])
    result[name] = np.exp(model.predict(valid))
    result[name + '_APE'] = abs(result[name] / result.posted_rate - 1) * 100
result['group'] = np.select([result.baseline_APE.le(10), result.baseline_APE.le(50),
                            result.baseline.gt(result.posted_rate)],
                           ['ordinary <=10%', 'moderate 10-50%', 'severe overprediction'],
                           default='severe underprediction')
result['actual_to_prediction'] = result.posted_rate / result.baseline
result['target_per_mile'] = result.posted_rate / result.distance
result['prediction_per_mile'] = result.baseline / result.distance
result['quote_total'] = result.distance * result.quote_signal
result['actual_to_quote'] = result.posted_rate / result.quote_total
for col in ['road_distance_miles', 'input_to_road_ratio', 'market_curve_residual']:
    result[col] = b[col]
groups = result.groupby('group').agg(rows=('load_id', 'size'),
    baseline_MAPE=('baseline_APE','mean'), actual_to_prediction=('actual_to_prediction','median'),
    target_per_mile=('target_per_mile','median'),prediction_per_mile=('prediction_per_mile','median'),
    actual_to_quote=('actual_to_quote','median'))
groups['share_total_APE_pct'] = 100 * result.groupby('group').baseline_APE.sum() / result.baseline_APE.sum()
for name in ['direction','route_load','lane_pricing','quote_interactions']:
    groups[name+'_MAPE'] = result.groupby('group')[name+'_APE'].mean()
groups.to_csv(OUT/'groups.csv')
features = ['distance','weight','market_index','quote_signal','road_distance_miles',
            'input_to_road_ratio','market_curve_residual']
result.groupby('group')[features].median().T.to_csv(OUT/'input-medians.csv')
cat=[]
for col in ['equipment','pickup','delivery']:
    t=result.assign(severe=result.baseline_APE.gt(50)).groupby(col).agg(rows=('load_id','size'),severe=('severe','sum'))
    t['severe_pct']=100*t.severe/t.rows
    cat.append(t.reset_index().rename(columns={col:'category'}).assign(feature=col))
pd.concat(cat).to_csv(OUT/'category-rates.csv',index=False)
peers=[]
reference=exp.dev.copy()
match=['distance','weight','quote_signal','market_index']
center=reference[match].median(); scale=(reference[match].quantile(.75)-reference[match].quantile(.25)).replace(0,1)
for _,row in result.loc[result.baseline_APE.gt(50)].iterrows():
    lane=reference.loc[reference.pickup.eq(row.pickup)&reference.delivery.eq(row.delivery)]
    pool=lane.loc[lane.equipment.eq(row.equipment)]
    query=row[match].astype(float).fillna(center)
    d=(((pool[match].fillna(center)-query)/scale)**2).sum(axis=1)
    nearest=pool.loc[d.nsmallest(5).index]
    # Price/mile comparison adjusts for varying distance within the named route.
    median_rpm=(nearest.posted_rate/nearest.distance).median()
    peers.append(dict(load_id=row.load_id,lane_train_rows=len(lane),same_equipment_rows=len(pool),
        nearest_rows=len(nearest),peer_rpm=median_rpm,actual_rpm=row.target_per_mile,
        predicted_rpm=row.prediction_per_mile,actual_to_peer=row.target_per_mile/median_rpm,
        peer_ids=';'.join(nearest.load_id),
        peer_distance_min=nearest.distance.min(),peer_distance_max=nearest.distance.max(),
        peer_weight_median=nearest.weight.median(),peer_quote_median=nearest.quote_signal.median(),
        peer_market_median=nearest.market_index.median()))
peer_table=pd.DataFrame(peers)
result=result.merge(peer_table,on='load_id',how='left',validate='one_to_one')
result.sort_values('baseline_APE',ascending=False).to_csv(OUT/'cases.csv',index=False)
print(groups.to_string())
print(result.nlargest(10,'baseline_APE')[['load_id','pickup','delivery','equipment','distance','weight',
 'quote_signal','market_index','posted_rate','baseline','baseline_APE','actual_to_peer','same_equipment_rows']].to_string(index=False))
print('Input medians\n',result.groupby('group')[features].median().T.to_string())
