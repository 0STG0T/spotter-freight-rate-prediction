"""Train-only target outlier ablation; validation and holdout stay complete."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import experiment_features as e

OUT = e.ROOT / 'outputs/target-outlier-experiment'
OUT.mkdir(exist_ok=True)
VARIANTS = {'baseline': None, 'rpm_mad_3': 3., 'rpm_mad_4': 4., 'rpm_mad_5': 5.}


def training_mask(frame, threshold):
    """Robust z score of log(price/mile) within equipment, fit on train only."""
    z = np.log(frame.posted_rate / frame.distance)
    medians = z.groupby(frame.equipment).median()
    center = frame.equipment.map(medians).astype(float)
    mad = (z-center).abs().groupby(frame.equipment).median()
    scale = frame.equipment.map(mad).astype(float) * 1.4826
    if (scale <= 0).any():
        raise ValueError('Degenerate training MAD')
    score = (z-center).abs() / scale
    keep = pd.Series(True,index=frame.index) if threshold is None else score.le(threshold)
    return keep, score, pd.DataFrame({'median_log_rpm':medians,'scaled_MAD':mad*1.4826})


def main():
    results, removed, bounds = [], [], []
    for number, (ti, vi) in enumerate(e.TimeSeriesCV(3).split(e.dev), 1):
        train, valid = e.dev.iloc[ti], e.dev.iloc[vi]
        # Keep unlabeled traffic inputs for preprocessing/history. Only the
        # supervised learner excludes rows; no validation target informs filtering.
        a,b=e.prepare(train,valid)
        for name, threshold in VARIANTS.items():
            keep,score,limits=training_mask(train,threshold)
            assert not set(train.index) & set(valid.index)
            assert len(b)==len(valid)
            metrics=e.fit_score(a.loc[keep],b,train.loc[keep],valid)
            row=dict(variant=name,fold=number,train_rows=len(train),removed=int((~keep).sum()),
                     removed_pct=float(100*(~keep).mean()),valid_rows=len(valid),**metrics)
            results.append(row)
            removed.append(train.loc[~keep,['load_id','equipment','posted_rate','distance']].assign(
                variant=name,fold=number,robust_z=score.loc[~keep]))
            bounds.append(limits.reset_index().assign(variant=name,fold=number,threshold=threshold))
            pd.DataFrame(results).to_csv(OUT/'fold-results.csv',index=False)
            print(row,flush=True)
    table=pd.DataFrame(results)
    summary=table.pivot(index='variant',columns='fold',values='MAPE_pct')
    summary['weighted_MAPE_pct']=summary[[1,2,3]].to_numpy() @ [.2,.3,.5]
    summary=summary.sort_values('weighted_MAPE_pct')
    summary.to_csv(OUT/'summary.csv')
    pd.concat(removed).to_csv(OUT/'removed-training-rows.csv',index=False)
    pd.concat(bounds).to_csv(OUT/'training-thresholds.csv',index=False)
    winner=summary.index[0]
    a,b=e.prepare(e.dev,e.test)
    holdout=[]
    for name in dict.fromkeys(['baseline',winner]):
        keep,_,_=training_mask(e.dev,VARIANTS[name])
        trees=int(table.loc[table.variant.eq(name),'trees'].median())
        holdout.append(dict(variant=name,rows=len(e.test),removed_train=int((~keep).sum()),
            **e.fit_score(a.loc[keep],b,e.dev.loc[keep],e.test,iterations=trees)))
    pd.DataFrame(holdout).to_csv(OUT/'holdout-results.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(variants=VARIANTS,params=e.params,
        selected_by_cv=winner,weights=[.2,.3,.5],preprocessing='All train covariates; fit model on retained targets',
        market_fit_scope=e.meta['market_fit_scope'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_train_sha256=hashlib.sha256((e.ROOT/'data/train-test.csv').read_bytes()).hexdigest()),indent=2))
    print(summary.to_string(),flush=True);print(holdout,flush=True)

if __name__=='__main__': main()
