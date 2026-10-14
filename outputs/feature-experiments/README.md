# Feature ablation experiment

All variants use the existing three temporal folds, log-target CatBoost, fixed incumbent hyperparameters, and weights 0.2 / 0.3 / 0.5. Early stopping uses validation log-RMSE, as in the incumbent pipeline. No pooled OOF metric is calculated.

The baseline exactly reproduces weighted CV MAPE 5.1118895563% and October holdout MAPE 5.5562156389%. None of 12 feature variants improves CV. Production models, notebooks, and prediction files are unchanged.

| Variant | Fold 1 | Fold 2 | Fold 3 | Weighted MAPE (%) |
|---|---:|---:|---:|---:|
| baseline | 6.2001 | 5.0063 | 4.7399 | 5.1119 |
| direction | 6.4887 | 4.9744 | 4.8835 | 5.2318 |
| route_load | 6.0583 | 5.1788 | 4.9407 | 5.2357 |
| weight_capacity_fraction | 6.3338 | 5.0679 | 4.9135 | 5.2439 |
| extra_road_miles | 6.2154 | 5.0926 | 4.9911 | 5.2664 |
| road_speed | 6.4508 | 5.1386 | 4.8952 | 5.2793 |
| ton_miles | 6.6004 | 5.2020 | 4.8380 | 5.2997 |
| weight_per_mile | 6.4604 | 5.1053 | 4.9552 | 5.3013 |
| lane_pricing | 6.2949 | 5.1063 | 5.0463 | 5.3140 |
| city_pricing | 6.5293 | 5.2212 | 4.8993 | 5.3219 |
| quote_total_only | 10.9363 | 5.1042 | 4.9287 | 6.1829 |
| quote_interactions | 11.1382 | 5.2588 | 5.3178 | 6.4641 |
| all | 11.0418 | 5.4544 | 5.3663 | 6.5278 |

Historical pricing features use smoothed log(rate / distance), grouped by endpoint city or directed lane (with/without equipment). Training features use strictly earlier dates, excluding same-day targets; validation mappings use training targets only. The prior is fixed at log(2 USD/mile), with smoothing strength 30. Focused checks confirmed that current/future-day and validation labels cannot affect these encodings.

The market curve retains the disclosed incumbent scope: known train plus supplied-validation market inputs, with no price targets. The 14-day holdout is evaluated only for the CV-selected variant (baseline in this run); it is not used to select feature variants. It has previously been inspected, so it is not a pristine final test.

This is evidence against these particular feature definitions with the current parameters, not proof that feature engineering cannot help. Additional tuning might change the comparison. The capacity fraction uses an illustrative 45,000 lb denominator and is simply a rescaling of weight.

Reproduce from the repository root:

```bash
python scripts/experiment_features.py
```

The script runs fresh and writes fold metrics, the ranked summary, holdout metrics and a configuration manifest in this directory. It requires the original assessment CSVs and the saved preprocessing bundle.
