# Post-hoc analysis of large errors

This diagnostic uses the October holdout after feature selection. It must not be interpreted as fresh model-selection evidence. No production artifacts were changed.

Baseline: 2,183 rows, MAPE 5.5562%. 2,138 rows have APE <=10% (mean 2.1632%); 38 rows have APE >50% and contribute 61.24% of total APE. The remaining seven rows have errors from 10.04% to 12.52%.

Among severe errors, 22 are overpredictions: median actual/prediction 0.248 and mean APE 286.16%. Sixteen are underpredictions: median actual/prediction 3.225 and mean APE 70.76%. MAPE amplifies the cheap-price cases because it divides by the actual rate.

All 38 severe cases have historical same-direction examples; 36 also have matching equipment (21 overpredictions and 15 underpredictions). The two unmatched equipment combinations are Syracuse to Raleigh / Flatbed and Corpus Christi to Las Vegas / Flatbed. Match up to five historical loads using distance, weight, quote and market (scaled by development IQR), then compare price per mile. The median actual/peer price-per-mile ratio is 0.238 for severe overpredictions and 3.185 for underpredictions. Peer samples are small and historical conditions can differ; these are comparisons, not counterfactual truth.

The five worst cases have actual/peer ratios 0.192-0.212. Their weights and observed market/quote inputs do not exhibit a similarly large deviation. Severe cases occur throughout the holdout and across all equipment types. This supports investigating an unobserved pricing factor, special conditions or target anomalies; it does not prove random noise or corrupted labels.

New features mostly worsen ordinary predictions: ordinary-row MAPE changes from 2.163% to 2.361% (direction), 2.356% (route/load), 2.436% (historical lane pricing), and 2.621% (quote interactions). Severe errors barely improve or worsen. Thus aggregate gains are absent: the added features do not explain the large price changes and perturb already accurate predictions.

A separate smaller issue affects short-distance trips: six of seven moderate errors have distance <=204 miles, with median 71.6 miles. Some named routes have large discrepancies between supplied distance and HERE city-center distance. This is a concrete subgroup for future investigation, but the subgroup was found post hoc and needs confirmation in temporal CV.

Files: cases.csv contains per-row inputs, predictions, errors and historical match IDs; groups.csv compares errors by baseline-defined group; input-medians.csv and category-rates.csv describe input differences. Reproduce with `python scripts/analyze_error_cases.py` after running the feature experiment. No pooled OOF calculation is performed.
