# Training target outlier experiment

Remove rows from supervised training only; all validation and holdout rows remain scored. Raw assessment data and production artifacts are unchanged.

For each equipment category, calculate `z = log(posted_rate / distance)` on the training partition. Retain rows with `abs(z - median(z)) / (1.4826 * median(abs(z - median(z)))) <= threshold`. Thresholds 3, 4 and 5 are compared using the existing three temporal folds and weights 0.2 / 0.3 / 0.5. Equipment medians and MADs are refitted inside each training partition. Covariate-only preprocessing and city traffic histories retain all training inputs. The incumbent shared market curve uses known train/validation market inputs without price labels, as previously disclosed.

| Variant | Fold 1 MAPE | Fold 2 MAPE | Fold 3 MAPE | Weighted MAPE |
|---|---:|---:|---:|---:|
| Baseline | 6.2001 | 5.0063 | 4.7399 | 5.1119 |
| MAD threshold 3 | 7.1076 | 4.9064 | 4.8071 | 5.2970 |
| MAD threshold 4 | 6.8122 | 4.8601 | 4.7148 | 5.1779 |
| MAD threshold 5 | 6.4336 | 4.7629 | 4.6801 | 5.0557 |

The CV-selected threshold is 5. It removes 181 / 323 / 477 training rows from the three folds (1.55% / 1.39% / 1.38%). Only the selected variant and baseline are evaluated on the complete 2,183-row October holdout:

- Baseline: MAPE 5.5562%, median APE 1.8970%, 285 trees.
- Selected filter: MAPE 5.4658%, median APE 1.8770%, 299 trees; 639 of 45,817 development labels excluded (1.39%).

The weighted-CV gain is 0.0562 percentage points, and holdout gain is 0.0904 percentage points. This is a modest improvement, not evidence of a resolved outlier mechanism. Fold 1 worsens. Single-seed evaluation and a previously inspected holdout do not establish statistical robustness. The filter is a heuristic: some valid short-distance shipments may have high price per mile. It does not identify corrupted labels with certainty.

CatBoost hyperparameters remain fixed; iteration counts use the existing early-stopping procedure on validation log-RMSE. Holdout tree count is the median across the corresponding CV folds. No pooled OOF metric is used.

Run `python scripts/experiment_target_outliers.py`. CSVs record every fold result, fitted thresholds, excluded training IDs and holdout scores; the manifest records parameters and source/data hashes. This experiment does not replace the saved production model or submissions.
