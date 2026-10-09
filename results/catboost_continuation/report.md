# CatBoost continuation check

## What was saved already

The fitted CatBoost `.cbm` models retain a per-tree **training-set RMSE** curve, available through CatBoost's `get_evals_result()`. The project also saved validation-selected parameters, best validation RMSE, and selected tree count in `results/experiment2/model_selection.csv`. It did **not** save a per-tree validation curve for the final refit model.

The continuation starts from the existing pooled F3 cross-annotator model with 453 trees (`models\experiment2\pooled_holdout_holdout_cross_catboost_F3.cbm`). Its selected settings are depth 4, L2 3, learning rate 0.03, RMSE loss, and 4 CPU threads.

The primary English-to-Bhojpuri cross-trained F3 folds in the existing selection file stopped at 96–214 trees (maximum 500). I selected the pooled cross-trained F3 model for this check because it had the most iteration headroom, at 453 trees, and a separate validation/test partition.

## Method

Continued the saved model for up to 1000 additional trees using the exact original training rows. The existing validation partition controlled early stopping (patience 100) and selection. The original held-out test partition was not used for tuning. Document, leakage-group, translation, and pair IDs were checked for overlap across train, validation, and test.

| Split | Model | Trees | Weighted MAE | Weighted RMSE |
|---|---|---:|---:|---:|
| validation | saved model | 453 | 16.840 | 21.732 |
| validation | continued candidate | 513 | 16.810 | 21.719 |
| test | saved model | 453 | 18.123 | 23.111 |
| test | continued candidate | 513 | 18.112 | 23.116 |
| test | selected using validation | 513 | 18.112 | 23.116 |

**Validation decision:** `continued_candidate` was selected; the test set was not involved in that choice. The saved model's test metrics reproduce the existing report within tolerance.

On the test set, candidate-minus-baseline MAE was -0.011 (95% document-bootstrap CI -0.021411 to -0.000005); RMSE difference was 0.005 (CI -0.006324 to 0.016764). These paired intervals quantify the small test difference.

## Interpretation

The continuation improved validation RMSE, so it was selected before test evaluation. It added 60 retained trees; early stopping observed 160 continuation rounds before stopping and pruning. The test MAE gain was just 0.011 points (its paired interval only barely excludes zero), while test RMSE rose by 0.005 and its interval includes zero. For the RMSE objective, this is no meaningful held-out improvement; keeping the original 453-tree model is the sensible choice. This is one controlled continuation check of one pooled F3 model, not evidence that adding iterations will help every language pair or fold. A lower training loss alone does not imply better held-out quality.

## Artifacts

- `loss_curve.csv`: saved model training curve and continuation training/validation curves.
- `loss_curve.png`: RMSE curves for visual inspection.
- `metrics.csv`: baseline and continued candidate metrics on validation and test.
- `test_delta_bootstrap.csv` and `test_predictions.csv`: paired document bootstrap and auditable held-out predictions.
- `continued_candidate.cbm`: new candidate only; the existing model under `models/` was left unchanged.
- `run_summary.json`: split counts, selected tree counts, parameters, and validation decision.
- `continue_catboost.py`: reproducible CPU-only continuation script.
