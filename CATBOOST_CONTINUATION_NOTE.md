# CatBoost continuation check

Date: 2026-10-09

## Was training loss stored?

Yes. The saved CatBoost `.cbm` file for the pooled F3 cross-annotator model retains 453 per-tree training RMSE values, available through CatBoost's `get_evals_result()`. The existing selection table also records the chosen tree count and best validation RMSE. The final refit model did not retain its per-tree validation curve.

## Continuation experiment

Continued `models/experiment2/pooled_holdout_holdout_cross_catboost_F3.cbm` using the same training rows and parameters. Validation data controlled early stopping and model selection. The held-out test data did not control any choices. Split checks found no overlap among documents, leakage groups, translation IDs, or pairs.

This pooled cross-trained F3 model was chosen because its selected 453 trees were closest to the original 500-tree cap and it has a separate validation/test partition. The primary English-to-Bhojpuri cross-trained F3 folds selected 96-214 trees in the existing validation runs; those stopping points suggest less iteration headroom, though they were not separately continued here.

- Original: 453 trees; validation weighted RMSE 21.7316; test weighted MAE 18.1227 and RMSE 23.1107.
- Continued candidate: 513 trees (60 retained additional trees); validation weighted RMSE 21.7187; test weighted MAE 18.1120 and RMSE 23.1156.
- Test MAE difference: -0.0108 points; 95% document-bootstrap interval -0.0214 to -0.000005 (barely below zero).
- Test RMSE difference: +0.0049 points; 95% document-bootstrap interval -0.0063 to +0.0168.

Conclusion: continuing produced a tiny MAE gain, but slightly worsened RMSE, the training objective. This is no meaningful improvement for this objective, so the existing 453-tree model should be kept. The original model files were not changed; the candidate and all new artifacts are in `results/catboost_continuation/`.

## Artifacts

- `results/catboost_continuation/report.md`: results and interpretation.
- `results/catboost_continuation/loss_curve.png` and `loss_curve.csv`: loss curves.
- `results/catboost_continuation/metrics.csv`: weighted validation/test metrics.
- `results/catboost_continuation/test_delta_bootstrap.csv`: paired document bootstrap comparison.
- `results/catboost_continuation/continued_candidate.cbm`: experimental continuation model.
- `results/catboost_continuation/continue_catboost.py`: reproducible CPU-only script.

The code uses CatBoost's documented `init_model` continuation option and evaluation history API: https://catboost.ai/docs/en/concepts/python-reference_catboost_fit and https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_evals_result.
