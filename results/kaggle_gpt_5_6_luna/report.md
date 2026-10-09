# GPT-5.6 Luna WMT25 ESA pilot

## Run

- Kaggle model: `openai/gpt-5.6-luna` (display slug `gpt-5.6-luna`)
- Kaggle task version: 2; notebook run: `Run #1`, completed
- Sample: 100 English→Bhojpuri translations from 29 documents; 19 systems; two human annotations per translation
- Prompt calls: 200 total (100 feature-only, 100 context-aware); all succeeded
- Reported Model Proxy cost: **$0.075141**; tokens: 95,607 input and 46,683 output
- Exact frozen-sample digest: `c10f77eb89f19955b07ad87e529aaeef00f8d85ff35806ccc3c1b1a9c1f375d2`

The feature-only prompt sees only the 16 numeric error/length features. The context-aware prompt also sees the English source and Bhojpuri translation. Both use the same error annotation features and neither sees human scores, annotator IDs, or system names.

## Results

Primary comparison uses annotator B's independent human score, matching the existing local experiment. The document bootstrap resamples the 29 document groups. It is descriptive at this sample size.

| Model | MAE (95% doc CI) | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| mean_cross | 25.418 (22.530, 28.555) | 30.972 | -0.102 | -0.141 |
| fixed_penalty | 28.580 (21.526, 37.082) | 41.551 | -0.037 | 0.132 |
| ridge_cross_F3 | 20.350 (16.663, 24.060) | 26.198 | 0.540 | 0.349 |
| catboost_cross_F3 | 20.091 (16.174, 23.686) | 26.269 | 0.534 | 0.375 |
| Luna_feature_only | 20.930 (15.809, 25.834) | 31.434 | 0.452 | 0.333 |
| Luna_context_aware | 20.780 (17.871, 25.127) | 27.554 | 0.471 | 0.441 |

On this sample, the lowest MAE was **catboost_cross_F3** (20.09). See `metrics.csv` for the same models against annotator A and the mean of A/B. `paired_bootstrap.csv` gives paired document-bootstrap intervals for Luna versus cross-annotator CatBoost and between prompt conditions. `pairwise_ranking.csv` reports within-source ranking against the mean human score; ties in human scores are excluded and predicted ties count as half-correct.

## Error and annotator review

Mean A/B absolute disagreement was 22.50 points (median 17.50; 90th percentile 50.10). Context-aware model errors against annotator B are not directly the same quantity as human disagreement.

- B assigned 80+ while marking at least one major error in **22** translations.
- B assigned 50 or below with no B-marked errors in **0** translations.
- `error_examples.csv` contains the ten largest errors for each Luna condition, plus any high-score/major-error or low-score/zero-error contradictions, with source, translation, and both annotators' spans.
- Spearman correlation between context-aware absolute error vs B and A/B disagreement is **0.292**. The top ten context-aware errors average 32.50 points of A/B disagreement, versus 22.50 overall. The high-B/major-error subset has mean context-aware error 21.05 (n=22). These are descriptive checks, not causal evidence.
- `human_disagreement_analysis.csv` records those group summaries.

## Scope and interpretation

This is a 100-translation, one-language-pair pilot, not the full WMT25 ESA dataset. The comparison is not a head-to-head model training experiment: Luna judges the supplied sample, while Ridge and CatBoost are existing cross-annotator local baselines. Human annotation disagreement is substantial, so these results do not establish a stable replacement scoring layer. The context-aware condition is particularly different from an error-detector-only scorer because it can use the source and translation text.

The Kaggle creation-time notebook run already produced the completed `Run #1` artifacts. The separate `tasks run` queue remains empty; no second run was launched, to avoid repeating the 200 paid-budget prompts.
