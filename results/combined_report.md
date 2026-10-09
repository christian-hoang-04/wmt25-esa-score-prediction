# Human-Aligned MT Scoring — Combined Results

## Experiment 2: independent-annotator generalization

The paired dataset contains **33,049 translations** across **14 ESA language pairs**. The Bhojpuri five-fold out-of-fold set contains 2,300 paired translations; its held-out fold document counts were 1, 10, 13, 12, and 14, so fold-level uncertainty is uneven. Pooled ESA evaluation uses 4,294 paired translations in the grouped test partition. No source document, duplicate-source leakage group, reciprocal pair, or translation crossed splits.

| experiment_scope | language_pair | regime | model | feature_set | n_translations | mae | mae_ci_low | mae_ci_high | rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en_bho_cv | en-bho_IN | same_target | mean | none | 2300.000 | 24.908 | 24.000 | 26.224 | 30.189 | -0.038 |
| en_bho_cv | en-bho_IN | cross_trained | mean | none | 2300.000 | 24.908 | 24.000 | 26.253 | 30.189 | -0.038 |
| en_bho_cv | en-bho_IN | same_target | fixed_penalty | none | 2300.000 | 22.251 | 20.095 | 24.040 | 35.384 | 0.562 |
| en_bho_cv | en-bho_IN | frozen_cross | fixed_penalty | none | 2300.000 | 24.949 | 23.271 | 27.120 | 36.751 | 0.282 |
| en_bho_cv | en-bho_IN | same_target | ridge | F3 | 2300.000 | 12.267 | 10.937 | 13.221 | 16.249 | 0.825 |
| en_bho_cv | en-bho_IN | frozen_cross | ridge | F3 | 2300.000 | 18.650 | 17.041 | 19.778 | 24.723 | 0.487 |
| en_bho_cv | en-bho_IN | cross_trained | ridge | F3 | 2300.000 | 18.862 | 17.314 | 19.968 | 23.366 | 0.490 |
| en_bho_cv | en-bho_IN | same_target | catboost | F3 | 2300.000 | 10.707 | 9.405 | 11.616 | 15.184 | 0.849 |
| en_bho_cv | en-bho_IN | frozen_cross | catboost | F3 | 2300.000 | 19.372 | 17.561 | 20.785 | 25.906 | 0.455 |
| en_bho_cv | en-bho_IN | cross_trained | catboost | F3 | 2300.000 | 18.723 | 17.310 | 19.739 | 23.263 | 0.488 |
| pooled_holdout | ALL_ESA | same_target | mean | none | 4294.000 | 23.873 | 21.800 | 26.370 | 30.151 | — |
| pooled_holdout | ALL_ESA | cross_trained | mean | none | 4294.000 | 23.873 | 21.703 | 26.756 | 30.151 | — |
| pooled_holdout | ALL_ESA | same_target | fixed_penalty | none | 4294.000 | 25.229 | 21.763 | 29.307 | 37.425 | 0.685 |
| pooled_holdout | ALL_ESA | frozen_cross | fixed_penalty | none | 4294.000 | 26.841 | 23.537 | 30.729 | 38.674 | 0.326 |
| pooled_holdout | ALL_ESA | same_target | ridge | F3 | 4294.000 | 12.217 | 11.306 | 13.295 | 16.811 | 0.783 |
| pooled_holdout | ALL_ESA | frozen_cross | ridge | F3 | 4294.000 | 18.643 | 17.512 | 19.782 | 24.825 | 0.452 |
| pooled_holdout | ALL_ESA | cross_trained | ridge | F3 | 4294.000 | 18.476 | 17.564 | 19.503 | 23.308 | 0.456 |
| pooled_holdout | ALL_ESA | same_target | catboost | F3 | 4294.000 | 11.708 | 10.931 | 12.653 | 16.023 | 0.799 |
| pooled_holdout | ALL_ESA | frozen_cross | catboost | F3 | 4294.000 | 18.457 | 17.335 | 19.512 | 24.807 | 0.458 |
| pooled_holdout | ALL_ESA | cross_trained | catboost | F3 | 4294.000 | 18.123 | 17.199 | 19.043 | 23.111 | 0.461 |
| en_bho_cv | en-bho_IN | human_baseline | input_annotator_a | human_score_a | 2300.000 | 22.704 | 20.098 | 24.823 | 30.445 | 0.387 |
| pooled_holdout | ALL_ESA | human_baseline | input_annotator_a | human_score_a | 4294.000 | 17.887 | 16.661 | 19.142 | 24.748 | 0.488 |

Pooled-language held-out results are shown where sample sizes allow:

| language_pair | model | n_translations | n_unique_documents | mae | mae_ci_low | mae_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| cs-de_DE | ridge | 382.000 | 19.000 | 18.172 | 16.579 | 20.308 |
| cs-de_DE | catboost | 382.000 | 19.000 | 17.867 | 16.494 | 19.717 |
| cs-uk_UA | ridge | 298.000 | 19.000 | 14.298 | 13.568 | 15.194 |
| cs-uk_UA | catboost | 298.000 | 19.000 | 14.318 | 13.483 | 15.132 |
| en-ar_EG | ridge | 235.000 | 10.000 | 22.787 | 19.971 | 24.570 |
| en-ar_EG | catboost | 235.000 | 10.000 | 20.566 | 18.678 | 22.361 |
| en-cs_CZ | ridge | 617.000 | 14.000 | 15.464 | 14.657 | 16.414 |
| en-cs_CZ | catboost | 617.000 | 14.000 | 14.949 | 13.883 | 15.849 |
| en-et_EE | ridge | 452.000 | 10.000 | 21.809 | 19.308 | 24.609 |
| en-et_EE | catboost | 452.000 | 10.000 | 21.614 | 19.286 | 24.241 |
| en-is_IS | ridge | 348.000 | 10.000 | 21.397 | 20.009 | 22.525 |
| en-is_IS | catboost | 348.000 | 10.000 | 22.417 | 21.179 | 23.375 |
| en-it_IT | ridge | 338.000 | 10.000 | 20.346 | 18.826 | 22.459 |
| en-it_IT | catboost | 338.000 | 10.000 | 20.393 | 18.798 | 22.903 |
| en-ja_JP | ridge | 427.000 | 9.000 | 15.255 | 14.735 | 15.587 |
| en-ja_JP | catboost | 427.000 | 9.000 | 14.897 | 14.464 | 15.250 |
| en-mas_KE | ridge | 128.000 | 10.000 | 21.761 | 19.882 | 22.695 |
| en-mas_KE | catboost | 128.000 | 10.000 | 23.052 | 19.980 | 25.532 |
| en-ru_RU | ridge | 267.000 | 10.000 | 19.235 | 17.166 | 22.832 |
| en-ru_RU | catboost | 267.000 | 10.000 | 19.016 | 16.813 | 23.013 |
| en-sr_Cyrl_RS | ridge | 286.000 | 9.000 | 26.575 | 23.943 | 29.423 |
| en-sr_Cyrl_RS | catboost | 286.000 | 9.000 | 24.515 | 23.040 | 26.231 |
| en-uk_UA | ridge | 233.000 | 10.000 | 10.272 | 9.343 | 11.098 |
| en-uk_UA | catboost | 233.000 | 10.000 | 9.784 | 9.307 | 10.317 |
| en-zh_CN | ridge | 283.000 | 9.000 | 16.357 | 15.276 | 18.895 |
| en-zh_CN | catboost | 283.000 | 9.000 | 15.997 | 15.011 | 18.063 |
| cs-de_DE | input_annotator_a | 382.000 | 19.000 | 16.613 | 14.092 | 19.438 |
| cs-uk_UA | input_annotator_a | 298.000 | 19.000 | 14.154 | 12.255 | 15.922 |
| en-ar_EG | input_annotator_a | 235.000 | 10.000 | 7.204 | 5.143 | 11.482 |
| en-cs_CZ | input_annotator_a | 617.000 | 14.000 | 17.280 | 15.589 | 18.690 |
| en-et_EE | input_annotator_a | 452.000 | 10.000 | 22.308 | 19.125 | 26.285 |
| en-is_IS | input_annotator_a | 348.000 | 10.000 | 19.187 | 17.462 | 21.428 |
| en-it_IT | input_annotator_a | 338.000 | 10.000 | 25.550 | 22.737 | 28.933 |
| en-ja_JP | input_annotator_a | 427.000 | 9.000 | 20.077 | 17.641 | 21.109 |
| en-mas_KE | input_annotator_a | 128.000 | 10.000 | 4.789 | 2.754 | 7.941 |
| en-ru_RU | input_annotator_a | 267.000 | 10.000 | 21.363 | 19.274 | 24.975 |
| en-sr_Cyrl_RS | input_annotator_a | 286.000 | 9.000 | 19.455 | 16.772 | 22.619 |
| en-uk_UA | input_annotator_a | 233.000 | 10.000 | 8.262 | 7.308 | 9.885 |
| en-zh_CN | input_annotator_a | 283.000 | 9.000 | 21.594 | 20.270 | 25.028 |

### Feature ablation

| model | feature_set | n_translations | mae | mae_ci_low | mae_ci_high | rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | F1 | 4294.000 | 23.084 | 20.923 | 25.908 | 29.684 | 0.308 |
| ridge | F2 | 4294.000 | 18.573 | 17.658 | 19.589 | 23.422 | 0.458 |
| ridge | F3 | 4294.000 | 18.476 | 17.564 | 19.503 | 23.308 | 0.456 |
| catboost | F1 | 4294.000 | 21.536 | 19.868 | 23.384 | 27.294 | 0.371 |
| catboost | F2 | 4294.000 | 18.083 | 17.187 | 19.155 | 22.957 | 0.467 |
| catboost | F3 | 4294.000 | 18.123 | 17.199 | 19.043 | 23.111 | 0.461 |

Human absolute score disagreement is summarized below. These human-to-human differences provide noise context; they are not directly identical to model-to-one-rater MAE.

| language_pair | translations | independent_annotator_pairs | directed_rows | weighted_mean_abs_difference | weighted_median_abs_difference | weighted_p90_abs_difference |
| --- | --- | --- | --- | --- | --- | --- |
| en-bho_IN | 2300.000 | 2300.000 | 4600.000 | 22.704 | 18.000 | 51.000 |
| ALL_ESA | 33049.000 | 33049.000 | 66098.000 | 18.343 | 13.000 | 43.000 |

## Experiment 3: frozen 100-item sample

The local matched results against independent annotator B are:

| model | n_translations | n_documents | mae | mae_ci_low | mae_ci_high | rmse | within_10_pct | delta_mae_vs_matched_catboost | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 100.000 | 29.000 | 25.418 | 22.368 | 28.555 | 30.972 | 14.000 | 5.327 | 3.025 | 8.538 |
| ridge_cross_F3 | 100.000 | 29.000 | 20.350 | 16.515 | 24.392 | 26.198 | 28.000 | 0.259 | -0.546 | 1.089 |
| catboost_cross_F3 | 100.000 | 29.000 | 20.091 | 16.311 | 23.723 | 26.269 | 29.000 | 0.000 | 0.000 | 0.000 |
| catboost_same_F3 | 100.000 | 29.000 | 21.988 | 17.843 | 25.283 | 30.432 | 40.000 | 1.897 | -0.106 | 3.700 |
| human_input_annotator_a | 100.000 | 29.000 | 22.500 | 19.362 | 25.720 | 30.551 | 41.000 | 2.409 | -0.729 | 5.911 |

The same local baselines against input annotator A are:

| model | n_translations | n_documents | mae | mae_ci_low | mae_ci_high | rmse | within_10_pct | delta_mae_vs_matched_catboost | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 100.000 | 29.000 | 24.299 | 20.392 | 29.105 | 29.668 | 20.000 | 14.323 | 10.294 | 19.727 |
| ridge_cross_F3 | 100.000 | 29.000 | 14.801 | 12.599 | 17.242 | 16.777 | 25.000 | 4.826 | 3.310 | 6.491 |
| catboost_cross_F3 | 100.000 | 29.000 | 16.085 | 14.175 | 18.298 | 18.225 | 18.000 | 6.109 | 4.565 | 7.818 |
| catboost_same_F3 | 100.000 | 29.000 | 9.975 | 8.137 | 11.431 | 14.024 | 62.000 | 0.000 | 0.000 | 0.000 |

Frontier API status: **request_error**; provider `gemini`, model `gemini-3.8-flash`, sanitized error category `rate_or_quota_limit`, HTTP status `429`. Synthetic-only calls: 3. Token-price estimate from observed usage: **$0.000019**. No sample source/translation text was sent; no LLM scores are fabricated. The unavailable rows in `experiment3/metrics.csv` document the blocked comparisons.

## Answers to the research questions

1. **Can human scores be predicted from error annotations?** Yes, to a useful but imperfect degree when training and evaluating on independent annotators: pooled cross-trained CatBoost F3 MAE is **18.12** points, versus **17.89** for using input annotator A's score as a human-to-human reference. This result conditions on gold human error annotations.
2. **Does coverage improve prediction?** Yes from counts-only to counts plus coverage for pooled CatBoost (F1 MAE **21.54**, F2 **18.08**); the full feature set is **18.12**, so text lengths and derived features add little beyond coverage in this pooled comparison.
3. **Does CatBoost beat a linear mapping?** The pooled CatBoost F3 MAE is **18.12** versus Ridge **18.48**. This is a small descriptive advantage; it does not establish a robust or statistically decisive nonlinear gain.
4. **How much annotator disagreement is there?** Across ESA pairs, the mean absolute difference is about **18.34** points (median 13, p90 43); for English→Bhojpuri it is about **22.70** (median 18, p90 51). On the 100-item sample, A/B disagreement has mean 22.50, median 17.50, and p90 50.10.
5. **Which annotations are hardest?** The largest held-out CatBoost residuals and high-disagreement or score/error-contradiction cases are in `experiment2/failure_cases.csv` and `experiment3/failure_cases.csv`. Current pooled failure-case category counts: `{'model_close_to_a_far_from_b': 811, 'model_closer_to_b_than_a': 654, 'high_score_with_major_errors': 372, 'high_annotator_disagreement': 252, 'largest_model_error': 11, 'low_score_no_marked_errors': 2}`. The sample CSV includes source, translation, both span lists, both scores, and local prediction for inspection.
6. **Is the learned function stable after an LLM error detector?** Not established. These models consume human error spans, not detector-generated annotations. The frontier score-judge comparison also has no result because its synthetic preflight hit a 429 rate/quota limit; the score layer needs a later end-to-end evaluation with detector outputs before use as a scoring layer.

## Artifacts

- [Experiment 2 report](experiment2/report.md), [metrics](experiment2/metrics.csv), [ablations](experiment2/ablations.csv), and [failure cases](experiment2/failure_cases.csv).
- [Experiment 3 report](experiment3/report.md), [local metrics](experiment3/metrics_local.csv), [comparison](experiment3/comparison.csv), [hosted score status](experiment3/llm_predictions.csv), and [API cost log](experiment3/api_costs.csv).
- Prompt payloads: `experiment3/prompts/`; plots: `experiment3/plots/`.
- Frozen sample IDs and digest: [sample100.csv](experiment3/sample100.csv) and [manifest](experiment3/sample100_manifest.json).
