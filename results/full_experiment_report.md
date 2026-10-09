# WMT25 ESA human-score prediction: consolidated experiment report

**Status:** completed report of the experiments already run. This document consolidates saved outputs; it does not launch training, Kaggle calls, or other model requests.

## Executive summary

We tested whether numeric features from human Error Span Annotations (ESA) can predict WMT25 human translation quality scores. We first used the same annotation’s spans and score, then ran a stricter experiment in which a model saw one annotator’s spans and predicted an independent annotator’s score. We also evaluated GPT-5.6 Luna through Kaggle on a frozen 100-translation sample and tested whether extending a saved CatBoost model improved held-out results.

The clearest result is that **error counts alone are weak, but span coverage carries much more signal**. With gold spans and same-annotation targets, CatBoost F3 reached MAE 9.92 on English→Bhojpuri and 11.31 on pooled ESA. On independent-annotator targets, cross-trained CatBoost F3 MAE was 18.72 and 18.12 respectively. Luna’s best result on the 100-item sample was MAE 20.78, close to but not better than the local CatBoost baseline’s 20.09. Continuing the pooled CatBoost model added 60 trees but changed test MAE by only −0.011 and slightly worsened RMSE.

These are proof-of-concept results using **human-marked gold error spans**. They do not establish performance when an LLM or another error detector supplies the spans.

## What was run

| Phase | Data and question | Outcome |
|---|---|---|
| 1. ESA-to-score prediction | All valid annotations in the 14 WMT25 ESA directions; English→Bhojpuri grouped cross-validation and a pooled document-group holdout. Compared mean, fixed penalty, Ridge, and CatBoost with F1/F2/F3 feature sets. | Coverage features sharply improved prediction. CatBoost slightly beat Ridge with full numeric features. |
| 2. Independent-annotator generalization | Paired annotator views of the same translations; trained on annotator A’s spans and scores or A’s spans/B’s scores, then evaluated against B. | Error-span mapping generalized less well to another annotator’s score; score calibration and disagreement matter. |
| 3. Frozen-sample judge pilots | Existing 100-item English→Bhojpuri sample. First, local baselines and a hosted-judge preflight; later, GPT-5.6 Luna on Kaggle in feature-only and context-aware conditions. | Kaggle completed all 200 prompts. Luna was close to the local learned baselines, without a clear win. An earlier separate hosted-judge preflight stopped on a 429 quota/rate response before any sample text was sent. |
| 4. CatBoost continuation | Continued the existing pooled cross-annotator F3 CatBoost model using its original train/validation/test partitions. | Validation improved slightly; held-out changes were negligible. The original model was retained. |

## Data and preparation

The source was the [official WMT25 General MT human-evaluation dataset](https://github.com/wmt-conference/wmt25-general-mt/blob/main/README.md#human-evaluation-data). We included only the 14 directions designated ESA and excluded MQM-only directions. Every valid human annotation was kept as a separate example for the initial experiment; annotation scores were not averaged for model fitting. Mean scores were used only where a translation-level ranking target was needed.

The parser was checked against the downloaded JSONL. Some error records contained the literal string `"missing"` for span offsets despite having a severity. Such an annotation has an unknown span, so it was reported and excluded rather than treated as error-free. Other invalid spans and unsupported severities were handled the same way. In total, **8,992 annotations** were excluded across ESA directions:

| Invalid error record reason | Count |
|---|---:|
| Missing/unknown offsets | 8,199 |
| Out-of-bounds or reversed offsets | 704 |
| Unsupported severity | 239 |

Coverage was calculated from unions of inclusive character intervals, so overlapping spans were not double-counted. Predictors were numeric error and length features only. Annotator ID, translation-system name, document ID, and raw text were retained for analysis/grouping but excluded from the initial models. Language pair and document group were parsed from IDs without relying on a single rigid ID pattern.

### Main annotation population by direction

Counts below are valid examples retained for the main model experiment; the full score/error distributions are in [dataset_summary.json](dataset_summary.json) and the detailed [main report](report.md).

| Language pair | Segments | Documents | Systems | Annotations | Annotators | Mean score | Score SD | Zero-error annotations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cs-de_DE | 231 | 128 | 21 | 8,655 | 12 | 83.93 | 19.89 | 27.58% |
| cs-uk_UA | 225 | 131 | 19 | 8,425 | 7 | 87.33 | 15.97 | 45.32% |
| en-ar_EG | 199 | 51 | 19 | 7,332 | 3 | 30.31 | 37.48 | 13.30% |
| en-bho_IN | 189 | 50 | 19 | 6,698 | 4 | 71.37 | 30.78 | 17.84% |
| en-cs_CZ | 212 | 60 | 20 | 7,545 | 17 | 78.58 | 19.20 | 37.97% |
| en-et_EE | 190 | 52 | 19 | 6,860 | 8 | 58.12 | 25.11 | 15.04% |
| en-is_IS | 202 | 53 | 19 | 6,978 | 5 | 45.80 | 28.74 | 9.72% |
| en-it_IT | 215 | 54 | 18 | 7,249 | 7 | 69.80 | 26.83 | 30.69% |
| en-ja_JP | 189 | 51 | 19 | 6,691 | 40 | 80.88 | 19.89 | 57.47% |
| en-mas_KE | 185 | 49 | 19 | 6,168 | 4 | 0.87 | 4.03 | 0.05% |
| en-ru_RU | 200 | 52 | 19 | 6,659 | 6 | 72.05 | 24.93 | 30.29% |
| en-sr_Cyrl_RS | 206 | 50 | 19 | 7,061 | 4 | 79.40 | 24.07 | 20.34% |
| en-uk_UA | 199 | 52 | 19 | 6,784 | 2 | 87.10 | 10.19 | 62.38% |
| en-zh_CN | 198 | 51 | 19 | 6,741 | 39 | 84.51 | 18.44 | 51.89% |

The pooled primary split contains 70,465 training, 14,974 validation, and 14,407 test annotations, grouped over 884 documents and 254 exact-source leakage groups. Shared exact source texts were assigned together. English→Bhojpuri has 50 documents and used five-fold GroupKFold out-of-fold predictions. Feature scaling was fit on training data, and model selection used validation data only.

## Phase 1: predicting a score from the same annotation’s error features

F1 uses only major/minor error counts. F2 adds minor, major, and total error coverage. F3 adds all requested numeric features, including lengths, span-size summaries, density, major fraction, and binary indicators. The fixed penalty is the requested heuristic, `clip(100 − 5 × n_major − n_minor, 0, 100)`; it is not the ESA scoring protocol.

| Scope / model | Features | N annotations | MAE | RMSE | Pearson | Spearman | Pairwise rank accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Bho CV — mean | — | 6,698 | 25.264 | 30.796 | −0.043 | −0.048 | 0.500 |
| Bho CV — fixed penalty | counts | 6,698 | 21.840 | 35.721 | 0.259 | 0.553 | 0.701 |
| Bho CV — Ridge | F1 | 6,698 | 23.860 | 29.867 | 0.242 | 0.518 | 0.698 |
| Bho CV — CatBoost | F1 | 6,698 | 19.335 | 25.276 | 0.571 | 0.631 | 0.739 |
| Bho CV — Ridge | F2 | 6,698 | 11.616 | 15.621 | 0.862 | 0.834 | 0.849 |
| Bho CV — CatBoost | F2 | 6,698 | 10.169 | 14.602 | 0.880 | 0.848 | 0.855 |
| **Bho CV — Ridge** | **F3** | **6,698** | **11.440** | **15.505** | **0.864** | **0.837** | **0.847** |
| **Bho CV — CatBoost** | **F3** | **6,698** | **9.921** | **14.281** | **0.886** | **0.858** | **0.857** |
| Pooled test — mean | — | 14,407 | 28.217 | 34.684 | — | — | 0.500 |
| Pooled test — fixed penalty | counts | 14,407 | 30.526 | 44.767 | 0.270 | 0.625 | 0.734 |
| Pooled test — Ridge | F1 | 14,407 | 26.080 | 33.441 | 0.271 | 0.612 | 0.732 |
| Pooled test — CatBoost | F1 | 14,407 | 19.299 | 25.861 | 0.670 | 0.700 | 0.738 |
| Pooled test — Ridge | F2 | 14,407 | 12.668 | 17.312 | 0.866 | 0.830 | 0.788 |
| Pooled test — CatBoost | F2 | 14,407 | 11.276 | 15.954 | 0.888 | 0.841 | 0.800 |
| Pooled test — Ridge | F3 | 14,407 | 11.853 | 16.438 | 0.880 | 0.826 | 0.793 |
| **Pooled test — CatBoost** | **F3** | **14,407** | **11.314** | **15.747** | **0.891** | **0.836** | **0.801** |

Pairwise accuracy is measured within source segments, after averaging independent scores for each translation for this ranking calculation only. The Bho comparison has 31,029 comparable pairs; pooled has 61,837. Document-bootstrap 95% intervals for CatBoost F3 MAE are 9.277–10.432 in Bho CV and 10.538–12.186 on pooled test.

### Ablation finding

Coverage produced the largest gain. CatBoost F1→F2 reduced MAE by 9.17 points on Bho and 8.02 pooled; Ridge improved by 12.24 and 13.41 points. The F3 extras yielded a small additional Bho improvement for CatBoost (10.169→9.921), while pooled F2 was marginally better than F3 (11.276 vs 11.314). Thus the full set is not uniformly best; coverage is the consistently important feature group.

On the pooled held-out test, per-language CatBoost F3 MAE ranged from 4.384 (en-uk_UA) to 20.392 (en-sr_Cyrl_RS). The direction-specific score scales and target distributions vary substantially; the full 14-direction table is in [results/report.md](report.md). In particular, en-mas_KE’s scores are concentrated near zero, so correlation/ranking should be read alongside its MAE and score distribution.

## Phase 2: independent-annotator generalization

This evaluation pairs distinct annotators’ judgments of the same translation. The paired population contains 33,049 translations across 14 ESA directions and 66,098 directed input/target rows. Bho has 2,300 paired translations; pooled held-out evaluation has 4,294. Repeated judgments by one annotator were collapsed within a translation before forming distinct-annotator pairs. Each translation’s total pair weight was normalized so translations with more annotators did not dominate. Documents, exact-source groups, translations, and reciprocal pairs were kept out of train/test overlap.

Three regimes distinguish training from evaluation: **same-target** trains and evaluates against the input annotator’s own score; **frozen-cross** applies that model to another annotator’s score; **cross-trained** directly trains using annotator A’s features and annotator B’s score. The last is the more relevant independent-rater test.

| Scope / regime | Model | Features | MAE (95% document CI) | RMSE | Spearman |
|---|---|---|---:|---:|---:|
| Bho — same-target | CatBoost | F3 | 10.707 (9.405–11.616) | 15.184 | 0.849 |
| Bho — frozen-cross | CatBoost | F3 | 19.372 (17.561–20.785) | 25.906 | 0.455 |
| Bho — cross-trained | Ridge | F3 | 18.862 (17.314–19.968) | 23.366 | 0.490 |
| **Bho — cross-trained** | **CatBoost** | **F3** | **18.723 (17.310–19.739)** | **23.263** | **0.488** |
| Pooled — same-target | CatBoost | F3 | 11.708 (10.931–12.653) | 16.023 | 0.799 |
| Pooled — frozen-cross | CatBoost | F3 | 18.457 (17.335–19.512) | 24.807 | 0.458 |
| Pooled — cross-trained | Ridge | F3 | 18.476 (17.564–19.503) | 23.308 | 0.456 |
| **Pooled — cross-trained** | **CatBoost** | **F3** | **18.123 (17.199–19.043)** | **23.111** | **0.461** |
| Human A score → human B score | — | — | 22.704 Bho; 17.887 pooled | 30.445; 24.748 | 0.387; 0.488 |

CatBoost’s cross-trained advantage over Ridge is small: 0.139 MAE points in Bho and 0.353 pooled. The same-target to frozen-cross gap is much larger: 8.665 points in Bho and 6.749 pooled for CatBoost F3. This indicates that predicting the score of a different rater is materially harder than predicting the score attached to the same spans, and that rater calibration is important.

The cross-trained CatBoost F1→F2 gain was 3.86 MAE points in Bho and 3.45 pooled; F3 changed pooled MAE slightly upward from 18.083 to 18.123. Again, coverage matters more consistently than the remaining derived features.

## Annotator agreement and difficult examples

For translations with multiple independent scores, all distinct-annotator score differences were measured. Across ESA, 33,049 translations had multiple annotators: mean absolute difference **18.343**, median **13**, and 90th percentile **43**. For Bho, the corresponding figures were **22.704**, **18**, and **51**. These are human-to-human differences, while model MAE compares a prediction with one rater; they are useful context, not identical measures.

The initial Bho error review found **5** annotations with score ≤50 and no marked errors, and **1,592** annotations with score ≥80 despite at least one major error. Five of the ten largest CatBoost residuals were low-score/no-error cases, and one had high score despite major errors. The top error table in the detailed report includes examples such as a score of 0 with predicted 98.02 on a zero-error annotation, and score 88 with predicted 27.91 when a major error was marked. These contradictions expose a limit of features derived only from spans: the scorer cannot infer an unmarked problem or know why a marked span coexists with a high holistic score.

Review the source, translation, span lists, human score, prediction, and residual in [results/error_analysis.csv](error_analysis.csv). The independent-rater counterpart is [experiment2/failure_cases.csv](experiment2/failure_cases.csv). The saved plots are [prediction vs human](plots/predicted_vs_human.png), [score by error counts](plots/score_by_error_counts.png), and [CatBoost feature importance](plots/catboost_feature_importance.png).

## Phase 3: 100-item score-judge pilots

### Local baselines and earlier hosted-judge preflight

Before the Kaggle run, local methods were compared on the same frozen 100 English→Bhojpuri translations (29 documents, 19 systems). Against independent annotator B, mean predictor MAE was 25.418, Ridge cross-trained F3 20.350, and CatBoost cross-trained F3 20.091. The small test had wide document-bootstrap intervals. A separate hosted-judge preflight made three synthetic-only calls; it stopped on an HTTP 429 rate/quota response. **No source or translation text was sent in that preflight and no hosted scores were made up.**

### GPT-5.6 Luna on Kaggle

At the user’s direction, GPT-5.6 Luna was run through Kaggle Benchmarks on the same frozen sample. All **200** prompts completed: 100 feature-only prompts using numeric features and 100 context-aware prompts also given source and translation text. The requests did not include human scores, annotator IDs, or system names. Reported Kaggle Model Proxy usage was **$0.075141**, 95,607 input tokens, and 46,683 output tokens.

| Model / condition | MAE vs annotator B (95% document CI) | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Mean predictor | 25.418 (22.530–28.555) | 30.972 | −0.102 | −0.141 |
| Fixed penalty | 28.580 (21.526–37.082) | 41.551 | −0.037 | 0.132 |
| Ridge cross-trained F3 | 20.350 (16.663–24.060) | 26.198 | 0.540 | 0.349 |
| CatBoost cross-trained F3 | **20.091 (16.174–23.686)** | 26.269 | 0.534 | 0.375 |
| Luna feature-only | 20.930 (15.809–25.834) | 31.434 | 0.452 | 0.333 |
| Luna context-aware | 20.780 (17.871–25.127) | 27.554 | 0.471 | 0.441 |

Luna clearly beat the fixed penalty on this sample, but the point estimates did not beat CatBoost. Paired document-bootstrap intervals for Luna-vs-CatBoost differences include zero; the feature-only and context-aware conditions also do not differ clearly. Ranking accuracy was 78.3% for either Luna condition, vs 83.3% Ridge and 76.7% CatBoost, but there were only 30 comparable pairs across 14 source groups. Treat these as pilot estimates, not a robust ranking.

Mean A/B disagreement on the sample was 22.50 points (median 17.50; p90 50.10). Context-aware error had Spearman correlation 0.292 with A/B disagreement. Its ten largest errors averaged 32.50 points of annotator disagreement. These are descriptive patterns from a small sample, not causal evidence.

The run used Kaggle model `openai/gpt-5.6-luna`, private task version 2, and completed notebook `Run #1`. A separate task-run queue was empty, so no second set of 200 requests was launched. Details, prompts, predictions, and example reviews are in [the Kaggle report](kaggle_gpt_5_6_luna/report.md) and [Kaggle note](../KAGGLE_LUNA_NOTE.md).

## Phase 4: did more CatBoost trees help?

The saved pooled cross-annotator F3 CatBoost model retained 453 per-tree training RMSE values. The saved model-selection table held the selected tree count and best validation RMSE, but the final refit did not retain a per-tree validation curve. We continued the 453-tree model from its original training rows, using validation for early stopping and selection and keeping test data out of tuning. The candidate retained 60 additional trees (513 total). The original model under `models/` was not modified.

| Split | Model | Trees | Weighted MAE | Weighted RMSE |
|---|---|---:|---:|---:|
| Validation | Saved model | 453 | 16.840 | 21.732 |
| Validation | Continued candidate | 513 | 16.810 | 21.719 |
| Test | Saved model | 453 | 18.123 | 23.111 |
| Test | Continued candidate | 513 | 18.112 | 23.116 |

Test candidate-minus-baseline MAE was −0.0108 (95% document-bootstrap CI −0.0214 to approximately 0); RMSE was +0.0049 (CI −0.0063 to +0.0168). So the continuation produced **no practically meaningful held-out improvement**, especially for the RMSE training objective. Keep the original 453-tree model. This was a single pooled model continuation check, not a sweep across every fold or language pair. See the [continuation report](catboost_continuation/report.md), [loss curves](catboost_continuation/loss_curve.png), and [saved candidate](catboost_continuation/continued_candidate.cbm).

## Answers to the original questions

1. **Can human-assigned ESA scores be predicted from error counts alone?** Somewhat. F1 CatBoost beats the mean predictor in the primary same-annotation tests, but its MAE remains high (19.34 Bho; 19.30 pooled), and the fixed formula’s performance is inconsistent. Counts alone miss important variation.
2. **Does coverage improve prediction?** Yes. Adding coverage reduces CatBoost MAE by 9.17 points in Bho and 8.02 points pooled for same-annotation evaluation, and by 3.86 and 3.45 points for independent-annotator cross-training.
3. **Does CatBoost outperform a simple linear mapping?** Slightly with gold same-annotation targets (F3 gains of 1.52 MAE points Bho and 0.54 pooled). The cross-annotator gains are smaller (0.139 Bho and 0.353 pooled), so the evidence supports a modest advantage, not a decisive general nonlinear win.
4. **How much human disagreement exists?** Across ESA, mean/median/p90 absolute inter-rater difference is 18.34/13/43 points. For Bho it is 22.70/18/51. On the 100-item pilot it is 22.50/17.50/50.10.
5. **Which annotations are hardest?** Largest residuals include low human scores without marked errors and high human scores despite major spans. These cases, along with high inter-annotator disagreement, are in the error/failure CSVs linked above.
6. **Is this stable enough as a scoring layer after an LLM error detector?** Not established. All local models were trained/evaluated with gold human spans. Luna was separately prompted with features (and in one condition text), but this does not test a pipeline where Luna or another detector produces error spans and a learned scorer consumes them. End-to-end testing with predicted spans across held-out documents and language pairs is still needed.

## Reproducibility and artifact index

- Main experiment: [detailed report](report.md), [metrics](metrics.csv), [ablations](ablations.csv), [dataset summary](dataset_summary.json), [splits](split_manifest.csv), [annotator agreement](annotator_agreement.csv), [error analysis](error_analysis.csv), and [plots](plots/).
- Independent annotators: [combined report](combined_report.md), [Experiment 2 report](experiment2/report.md), [metrics](experiment2/metrics.csv), [ablations](experiment2/ablations.csv), [failure cases](experiment2/failure_cases.csv), and fitted models under `../models/experiment2/`.
- Earlier frozen-sample pilot: [Experiment 3 report](experiment3/report.md), [sample manifest](experiment3/sample100_manifest.json), [local metrics](experiment3/metrics_local.csv), [failure cases](experiment3/failure_cases.csv), and [hosted preflight status](experiment3/llm_predictions.csv).
- Kaggle Luna: [report](kaggle_gpt_5_6_luna/report.md), [metrics](kaggle_gpt_5_6_luna/metrics.csv), [paired comparisons](kaggle_gpt_5_6_luna/paired_bootstrap.csv), [error examples](kaggle_gpt_5_6_luna/error_examples.csv), [run note](../KAGGLE_LUNA_NOTE.md), and source in `../execution/kaggle_benchmarks/`.
- CatBoost continuation: [report](catboost_continuation/report.md), [loss curve data](catboost_continuation/loss_curve.csv), [test bootstrap](catboost_continuation/test_delta_bootstrap.csv), [candidate model](catboost_continuation/continued_candidate.cbm), [script](catboost_continuation/continue_catboost.py), and [run note](../CATBOOST_CONTINUATION_NOTE.md).
- Original experiment scripts and fitted models remain in the project; no original model or report was overwritten to create this consolidation.
