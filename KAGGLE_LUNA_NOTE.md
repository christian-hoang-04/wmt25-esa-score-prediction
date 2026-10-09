# Kaggle / Luna request — status note

Date: 2026-10-09

The project already has a Kaggle kernel package at `execution/kaggle_kernel/`:

- Kernel ref: `namlh2004/wmt25-esa-score-prediction-experiment`
- Entry point: `main.py`, generated from the project scripts by `build_kaggle_kernel.py`
- Kaggle settings: private, CPU, internet enabled
- Existing orchestration uses the `primary` profile and scripts under `D:\projects\personal\kaggle-agent-compute\execution\scripts`
- A prior run-output folder exists under `execution/run-output/`

Clarification: “Luna” means OpenAI GPT-5.6 Luna. The live Kaggle Benchmarks model list exposes it as `gpt-5.6-luna`. The WMT25 test is assumed to use the existing frozen 100-translation sample and both existing prompt conditions (feature-only and context-aware).

The primary profile is authenticated. Its local account snapshot is stale and marks Benchmarks spend as unknown; the installed Kaggle CLI 2.2.3 can list models and tasks but does not implement `kaggle benchmarks quota`, so the current remaining dollar balance could not be read locally. The requested run is limited to the 200 frozen prompt rows and will stop on the first request error. Kaggle, rather than a local dollar cap, will enforce the account's available Model Proxy budget.

Existing Kaggle Benchmarks history includes completed WMT24 Luna-named tasks, but those used a different model and do not constitute this WMT25 run. The WMT25 Luna test was run as requested. The source and results are confined to `execution/kaggle_benchmarks/`, `results/kaggle_gpt_5_6_luna/`, and this note.

## Completed WMT25 run

- Kaggle model: `openai/gpt-5.6-luna` (display slug `gpt-5.6-luna`)
- Private task version 2: https://www.kaggle.com/benchmarks/tasks/namlh2004/wmt25-esa-gpt-56-luna/2
- Backing notebook: `namlh2004/new-benchmark-task-251bc`
- Kaggle notebook run: `Run #1`, completed 2026-10-08 20:34:40Z to 20:44:24Z
- Sample: the existing frozen 100 English-to-Bhojpuri translations, digest `c10f77eb89f19955b07ad87e529aaeef00f8d85ff35806ccc3c1b1a9c1f375d2`
- Requests: 200 total, 100 feature-only and 100 context-aware; all 200 succeeded
- Reported Model Proxy cost: $0.075141 (95,607 input tokens; 46,683 output tokens)

The Kaggle notebook run happened during task creation and produced a completed run artifact. The separate `tasks run` queue reports no runs; no second run was launched because that would repeat the 200 model requests.

Primary test metrics against independent annotator B (document-bootstrap MAE 95% intervals are in the report): mean predictor 25.42 MAE, fixed penalty 28.58, cross Ridge F3 20.35, cross CatBoost F3 20.09, Luna feature-only 20.93, Luna context-aware 20.78. This is a 100-translation pilot, not the full WMT25 dataset. Mean absolute human A/B disagreement is 22.50 points (median 17.50; p90 50.10), so the pilot does not establish Luna as a stable scoring layer.

Artifacts:

- Kaggle task source: `execution/kaggle_benchmarks/wmt25_esa_gpt_5_6_luna.py`
- Downloaded predictions and Kaggle run metadata/logs: `results/kaggle_gpt_5_6_luna/kernel-output/`
- Reproducible local comparison: `results/kaggle_gpt_5_6_luna/analyze_results.py`
- Metrics, pairwise ranking, error examples, and human-disagreement checks: `results/kaggle_gpt_5_6_luna/`
- Human-readable summary: `results/kaggle_gpt_5_6_luna/report.md`
