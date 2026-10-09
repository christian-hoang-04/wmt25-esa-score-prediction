# WMT25 ESA score prediction from error annotations

This repository contains CPU experiments for WMT25 human Error Span Annotation (ESA) scores. Experiment 1 predicts scores from one human annotation's features. Experiment 2 evaluates cross-annotator generalization. Experiment 3 freezes a small sample and compares local models with feature-only and text-context frontier-LLM judgments when a configured API is available. No GPU is used.

## Data and protocol

The input is the official [`wmt25-genmt-humeval.jsonl`](https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/data/wmt25-genmt-humeval.jsonl) file from the [WMT25 General MT repository](https://github.com/wmt-conference/wmt25-general-mt). The official schema has one JSON object per source segment, translations keyed by system, and one or more human score/error annotations per translation. WMT describes `doc_id` as language-pair, domain, document, and segment components separated by `_#_`.

Protocol inclusion follows the [WMT25 segment-level error span task](https://www2.statmt.org/wmt25/mteval-subtask2.html): only its 14 ESA score directions are retained. The MQM-only directions are excluded. In the downloaded file, some error objects have `start_i: "missing"` and `end_i: "missing"` despite having a severity. These are not clean/no-error annotations: the full annotation is written to `results/invalid_annotations.csv` and omitted from every model so the model matrix does not treat unknown span coverage as zero. Missing scores, annotators, translations, malformed error records, out-of-bounds/reversed offsets, and unsupported severities are handled the same way. The detailed reason counts are in `results/dataset_summary.json`. Valid span offsets are inclusive; coverage uses the union of character intervals.

## Run locally

Python 3.10+ is recommended. Install the CPU dependencies, then download and prepare the dataset:

```powershell
python -m pip install -r requirements.txt
python download_data.py
python prepare_data.py
python train.py
python evaluate.py
```

The project also uses `uv` with a lockfile for a reproducible Python environment:

```powershell
uv sync --python 3.12
uv run python experiment2.py
```

`experiment2.py` reuses `results/split_manifest.csv` from Experiment 1. It writes a directed A→B/B→A paired set, keeps the five Bhojpuri folds and pooled 70/15/15 assignments document-grouped, and fits matched same-annotator and cross-trained Ridge/CatBoost models with F1/F2/F3 ablations. If fitting has already completed and only plots/reports/sample outputs need regeneration, use `uv run python experiment2.py --postprocess-only`.

### Frontier judge (optional API run)

Experiment 3 creates and hashes `results/experiment3/sample100.csv` before any hosted request. The ID sample is random, contains 100 distinct en→bho translations, and is selected without scores or prediction errors. Local matched predictions are in `sample100_local_baselines.csv`.

The live script uses the Meddies provider pool and its OS-keyring credentials. Run a synthetic, data-free preflight first, then run the frozen sample only after that exact model succeeds:

```powershell
uv run --project D:\projects\meddies\meddies-llm-runtime --with pandas `
  python run_frontier_judge.py --preflight-only --model gemini-3.8-flash

uv run --project D:\projects\meddies\meddies-llm-runtime --with pandas `
  python run_frontier_judge.py --confirmed-preflight --model gemini-3.8-flash

uv run python evaluate_frontier.py
```

The runner sends two separate requests per translation, serially: F3 features only, then source/translation plus input spans and F3. It excludes human scores, annotator IDs, and system names from both prompts, validates parsed scores as integer JSON from 0 to 100, stores no raw prompt or response, and stops before a request if its conservative token-price reserve could exceed the $20 cap. It does not switch to another provider or model after an error. If preflight fails, run `uv run python evaluate_frontier.py` to record the block and keep the local results.

If you already have the JSONL file, pass it to the downloader instead:

```powershell
python download_data.py --input D:\data\wmt25-genmt-humeval.jsonl
```

The downloader validates that the file is real JSONL rather than an HTML response or Git LFS pointer. If the ordinary GitHub raw endpoint returns a pointer or fails, it tries GitHub's LFS media endpoint.

## Run on Kaggle

The `execution/kaggle_kernel/` folder is the Kaggle upload package. Kaggle's script runner executes only the designated code file, so `build_kaggle_kernel.py` embeds the canonical root scripts into one self-contained entrypoint. It downloads the public JSONL from within a private CPU Kaggle kernel with internet enabled, runs preparation, training, and evaluation, and writes inspectable artifacts under `/kaggle/working`.

From this repository, use the shared orchestration scripts:

```powershell
python build_kaggle_kernel.py
& 'D:\projects\personal\kaggle-agent-compute\execution\scripts\Submit-KaggleKernelJob.ps1' `
  -Alias primary `
  -KernelPath (Resolve-Path execution\kaggle_kernel).Path `
  -TimeoutSeconds 7200 `
  -Note 'Run WMT25 ESA score prediction experiment on CPU'
```

## Experiment design

- Each valid human annotation is an independent row. No annotator ID, system name, document ID, or raw text is a model feature.
- Features are F1 counts; F2 counts plus union span coverage; F3 all requested numeric count, length, span, density, fraction, and indicator features.
- Mean and fixed `100 - 5 × major - minor` predictors are reference baselines. Ridge uses `StandardScaler + Ridge`, with alpha selected by validation RMSE. CatBoost uses CPU regression, 500 maximum iterations, depth and L2 regularization candidates, and validation early stopping.
- All document IDs are grouped, and any documents sharing an exact source text are joined into the same leakage group. En→bho uses five-fold grouped cross-validation because its 50 documents are too few for a dependable single held-out test. The pooled ESA model uses grouped 70/15/15 train/validation/test partitions, with per-language-pair count balance checked during split selection.
- The pooled human-score ranking target is the mean independent score per source segment and system. Pairwise ranking accuracy excludes score ties. Document-cluster bootstrap intervals quantify uncertainty in MAE and paired MAE differences.
- Human disagreement averages repeated scores from the same annotator before calculating between-annotator differences.

Experiment 2 uses a stable translation key derived from language pair, full segment `doc_id`, and system. Duplicate rows from one annotator on a translation are reduced to one view: its score is averaged and the earliest valid annotation supplies its real error features/spans. Distinct annotator pairs produce both directions. Each translation has total pair weight 1, so translations with more annotators do not dominate. The frozen estimator is trained on same-annotator feature/score rows; the cross-trained estimator is trained on A's features with B's score. Both are evaluated on identical outer test pairs. The original Bhojpuri folds are retained, including their uneven document counts; the report calls out the one-document first fold.

Experiment 3 uses the same frozen 100 Bhojpuri translation pairs for all local baselines and any frontier prompts. It reports independent annotator B as the primary target, annotator A as a secondary target, and document-cluster bootstrap intervals for paired MAE differences. The API path is optional; no LLM score rows are created when the configured model cannot pass preflight.

## Deliverables

- `results/metrics.csv` and `results/ablations.csv`: overall and per-direction metrics and feature ablations.
- `results/report.md`: results, plots, limitations, and answers to the six experiment questions.
- `results/error_analysis.csv`: largest CatBoost errors and available score/span contradictions, with texts and serialized spans.
- `results/plots/`: predicted-vs-human, human score by error counts, and CatBoost importance plots.
- `models/`: saved Ridge and CatBoost fits. Bhojpuri full-data models are refit after out-of-fold scoring and are not the source of their reported OOF metrics.
- `data/processed/annotations.csv.gz`: valid flattened examples; raw downloaded data is kept separately under `data/raw/`.
- `results/experiment2/`: paired annotation population, no-leakage split audit, matched predictions/metrics/ablations, bootstrap generalization gap, human disagreement, failure cases, plots, report and fold-specific saved models.
- `results/experiment3/`: frozen sample IDs and SHA-256, local baseline rows/metrics, preflight and sanitized request logs, frontier metrics and report when available.
- `results/combined_report.md`: concise Experiment 2 and 3 summary with API status.

See `results/report.md` for measured outcomes. Results should be interpreted as score prediction from gold human spans; this experiment does not measure the noise introduced by an automatic or LLM-based error detector.
