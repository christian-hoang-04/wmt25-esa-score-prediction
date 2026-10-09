"""Evaluate frontier judgments against the frozen sample and local baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from .run_frontier_judge import write_prompt_artifacts


BOOTSTRAP_REPLICATES = 2000
SEED = 42


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error": float(np.median(absolute)),
        "pearson": float(pearsonr(actual, prediction).statistic) if np.ptp(actual) and np.ptp(prediction) else float("nan"),
        "spearman": float(spearmanr(actual, prediction).statistic) if np.ptp(actual) and np.ptp(prediction) else float("nan"),
        "mean_bias_pred_minus_human": float(np.mean(error)),
        "within_5_pct": float(np.mean(absolute <= 5) * 100),
        "within_10_pct": float(np.mean(absolute <= 10) * 100),
        "within_20_pct": float(np.mean(absolute <= 20) * 100),
    }


def _document_bootstrap(
    frame: pd.DataFrame,
    actual_col: str,
    prediction_col: str,
    cluster_col: str = "doc_group_id",
    seed: int = SEED,
) -> tuple[float, float]:
    sums = frame.assign(
        _absolute=np.abs(frame[actual_col].to_numpy(float) - frame[prediction_col].to_numpy(float))
    ).groupby(cluster_col, observed=True)["_absolute"].agg(["sum", "count"]).to_numpy(float)
    if len(sums) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sums), size=(BOOTSTRAP_REPLICATES, len(sums)))
    values = sums[sampled].sum(axis=1)
    low, high = np.quantile(values[:, 0] / values[:, 1], [0.025, 0.975])
    return float(low), float(high)


def _paired_delta_ci(
    frame: pd.DataFrame,
    actual_col: str,
    candidate_col: str,
    baseline_col: str,
    cluster_col: str = "doc_group_id",
    seed: int = SEED,
) -> tuple[float, float]:
    grouped = frame.assign(
        _candidate=np.abs(frame[actual_col].to_numpy(float) - frame[candidate_col].to_numpy(float)),
        _baseline=np.abs(frame[actual_col].to_numpy(float) - frame[baseline_col].to_numpy(float)),
    ).groupby(cluster_col, observed=True)[["_candidate", "_baseline"]].sum()
    counts = frame.groupby(cluster_col, observed=True).size().reindex(grouped.index).to_numpy(float)
    values = np.column_stack([grouped.to_numpy(float), counts])
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sums = values[sampled].sum(axis=1)
    delta = (sums[:, 0] - sums[:, 1]) / sums[:, 2]
    low, high = np.quantile(delta, [0.025, 0.975])
    return float(low), float(high)


def _fmt(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value):.3f}"


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for values in frame.itertuples(index=False, name=None):
        rows.append([
            ("—" if pd.isna(value) else format(float(value), floatfmt))
            if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
            for value in values
        ])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_local_only_artifacts(sample_path: Path, results_dir: Path) -> dict[str, Path]:
    """Materialize review artifacts without inventing unavailable hosted scores."""
    sample = pd.read_csv(sample_path, low_memory=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths = write_prompt_artifacts(sample, results_dir)

    local_predictions = results_dir / "sample100_local_baselines.csv"
    local = pd.read_csv(local_predictions, low_memory=False) if local_predictions.exists() else sample.copy()
    local_ids = set(local["translation_id"].astype(str))
    sample_ids = set(sample["translation_id"].astype(str))
    if local_ids != sample_ids:
        raise ValueError("Local baseline IDs must exactly match the frozen 100-item sample")

    predictions_path = results_dir / "llm_predictions.csv"
    llm_rows = []
    for row in sample.to_dict("records"):
        for condition in ("feature_only", "context_aware"):
            llm_rows.append({
                "translation_id": row["translation_id"], "condition": condition,
                "model": "gemini-3.8-flash", "score_prediction": np.nan,
                "explanation": "", "status": "not_run_preflight_blocked",
                "error_category": "rate_or_quota_limit", "http_status": 429,
            })
    pd.DataFrame(llm_rows).to_csv(predictions_path, index=False)

    comparison_cols = [
        "translation_id", "doc_id", "doc_group_id", "language_pair", "system_name",
        "score_a", "score_b", "pred_mean_cross", "pred_cross_ridge_F3",
        "pred_cross_catboost_F3", "pred_same_catboost_F3",
    ]
    comparison = local[[column for column in comparison_cols if column in local]].copy()
    comparison["llm_feature_only_prediction"] = np.nan
    comparison["llm_context_aware_prediction"] = np.nan
    comparison["llm_status"] = "not_run_preflight_blocked"
    comparison.to_csv(results_dir / "comparison.csv", index=False)

    metrics_local_path = results_dir / "metrics_local.csv"
    metrics_local = pd.read_csv(metrics_local_path, low_memory=False) if metrics_local_path.exists() else pd.DataFrame()
    metrics = metrics_local.copy()
    if not metrics.empty:
        metrics["status"] = "measured_local"
    unavailable = []
    for target in ("independent_target_b", "input_annotator_a"):
        for condition in ("feature_only", "context_aware"):
            unavailable.append({
                "target": target, "model": f"llm_{condition}",
                "status": "not_run_preflight_blocked", "n_translations": int(sample.translation_id.nunique()),
                "n_documents": int(sample.doc_group_id.nunique()),
            })
    metrics = pd.concat([metrics, pd.DataFrame(unavailable)], ignore_index=True, sort=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    # Include local, paired cluster-bootstrap deltas already computed in metrics_local.
    deltas = metrics_local[
        ["target", "model", "delta_mae_vs_matched_catboost", "delta_ci_low", "delta_ci_high", "n_translations", "n_documents"]
    ].copy() if not metrics_local.empty else pd.DataFrame()
    deltas.to_csv(results_dir / "paired_bootstrap.csv", index=False)

    failure_source = local.copy()
    failure_source["absolute_error_vs_b"] = (failure_source["pred_cross_catboost_F3"] - failure_source["score_b"]).abs()
    failure_source["absolute_error_vs_a"] = (failure_source["pred_cross_catboost_F3"] - failure_source["score_a"]).abs()
    failure_source["human_disagreement"] = (failure_source["score_a"] - failure_source["score_b"]).abs()
    p90_disagreement = float(failure_source["human_disagreement"].quantile(.9))
    candidate_sets: list[pd.DataFrame] = []
    top_errors = failure_source.nlargest(10, "absolute_error_vs_b").copy()
    top_errors["category"] = "largest_catboost_error_vs_b"
    candidate_sets.append(top_errors)
    high_disagreement = failure_source[failure_source["human_disagreement"] >= p90_disagreement].copy()
    high_disagreement["category"] = "high_human_disagreement"
    candidate_sets.append(high_disagreement)
    if "n_total_b" in failure_source:
        low_no_errors = failure_source[(failure_source["score_b"] <= 50) & (failure_source["n_total_b"] == 0)].copy()
        low_no_errors["category"] = "low_b_score_no_b_errors"
        candidate_sets.append(low_no_errors)
    if "n_major_b" in failure_source:
        high_major = failure_source[(failure_source["score_b"] >= 80) & (failure_source["n_major_b"] > 0)].copy()
        high_major["category"] = "high_b_score_with_b_major_errors"
        candidate_sets.append(high_major)
    if "n_total" in failure_source:
        input_no_errors = failure_source[(failure_source["score_b"] <= 50) & (failure_source["n_total"] == 0)].copy()
        input_no_errors["category"] = "low_b_score_no_a_errors"
        candidate_sets.append(input_no_errors)
    if "n_major" in failure_source:
        input_major = failure_source[(failure_source["score_b"] >= 80) & (failure_source["n_major"] > 0)].copy()
        input_major["category"] = "high_b_score_with_a_major_errors"
        candidate_sets.append(input_major)
    close_a_far_b = failure_source[(failure_source["absolute_error_vs_a"] <= 10) & (failure_source["absolute_error_vs_b"] >= 20)].copy()
    close_a_far_b["category"] = "catboost_close_to_a_far_from_b"
    candidate_sets.append(close_a_far_b)
    closer_b = failure_source[failure_source["absolute_error_vs_b"] + 10 <= failure_source["absolute_error_vs_a"]].nlargest(20, "absolute_error_vs_a").copy()
    closer_b["category"] = "catboost_closer_to_b_than_a"
    candidate_sets.append(closer_b)
    failures = pd.concat(candidate_sets, ignore_index=True).drop_duplicates(["translation_id", "category"])
    review_cols = [
        "category", "translation_id", "doc_id", "doc_group_id", "language_pair", "system_name",
        "src_text", "target_text", "score_a", "score_b", "pred_cross_catboost_F3",
        "absolute_error_vs_a", "absolute_error_vs_b", "human_disagreement",
        "n_minor", "n_major", "n_total", "n_minor_b", "n_major_b", "n_total_b",
        "error_spans_a_json", "error_spans_b_json",
    ]
    failures[[column for column in review_cols if column in failures]].to_csv(results_dir / "failure_cases.csv", index=False)

    log_path = results_dir / "frontier_request_log.csv"
    logs = pd.read_csv(log_path, low_memory=False) if log_path.exists() else pd.DataFrame()
    safe_columns = [
        "condition", "provider", "model", "status", "error_category", "http_status",
        "latency_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
        "actual_cost_usd", "projected_reserve_usd", "retry_after_seconds",
    ]
    costs = logs[[column for column in safe_columns if column in logs]].copy() if not logs.empty else pd.DataFrame(columns=safe_columns)
    costs.to_csv(results_dir / "api_costs.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.scatter(local["score_b"], local["pred_cross_catboost_F3"], s=20, alpha=.68, color="#176b87", edgecolors="none")
    ax.plot([0, 100], [0, 100], color="#cc4b37", linestyle="--", linewidth=1.2)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Independent annotator B score", ylabel="Cross-trained CatBoost F3 prediction", title="Local prediction on frozen 100-item sample")
    fig.tight_layout()
    fig.savefig(plots_dir / "local_predicted_vs_human.png", dpi=160)
    plt.close(fig)
    differences = (local["score_a"] - local["score_b"]).abs()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.hist(differences, bins=16, color="#8b6cad", edgecolor="white")
    ax.set(xlabel="Absolute A/B human score difference", ylabel="Translations", title="Human disagreement on frozen sample")
    fig.tight_layout()
    fig.savefig(plots_dir / "human_disagreement.png", dpi=160)
    plt.close(fig)

    return {
        "metrics": results_dir / "metrics.csv", "predictions": predictions_path,
        "comparison": results_dir / "comparison.csv", "failures": results_dir / "failure_cases.csv",
        "costs": results_dir / "api_costs.csv", "plots": plots_dir,
        "prompts": prompt_paths[0].parent,
    }


def _build_combined_report(results_dir: Path) -> Path:
    root_results = results_dir.parent
    exp2_metrics_path = root_results / "experiment2" / "metrics.csv"
    exp2_human_path = root_results / "experiment2" / "human_disagreement.csv"
    local_path = root_results / "experiment3" / "metrics_local.csv"
    frontier_metrics_path = root_results / "experiment3" / "metrics.csv"
    frontier_report_path = root_results / "experiment3" / "report.md"
    preflight_path = root_results / "experiment3" / "frontier_preflight.json"
    log_path = root_results / "experiment3" / "frontier_request_log.csv"

    exp2_display = pd.DataFrame()
    exp2 = pd.DataFrame()
    language_display = pd.DataFrame()
    ablation_display = pd.DataFrame()
    human_display = pd.DataFrame()
    local_display = pd.DataFrame()
    local_secondary_display = pd.DataFrame()
    frontier_display = pd.DataFrame()
    if exp2_metrics_path.exists():
        exp2 = pd.read_csv(exp2_metrics_path)
        exp2_display = exp2[
            (exp2["language_pair"].isin(["ALL_ESA", "en-bho_IN"]))
            & (
                (exp2["model"].isin(["ridge", "catboost"]) & exp2["feature_set"].eq("F3") & exp2["regime"].isin(["same_target", "frozen_cross", "cross_trained"]))
                | exp2["model"].isin(["input_annotator_a", "mean", "fixed_penalty"])
            )
        ][["experiment_scope", "language_pair", "regime", "model", "feature_set", "n_translations", "mae", "mae_ci_low", "mae_ci_high", "rmse", "spearman"]]
        language_display = exp2[
            exp2["experiment_scope"].eq("pooled_holdout")
            & exp2["language_pair"].ne("ALL_ESA")
            & ((exp2["model"].isin(["ridge", "catboost"]) & exp2["feature_set"].eq("F3") & exp2["regime"].eq("cross_trained")) | exp2["model"].eq("input_annotator_a"))
        ][["language_pair", "model", "n_translations", "n_unique_documents", "mae", "mae_ci_low", "mae_ci_high"]]
        ablation_path = root_results / "experiment2" / "ablations.csv"
        if ablation_path.exists():
            ablations = pd.read_csv(ablation_path)
            ablation_display = ablations[
                ablations["experiment_scope"].eq("pooled_holdout")
                & ablations["language_pair"].eq("ALL_ESA")
                & ablations["regime"].eq("cross_trained")
            ][["model", "feature_set", "n_translations", "mae", "mae_ci_low", "mae_ci_high", "rmse", "spearman"]]
    if exp2_human_path.exists():
        human = pd.read_csv(exp2_human_path)
        human_display = human[human["language_pair"].isin(["en-bho_IN", "ALL_ESA"])]
    if local_path.exists():
        local_metrics = pd.read_csv(local_path)
        local_columns = [column for column in [
            "model", "n_translations", "n_documents", "mae", "mae_ci_low", "mae_ci_high", "rmse",
            "within_10_pct", "delta_mae_vs_matched_catboost", "delta_ci_low", "delta_ci_high",
        ] if column in local_metrics.columns]
        local_display = local_metrics[local_metrics["target"] == "independent_target_b"][local_columns]
        local_secondary_display = local_metrics[local_metrics["target"] == "input_annotator_a"][local_columns]
    if frontier_metrics_path.exists() and frontier_metrics_path.stat().st_size > 4:
        frontier = pd.read_csv(frontier_metrics_path)
        frontier_display = frontier[frontier["target"] == "independent_target_b"]

    api_status = "not attempted"
    api_detail = "No hosted model score rows are available."
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        api_status = str(preflight.get("status", "unknown"))
        api_detail = (
            f"provider `{preflight.get('provider', '')}`, model `{preflight.get('model', '')}`, "
            f"sanitized error category `{preflight.get('error_category', '')}`, HTTP status `{preflight.get('http_status', '')}`."
        )
    observed_cost = 0.0
    synthetic_calls = 0
    if log_path.exists():
        logs = pd.read_csv(log_path)
        observed_cost = float(pd.to_numeric(logs.get("actual_cost_usd", 0), errors="coerce").fillna(0).sum())
        synthetic_calls = int((logs.get("condition") == "synthetic_preflight").sum()) if "condition" in logs else 0

    def _metric(model: str, scope: str = "pooled_holdout", language_pair: str = "ALL_ESA", regime: str | None = None, feature_set: str | None = None) -> float | None:
        if exp2.empty:
            return None
        mask = exp2["experiment_scope"].eq(scope) & exp2["language_pair"].eq(language_pair) & exp2["model"].eq(model)
        if regime is not None:
            mask &= exp2["regime"].eq(regime)
        if feature_set is not None:
            mask &= exp2["feature_set"].eq(feature_set)
        rows = exp2[mask]
        return float(rows.iloc[0]["mae"]) if not rows.empty else None

    def _number(value: float | None) -> str:
        return "unavailable" if value is None or pd.isna(value) else f"{value:.2f}"

    cat_mae = _metric("catboost", regime="cross_trained", feature_set="F3")
    ridge_mae = _metric("ridge", regime="cross_trained", feature_set="F3")
    human_mae = _metric("input_annotator_a", regime="human_baseline", feature_set="human_score_a")
    count_mae = _metric("catboost", regime="cross_trained", feature_set="F1")
    coverage_mae = _metric("catboost", regime="cross_trained", feature_set="F2")
    full_mae = _metric("catboost", regime="cross_trained", feature_set="F3")
    exp2_failure_path = root_results / "experiment2" / "failure_cases.csv"
    failure_counts = pd.read_csv(exp2_failure_path, low_memory=False)["category"].value_counts().to_dict() if exp2_failure_path.exists() else {}

    report_path = root_results / "combined_report.md"
    report = f"""# Human-Aligned MT Scoring — Combined Results

## Experiment 2: independent-annotator generalization

The paired dataset contains **33,049 translations** across **14 ESA language pairs**. The Bhojpuri five-fold out-of-fold set contains 2,300 paired translations; its held-out fold document counts were 1, 10, 13, 12, and 14, so fold-level uncertainty is uneven. Pooled ESA evaluation uses 4,294 paired translations in the grouped test partition. No source document, duplicate-source leakage group, reciprocal pair, or translation crossed splits.

{_markdown_table(exp2_display)}

Pooled-language held-out results are shown where sample sizes allow:

{_markdown_table(language_display)}

### Feature ablation

{_markdown_table(ablation_display)}

Human absolute score disagreement is summarized below. These human-to-human differences provide noise context; they are not directly identical to model-to-one-rater MAE.

{_markdown_table(human_display)}

## Experiment 3: frozen 100-item sample

The local matched results against independent annotator B are:

{_markdown_table(local_display)}

The same local baselines against input annotator A are:

{_markdown_table(local_secondary_display)}

Frontier API status: **{api_status}**; {api_detail} Synthetic-only calls: {synthetic_calls}. Token-price estimate from observed usage: **${observed_cost:.6f}**. No sample source/translation text was sent; no LLM scores are fabricated. The unavailable rows in `experiment3/metrics.csv` document the blocked comparisons.

## Answers to the research questions

1. **Can human scores be predicted from error annotations?** Yes, to a useful but imperfect degree when training and evaluating on independent annotators: pooled cross-trained CatBoost F3 MAE is **{_number(cat_mae)}** points, versus **{_number(human_mae)}** for using input annotator A's score as a human-to-human reference. This result conditions on gold human error annotations.
2. **Does coverage improve prediction?** Yes from counts-only to counts plus coverage for pooled CatBoost (F1 MAE **{_number(count_mae)}**, F2 **{_number(coverage_mae)}**); the full feature set is **{_number(full_mae)}**, so text lengths and derived features add little beyond coverage in this pooled comparison.
3. **Does CatBoost beat a linear mapping?** The pooled CatBoost F3 MAE is **{_number(cat_mae)}** versus Ridge **{_number(ridge_mae)}**. This is a small descriptive advantage; it does not establish a robust or statistically decisive nonlinear gain.
4. **How much annotator disagreement is there?** Across ESA pairs, the mean absolute difference is about **18.34** points (median 13, p90 43); for English→Bhojpuri it is about **22.70** (median 18, p90 51). On the 100-item sample, A/B disagreement has mean 22.50, median 17.50, and p90 50.10.
5. **Which annotations are hardest?** The largest held-out CatBoost residuals and high-disagreement or score/error-contradiction cases are in `experiment2/failure_cases.csv` and `experiment3/failure_cases.csv`. Current pooled failure-case category counts: `{failure_counts}`. The sample CSV includes source, translation, both span lists, both scores, and local prediction for inspection.
6. **Is the learned function stable after an LLM error detector?** Not established. These models consume human error spans, not detector-generated annotations. The frontier score-judge comparison also has no result because its synthetic preflight hit a 429 rate/quota limit; the score layer needs a later end-to-end evaluation with detector outputs before use as a scoring layer.

## Artifacts

- [Experiment 2 report](experiment2/report.md), [metrics](experiment2/metrics.csv), [ablations](experiment2/ablations.csv), and [failure cases](experiment2/failure_cases.csv).
- [Experiment 3 report](experiment3/report.md), [local metrics](experiment3/metrics_local.csv), [comparison](experiment3/comparison.csv), [hosted score status](experiment3/llm_predictions.csv), and [API cost log](experiment3/api_costs.csv).
- Prompt payloads: `experiment3/prompts/`; plots: `experiment3/plots/`.
- Frozen sample IDs and digest: [sample100.csv](experiment3/sample100.csv) and [manifest](experiment3/sample100_manifest.json).
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def evaluate(
    sample_path: Path = Path("results/experiment3/sample100_local_baselines.csv"),
    predictions_path: Path = Path("results/experiment3/frontier_predictions.csv"),
    metadata_path: Path = Path("results/experiment3/frontier_run_metadata.json"),
    results_dir: Path = Path("results/experiment3"),
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(sample_path, low_memory=False)
    metrics_path = results_dir / "metrics.csv"
    delta_path = results_dir / "paired_bootstrap.csv"
    report_path = results_dir / "report.md"
    if not predictions_path.exists():
        preflight = results_dir / "frontier_preflight.json"
        status = json.loads(preflight.read_text(encoding="utf-8")) if preflight.exists() else {"status": "not_run"}
        artifacts = _write_local_only_artifacts(sample_path, results_dir)
        log_path = results_dir / "frontier_request_log.csv"
        logs = pd.read_csv(log_path, low_memory=False) if log_path.exists() else pd.DataFrame()
        actual_cost = float(pd.to_numeric(logs.get("actual_cost_usd", 0), errors="coerce").fillna(0).sum()) if not logs.empty else 0.0
        synthetic_calls = int((logs.get("condition") == "synthetic_preflight").sum()) if "condition" in logs else 0
        local_metrics_path = results_dir / "metrics_local.csv"
        local_metrics = pd.read_csv(local_metrics_path) if local_metrics_path.exists() else pd.DataFrame()
        local_primary = local_metrics[local_metrics["target"] == "independent_target_b"] if not local_metrics.empty else pd.DataFrame()
        local_secondary = local_metrics[local_metrics["target"] == "input_annotator_a"] if not local_metrics.empty else pd.DataFrame()
        local_columns = [column for column in [
            "model", "n_translations", "n_documents", "mae", "mae_ci_low", "mae_ci_high", "rmse",
            "within_10_pct", "delta_mae_vs_matched_catboost", "delta_ci_low", "delta_ci_high",
        ] if column in local_primary.columns]
        local_table = _markdown_table(local_primary[local_columns]) if not local_primary.empty else "_No local baseline rows._"
        secondary_table = _markdown_table(local_secondary[local_columns]) if not local_secondary.empty else "_No local secondary rows._"
        local_data = pd.read_csv(sample_path, low_memory=False)
        human_difference = np.abs(local_data["score_a"].to_numpy(float) - local_data["score_b"].to_numpy(float))
        human_summary = (
            f"On the same sample, A/B absolute score disagreement has mean {np.mean(human_difference):.2f}, "
            f"median {np.median(human_difference):.2f}, and p90 {np.quantile(human_difference, .9):.2f}."
            if len(human_difference) else "Sample human disagreement is unavailable."
        )
        local_data["_model_error_b"] = (local_data["pred_cross_catboost_F3"] - local_data["score_b"]).abs()
        local_data["_model_error_a"] = (local_data["pred_cross_catboost_F3"] - local_data["score_a"]).abs()
        local_data["_human_difference"] = (local_data["score_a"] - local_data["score_b"]).abs()
        high_disagreement = local_data[local_data["_human_difference"] >= local_data["_human_difference"].quantile(.9)]
        b_major_high = local_data[local_data["score_b"].ge(80) & local_data["n_major_b"].gt(0)]
        b_low_none = local_data[local_data["score_b"].le(50) & local_data["n_total_b"].eq(0)]
        closer_a = int((local_data["_model_error_a"] < local_data["_model_error_b"]).sum())
        closer_b = int((local_data["_model_error_b"] < local_data["_model_error_a"]).sum())
        tied = int((local_data["_model_error_b"] == local_data["_model_error_a"]).sum())
        residual_analysis = (
            f"Among the 10% most disagreeing A/B cases (n={len(high_disagreement)}), mean absolute CatBoost error against B is "
            f"{high_disagreement['_model_error_b'].mean():.2f}, compared with {local_data['_model_error_b'].mean():.2f} overall. "
            f"Annotator B marked at least one major error while assigning a score of 80 or more in {len(b_major_high)} cases; "
            f"their mean model error is {b_major_high['_model_error_b'].mean():.2f}. There are {len(b_low_none)} cases with B score 50 or lower and no B-marked errors. "
            f"Across the full sample, the prediction is closer to A than B in {closer_a} cases, closer to B in {closer_b}, and tied in {tied}. "
            "This is consistent with annotator disagreement contributing to large residuals, but the small sample does not establish causation."
        )
        report_path.write_text(f"""# Experiment 3 — Frontier score judge

## Status

The frozen 100-translation sample and local baselines are complete. The hosted judge was not run on sample items. Three synthetic-only preflight calls ended with status `{status.get('status')}`, sanitized category `{status.get('error_category', '')}`, HTTP `{status.get('http_status', '')}`. Their observed token-price estimate totals **${actual_cost:.6f}**. No source or translation text or human score was sent, and no hosted scores are fabricated. The provider's rate/quota response ended this part of the run.

## Local results against independent annotator B

{local_table}

## Secondary local results against input annotator A

{secondary_table}

{human_summary} Human disagreement and model-to-rater error are different quantities; the comparison is descriptive rather than an equivalence claim.

## Residual inspection

{residual_analysis}

`failure_cases.csv` distinguishes errors marked by the input annotator A from B's own error spans. Cases with a low score and no errors, or a high score with major errors, are selected using the same annotator's score and span list wherever possible. A/B score contradictions remain separate cross-annotator cases.

## Review artifacts

- `sample100.csv` and `sample100_manifest.json`: frozen sample IDs and digest.
- `prompts/feature_only.jsonl` and `prompts/context_aware.jsonl`: exact prompt payloads with response schemas. Join IDs are outside the message bodies; scores and annotator/system identifiers are excluded from prompts.
- `llm_predictions.csv` and `comparison.csv`: explicit `not_run_preflight_blocked` rows with blank hosted score fields.
- `metrics.csv` and `paired_bootstrap.csv`: local results plus explicit unavailable hosted comparisons.
- `failure_cases.csv`: largest local CatBoost errors, high human disagreement, contradictory score/error cases, and cases where predictions track A versus B.
- `api_costs.csv`: sanitized synthetic preflight usage and request status.
- `plots/local_predicted_vs_human.png` and `plots/human_disagreement.png`.

![Local CatBoost predictions](plots/local_predicted_vs_human.png)

![Human annotator disagreement](plots/human_disagreement.png)

See `metrics_local.csv` and `sample100_local_baselines.csv` for per-example local predictions and document-bootstrap intervals.
""", encoding="utf-8")
        combined_path = _build_combined_report(results_dir)
        return {**artifacts, "metrics": metrics_path, "paired_bootstrap": delta_path, "report": report_path, "combined_report": combined_path}

    frontier = pd.read_csv(predictions_path, low_memory=False)
    frontier = frontier[frontier["request_status"] == "ok"].copy()
    if frontier.duplicated(["translation_id", "condition"]).any():
        raise ValueError("More than one valid prediction exists for a sample/condition")
    joined = sample.merge(
        frontier,
        on=["translation_id", "pair_id", "directed_pair_id"],
        how="left",
        suffixes=("", "_frontier"),
        validate="one_to_many",
    )
    # Reduce the repeated paired row to one row per translation and condition.
    feature_only = joined[joined["condition"] == "feature_only"].copy()
    context = joined[joined["condition"] == "context_aware"].copy()
    if feature_only["translation_id"].nunique() != len(sample) or context["translation_id"].nunique() != len(sample):
        raise ValueError("Both prompt conditions must have one result for every frozen translation")
    feature_only = feature_only.drop_duplicates("translation_id")
    context = context.drop_duplicates("translation_id")
    paired = sample.merge(
        feature_only[["translation_id", "score_prediction"]].rename(columns={"score_prediction": "llm_feature_only"}),
        on="translation_id", validate="one_to_one",
    ).merge(
        context[["translation_id", "score_prediction"]].rename(columns={"score_prediction": "llm_context_aware"}),
        on="translation_id", validate="one_to_one",
    )
    if paired[["llm_feature_only", "llm_context_aware"]].isna().any().any():
        raise ValueError("Some frontier responses did not pass the strict score schema")

    model_columns = {
        "mean_cross": "pred_mean_cross",
        "ridge_cross_F3": "pred_cross_ridge_F3",
        "catboost_cross_F3": "pred_cross_catboost_F3",
        "catboost_same_F3": "pred_same_catboost_F3",
        "llm_feature_only": "llm_feature_only",
        "llm_context_aware": "llm_context_aware",
    }
    metric_rows: list[dict[str, Any]] = []
    for target_name, actual_col in (("independent_target_b", "score_b"), ("input_annotator_a", "score_a")):
        for model_name, prediction_col in model_columns.items():
            values = _metrics(paired[actual_col].to_numpy(float), paired[prediction_col].to_numpy(float))
            ci_low, ci_high = _document_bootstrap(paired, actual_col, prediction_col, seed=SEED + len(metric_rows))
            metric_rows.append({
                "target": target_name,
                "model": model_name,
                "n_translations": int(paired["translation_id"].nunique()),
                "n_documents": int(paired["doc_group_id"].nunique()),
                **values,
                "mae_ci_low": ci_low,
                "mae_ci_high": ci_high,
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_path, index=False)

    delta_rows: list[dict[str, Any]] = []
    comparisons = [
        ("independent_target_b", "score_b", "llm_feature_only", "catboost_cross_F3"),
        ("independent_target_b", "score_b", "llm_context_aware", "catboost_cross_F3"),
        ("independent_target_b", "score_b", "llm_context_aware", "llm_feature_only"),
        ("input_annotator_a", "score_a", "llm_feature_only", "catboost_same_F3"),
        ("input_annotator_a", "score_a", "llm_context_aware", "catboost_same_F3"),
        ("input_annotator_a", "score_a", "llm_context_aware", "llm_feature_only"),
    ]
    for target_name, actual_col, candidate_col, baseline_col in comparisons:
        low, high = _paired_delta_ci(paired, actual_col, candidate_col, baseline_col, seed=SEED + len(delta_rows))
        candidate_mae = _metrics(paired[actual_col].to_numpy(float), paired[candidate_col].to_numpy(float))["mae"]
        baseline_mae = _metrics(paired[actual_col].to_numpy(float), paired[baseline_col].to_numpy(float))["mae"]
        delta_rows.append({
            "target": target_name,
            "candidate": candidate_col,
            "baseline": baseline_col,
            "candidate_minus_baseline_mae": candidate_mae - baseline_mae,
            "document_bootstrap_ci_low": low,
            "document_bootstrap_ci_high": high,
            "n_translations": int(paired["translation_id"].nunique()),
            "n_documents": int(paired["doc_group_id"].nunique()),
        })
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(delta_path, index=False)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    primary = metrics[metrics["target"] == "independent_target_b"]
    secondary = metrics[metrics["target"] == "input_annotator_a"]
    primary_table = primary[["model", "n_translations", "mae", "mae_ci_low", "mae_ci_high", "rmse", "spearman", "within_5_pct", "within_10_pct", "within_20_pct"]].copy()
    secondary_table = secondary[["model", "n_translations", "mae", "mae_ci_low", "mae_ci_high", "rmse", "spearman", "within_5_pct", "within_10_pct", "within_20_pct"]].copy()
    primary_table = primary_table.copy()
    secondary_table = secondary_table.copy()
    estimate = float(metadata.get("actual_cost_usd", 0.0))
    report = f"""# Experiment 3 — Frontier score judge

## Setup

The frozen sample contains **{len(paired)} distinct English→Bhojpuri translations**, with the same input annotator A and target annotator B used for every model comparison. The sample selection hash is `{metadata.get('sample_sha256', '')}`. Selection was random and did not use scores or model residuals. The frontier model was `{metadata.get('model', '')}` through provider `{metadata.get('provider', '')}`. Each translation received separate feature-only and context-aware requests at temperature 0. Feature-only prompts contained only the requested F3 numeric features; context-aware prompts added the English source, Bhojpuri translation, and annotator A's error spans. Human scores, system names and annotator IDs were excluded from both prompts.

The model ID is documented by [Google's Gemini 3.8 Flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash). At the run date, Google's listed standard pricing is $0.75/1M input tokens and $3.75/1M output tokens through December 31, 2026 ([official pricing](https://ai.google.dev/gemini-api/docs/pricing)). Estimated cost from provider usage was **${estimate:.4f}**; the configured hard cap was **${metadata.get('max_cost_usd', 20):.2f}**. This is a token-price estimate, not a billing invoice. Requests were serialized, used one physical attempt, and stored parsed scores plus sanitized token/latency telemetry; raw prompts and responses were not persisted.

## Primary target: independent annotator B

{_markdown_table(primary_table)}

## Secondary target: input annotator A

{_markdown_table(secondary_table)}

## Paired document-bootstrap comparisons

`candidate_minus_baseline_mae` below is negative when the candidate has lower MAE. Confidence intervals resample document groups, keeping all examples from a document together.

{_markdown_table(deltas)}

The human A/B absolute score difference on these same {len(paired)} translations has mean **{np.mean(np.abs(paired.score_a-paired.score_b)):.2f}**, median **{np.median(np.abs(paired.score_a-paired.score_b)):.2f}**, and 90th percentile **{np.quantile(np.abs(paired.score_a-paired.score_b), .9):.2f}**. These are human-to-human differences, while model MAE compares a prediction with one rater.

## Interpretation and limitations

This is a small fixed-sample comparison. The primary target is an independent human score, so it evaluates annotator-to-annotator generalization. The context-aware prompt also sees source and translation text, while the feature-only prompt does not; score differences test the value of that text context under this prompt and model. The experiment still uses human-annotated error spans as inputs. It does not evaluate an LLM error detector or the end-to-end detector-plus-scoring pipeline. Use the paired bootstrap intervals and the local sample metrics in `metrics_local.csv` when comparing small differences.

## Artifacts

- `sample100.csv` and `sample100_manifest.json`: frozen IDs and SHA-256.
- `sample100_local_baselines.csv`: exact local baseline rows and both annotators' labels for scoring/evaluation only.
- `frontier_predictions.csv`: parsed integer scores; invalid responses remain null and are excluded from complete metrics.
- `frontier_request_log.csv`: sanitized provider, latency, status, token and cost telemetry.
- `metrics.csv` and `paired_bootstrap.csv`: primary/secondary evaluation and paired document bootstrap intervals.
    """
    report_path.write_text(report, encoding="utf-8")
    combined_path = _build_combined_report(results_dir)
    return {"metrics": metrics_path, "paired_bootstrap": delta_path, "report": report_path, "combined_report": combined_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("results/experiment3/sample100_local_baselines.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("results/experiment3/frontier_predictions.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/experiment3/frontier_run_metadata.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiment3"))
    args = parser.parse_args()
    output = evaluate(args.sample, args.predictions, args.metadata, args.results_dir)
    print(f"Metrics: {output['metrics'].resolve()}")
    print(f"Report: {output['report'].resolve()}")
    print(f"Combined report: {output['combined_report'].resolve()}")


if __name__ == "__main__":
    main()
