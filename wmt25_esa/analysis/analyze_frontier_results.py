"""Compare the completed Kaggle Luna pilot against the frozen local baselines."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "kaggle_gpt_5_6_luna"
BASELINES = ROOT / "results" / "experiment3" / "sample100_local_baselines.csv"
PREDICTIONS = OUT / "kernel-output" / "gpt_5_6_luna_predictions.csv"
RUN_SUMMARY = OUT / "kernel-output" / "gpt_5_6_luna_run.json"
REPLICATES = 2000
SEED = 42


MODEL_COLUMNS = {
    "mean_cross": "pred_mean_cross",
    "fixed_penalty": "pred_fixed_penalty",
    "ridge_cross_F3": "pred_cross_ridge_F3",
    "catboost_cross_F3": "pred_cross_catboost_F3",
    "Luna_feature_only": "feature_only",
    "Luna_context_aware": "context_aware",
}


def pearson_corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def spearman_corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_ranks = pd.Series(actual).rank(method="average").to_numpy(float)
    predicted_ranks = pd.Series(predicted).rank(method="average").to_numpy(float)
    return pearson_corr(actual_ranks, predicted_ranks)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    absolute = np.abs(errors)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "pearson": pearson_corr(actual, predicted),
        "spearman": spearman_corr(actual, predicted),
        "mean_bias_pred_minus_human": float(errors.mean()),
        "within_10_points_pct": float((absolute <= 10).mean() * 100),
    }


def bootstrap_mae(frame: pd.DataFrame, actual: str, predicted: str, seed: int) -> tuple[float, float]:
    grouped = frame.assign(
        _abs=np.abs(frame[actual].to_numpy(float) - frame[predicted].to_numpy(float))
    ).groupby("doc_group_id", observed=True)["_abs"].agg(["sum", "count"]).to_numpy(float)
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(grouped), size=(REPLICATES, len(grouped)))
    sampled = grouped[indexes].sum(axis=1)
    low, high = np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975])
    return float(low), float(high)


def bootstrap_delta(
    frame: pd.DataFrame, actual: str, candidate: str, baseline: str, seed: int
) -> tuple[float, float]:
    grouped = frame.assign(
        _candidate=np.abs(frame[actual].to_numpy(float) - frame[candidate].to_numpy(float)),
        _baseline=np.abs(frame[actual].to_numpy(float) - frame[baseline].to_numpy(float)),
    ).groupby("doc_group_id", observed=True)[["_candidate", "_baseline"]].sum()
    counts = frame.groupby("doc_group_id", observed=True).size().reindex(grouped.index).to_numpy(float)
    values = np.column_stack([grouped.to_numpy(float), counts])
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    sampled = values[indexes].sum(axis=1)
    delta = (sampled[:, 0] - sampled[:, 1]) / sampled[:, 2]
    low, high = np.quantile(delta, [0.025, 0.975])
    return float(low), float(high)


def pairwise_ranking(frame: pd.DataFrame, target: str, predicted: str) -> tuple[float, int]:
    correct = 0.0
    comparable = 0
    for _, group in frame.groupby("doc_id", observed=True):
        rows = group[[target, predicted]].to_numpy(float)
        for left, right in combinations(rows, 2):
            human_delta = left[0] - right[0]
            if human_delta == 0:
                continue
            model_delta = left[1] - right[1]
            comparable += 1
            correct += 1.0 if model_delta * human_delta > 0 else 0.5 if model_delta == 0 else 0.0
    return (100.0 * correct / comparable if comparable else float("nan"), comparable)


def main() -> None:
    local = pd.read_csv(BASELINES, low_memory=False)
    raw = pd.read_csv(PREDICTIONS, low_memory=False)
    if raw["status"].ne("ok").any():
        raise ValueError("The downloaded run contains failed prediction rows")
    wide = raw.pivot(index="translation_id", columns="condition", values="score_prediction")
    if set(wide.columns) != {"feature_only", "context_aware"}:
        raise ValueError("Expected both prompt conditions")
    if local["translation_id"].nunique() != len(local) or set(local["translation_id"]) != set(wide.index):
        raise ValueError("Prediction IDs do not exactly match the frozen 100-translation sample")
    frame = local.merge(wide.reset_index(), on="translation_id", validate="one_to_one")
    frame["human_mean_ab"] = (frame["score_a"].astype(float) + frame["score_b"].astype(float)) / 2

    target_columns = {
        "independent_annotator_b": "score_b",
        "input_annotator_a": "score_a",
        "mean_of_a_and_b": "human_mean_ab",
    }
    metric_rows: list[dict[str, object]] = []
    for target_name, actual in target_columns.items():
        for model_name, predicted in MODEL_COLUMNS.items():
            low, high = bootstrap_mae(frame, actual, predicted, SEED + len(metric_rows))
            metric_rows.append({
                "target": target_name,
                "model": model_name,
                "n_translations": int(frame["translation_id"].nunique()),
                "n_documents": int(frame["doc_group_id"].nunique()),
                **metrics(frame[actual].to_numpy(float), frame[predicted].to_numpy(float)),
                "document_bootstrap_mae_ci_low": low,
                "document_bootstrap_mae_ci_high": high,
            })
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(OUT / "metrics.csv", index=False)

    delta_rows = []
    for target_name, actual in target_columns.items():
        for candidate, baseline in (
            ("Luna_feature_only", "catboost_cross_F3"),
            ("Luna_context_aware", "catboost_cross_F3"),
            ("Luna_context_aware", "Luna_feature_only"),
        ):
            low, high = bootstrap_delta(
                frame, actual, MODEL_COLUMNS[candidate], MODEL_COLUMNS[baseline], SEED + len(delta_rows)
            )
            candidate_mae = metrics(frame[actual].to_numpy(float), frame[MODEL_COLUMNS[candidate]].to_numpy(float))["mae"]
            baseline_mae = metrics(frame[actual].to_numpy(float), frame[MODEL_COLUMNS[baseline]].to_numpy(float))["mae"]
            delta_rows.append({
                "target": target_name,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_minus_baseline_mae": candidate_mae - baseline_mae,
                "document_bootstrap_ci_low": low,
                "document_bootstrap_ci_high": high,
                "n_translations": len(frame),
                "n_documents": int(frame["doc_group_id"].nunique()),
            })
    pd.DataFrame(delta_rows).to_csv(OUT / "paired_bootstrap.csv", index=False)

    ranking_rows = []
    for model_name, predicted in MODEL_COLUMNS.items():
        accuracy, count = pairwise_ranking(frame, "human_mean_ab", predicted)
        ranking_rows.append({
            "model": model_name,
            "pairwise_accuracy_pct": accuracy,
            "comparable_pairs": count,
            "pair_groups": int((frame.groupby("doc_id").size() > 1).sum()),
            "target": "mean_of_a_and_b",
        })
    pd.DataFrame(ranking_rows).to_csv(OUT / "pairwise_ranking.csv", index=False)

    strata = {
        "major_errors_in_input_annotation": frame["n_major"].astype(float) > 0,
        "minor_only_in_input_annotation": (frame["n_major"].astype(float) == 0) & (frame["n_minor"].astype(float) > 0),
        "zero_errors_in_input_annotation": frame["n_total"].astype(float) == 0,
    }
    strata_rows = []
    for stratum, mask in strata.items():
        subset = frame.loc[mask]
        if subset.empty:
            continue
        for model_name, predicted in MODEL_COLUMNS.items():
            strata_rows.append({
                "target": "independent_annotator_b",
                "stratum": stratum,
                "model": model_name,
                "n_translations": len(subset),
                **metrics(subset["score_b"].to_numpy(float), subset[predicted].to_numpy(float)),
            })
    pd.DataFrame(strata_rows).to_csv(OUT / "error_strata.csv", index=False)

    frame["human_disagreement"] = (frame["score_a"].astype(float) - frame["score_b"].astype(float)).abs()
    examples: list[pd.DataFrame] = []
    review_columns = [
        "translation_id", "doc_id", "doc_group_id", "language_pair", "system_name",
        "src_text", "target_text", "score_a", "score_b", "human_mean_ab",
        "n_minor", "n_major", "error_spans_a_json", "error_spans_b_json",
        "feature_only", "context_aware", "human_disagreement",
    ]
    for condition in ("context_aware", "feature_only"):
        difficult = frame.assign(
            absolute_error_vs_b=(frame[condition].astype(float) - frame["score_b"].astype(float)).abs()
        ).nlargest(10, "absolute_error_vs_b").copy()
        difficult["category"] = f"largest_{condition}_error_vs_independent_annotator_b"
        examples.append(difficult[review_columns + ["absolute_error_vs_b", "category"]])
    contradiction_specs = [
        ("low_b_score_no_b_errors", (frame["score_b"] <= 50) & (frame["n_total_b"] == 0)),
        ("high_b_score_with_b_major_error", (frame["score_b"] >= 80) & (frame["n_major_b"] > 0)),
        ("low_a_score_no_a_errors", (frame["score_a"] <= 50) & (frame["n_total"] == 0)),
        ("high_b_score_with_a_major_error", (frame["score_b"] >= 80) & (frame["n_major"] > 0)),
    ]
    for category, mask in contradiction_specs:
        selected = frame.loc[mask].copy()
        if selected.empty:
            continue
        selected["absolute_error_vs_b"] = (selected["context_aware"] - selected["score_b"]).abs()
        selected["category"] = category
        examples.append(selected[review_columns + ["absolute_error_vs_b", "category"]])
    pd.concat(examples, ignore_index=True).to_csv(OUT / "error_examples.csv", index=False)

    run = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    cost_usd = float(run["reported_total_cost_nanodollars"]) / 1_000_000_000
    token_totals = raw[["input_tokens", "output_tokens"]].sum(numeric_only=True)
    disagreement = frame["human_disagreement"]
    major_high_b = int(((frame["score_b"] >= 80) & (frame["n_major_b"] > 0)).sum())
    low_no_b = int(((frame["score_b"] <= 50) & (frame["n_total_b"] == 0)).sum())
    frame["context_absolute_error_vs_b"] = (frame["context_aware"].astype(float) - frame["score_b"].astype(float)).abs()
    top_context = frame.nlargest(10, "context_absolute_error_vs_b")
    high_b_major = frame[(frame["score_b"] >= 80) & (frame["n_major_b"] > 0)]
    context_disagreement_rho = spearman_corr(
        disagreement.to_numpy(float), frame["context_absolute_error_vs_b"].to_numpy(float)
    )
    disagreement_rows = [
        {"group": "all_translations", "n_translations": len(frame),
         "mean_human_disagreement": float(disagreement.mean()),
         "mean_context_error_vs_b": float(frame["context_absolute_error_vs_b"].mean())},
        {"group": "top_10_context_errors_vs_b", "n_translations": len(top_context),
         "mean_human_disagreement": float(top_context["human_disagreement"].mean()),
         "mean_context_error_vs_b": float(top_context["context_absolute_error_vs_b"].mean())},
        {"group": "b_score_80_plus_with_b_major_error", "n_translations": len(high_b_major),
         "mean_human_disagreement": float(high_b_major["human_disagreement"].mean()),
         "mean_context_error_vs_b": float(high_b_major["context_absolute_error_vs_b"].mean())},
    ]
    pd.DataFrame(disagreement_rows).to_csv(OUT / "human_disagreement_analysis.csv", index=False)
    best_b = metrics_frame[metrics_frame["target"] == "independent_annotator_b"].sort_values("mae").iloc[0]
    main_table = metrics_frame[metrics_frame["target"] == "independent_annotator_b"][
        ["model", "mae", "document_bootstrap_mae_ci_low", "document_bootstrap_mae_ci_high", "rmse", "pearson", "spearman"]
    ]
    fmt = lambda x: "—" if pd.isna(x) else f"{float(x):.3f}"
    table = [
        "| Model | MAE (95% doc CI) | RMSE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_table.to_dict("records"):
        table.append(
            f"| {row['model']} | {fmt(row['mae'])} ({fmt(row['document_bootstrap_mae_ci_low'])}, {fmt(row['document_bootstrap_mae_ci_high'])}) | {fmt(row['rmse'])} | {fmt(row['pearson'])} | {fmt(row['spearman'])} |"
        )

    report = f"""# GPT-5.6 Luna WMT25 ESA pilot

## Run

- Kaggle model: `openai/gpt-5.6-luna` (display slug `gpt-5.6-luna`)
- Kaggle task version: 2; notebook run: `Run #1`, completed
- Sample: {frame['translation_id'].nunique()} English→Bhojpuri translations from {frame['doc_group_id'].nunique()} documents; {frame['system_name'].nunique()} systems; two human annotations per translation
- Prompt calls: {len(raw)} total ({int((raw['condition'] == 'feature_only').sum())} feature-only, {int((raw['condition'] == 'context_aware').sum())} context-aware); all succeeded
- Reported Model Proxy cost: **${cost_usd:.6f}**; tokens: {int(token_totals.get('input_tokens', 0)):,} input and {int(token_totals.get('output_tokens', 0)):,} output
- Exact frozen-sample digest: `{run['sample_sha256']}`

The feature-only prompt sees only the 16 numeric error/length features. The context-aware prompt also sees the English source and Bhojpuri translation. Both use the same error annotation features and neither sees human scores, annotator IDs, or system names.

## Results

Primary comparison uses annotator B's independent human score, matching the existing local experiment. The document bootstrap resamples the {frame['doc_group_id'].nunique()} document groups. It is descriptive at this sample size.

{chr(10).join(table)}

On this sample, the lowest MAE was **{best_b['model']}** ({best_b['mae']:.2f}). See `metrics.csv` for the same models against annotator A and the mean of A/B. `paired_bootstrap.csv` gives paired document-bootstrap intervals for Luna versus cross-annotator CatBoost and between prompt conditions. `pairwise_ranking.csv` reports within-source ranking against the mean human score; ties in human scores are excluded and predicted ties count as half-correct.

## Error and annotator review

Mean A/B absolute disagreement was {disagreement.mean():.2f} points (median {disagreement.median():.2f}; 90th percentile {disagreement.quantile(.9):.2f}). Context-aware model errors against annotator B are not directly the same quantity as human disagreement.

- B assigned 80+ while marking at least one major error in **{major_high_b}** translations.
- B assigned 50 or below with no B-marked errors in **{low_no_b}** translations.
- `error_examples.csv` contains the ten largest errors for each Luna condition, plus any high-score/major-error or low-score/zero-error contradictions, with source, translation, and both annotators' spans.
- Spearman correlation between context-aware absolute error vs B and A/B disagreement is **{context_disagreement_rho:.3f}**. The top ten context-aware errors average {top_context['human_disagreement'].mean():.2f} points of A/B disagreement, versus {disagreement.mean():.2f} overall. The high-B/major-error subset has mean context-aware error {high_b_major['context_absolute_error_vs_b'].mean():.2f} (n={len(high_b_major)}). These are descriptive checks, not causal evidence.
- `human_disagreement_analysis.csv` records those group summaries.

## Scope and interpretation

This is a 100-translation, one-language-pair pilot, not the full WMT25 ESA dataset. The comparison is not a head-to-head model training experiment: Luna judges the supplied sample, while Ridge and CatBoost are existing cross-annotator local baselines. Human annotation disagreement is substantial, so these results do not establish a stable replacement scoring layer. The context-aware condition is particularly different from an error-detector-only scorer because it can use the source and translation text.

The Kaggle creation-time notebook run already produced the completed `Run #1` artifacts. The separate `tasks run` queue remains empty; no second run was launched, to avoid repeating the 200 paid-budget prompts.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
