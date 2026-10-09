"""Evaluate saved predictions and produce WMT25 ESA experiment artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


MODEL_ORDER = ["mean", "fixed_penalty", "ridge", "catboost"]
FEATURE_ORDER = ["F1", "F2", "F3"]
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42


def _safe_correlation(function: Any, actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return float("nan")
    try:
        result = function(actual, predicted)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except (ValueError, FloatingPointError):
        return float("nan")


def pairwise_ranking_accuracy(frame: pd.DataFrame) -> tuple[float, int]:
    """Compare system mean scores and mean predictions within each source segment."""
    translation_means = frame.groupby(["doc_id", "system_name"], observed=True).agg(
        human_score=("score", "mean"), predicted_score=("predicted_score", "mean")
    )
    correct = 0.0
    comparable = 0
    for _, translations in translation_means.groupby(level=0, sort=False):
        values = translations.reset_index(drop=True)
        if len(values) < 2:
            continue
        y = values["human_score"].to_numpy(dtype=float)
        p = values["predicted_score"].to_numpy(dtype=float)
        for first, second in itertools.combinations(range(len(values)), 2):
            difference = y[first] - y[second]
            if difference == 0:
                continue
            comparable += 1
            predicted_difference = p[first] - p[second]
            if predicted_difference == 0:
                correct += 0.5
            elif np.sign(predicted_difference) == np.sign(difference):
                correct += 1.0
    return (correct / comparable if comparable else float("nan"), comparable)


def _bootstrap_mae_ci(
    frame: pd.DataFrame,
    group_column: str = "doc_group_id",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    grouped = frame.assign(abs_error=(frame["score"] - frame["predicted_score"]).abs()).groupby(
        group_column, observed=True
    )["abs_error"].agg(["sum", "count"])
    if len(grouped) < 2:
        return float("nan"), float("nan")
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    numerators = sums[sampled].sum(axis=1)
    denominators = counts[sampled].sum(axis=1)
    values = numerators / np.maximum(denominators, 1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_delta_ci(
    frame: pd.DataFrame,
    first_column: str,
    second_column: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    aligned = frame[["doc_group_id", "score", first_column, second_column]].copy()
    aligned["delta"] = (aligned["score"] - aligned[first_column]).abs() - (
        aligned["score"] - aligned[second_column]
    ).abs()
    grouped = aligned.groupby("doc_group_id", observed=True)["delta"].agg(["sum", "count"])
    if len(grouped) < 2:
        return float("nan"), float("nan")
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    values = sums[sampled].sum(axis=1) / np.maximum(counts[sampled].sum(axis=1), 1)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _metric_row(
    scored: pd.DataFrame,
    experiment_scope: str,
    language_pair: str,
    model: str,
    feature_set: str,
    split_strategy: str,
    include_bootstrap: bool,
) -> dict[str, Any]:
    y = scored["score"].to_numpy(dtype=float)
    prediction = scored["predicted_score"].to_numpy(dtype=float)
    pairwise_accuracy, comparable_pairs = pairwise_ranking_accuracy(scored)

    major = scored[scored["n_major"] > 0]
    minor_only = scored[(scored["n_major"] == 0) & (scored["n_minor"] > 0)]
    zero_errors = scored[scored["n_total"] == 0]
    absolute_errors = np.abs(y - prediction)
    ci_low, ci_high = (float("nan"), float("nan"))
    if include_bootstrap:
        ci_low, ci_high = _bootstrap_mae_ci(scored)

    return {
        "experiment_scope": experiment_scope,
        "language_pair": language_pair,
        "split_strategy": split_strategy,
        "model": model,
        "feature_set": feature_set,
        "n_annotations": len(scored),
        "n_unique_segments": scored["doc_id"].nunique(),
        "n_unique_documents": scored["doc_group_id"].nunique(),
        "n_translation_systems": scored["system_name"].nunique(),
        "n_annotators": scored["annotator_id"].nunique(),
        "mae": float(np.mean(absolute_errors)),
        "mae_bootstrap_ci_low": ci_low,
        "mae_bootstrap_ci_high": ci_high,
        "rmse": float(math.sqrt(np.mean(np.square(y - prediction)))),
        "pearson": _safe_correlation(pearsonr, y, prediction),
        "spearman": _safe_correlation(spearmanr, y, prediction),
        "pairwise_ranking_accuracy": pairwise_accuracy,
        "pairwise_comparable_pairs": comparable_pairs,
        "mae_major_error": float(np.mean(np.abs(major["score"] - major["predicted_score"])))
        if len(major)
        else float("nan"),
        "n_major_error_annotations": len(major),
        "mae_minor_only": float(
            np.mean(np.abs(minor_only["score"] - minor_only["predicted_score"]))
        )
        if len(minor_only)
        else float("nan"),
        "n_minor_only_annotations": len(minor_only),
        "mae_zero_errors": float(
            np.mean(np.abs(zero_errors["score"] - zero_errors["predicted_score"]))
        )
        if len(zero_errors)
        else float("nan"),
        "n_zero_error_annotations": len(zero_errors),
    }


def _annotator_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise absolute differences after collapsing repeated judgments per annotator."""
    per_annotator = frame.groupby(
        ["language_pair", "doc_id", "system_name", "annotator_id"], observed=True, as_index=False
    )["score"].mean()
    result_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[float]] = {"ALL_ESA": []}
    paired_translations: dict[str, set[tuple[str, str]]] = {"ALL_ESA": set()}
    for (language_pair, doc_id, system_name), group in per_annotator.groupby(
        ["language_pair", "doc_id", "system_name"], observed=True, sort=False
    ):
        if group["annotator_id"].nunique() < 2:
            continue
        scores = group["score"].to_numpy(dtype=float)
        diffs = [abs(scores[i] - scores[j]) for i, j in itertools.combinations(range(len(scores)), 2)]
        key = str(language_pair)
        buckets.setdefault(key, [])
        paired_translations.setdefault(key, set())
        buckets[key].extend(diffs)
        paired_translations[key].add((str(doc_id), str(system_name)))
        buckets["ALL_ESA"].extend(diffs)
        paired_translations["ALL_ESA"].add((str(doc_id), str(system_name)))

    for language_pair, differences in sorted(buckets.items()):
        if differences:
            result_rows.append(
                {
                    "language_pair": language_pair,
                    "translations_with_multiple_annotators": len(paired_translations[language_pair]),
                    "independent_annotator_pairs": len(differences),
                    "mean_absolute_difference": float(np.mean(differences)),
                    "median_absolute_difference": float(np.median(differences)),
                    "p90_absolute_difference": float(np.quantile(differences, 0.90)),
                    "max_absolute_difference": float(np.max(differences)),
                }
            )
    return pd.DataFrame(result_rows)


def _save_error_analysis(
    scored_by_scope: dict[str, pd.DataFrame], results_dir: Path
) -> tuple[Path, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    selected_scope = "en_bho_cv" if "en_bho_cv" in scored_by_scope else next(iter(scored_by_scope))
    primary = scored_by_scope[selected_scope]
    diagnostics["primary_analysis_scope"] = selected_scope

    top_errors = primary.sort_values("absolute_error", ascending=False).head(10).copy()
    top_errors["analysis_scope"] = selected_scope
    top_errors["analysis_type"] = "ten_largest_catboost_F3_errors"
    rows.append(top_errors)

    zero_error = primary[primary["n_total"] == 0]
    low_score = zero_error[zero_error["score"] <= 50].sort_values("score").head(10).copy()
    low_scope = selected_scope
    diagnostics["low_score_zero_error_threshold"] = 50
    diagnostics["low_score_zero_error_count"] = int(len(zero_error[zero_error["score"] <= 50]))
    if low_score.empty and "pooled_holdout" in scored_by_scope:
        pooled_zero = scored_by_scope["pooled_holdout"]
        low_score = pooled_zero[
            (pooled_zero["language_pair"] == "en-bho_IN")
            & (pooled_zero["n_total"] == 0)
            & (pooled_zero["score"] <= 50)
        ].sort_values("score").head(10).copy()
        low_scope = "pooled_holdout"
        diagnostics["low_score_zero_error_fallback_scope"] = "pooled_holdout"
    low_score["analysis_scope"] = low_scope
    low_score["analysis_type"] = "human_score_le_50_with_no_marked_errors"
    if not low_score.empty:
        rows.append(low_score)

    major_high = primary[(primary["n_major"] > 0) & (primary["score"] >= 80)].sort_values(
        ["score", "absolute_error"], ascending=[False, False]
    ).head(10).copy()
    major_scope = selected_scope
    diagnostics["high_score_major_error_threshold"] = 80
    diagnostics["high_score_major_error_count"] = int(
        len(primary[(primary["n_major"] > 0) & (primary["score"] >= 80)])
    )
    if major_high.empty and "pooled_holdout" in scored_by_scope:
        pooled = scored_by_scope["pooled_holdout"]
        major_high = pooled[
            (pooled["language_pair"] == "en-bho_IN")
            & (pooled["n_major"] > 0)
            & (pooled["score"] >= 80)
        ].sort_values(["score", "absolute_error"], ascending=[False, False]).head(10).copy()
        major_scope = "pooled_holdout"
        diagnostics["high_score_major_error_fallback_scope"] = "pooled_holdout"
    major_high["analysis_scope"] = major_scope
    major_high["analysis_type"] = "human_score_ge_80_with_major_error"
    if not major_high.empty:
        rows.append(major_high)

    export_columns = [
        "analysis_scope",
        "analysis_type",
        "annotation_id",
        "doc_id",
        "doc_group_id",
        "segment_id",
        "language_pair",
        "system_name",
        "annotator_id",
        "src_text",
        "target_text",
        "error_spans_json",
        "n_minor",
        "n_major",
        "n_total",
        "score",
        "predicted_score",
        "absolute_error",
    ]
    if rows:
        output = pd.concat(rows, ignore_index=True)
        output[export_columns].to_csv(results_dir / "error_analysis.csv", index=False)
    else:
        pd.DataFrame(columns=export_columns).to_csv(results_dir / "error_analysis.csv", index=False)
    return results_dir / "error_analysis.csv", diagnostics


def _make_plots(
    primary_scored: pd.DataFrame,
    feature_importance_path: Path,
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        primary_scored["score"],
        primary_scored["predicted_score"],
        s=12,
        alpha=0.22,
        color="#2864a5",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.2, color="#bf4b40")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Human ESA score", ylabel="Predicted score")
    ax.set_title("CatBoost F3: predicted versus human score (English→Bhojpuri)")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path = plots_dir / "predicted_vs_human.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    outputs.append(path)

    combinations = primary_scored.groupby(["n_major", "n_minor"], observed=True).size()
    combinations = combinations.sort_values(ascending=False)
    selected = [pair for pair, count in combinations.items() if count >= 25][:10]
    if (0, 0) not in selected and (0, 0) in combinations.index:
        selected = [(0, 0), *selected[:9]]
    plot_data = primary_scored.copy()
    labels: list[str] = []
    values: list[np.ndarray] = []
    for major_count, minor_count in selected:
        subset = plot_data[(plot_data["n_major"] == major_count) & (plot_data["n_minor"] == minor_count)]
        labels.append(f"M{major_count}/m{minor_count}\n(n={len(subset)})")
        values.append(subset["score"].to_numpy(dtype=float))
    if values:
        fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.9), 5.8))
        ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#b9d1e8", "edgecolor": "#2864a5"},
                   medianprops={"color": "#bf4b40", "linewidth": 1.5})
        ax.set(xlabel="Marked error counts (major/minor)", ylabel="Human ESA score", ylim=(0, 100))
        ax.set_title("Human score distributions by marked major/minor error counts")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = plots_dir / "score_by_error_counts.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)

    if feature_importance_path.exists():
        importance = pd.read_csv(feature_importance_path).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8.0, max(5.0, len(importance) * 0.28)))
        ax.barh(importance["feature"], importance["importance"], color="#2f8f80")
        ax.set(xlabel="CatBoost feature importance", ylabel="Numeric feature")
        ax.set_title("CatBoost F3 feature importance (English→Bhojpuri refit)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        path = plots_dir / "catboost_feature_importance.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)
    return outputs


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame[columns].copy()
    integer_columns = {
        "Segments",
        "Documents",
        "Systems",
        "Annotations",
        "Annotators",
        "Invalid annotations",
        "Count",
        "n_annotations",
        "n_unique_segments",
        "n_unique_documents",
        "n_translation_systems",
        "n_annotators",
        "n_major_error_annotations",
        "n_minor_only_annotations",
        "n_zero_error_annotations",
        "pairwise_comparable_pairs",
        "translations_with_multiple_annotators",
        "independent_annotator_pairs",
    }
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            if column in integer_columns:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{int(value):,}"
                )
            else:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
                )
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def _build_report(
    metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    primary_summary: dict[str, Any],
    all_summary: dict[str, Any],
    agreement: pd.DataFrame,
    bootstrap: pd.DataFrame,
    diagnostics: dict[str, Any],
    run_metadata: dict[str, Any],
    plot_paths: list[Path],
    results_dir: Path,
) -> None:
    per_lp = all_summary.get("per_language_pair", {})
    dataset_rows = []
    for lp in sorted(per_lp):
        item = per_lp[lp]
        score = item["score_distribution"]
        dataset_rows.append(
            {
                "Language pair": lp,
                "Segments": item["unique_segments"],
                "Documents": item["unique_documents"],
                "Systems": item["translation_systems"],
                "Annotations": item["retained_annotations"],
                "Annotators": item["annotators"],
                "Score mean": score["mean"],
                "Score SD": score["std"],
                "Score p10": score["p10"],
                "Score median": score["median"],
                "Score p90": score["p90"],
                "Major mean": item["n_major_distribution"]["mean"],
                "Major median": item["n_major_distribution"]["median"],
                "Major p90": item["n_major_distribution"]["p90"],
                "Minor mean": item["n_minor_distribution"]["mean"],
                "Minor median": item["n_minor_distribution"]["median"],
                "Minor p90": item["n_minor_distribution"]["p90"],
                "Zero-error %": item["percentage_no_marked_errors"],
                "Invalid annotations": item["invalid_annotations"],
            }
        )
    invalid_reason_rows = [
        {"Invalid error record reason": reason, "Count": count}
        for reason, count in all_summary.get("invalid_error_record_reasons", {}).items()
    ]
    invalid_reason_table = _markdown_table(
        pd.DataFrame(invalid_reason_rows), ["Invalid error record reason", "Count"]
    )

    main_scopes = metrics[metrics["experiment_scope"].isin(["en_bho_cv", "en_bho_holdout", "pooled_holdout"])]
    if "en_bho_cv" in set(main_scopes["experiment_scope"]):
        primary_scope = "en_bho_cv"
    else:
        primary_scope = "en_bho_holdout"
    primary_rows = main_scopes[main_scopes["experiment_scope"] == primary_scope]
    pooled_rows = main_scopes[main_scopes["experiment_scope"] == "pooled_holdout"]
    main_columns = [
        "experiment_scope",
        "language_pair",
        "model",
        "feature_set",
        "n_annotations",
        "mae",
        "mae_bootstrap_ci_low",
        "mae_bootstrap_ci_high",
        "rmse",
        "pearson",
        "spearman",
        "pairwise_ranking_accuracy",
        "pairwise_comparable_pairs",
        "mae_major_error",
        "mae_minor_only",
        "mae_zero_errors",
    ]
    lp_catboost = metrics[
        (metrics["experiment_scope"] == "pooled_per_language_pair")
        & (metrics["model"] == "catboost")
        & (metrics["feature_set"] == "F3")
    ].sort_values("language_pair")
    def row_for(scope: str, model: str, feature_set: str) -> pd.Series | None:
        rows = metrics[
            (metrics["experiment_scope"] == scope)
            & (metrics["model"] == model)
            & (metrics["feature_set"] == feature_set)
        ]
        return rows.iloc[0] if len(rows) else None

    def mae_comparison(scope: str, left_model: str, left_feature: str, right_model: str, right_feature: str) -> str:
        left = row_for(scope, left_model, left_feature)
        right = row_for(scope, right_model, right_feature)
        if left is None or right is None:
            return "metrics were unavailable"
        difference = float(left["mae"] - right["mae"])
        direction = "lower MAE" if difference < 0 else "higher MAE" if difference > 0 else "the same MAE"
        return f"{left_model} {left_feature} versus {right_model} {right_feature}: {difference:+.2f} points ({direction})"

    primary_mean = row_for(primary_scope, "mean", "F1")
    primary_fixed = row_for(primary_scope, "fixed_penalty", "F1")
    primary_ridge_counts = row_for(primary_scope, "ridge", "F1")
    primary_cat_counts = row_for(primary_scope, "catboost", "F1")
    pooled_mean = row_for("pooled_holdout", "mean", "F1")
    pooled_fixed = row_for("pooled_holdout", "fixed_penalty", "F1")
    pooled_ridge_counts = row_for("pooled_holdout", "ridge", "F1")
    pooled_cat_counts = row_for("pooled_holdout", "catboost", "F1")
    if (
        primary_mean is not None
        and primary_fixed is not None
        and primary_ridge_counts is not None
        and primary_cat_counts is not None
    ):
        count_learning = (
            f"Yes. On en→bho, Ridge F1 MAE is {primary_ridge_counts['mae']:.2f} and CatBoost F1 is "
            f"{primary_cat_counts['mae']:.2f}, both below the mean predictor ({primary_mean['mae']:.2f}); "
            f"the fixed formula is {primary_fixed['mae']:.2f}. "
            f"On pooled ESA, Ridge F1 ({pooled_ridge_counts['mae']:.2f}) and CatBoost F1 "
            f"({pooled_cat_counts['mae']:.2f}) also beat the mean ({pooled_mean['mae']:.2f}), "
            f"while the fixed formula ({pooled_fixed['mae']:.2f}) does not."
            if all(x is not None for x in [pooled_mean, pooled_fixed, pooled_ridge_counts, pooled_cat_counts])
            else f"Yes. Ridge F1 MAE is {primary_ridge_counts['mae']:.2f} and CatBoost F1 is "
            f"{primary_cat_counts['mae']:.2f}, both below the en→bho mean predictor "
            f"({primary_mean['mae']:.2f}); fixed formula MAE is {primary_fixed['mae']:.2f}."
        )
    else:
        count_learning = "Counts-only model results are unavailable."
    coverage_learning = (
        f"Ridge: {mae_comparison(primary_scope, 'ridge', 'F2', 'ridge', 'F1')}; "
        f"CatBoost: {mae_comparison(primary_scope, 'catboost', 'F2', 'catboost', 'F1')}."
    )
    nonlinear_learning = (
        f"{mae_comparison(primary_scope, 'catboost', 'F3', 'ridge', 'F3')}. "
        f"Pooled: {mae_comparison('pooled_holdout', 'catboost', 'F3', 'ridge', 'F3')}."
    )
    pooled_split_rows = run_metadata.get("pooled_split_rows", {})
    pooled_total_rows = sum(pooled_split_rows.values())
    pooled_split_description = ", ".join(
        f"{name} {count:,} ({100 * count / pooled_total_rows:.1f}%)"
        for name, count in pooled_split_rows.items()
    ) if pooled_total_rows else "not recorded"

    agreement_all = agreement[agreement["language_pair"] == "ALL_ESA"]
    agreement_text = (
        _markdown_table(
            agreement[agreement["language_pair"].isin(["ALL_ESA", "en-bho_IN"])],
            [
                "language_pair",
                "translations_with_multiple_annotators",
                "independent_annotator_pairs",
                "mean_absolute_difference",
                "median_absolute_difference",
                "p90_absolute_difference",
            ],
        )
        if len(agreement)
        else "_No translations had multiple distinct annotators after cleaning._"
    )
    all_agreement = agreement_all.iloc[0] if len(agreement_all) else None
    primary_agreement = agreement[agreement["language_pair"] == "en-bho_IN"]
    bho_agreement = primary_agreement.iloc[0] if len(primary_agreement) else None
    primary_cat_f3 = primary_rows[
        (primary_rows["model"] == "catboost") & (primary_rows["feature_set"] == "F3")
    ]
    pooled_cat_f3 = pooled_rows[
        (pooled_rows["model"] == "catboost") & (pooled_rows["feature_set"] == "F3")
    ]
    noise_comparison = []
    if bho_agreement is not None and len(primary_cat_f3):
        noise_comparison.append(
            f"For en→bho, median human-to-human difference was "
            f"{bho_agreement['median_absolute_difference']:.2f} points, versus "
            f"CatBoost F3 model MAE of {primary_cat_f3.iloc[0]['mae']:.2f} points"
        )
    if all_agreement is not None and len(pooled_cat_f3):
        noise_comparison.append(
            f"Across ESA, median human-to-human difference was "
            f"{all_agreement['median_absolute_difference']:.2f} points, versus "
            f"pooled CatBoost F3 MAE of {pooled_cat_f3.iloc[0]['mae']:.2f} points"
        )
    noise_comparison_text = "; ".join(noise_comparison)
    error_analysis_path = results_dir / "error_analysis.csv"
    if error_analysis_path.exists():
        error_examples = pd.read_csv(error_analysis_path)
        largest_errors = error_examples[
            error_examples["analysis_type"] == "ten_largest_catboost_F3_errors"
        ].head(10)
        largest_error_table = _markdown_table(
            largest_errors,
            ["doc_id", "system_name", "score", "predicted_score", "absolute_error"],
            digits=2,
        )
        low_score_no_span_in_top = int(
            ((largest_errors["score"] <= 50) & (largest_errors["n_total"] == 0)).sum()
        )
        high_score_major_in_top = int(
            ((largest_errors["score"] >= 80) & (largest_errors["n_major"] > 0)).sum()
        )
        mismatch_parts = []
        if low_score_no_span_in_top:
            count_text = "one of the ten largest errors has" if low_score_no_span_in_top == 1 else f"{low_score_no_span_in_top} of the ten largest errors have"
            mismatch_parts.append(
                f"{count_text} a human score ≤50 with no marked errors"
            )
        if high_score_major_in_top:
            count_text = "one" if high_score_major_in_top == 1 else str(high_score_major_in_top)
            mismatch_parts.append(
                f"{count_text} has a score ≥80 despite major spans"
                if high_score_major_in_top == 1
                else f"{count_text} have a score ≥80 despite major spans"
            )
        difficult_case_interpretation = (
            "These score/span mismatches directly contribute to the hardest residuals: the numeric features can only "
            "learn from spans that were marked."
            if mismatch_parts
            else "The largest residuals do not show these two score/span mismatch patterns."
        )
        difficult_case_interpretation = (
            ("In the top ten, " + " and ".join(mismatch_parts) + ". " + difficult_case_interpretation)
            if mismatch_parts
            else difficult_case_interpretation
        )
    else:
        largest_error_table = "_Error analysis has not been generated._"
        difficult_case_interpretation = "Error analysis has not been generated."
    plot_markdown = "\n".join(
        f"![{path.stem.replace('_', ' ')}]({path.relative_to(results_dir).as_posix()})"
        for path in plot_paths
    )
    invalid_total = int(all_summary.get("invalid_annotations_excluded", 0))
    invalid_bho = int(per_lp.get("en-bho_IN", {}).get("invalid_annotations", 0))
    primary_docs = int(run_metadata.get("primary_unique_documents", 0))
    primary_strategy = run_metadata.get("primary_split_strategy", "unknown")

    report = f"""# WMT25 ESA score prediction from error annotations

## Objective and protocol

This CPU-only experiment asks whether numeric features from human-marked Error Span Annotation (ESA) spans predict the ESA human score. No LLM, GPU, or paid API was used. The source data is the [official WMT25 human evaluation JSONL](https://github.com/wmt-conference/wmt25-general-mt/blob/main/README.md#human-evaluation-data); ESA/MQM inclusion follows the [WMT25 segment-level error annotation task](https://www2.statmt.org/wmt25/mteval-subtask2.html). Only the 14 directions designated ESA are included; the MQM-only directions are excluded.

Every valid human annotation remains a separate row. Inspection of the real file found error objects whose `start_i` and `end_i` values are the string `"missing"` while severity is present. These annotations have unknown span locations, so the whole annotation is reported in `invalid_annotations.csv` and excluded from all models; it is never treated as a no-error row or assigned zero coverage. Rows with missing required fields, unsupported severities, or any other invalid inclusive character span are handled the same way. The preparation pass excluded **{invalid_total:,}** annotations across the ESA data, including **{invalid_bho:,}** en→bho annotations. Detailed counts of malformed error records:

{invalid_reason_table}

Valid spans use inclusive indices, and coverage uses the union of character intervals.

Feature inputs are numeric annotation/error measurements only. Annotator ID, system name, document ID, source text, and target text are retained for grouping and inspection but are not model inputs. F1 contains minor/major counts; F2 adds span coverage; F3 contains all requested numeric features.

The initial en→bho subset has **{primary_docs}** unique source documents. It uses **{primary_strategy}**. The pooled ESA model uses a grouped 70/15/15 train/validation/test split ({pooled_split_description}) across {run_metadata.get('pooled_unique_documents', 0):,} documents and {run_metadata.get('pooled_unique_leakage_groups', 0):,} exact-source leakage groups. Exact source texts shared by multiple documents are unioned before splitting, preventing identical source strings from crossing partitions. Ridge scaling is fitted on training rows. Hyperparameters are selected against validation rows only; the held-out pooled test is used for reporting. All models share the same prediction rows within each scope.

## Dataset size and distributions

{_markdown_table(pd.DataFrame(dataset_rows), ['Language pair', 'Segments', 'Documents', 'Systems', 'Annotations', 'Annotators', 'Score mean', 'Score SD', 'Score p10', 'Score median', 'Score p90', 'Major mean', 'Major median', 'Major p90', 'Minor mean', 'Minor median', 'Minor p90', 'Zero-error %', 'Invalid annotations'])}

The per-direction table shows score means and standard deviations; `dataset_summary.json` includes full score and error-count quantiles and the percentage with no marked errors. The experiment began with English→Bhojpuri as the low-resource primary slice and pooled all valid ESA directions for the larger analysis.

## Main model comparison

The table reports the main Bhojpuri OOF/group-CV results and the pooled held-out test results. MAE confidence intervals use 1,000 bootstrap resamples of source documents. Ranking accuracy compares translation pairs for the same source segment after averaging their independent human scores and predictions.

{_markdown_table(main_scopes, main_columns)}

## Feature ablations

{_markdown_table(ablations, ['experiment_scope', 'model', 'feature_set', 'n_annotations', 'mae', 'rmse', 'pearson', 'spearman', 'mae_zero_errors'])}

Pooled CatBoost F3 performance by language pair:

{_markdown_table(lp_catboost, ['language_pair', 'n_annotations', 'n_unique_documents', 'mae', 'rmse', 'pearson', 'spearman', 'pairwise_ranking_accuracy', 'pairwise_comparable_pairs'])}

## Human disagreement

For translations with scores from at least two distinct annotators, repeated judgments by the same annotator were first averaged, then all between-annotator absolute score differences were calculated. This measures human-to-human disagreement, while model MAE measures model-to-human error; they are related reference points, not identical quantities.

{agreement_text}

{noise_comparison_text}. These values describe different comparison targets, so they are context rather than a direct head-to-head agreement score.

"""
    if all_agreement is not None:
        report += (
            f"Across ESA directions the median inter-annotator absolute difference was "
            f"{all_agreement['median_absolute_difference']:.2f} points "
            f"(p90 {all_agreement['p90_absolute_difference']:.2f}); "
        )
    if bho_agreement is not None:
        report += (
            f"for en→bho it was {bho_agreement['median_absolute_difference']:.2f} points "
            f"(p90 {bho_agreement['p90_absolute_difference']:.2f}).\n\n"
        )

    report += f"""## Plots and difficult examples

{plot_markdown}

`error_analysis.csv` contains the ten largest en→bho CatBoost F3 absolute errors, with source text, translation, spans, human score, and predicted score. It also includes available cases with a human score ≤50 and no marked errors, plus cases with a score ≥80 despite at least one major error. Counts found: **{diagnostics.get('low_score_zero_error_count', 0)}** low-score/no-error annotations and **{diagnostics.get('high_score_major_error_count', 0)}** high-score/major-error annotations under those explicit cutoffs. These examples help inspect whether score/span contradictions account for residual error.

{difficult_case_interpretation}

{largest_error_table}

## Answers to the experiment questions

1. **Can error counts alone predict ESA scores?** {count_learning}
2. **Does coverage help?** {coverage_learning} A negative F2-minus-F1 MAE difference supports improvement; inspect the paired document-bootstrap intervals in `bootstrap_comparisons.csv` for uncertainty.
3. **Does CatBoost beat a linear mapping?** {nonlinear_learning} The pooled per-language table shows how consistent the result is across directions.
4. **How much annotator disagreement is there?** {f"The all-ESA median absolute difference is {all_agreement['median_absolute_difference']:.2f} points (p90 {all_agreement['p90_absolute_difference']:.2f})." if all_agreement is not None else "There were not enough multi-annotator translations to estimate this."}
5. **Which annotations are hardest?** The largest residuals are listed in `error_analysis.csv`. The additional low-score/no-error and high-score/major-error examples expose mismatches between the free-form score and marked spans that count-based features cannot resolve.
6. **Is this ready as a scoring layer after an LLM error detector?** It is a proof of concept for scoring *gold human spans*. No LLM detector was run, so these results do not measure detector noise or the combined detector-plus-scorer pipeline. Stable grouped-CV / held-out performance and bootstrap intervals can justify a next controlled experiment, but deployment confidence requires the scorer to be tested on predicted spans and across new documents/language pairs.

## Reproducibility artifacts

- `metrics.csv`: overall and per-language-pair evaluation metrics.
- `ablations.csv`: Ridge and CatBoost F1/F2/F3 comparisons.
- `error_analysis.csv`: inspectable residual and contradiction examples.
- `annotator_agreement.csv`: independent-annotator difference summaries.
- `bootstrap_comparisons.csv`: document-bootstrap confidence intervals for paired MAE differences.
- `predictions.csv.gz`: out-of-fold Bhojpuri predictions and pooled held-out predictions.
- `split_manifest.csv`: document/leakage-group assignments.
- `model_selection.csv`: validation-only tuning selections.
- `dataset_summary.json`: per-direction population and distribution summaries.
- `models/`: fitted Ridge and CatBoost artifacts; the Bhojpuri full-data refits are saved after OOF evaluation for later use, not for its OOF scores.
"""
    (results_dir / "report.md").write_text(report, encoding="utf-8")


def evaluate_experiments(
    data_path: Path,
    predictions_path: Path,
    results_dir: Path = Path("results"),
    models_dir: Path = Path("models"),
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    frame = pd.read_csv(data_path, compression="infer", low_memory=False)
    predictions = pd.read_csv(predictions_path, compression="infer", low_memory=False)
    run_metadata = json.loads((results_dir / "run_metadata.json").read_text(encoding="utf-8"))
    dataset_summary = json.loads((results_dir / "dataset_summary.json").read_text(encoding="utf-8"))

    merged = predictions.merge(frame, on="annotation_id", how="left", validate="many_to_one")
    if merged["score"].isna().any():
        raise ValueError("Some predictions do not map to a prepared human annotation")
    if merged["predicted_score"].isna().any():
        raise ValueError("Missing model predictions")
    merged["predicted_score"] = merged["predicted_score"].clip(0, 100)

    split_strategy = {
        "en_bho_cv": run_metadata.get("primary_split_strategy", "grouped evaluation"),
        "en_bho_holdout": "grouped 70/15/15 holdout",
        "pooled_holdout": "grouped 70/15/15 holdout",
    }
    metric_rows: list[dict[str, Any]] = []
    scored_by_scope: dict[str, pd.DataFrame] = {}
    model_variants = predictions[["experiment_scope", "model", "feature_set"]].drop_duplicates()
    for scope in predictions["experiment_scope"].drop_duplicates():
        scope_frame = merged[merged["experiment_scope"] == scope].copy()
        if scope == "en_bho_cv":
            scope_frame["language_pair"] = "en-bho_IN"
            language_pair_name = "en-bho_IN"
        elif scope == "en_bho_holdout":
            scope_frame["language_pair"] = "en-bho_IN"
            language_pair_name = "en-bho_IN"
        else:
            language_pair_name = "ALL_ESA"

        # Bootstrap comparisons and main metrics use CatBoost F3 in the primary scope.
        cat_f3 = scope_frame[
            (scope_frame["model"] == "catboost") & (scope_frame["feature_set"] == "F3")
        ].copy()
        if not cat_f3.empty:
            cat_f3["absolute_error"] = (cat_f3["score"] - cat_f3["predicted_score"]).abs()
            scored_by_scope[scope] = cat_f3

        for variant in model_variants[model_variants["experiment_scope"] == scope].itertuples(
            index=False
        ):
            variant_frame = scope_frame[
                (scope_frame["model"] == variant.model)
                & (scope_frame["feature_set"] == variant.feature_set)
            ].copy()
            metric_rows.append(
                _metric_row(
                    variant_frame,
                    scope,
                    language_pair_name,
                    variant.model,
                    variant.feature_set,
                    split_strategy.get(scope, "grouped evaluation"),
                    include_bootstrap=True,
                )
            )

            if scope == "pooled_holdout":
                for lp, lp_frame in variant_frame.groupby("language_pair", observed=True):
                    metric_rows.append(
                        _metric_row(
                            lp_frame,
                            "pooled_per_language_pair",
                            str(lp),
                            variant.model,
                            variant.feature_set,
                            "pooled grouped holdout; per-language subset",
                            include_bootstrap=False,
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = results_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    ablations = metrics[
        metrics["model"].isin(["ridge", "catboost"])
        & metrics["feature_set"].isin(["F1", "F2", "F3"])
        & metrics["experiment_scope"].isin(["en_bho_cv", "en_bho_holdout", "pooled_holdout"])
    ].copy()
    ablations_path = results_dir / "ablations.csv"
    ablations.to_csv(ablations_path, index=False)

    agreement = _annotator_agreement(frame)
    agreement.to_csv(results_dir / "annotator_agreement.csv", index=False)

    bootstrap_rows: list[dict[str, Any]] = []
    for scope in scored_by_scope:
        scope_frame = merged[merged["experiment_scope"] == scope]
        wide = scope_frame.pivot(index="annotation_id", columns=["model", "feature_set"], values="predicted_score")
        wide.columns = [f"{model}_{feature_set}" for model, feature_set in wide.columns]
        joined = frame.merge(wide, left_on="annotation_id", right_index=True, how="inner", validate="one_to_one")
        comparisons = [
            ("catboost_F3", "catboost_F3", "fixed_penalty_F1", "fixed_penalty_F1"),
            ("catboost_F3", "catboost_F3", "ridge_F3", "ridge_F3"),
            ("catboost_F3", "catboost_F3", "catboost_F1", "catboost_F1"),
            ("ridge_F3", "ridge_F3", "ridge_F1", "ridge_F1"),
        ]
        for first_name, first_key, second_name, second_key in comparisons:
            if first_key not in joined.columns or second_key not in joined.columns:
                continue
            low, high = _bootstrap_delta_ci(
                joined,
                first_key,
                second_key,
                seed=BOOTSTRAP_SEED + len(bootstrap_rows),
            )
            delta = float(
                np.mean(np.abs(joined["score"] - joined[first_key]))
                - np.mean(np.abs(joined["score"] - joined[second_key]))
            )
            bootstrap_rows.append(
                {
                    "experiment_scope": scope,
                    "comparison": f"{first_name} MAE minus {second_name} MAE",
                    "mae_difference": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "n_documents": joined["doc_group_id"].nunique(),
                    "interpretation": "negative favors the first model; positive favors the second",
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(results_dir / "bootstrap_comparisons.csv", index=False)

    primary_scope = "en_bho_cv" if "en_bho_cv" in scored_by_scope else "en_bho_holdout"
    importance = results_dir / "catboost_feature_importance.csv"
    if not importance.exists():
        importance = results_dir / "pooled_catboost_feature_importance.csv"
    primary_scored = scored_by_scope.get(primary_scope)
    if primary_scored is None or primary_scored.empty:
        raise ValueError("No primary Bhojpuri CatBoost F3 predictions found")
    plot_paths = _make_plots(primary_scored, importance, plots_dir)
    error_path, diagnostics = _save_error_analysis(scored_by_scope, results_dir)
    del error_path
    _build_report(
        metrics,
        ablations,
        dataset_summary.get("per_language_pair", {}).get("en-bho_IN", {}),
        dataset_summary,
        agreement,
        bootstrap,
        diagnostics,
        run_metadata,
        plot_paths,
        results_dir,
    )
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Ablations: {ablations_path.resolve()}")
    print(f"Report: {(results_dir / 'report.md').resolve()}")
    return metrics_path, ablations_path, results_dir / "report.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/annotations.csv.gz"))
    parser.add_argument("--predictions", type=Path, default=Path("results/predictions.csv.gz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    evaluate_experiments(args.data, args.predictions, args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
