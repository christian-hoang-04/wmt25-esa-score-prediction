"""Cross-annotator generalization experiments for WMT25 ESA annotations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from prepare_data import FEATURE_SETS, NUMERIC_FEATURES
from train import CATBOOST_EARLY_STOPPING, CATBOOST_GRID, CATBOOST_ITERATIONS, CATBOOST_THREADS, RIDGE_ALPHAS


SEED = 42
BOOTSTRAP_REPLICATES = 1000
PRIMARY_LP = "en-bho_IN"


def stable_translation_id(language_pair: str, doc_id: str, system_name: str) -> str:
    """Hash all stable translation identity fields, including document and system."""
    raw = "\x1f".join((str(language_pair), str(doc_id), str(system_name))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _as_spans(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_annotator_views(annotations: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated rows within a translation/annotator into one independent view."""
    identity = ["language_pair", "doc_id", "system_name"]
    annotations = annotations.copy()
    annotations["translation_id"] = [
        stable_translation_id(lp, doc, system)
        for lp, doc, system in annotations[identity].itertuples(index=False, name=None)
    ]
    group_columns = ["translation_id", "annotator_id"]
    # Error features come from one real annotation, not a fractional blend.
    representatives = annotations.sort_values("annotation_order", kind="stable").drop_duplicates(
        group_columns, keep="first"
    )
    aggregates = annotations.groupby(group_columns, sort=False, observed=True).agg(
        score=("score", "mean"),
        n_raw_rows_for_annotator=("annotation_id", "size"),
        annotation_ids=("annotation_id", list),
    ).reset_index()
    views = representatives[[
        "translation_id", "doc_id", "doc_group_id", "segment_id", "language_pair", "domain",
        "system_name", "src_text", "target_text", "annotation_order", *NUMERIC_FEATURES,
        "annotator_id", "error_spans_json",
    ]].merge(aggregates, on=group_columns, how="inner", validate="one_to_one")
    views["annotator_id"] = views["annotator_id"].astype(str)
    views["annotation_ids_json"] = views["annotation_ids"].map(
        lambda values: json.dumps(sorted(map(str, values)), ensure_ascii=False)
    )
    views["error_spans_json"] = views["error_spans_json"].map(
        lambda value: json.dumps(_as_spans(value), ensure_ascii=False, separators=(",", ":"))
    )
    view_frame = views.drop(columns="annotation_ids")
    counts = view_frame.groupby("translation_id")["annotator_id"].transform("nunique")
    view_frame["n_annotators_for_translation"] = counts.astype(int)
    return view_frame


def build_directed_pairs(views: pd.DataFrame) -> pd.DataFrame:
    """Create both directed inputs for every unordered pair of distinct annotators."""
    left = views.add_prefix("a_")
    right = views.add_prefix("b_")
    unordered = left.merge(right, left_on="a_translation_id", right_on="b_translation_id", how="inner")
    unordered = unordered[unordered["a_annotator_id"].astype(str) < unordered["b_annotator_id"].astype(str)].copy()
    if unordered.empty:
        return pd.DataFrame()
    unordered["n_pairs"] = unordered.groupby("a_translation_id")["a_annotator_id"].transform("size")
    unordered["pair_weight"] = 1.0 / (2.0 * unordered["n_pairs"].astype(float))
    unordered["pair_id"] = [
        hashlib.sha256(f"{translation}\x1f{annotator_a}\x1f{annotator_b}".encode("utf-8")).hexdigest()[:24]
        for translation, annotator_a, annotator_b in unordered[["a_translation_id", "a_annotator_id", "b_annotator_id"]].itertuples(index=False, name=None)
    ]

    def orient(source_prefix: str, target_prefix: str) -> pd.DataFrame:
        source_columns = {
            "translation_id": f"{source_prefix}_translation_id",
            "doc_id": f"{source_prefix}_doc_id",
            "doc_group_id": f"{source_prefix}_doc_group_id",
            "segment_id": f"{source_prefix}_segment_id",
            "language_pair": f"{source_prefix}_language_pair",
            "domain": f"{source_prefix}_domain",
            "system_name": f"{source_prefix}_system_name",
            "annotator_a": f"{source_prefix}_annotator_id",
            "annotator_b": f"{target_prefix}_annotator_id",
            "score_a": f"{source_prefix}_score",
            "score_b": f"{target_prefix}_score",
            "src_text": f"{source_prefix}_src_text",
            "target_text": f"{source_prefix}_target_text",
            "annotation_ids_a_json": f"{source_prefix}_annotation_ids_json",
            "annotation_ids_b_json": f"{target_prefix}_annotation_ids_json",
            "error_spans_a_json": f"{source_prefix}_error_spans_json",
            "error_spans_b_json": f"{target_prefix}_error_spans_json",
            "n_raw_rows_a": f"{source_prefix}_n_raw_rows_for_annotator",
            "n_raw_rows_b": f"{target_prefix}_n_raw_rows_for_annotator",
            "n_minor_b": f"{target_prefix}_n_minor",
            "n_major_b": f"{target_prefix}_n_major",
            "n_total_b": f"{target_prefix}_n_total",
            "pair_weight": "pair_weight",
            "pair_id": "pair_id",
        }
        out = unordered[list(source_columns.values())].rename(columns={value: key for key, value in source_columns.items()})
        for feature in NUMERIC_FEATURES:
            out[feature] = unordered[f"{source_prefix}_{feature}"].to_numpy(float)
        out["directed_pair_id"] = out["pair_id"] + ":" + out["annotator_a"].astype(str) + ":" + out["annotator_b"].astype(str)
        return out

    pairs = pd.concat([orient("a", "b"), orient("b", "a")], ignore_index=True)
    pair_weight_sums = pairs.groupby("translation_id")["pair_weight"].sum()
    if not np.allclose(pair_weight_sums.to_numpy(), 1.0):
        raise AssertionError("Directed pair weights must sum to one per translation")
    return pairs


def attach_saved_splits(pairs: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach the original Experiment 1 fold/holdout assignments to each document."""
    pairs = pairs.copy()
    pairs["experiment_scope"] = np.where(pairs["language_pair"].eq(PRIMARY_LP), "en_bho_cv", "pooled_holdout")
    assignments: list[pd.DataFrame] = []
    bho = manifest[(manifest["experiment_scope"] == "en_bho_cv") & (manifest["split"] == "outer_test")]
    bho = bho[["doc_group_id", "leakage_group_id", "fold"]].drop_duplicates()
    if bho["doc_group_id"].duplicated().any():
        raise ValueError("A Bho document appears as outer test in multiple folds")
    bho = bho.assign(experiment_scope="en_bho_cv", split="cv")
    assignments.append(bho)
    pooled = manifest[manifest["experiment_scope"] == "pooled_holdout"]
    pooled = pooled[["doc_group_id", "leakage_group_id", "split"]].drop_duplicates()
    if pooled["doc_group_id"].duplicated().any():
        raise ValueError("A pooled document has multiple split assignments")
    pooled = pooled.assign(fold="holdout", experiment_scope="pooled_holdout")
    assignments.append(pooled)
    assignment = pd.concat(assignments, ignore_index=True)
    pairs = pairs.merge(
        assignment, on=["doc_group_id", "experiment_scope"], how="left", validate="many_to_one"
    )
    if pairs[["leakage_group_id", "split", "fold"]].isna().any().any():
        missing = pairs.loc[pairs["split"].isna(), "doc_group_id"].nunique()
        raise ValueError(f"{missing} paired documents are missing saved split assignments")
    return pairs


def validate_splits(pairs: pd.DataFrame) -> dict[str, Any]:
    """Fail if source documents, duplicate-source leakage groups or reciprocal pairs cross splits."""
    audit: dict[str, Any] = {}
    bho = pairs[pairs["experiment_scope"] == "en_bho_cv"]
    for fold, test in bho.groupby("fold"):
        train = bho[bho["fold"] != fold]
        for key in ("doc_group_id", "leakage_group_id", "src_text", "translation_id", "pair_id"):
            overlap = set(train[key].astype(str)).intersection(test[key].astype(str))
            if overlap:
                raise ValueError(f"Bho fold {fold} leaks {len(overlap)} {key} values")
        audit[str(fold)] = {"train_docs": int(train.doc_group_id.nunique()), "test_docs": int(test.doc_group_id.nunique())}
    pooled = pairs[pairs["experiment_scope"] == "pooled_holdout"]
    partitions = {name: pooled[pooled["split"] == name] for name in ("train", "validation", "test")}
    for left, right in itertools.combinations(partitions, 2):
        for key in ("doc_group_id", "leakage_group_id", "src_text", "translation_id", "pair_id"):
            overlap = set(partitions[left][key].astype(str)).intersection(partitions[right][key].astype(str))
            if overlap:
                raise ValueError(f"Pooled {left}/{right} leak {len(overlap)} {key} values")
    audit["pooled"] = {name: int(part.doc_group_id.nunique()) for name, part in partitions.items()}
    return audit


def _group_inner_split(training_pairs: pd.DataFrame) -> tuple[set[str], set[str]]:
    groups = training_pairs[["doc_group_id", "leakage_group_id"]].drop_duplicates()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(groups, groups=groups["leakage_group_id"]))
    return set(groups.iloc[train_idx]["doc_group_id"]), set(groups.iloc[val_idx]["doc_group_id"])


def _weighted_rmse(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.square(y - pred), weights=weights)))


def _ridge_fit_predict(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str], alpha: float
) -> tuple[np.ndarray, StandardScaler, Ridge]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features], sample_weight=train["fit_weight"])
    model = Ridge(alpha=float(alpha))
    model.fit(x_train, train["target_score"], sample_weight=train["fit_weight"])
    return np.clip(model.predict(scaler.transform(test[features])), 0, 100), scaler, model


def _tune_ridge(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[float, float]:
    best_alpha, best_rmse = None, float("inf")
    for alpha in RIDGE_ALPHAS:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[features], sample_weight=train["fit_weight"])
        model = Ridge(alpha=float(alpha)).fit(
            x_train, train["target_score"], sample_weight=train["fit_weight"]
        )
        prediction = np.clip(model.predict(scaler.transform(validation[features])), 0, 100)
        value = _weighted_rmse(
            validation["target_score"].to_numpy(float), prediction, validation["fit_weight"].to_numpy(float)
        )
        if value < best_rmse:
            best_alpha, best_rmse = float(alpha), value
    assert best_alpha is not None
    return best_alpha, best_rmse


def _make_catboost(params: dict[str, Any], iterations: int = CATBOOST_ITERATIONS) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=int(params["depth"]),
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=SEED,
        thread_count=CATBOOST_THREADS,
        allow_writing_files=False,
        verbose=False,
    )


def _tune_catboost(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[dict[str, Any], float, int]:
    best_params: dict[str, Any] | None = None
    best_rmse, best_iterations = float("inf"), CATBOOST_ITERATIONS
    train_pool = Pool(train[features], train["target_score"], weight=train["fit_weight"])
    val_pool = Pool(validation[features], validation["target_score"], weight=validation["fit_weight"])
    for params in CATBOOST_GRID:
        model = _make_catboost(params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=CATBOOST_EARLY_STOPPING, use_best_model=True)
        prediction = np.clip(model.predict(validation[features]), 0, 100)
        value = _weighted_rmse(
            validation["target_score"].to_numpy(float), prediction, validation["fit_weight"].to_numpy(float)
        )
        if value < best_rmse:
            best_params = dict(params)
            best_rmse = value
            best_iterations = max(1, int(model.get_best_iteration()) + 1)
    assert best_params is not None
    return best_params, best_rmse, best_iterations


def _fit_final(
    model_type: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_dir: Path,
    artifact_name: str,
    refit_with_validation: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if model_type == "ridge":
        alpha, validation_rmse = _tune_ridge(train, validation, features)
        final = pd.concat([train, validation], ignore_index=True) if refit_with_validation else train
        scaler = StandardScaler().fit(final[features], sample_weight=final["fit_weight"])
        estimator = Ridge(alpha=alpha).fit(
            scaler.transform(final[features]), final["target_score"], sample_weight=final["fit_weight"]
        )
        prediction = np.clip(estimator.predict(scaler.transform(test[features])), 0, 100)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "model": estimator, "features": features}, model_dir / f"{artifact_name}.joblib")
        return prediction, {"alpha": alpha, "validation_rmse": validation_rmse, "iterations": None}, None

    params, validation_rmse, iterations = _tune_catboost(train, validation, features)
    final = pd.concat([train, validation], ignore_index=True) if refit_with_validation else train
    estimator = _make_catboost(params, iterations=iterations)
    estimator.fit(Pool(final[features], final["target_score"], weight=final["fit_weight"]))
    prediction = np.clip(estimator.predict(test[features]), 0, 100)
    model_dir.mkdir(parents=True, exist_ok=True)
    estimator.save_model(str(model_dir / f"{artifact_name}.cbm"))
    importance = estimator.get_feature_importance()
    return prediction, {
        "depth": params["depth"], "l2_leaf_reg": params["l2_leaf_reg"],
        "validation_rmse": validation_rmse, "iterations": iterations,
    }, importance


def _target_frame(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    result = frame.copy()
    if regime == "same":
        result["target_score"] = result["score"]
        result["fit_weight"] = 1.0 / result["n_annotators_for_translation"]
    else:
        result["target_score"] = result["score_b"]
        result["fit_weight"] = result["pair_weight"]
    return result


def _weighted_corr(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    if len(actual) < 2 or np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        return float("nan")
    mean_a = np.average(actual, weights=weights)
    mean_p = np.average(predicted, weights=weights)
    covariance = np.average((actual - mean_a) * (predicted - mean_p), weights=weights)
    variance_a = np.average(np.square(actual - mean_a), weights=weights)
    variance_p = np.average(np.square(predicted - mean_p), weights=weights)
    return float(covariance / math.sqrt(variance_a * variance_p)) if variance_a > 0 and variance_p > 0 else float("nan")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_quantile(values, weights, 0.5)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def _metric_values(frame: pd.DataFrame, actual: str, prediction: str, weight: str) -> dict[str, float]:
    actual_values = frame[actual].to_numpy(float)
    predicted_values = frame[prediction].to_numpy(float)
    weights = frame[weight].to_numpy(float)
    absolute = np.abs(actual_values - predicted_values)
    return {
        "mae": float(np.average(absolute, weights=weights)),
        "rmse": _weighted_rmse(actual_values, predicted_values, weights),
        "median_absolute_error": _weighted_median(absolute, weights),
        "pearson": _weighted_corr(actual_values, predicted_values, weights),
        "spearman": _weighted_corr(rankdata(actual_values, method="average"), rankdata(predicted_values, method="average"), weights),
        "mean_bias_pred_minus_human": float(np.average(predicted_values - actual_values, weights=weights)),
        "within_10_pct": float(np.average(absolute <= 10, weights=weights) * 100.0),
    }


def _bootstrap_mae_ci(
    frame: pd.DataFrame, actual: str, prediction: str, weight: str, cluster: str, seed: int
) -> tuple[float, float]:
    grouped = frame.assign(
        _weighted_abs=np.abs(frame[actual].to_numpy(float) - frame[prediction].to_numpy(float)) * frame[weight].to_numpy(float),
        _weight=frame[weight].to_numpy(float),
    ).groupby(cluster, observed=True)[["_weighted_abs", "_weight"]].sum()
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = grouped.to_numpy(float)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sums = values[sampled].sum(axis=1)
    maes = sums[:, 0] / sums[:, 1]
    low, high = np.quantile(maes, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_delta_mae_ci(
    frame: pd.DataFrame, actual: str, left: str, right: str, weight: str, cluster: str, seed: int
) -> tuple[float, float]:
    grouped = frame.assign(
        _left=np.abs(frame[actual].to_numpy(float) - frame[left].to_numpy(float)) * frame[weight].to_numpy(float),
        _right=np.abs(frame[actual].to_numpy(float) - frame[right].to_numpy(float)) * frame[weight].to_numpy(float),
        _weight=frame[weight].to_numpy(float),
    ).groupby(cluster, observed=True)[["_left", "_right", "_weight"]].sum()
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = grouped.to_numpy(float)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sums = values[sampled].sum(axis=1)
    deltas = sums[:, 0] / sums[:, 2] - sums[:, 1] / sums[:, 2]
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_two_target_gap_ci(
    frame: pd.DataFrame, prediction: str, weight: str, cluster: str, seed: int
) -> tuple[float, float]:
    grouped = frame.assign(
        _cross=np.abs(frame["score_b"].to_numpy(float) - frame[prediction].to_numpy(float)) * frame[weight].to_numpy(float),
        _same=np.abs(frame["score_a"].to_numpy(float) - frame[prediction].to_numpy(float)) * frame[weight].to_numpy(float),
        _weight=frame[weight].to_numpy(float),
    ).groupby(cluster, observed=True)[["_cross", "_same", "_weight"]].sum()
    if len(grouped) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = grouped.to_numpy(float)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    sums = values[sampled].sum(axis=1)
    gaps = (sums[:, 0] - sums[:, 1]) / sums[:, 2]
    low, high = np.quantile(gaps, [0.025, 0.975])
    return float(low), float(high)


def _prepare_fit_data(
    views: pd.DataFrame, pairs: pd.DataFrame, train_docs: set[str], val_docs: set[str], test_docs: set[str]
) -> dict[str, Any]:
    same = views.copy()
    same["target_score"] = same["score"].astype(float)
    same["fit_weight"] = 1.0 / same["n_annotators_for_translation"].astype(float)
    cross = pairs.copy()
    cross["target_score"] = cross["score_b"].astype(float)
    cross["fit_weight"] = cross["pair_weight"].astype(float)
    return {
        "same_train": same[same["doc_group_id"].astype(str).isin(train_docs)].copy(),
        "same_val": same[same["doc_group_id"].astype(str).isin(val_docs)].copy(),
        "same_test": same[same["doc_group_id"].astype(str).isin(test_docs)].copy(),
        "cross_train": cross[cross["doc_group_id"].astype(str).isin(train_docs)].copy(),
        "cross_val": cross[cross["doc_group_id"].astype(str).isin(val_docs)].copy(),
        "cross_test": cross[cross["doc_group_id"].astype(str).isin(test_docs)].copy(),
    }


def _run_one_fold(
    views: pd.DataFrame,
    pairs: pd.DataFrame,
    scope: str,
    fold_name: str,
    train_docs: set[str],
    val_docs: set[str],
    test_docs: set[str],
    model_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    data = _prepare_fit_data(views, pairs, train_docs, val_docs, test_docs)
    test_pairs = data["cross_test"].copy().reset_index(drop=True)
    if test_pairs.empty:
        raise ValueError(f"No paired test annotations for {scope}/{fold_name}")

    refit_with_validation = scope == "en_bho_cv"
    if refit_with_validation:
        same_mean_data = pd.concat([data["same_train"], data["same_val"]], ignore_index=True)
        cross_mean_data = pd.concat([data["cross_train"], data["cross_val"]], ignore_index=True)
    else:
        same_mean_data, cross_mean_data = data["same_train"], data["cross_train"]
    same_mean = float(np.average(same_mean_data["target_score"], weights=same_mean_data["fit_weight"]))
    cross_mean = float(np.average(cross_mean_data["target_score"], weights=cross_mean_data["fit_weight"]))
    test_pairs["pred_mean_same"] = same_mean
    test_pairs["pred_mean_cross"] = cross_mean
    fixed = np.clip(100.0 - 5.0 * test_pairs["n_major"].to_numpy(float) - test_pairs["n_minor"].to_numpy(float), 0, 100)
    test_pairs["pred_fixed_penalty"] = fixed

    selections: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for feature_set, features in FEATURE_SETS.items():
        for regime in ("same", "cross"):
            train = data[f"{regime}_train"]
            validation = data[f"{regime}_val"]
            test_frame = test_pairs
            for model_type in ("ridge", "catboost"):
                artifact_name = f"{scope}_{fold_name}_{regime}_{model_type}_{feature_set}"
                prediction, params, importance = _fit_final(
                    model_type, train, validation, test_frame, features, model_dir,
                    artifact_name, refit_with_validation,
                )
                prefix = f"pred_{regime}_{model_type}_{feature_set}"
                test_pairs[prefix] = prediction
                selections.append({
                    "experiment_scope": scope, "fold": fold_name, "regime": regime,
                    "model": model_type, "feature_set": feature_set, **params,
                    "train_documents": len(train_docs), "validation_documents": len(val_docs),
                    "test_documents": len(test_docs), "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)), "test_pairs": int(len(test_pairs)),
                })
                if importance is not None:
                    importances.extend({
                        "experiment_scope": scope, "fold": fold_name, "regime": regime,
                        "feature_set": feature_set, "feature": feature, "importance": float(value),
                    } for feature, value in zip(features, importance, strict=True))

    test_pairs["split_scope"] = scope
    test_pairs["evaluation_fold"] = fold_name
    return test_pairs, selections, importances


def _make_sample100(pairs: pd.DataFrame, predictions: pd.DataFrame, results_dir: Path) -> tuple[Path, Path]:
    """Freeze an unbiased, seeded Bhojpuri sample before any frontier model call."""
    experiment3_dir = results_dir / "experiment3"
    experiment3_dir.mkdir(parents=True, exist_ok=True)
    eligible = pairs[(pairs["language_pair"] == PRIMARY_LP) & (pairs["experiment_scope"] == "en_bho_cv")]
    candidate_translation_ids = sorted(eligible["translation_id"].unique())
    if len(candidate_translation_ids) < 100:
        raise ValueError(f"Experiment 3 requires 100 eligible Bho translations; found {len(candidate_translation_ids)}")
    rng = np.random.default_rng(SEED)
    selected_translations = sorted(rng.choice(candidate_translation_ids, size=100, replace=False).tolist())
    frozen_rows: list[dict[str, Any]] = []
    for translation_id in selected_translations:
        options = eligible[eligible["translation_id"] == translation_id].sort_values("directed_pair_id")
        pair_ids = options["pair_id"].drop_duplicates().tolist()
        pair_id = str(rng.choice(pair_ids))
        directed = options[options["pair_id"] == pair_id]
        row = directed.iloc[int(rng.integers(0, len(directed)))].to_dict()
        frozen_rows.append({
            "translation_id": translation_id,
            "pair_id": pair_id,
            "directed_pair_id": row["directed_pair_id"],
            "doc_id": row["doc_id"],
            "doc_group_id": row["doc_group_id"],
            "language_pair": row["language_pair"],
            "system_name": row["system_name"],
            "annotator_a": row["annotator_a"],
            "annotator_b": row["annotator_b"],
            "selection_seed": SEED,
            "selection_scope": "eligible en-bho translations, random distinct translation sample; annotator pair/orientation random",
        })
    frozen = pd.DataFrame(frozen_rows).sort_values("translation_id").reset_index(drop=True)
    sample_path = experiment3_dir / "sample100.csv"
    frozen.to_csv(sample_path, index=False)
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    metadata = {
        "sample_size": int(len(frozen)),
        "seed": SEED,
        "language_pair": PRIMARY_LP,
        "eligible_translations": len(candidate_translation_ids),
        "selection_rule": "Uniform random sample of distinct eligible translation IDs; no scores, features or model errors used to choose IDs. One eligible unordered annotator pair and orientation chosen uniformly per selected translation.",
        "sample_file_sha256": digest,
        "frozen_ids": frozen["translation_id"].tolist(),
    }
    metadata_path = experiment3_dir / "sample100_manifest.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_columns = [
        "translation_id", "pair_id", "directed_pair_id", "score_a", "score_b", "pair_weight",
        "src_text", "target_text", "error_spans_a_json", "error_spans_b_json", *NUMERIC_FEATURES,
        "n_minor_b", "n_major_b", "n_total_b", "pred_mean_same", "pred_mean_cross",
        "pred_fixed_penalty",
        *[f"pred_{regime}_{model}_{feature_set}" for regime in ("same", "cross") for model in ("ridge", "catboost") for feature_set in FEATURE_SETS],
    ]
    local = frozen.merge(
        predictions[prediction_columns],
        on=["translation_id", "pair_id", "directed_pair_id"],
        how="left", validate="one_to_one",
    )
    if local["pred_cross_catboost_F3"].isna().any():
        raise ValueError("Some frozen sample rows lack held-out CatBoost predictions")
    local_path = experiment3_dir / "sample100_local_baselines.csv"
    local.to_csv(local_path, index=False)
    return sample_path, local_path


def _local_baseline_metric_rows(local_sample: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_columns = [
        ("mean", "pred_mean_cross"),
        ("ridge_cross_F3", "pred_cross_ridge_F3"),
        ("catboost_cross_F3", "pred_cross_catboost_F3"),
        ("catboost_same_F3", "pred_same_catboost_F3"),
    ]
    for target_name, target_col, matched_baseline in (
        ("independent_target_b", "score_b", "pred_cross_catboost_F3"),
        ("input_annotator_a", "score_a", "pred_same_catboost_F3"),
    ):
        for model_name, prediction_column in model_columns:
            actual = local_sample[target_col].to_numpy(float)
            predicted = local_sample[prediction_column].to_numpy(float)
            values = _metric_values(local_sample, target_col, prediction_column, "pair_weight")
            low, high = _bootstrap_mae_ci(
                local_sample, target_col, prediction_column, "pair_weight", "doc_group_id",
                SEED + len(rows),
            )
            delta_low, delta_high = _bootstrap_delta_mae_ci(
                local_sample, target_col, prediction_column, matched_baseline, "pair_weight", "doc_group_id",
                SEED + 500 + len(rows),
            )
            rows.append({
                "target": target_name, "model": model_name,
                "n_translations": int(local_sample["translation_id"].nunique()),
                "n_documents": int(local_sample["doc_group_id"].nunique()),
                **values, "mae_ci_low": low, "mae_ci_high": high,
                "matched_catboost_baseline": matched_baseline,
                "delta_mae_vs_matched_catboost": values["mae"] - _metric_values(
                    local_sample, target_col, matched_baseline, "pair_weight"
                )["mae"],
                "delta_ci_low": delta_low, "delta_ci_high": delta_high,
            })
    # A second human's score is a useful practical reference for the learned
    # cross-annotator mapping, even though it is not an automatic model.
    actual = local_sample["score_b"].to_numpy(float)
    predicted = local_sample["score_a"].to_numpy(float)
    values = _metric_values(local_sample, "score_b", "score_a", "pair_weight")
    low, high = _bootstrap_mae_ci(
        local_sample, "score_b", "score_a", "pair_weight", "doc_group_id", SEED + len(rows)
    )
    delta_low, delta_high = _bootstrap_delta_mae_ci(
        local_sample, "score_b", "score_a", "pred_cross_catboost_F3", "pair_weight",
        "doc_group_id", SEED + 500 + len(rows),
    )
    rows.append({
        "target": "independent_target_b", "model": "human_input_annotator_a",
        "n_translations": int(local_sample["translation_id"].nunique()),
        "n_documents": int(local_sample["doc_group_id"].nunique()),
        **values, "mae_ci_low": low, "mae_ci_high": high,
        "matched_catboost_baseline": "pred_cross_catboost_F3",
        "delta_mae_vs_matched_catboost": values["mae"] - _metric_values(
            local_sample, "score_b", "pred_cross_catboost_F3", "pair_weight"
        )["mae"],
        "delta_ci_low": delta_low, "delta_ci_high": delta_high,
    })
    return rows


def _append_human_input_baseline(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Add score_a as a transparent reference prediction for target score_b."""
    metrics = metrics[metrics["model"] != "input_annotator_a"].copy()
    rows: list[dict[str, Any]] = []
    for scope, scope_frame in predictions.groupby("split_scope", sort=True):
        groups = [("ALL_ESA" if scope == "pooled_holdout" else PRIMARY_LP, scope_frame)]
        if scope == "pooled_holdout":
            groups.extend(
                (str(language_pair), language_frame)
                for language_pair, language_frame in scope_frame.groupby("language_pair", sort=True)
                if language_frame["translation_id"].nunique() >= 5
            )
        for language_pair, frame in groups:
            values = _metric_values(frame, "score_b", "score_a", "pair_weight")
            low, high = _bootstrap_mae_ci(
                frame, "score_b", "score_a", "pair_weight", "doc_group_id", SEED + len(rows)
            )
            rows.append({
                "experiment_scope": scope, "language_pair": language_pair,
                "regime": "human_baseline", "model": "input_annotator_a",
                "feature_set": "human_score_a", "n_directed_pairs": int(len(frame)),
                "n_translations": int(frame["translation_id"].nunique()),
                "n_unique_documents": int(frame["doc_group_id"].nunique()),
                "n_annotators": int(len(set(frame["annotator_a"]) | set(frame["annotator_b"]))),
                **values, "mae_ci_low": low, "mae_ci_high": high,
            })
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def _failure_cases(predictions: pd.DataFrame, output: Path) -> pd.DataFrame:
    pooled = predictions[(predictions["split_scope"] == "pooled_holdout")].copy()
    pooled["predicted_score"] = pooled["pred_cross_catboost_F3"]
    pooled["absolute_error"] = np.abs(pooled["score_b"] - pooled["predicted_score"])
    pooled["signed_error_pred_minus_human"] = pooled["predicted_score"] - pooled["score_b"]
    pooled["category"] = ""
    pooled.loc[(pooled["score_b"] <= 50) & (pooled["n_total_b"] == 0), "category"] = "low_score_no_marked_errors"
    pooled.loc[(pooled["score_b"] >= 80) & (pooled["n_major_b"] > 0), "category"] = "high_score_with_major_errors"
    score_a_error = (pooled["predicted_score"] - pooled["score_a"]).abs()
    score_b_error = (pooled["predicted_score"] - pooled["score_b"]).abs()
    pooled.loc[(score_a_error <= 10) & (score_b_error >= 20) & pooled["category"].eq(""), "category"] = "model_close_to_a_far_from_b"
    pooled.loc[(score_b_error + 10 <= score_a_error) & pooled["category"].eq(""), "category"] = "model_closer_to_b_than_a"
    disagreement_cut = float(pooled["score_a"].sub(pooled["score_b"]).abs().quantile(0.9))
    high_disagreement = pooled["score_a"].sub(pooled["score_b"]).abs() >= disagreement_cut
    pooled.loc[high_disagreement & pooled["category"].eq(""), "category"] = "high_annotator_disagreement"
    pooled.loc[pooled["category"].eq(""), "category"] = "largest_model_error"
    # One hard case per translation avoids showing reciprocal directions as duplicates.
    top = pooled.sort_values("absolute_error", ascending=False).drop_duplicates("translation_id").head(30)
    important = pooled[pooled["category"].isin([
        "low_score_no_marked_errors", "high_score_with_major_errors", "high_annotator_disagreement",
        "model_close_to_a_far_from_b", "model_closer_to_b_than_a",
    ])].sort_values("absolute_error", ascending=False).drop_duplicates("translation_id")
    combined = pd.concat([top, important], ignore_index=True).drop_duplicates("translation_id")
    keep = [
        "category", "translation_id", "pair_id", "doc_id", "doc_group_id", "language_pair", "system_name",
        "annotator_a", "annotator_b", "src_text", "target_text", "score_a", "score_b", "predicted_score",
        "absolute_error", "signed_error_pred_minus_human", "score_a_minus_b", "n_minor", "n_major", "n_total",
        "n_minor_b", "n_major_b", "n_total_b", "error_spans_a_json", "error_spans_b_json",
    ]
    combined["score_a_minus_b"] = combined["score_a"] - combined["score_b"]
    output_rows = combined[[column for column in keep if column in combined]].copy()
    output_rows.to_csv(output, index=False)
    return output_rows


def _plot_outputs(predictions: pd.DataFrame, importance: pd.DataFrame, all_annotations: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    pooled = predictions[predictions["split_scope"] == "pooled_holdout"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.scatter(pooled["score_b"], pooled["pred_cross_catboost_F3"], s=7, alpha=0.22, color="#176b87", edgecolors="none")
    ax.plot([0, 100], [0, 100], color="#cc4b37", linestyle="--", linewidth=1.3)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Target annotator human score", ylabel="Cross-annotator CatBoost F3 prediction", title="Pooled ESA held-out predictions")
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_human.png", dpi=170)
    plt.close(fig)

    scored = all_annotations.copy()
    scored["major_group"] = pd.cut(scored["n_major"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    scored["minor_group"] = pd.cut(scored["n_minor"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharey=True)
    for ax, col, title in zip(axes, ["major_group", "minor_group"], ["Major error count", "Minor error count"], strict=True):
        groups = [scored.loc[scored[col] == category, "score"].to_numpy(float) for category in ["0", "1", "2", "3+"]]
        ax.boxplot(groups, tick_labels=["0", "1", "2", "3+"], showfliers=False)
        ax.set(title=title, xlabel="Number of marked errors")
    axes[0].set_ylabel("Human ESA score")
    fig.suptitle("Human score distributions by error counts")
    fig.tight_layout()
    fig.savefig(plot_dir / "score_by_error_counts.png", dpi=170)
    plt.close(fig)

    cat = importance[(importance["regime"] == "cross") & (importance["feature_set"] == "F3")]
    average = cat.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.barh(average["feature"], average["importance"], color="#4185a5")
    ax.set(xlabel="Mean CatBoost importance", title="Cross-annotator CatBoost F3 feature importance")
    fig.tight_layout()
    fig.savefig(plot_dir / "catboost_feature_importance.png", dpi=170)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame[columns].copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def _build_report(
    views: pd.DataFrame, pairs: pd.DataFrame, predictions: pd.DataFrame,
    metrics: pd.DataFrame, gaps: pd.DataFrame, ablations: pd.DataFrame,
    disagreement: pd.DataFrame, failures: pd.DataFrame, audit: dict[str, Any],
    path: Path,
) -> None:
    primary = metrics[(metrics["experiment_scope"] == "en_bho_cv") & (metrics["language_pair"] == PRIMARY_LP)]
    pooled = metrics[(metrics["experiment_scope"] == "pooled_holdout") & (metrics["language_pair"] == "ALL_ESA")]
    score_stats = views.groupby("language_pair")["score"].agg(["count", "mean", "std", "median"]).reset_index()
    sizes = views.groupby("language_pair", as_index=False).agg(
        translations=("translation_id", "nunique"), documents=("doc_group_id", "nunique"),
        systems=("system_name", "nunique"), annotators=("annotator_id", "nunique"),
    )
    sizes = sizes.merge(score_stats, on="language_pair", how="left")
    disagreement_display = disagreement.sort_values("language_pair")
    per_language = metrics[
        (metrics["experiment_scope"] == "pooled_holdout")
        & (
            ((metrics["regime"] == "cross_trained") & (metrics["model"].isin(["ridge", "catboost"])) & (metrics["feature_set"] == "F3"))
            | (metrics["model"] == "input_annotator_a")
        )
        & (metrics["language_pair"] != "ALL_ESA")
    ].copy()
    bho_fold_counts = {fold: values["test_docs"] for fold, values in audit.items() if fold.startswith("fold_")}
    imbalance_note = (
        f"The saved Bho fold assignment has held-out document counts {bho_fold_counts}; one fold contains fewer than five documents, so fold-level uncertainty is uneven."
        if any(value < 5 for value in bho_fold_counts.values())
        else f"The saved Bho fold assignment has held-out document counts {bho_fold_counts}."
    )
    report = f"""# Experiment 2 — Cross-annotator score generalization

## Design and data

This experiment reuses the cleaned WMT25 ESA annotation table and Experiment 1 document assignments. No Experiment 1 model run was repeated. Each translation is identified by a SHA-256-derived key over language pair, full segment `doc_id`, and translation system. Repeated rows from one annotator on one translation are collapsed to a single annotator view: score averaged, error features taken from the deterministically earliest annotation. Only distinct annotators form a human pair. Every unordered annotator pair contributes both directed views; each translation's directed weights sum to 1, so translations with more annotators are not given more total evaluation/training weight.

The frozen model is trained on input annotations and their own annotator scores; it is evaluated against both the input annotator score and the independent target annotator score. The cross-trained model is trained on input features from annotator A and the score from annotator B. Both use the same document-level outer assignments, same held-out directed pairs, F1/F2/F3, and no annotator/system/document IDs as numeric predictors. Inner validation and all preprocessing/tuning use training documents only. Pooled results use the original grouped train/validation/test split. Bhojpuri results use the original five grouped outer folds. Document, exact-source leakage-group, reciprocal-pair, and translation overlap checks passed: `{json.dumps(audit, ensure_ascii=False)}`.

{imbalance_note} The old assignments were retained to match Experiment 1; no held-out document was reassigned to improve the balance.

The paired set has **{pairs.translation_id.nunique():,} translations** and **{len(pairs):,} directed input-target rows** across **{pairs.doc_group_id.nunique():,} document groups**, **{pairs.system_name.nunique():,} systems**, and **{pairs.language_pair.nunique()} ESA language pairs**. Metrics are pair-weighted so each translation sums to one. Confidence intervals resample document groups. Human disagreement is descriptive and uses the same weights. The `input_annotator_a` row is a human-to-human reference: A's score predicts independent annotator B's score.

### Paired annotation population by language pair

{_markdown_table(sizes, ['language_pair','translations','documents','systems','annotators','count','mean','std','median'])}

## Cross-annotator prediction metrics

`same` rows measure same-annotator targets; `frozen_cross` measures transfer from A's errors to B's score using the same-annotator model; `cross_trained` trains directly on A-features/B-score pairs. MAE/RMSE, median absolute error, correlation, bias and ±10 accuracy are weighted by translation. The confidence interval columns are document-cluster bootstrap intervals for MAE.

### English→Bhojpuri grouped out-of-fold results

{_markdown_table(primary, ['regime','model','feature_set','n_translations','mae','mae_ci_low','mae_ci_high','rmse','median_absolute_error','pearson','spearman','mean_bias_pred_minus_human','within_10_pct'])}

### Pooled ESA held-out results

{_markdown_table(pooled, ['regime','model','feature_set','n_translations','mae','mae_ci_low','mae_ci_high','rmse','median_absolute_error','pearson','spearman','mean_bias_pred_minus_human','within_10_pct'])}

## Generalization gap and feature ablation

The frozen generalization gap is cross-target MAE minus same-target MAE for a model trained on same-annotator data. A positive value means independent annotator targets are harder. `cross_trained` rows are the directly trained A→B alternative.

{_markdown_table(gaps, ['experiment_scope','model','feature_set','same_target_mae','frozen_cross_target_mae','frozen_generalization_gap','gap_ci_low','gap_ci_high','cross_trained_target_mae'])}

{_markdown_table(ablations[ablations['experiment_scope'].isin(['en_bho_cv','pooled_holdout'])], ['experiment_scope','language_pair','regime','model','feature_set','n_translations','mae','rmse','spearman'])}

### Pooled cross-trained F3 performance by language pair

{_markdown_table(per_language, ['language_pair','model','n_translations','n_unique_documents','mae','mae_ci_low','mae_ci_high','rmse','spearman'])}

## Human annotator disagreement

{_markdown_table(disagreement_display, ['language_pair','translations','independent_annotator_pairs','weighted_mean_abs_difference','weighted_median_abs_difference','weighted_p90_abs_difference'])}

These human-to-human differences compare two raters on the same translation. Model MAE compares a prediction with one rater. They provide context but are not the same evaluation quantity.

## Difficult and contradictory examples

The failure-case CSV contains the 30 largest pooled held-out CatBoost F3 errors (one row per translation) and eligible low-score/no-error, high-score/major-error, high-disagreement, and A-versus-B residual contrast examples. Each row includes source, translation, both annotators' scores, predicted score, and both span lists. The category counts in this export are: `{failures['category'].value_counts().to_dict()}`. These cases make it possible to inspect whether disagreements between free-form score and marked spans account for model residuals.

![Predicted versus target human score](plots/predicted_vs_human.png)

![Human scores by major and minor error counts](plots/score_by_error_counts.png)

![CatBoost feature importance](plots/catboost_feature_importance.png)

## Experiment 3 local sample

The independent 100-translation Bhojpuri sample is frozen in `../experiment3/sample100.csv` with seed 42 and a SHA-256 manifest before any hosted model call. It was sampled without using labels, features, or model residuals. The local baseline comparison is in `../experiment3/metrics_local.csv` and uses the same held-out OOF examples. Frontier judging is a separate script and is not implied by these local results.

## Limitations

These experiments use human error annotations as model inputs, so they estimate score prediction conditional on supplied spans, not a complete automatic error-detection pipeline. The paired sample is restricted to translations with at least two valid, distinct annotators; invalid/missing spans were already excluded by Experiment 1. Annotator scores are averaged only when one annotator has repeated rows for a translation. Pooled outcomes combine different language directions and scoring distributions; per-language rows should be read alongside the pooled summary. Stable use after an LLM detector requires evaluation with detector-generated errors and a wider frozen sample.
"""
    path.write_text(report, encoding="utf-8")


def run_experiment2(
    data_path: Path = Path("data/processed/annotations.csv.gz"),
    manifest_path: Path = Path("results/split_manifest.csv"),
    results_dir: Path = Path("results/experiment2"),
    models_dir: Path = Path("models/experiment2"),
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(data_path, compression="infer", low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    print(f"Loaded {len(annotations):,} valid annotation rows", flush=True)
    views_all = build_annotator_views(annotations)
    print(f"Collapsed to {len(views_all):,} annotator views", flush=True)
    n_annotators = views_all.groupby("translation_id")["annotator_id"].nunique()
    eligible_ids = set(n_annotators[n_annotators >= 2].index)
    views = views_all[views_all["translation_id"].isin(eligible_ids)].copy().reset_index(drop=True)
    pairs = attach_saved_splits(build_directed_pairs(views), manifest)
    print(f"Built {pairs.translation_id.nunique():,} paired translations / {len(pairs):,} directed rows", flush=True)
    audit = validate_splits(pairs)

    # Rows are stored before fitting to make the independent-annotator population auditable.
    pair_path = results_dir / "paired_annotations.csv"
    pairs.to_csv(pair_path, index=False)
    (results_dir / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    all_predictions: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    bho_pairs = pairs[pairs["experiment_scope"] == "en_bho_cv"].copy()
    bho_views = views[views["language_pair"] == PRIMARY_LP].copy()
    for fold in sorted(bho_pairs["fold"].unique(), key=lambda value: int(str(value).split("_")[-1])):
        test_docs = set(bho_pairs.loc[bho_pairs["fold"] == fold, "doc_group_id"].astype(str))
        outer_train_pairs = bho_pairs[bho_pairs["fold"] != fold]
        train_docs, val_docs = _group_inner_split(outer_train_pairs)
        if test_docs.intersection(train_docs | val_docs) or train_docs.intersection(val_docs):
            raise AssertionError(f"Bho {fold} document split collision")
        prediction, selections, importances = _run_one_fold(
            bho_views, bho_pairs, "en_bho_cv", str(fold), train_docs, val_docs, test_docs, models_dir
        )
        all_predictions.append(prediction)
        selection_rows.extend(selections)
        importance_rows.extend(importances)
        print(
            f"Experiment 2 Bho {fold}: train_docs={len(train_docs)}, validation_docs={len(val_docs)}, "
            f"test_docs={len(test_docs)}, directed_pairs={len(prediction):,}", flush=True
        )

    pooled_pairs = pairs[pairs["experiment_scope"] == "pooled_holdout"].copy()
    pooled_views = views[views["language_pair"] != PRIMARY_LP].copy()
    pooled_train = set(pooled_pairs.loc[pooled_pairs["split"] == "train", "doc_group_id"].astype(str))
    pooled_val = set(pooled_pairs.loc[pooled_pairs["split"] == "validation", "doc_group_id"].astype(str))
    pooled_test = set(pooled_pairs.loc[pooled_pairs["split"] == "test", "doc_group_id"].astype(str))
    pooled_prediction, selections, importances = _run_one_fold(
        pooled_views, pooled_pairs, "pooled_holdout", "holdout", pooled_train, pooled_val, pooled_test, models_dir
    )
    all_predictions.append(pooled_prediction)
    selection_rows.extend(selections)
    importance_rows.extend(importances)
    print(
        f"Experiment 2 pooled: train_docs={len(pooled_train)}, validation_docs={len(pooled_val)}, "
        f"test_docs={len(pooled_test)}, directed_pairs={len(pooled_prediction):,}", flush=True
    )

    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_path = results_dir / "predictions.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    selections_frame = pd.DataFrame(selection_rows)
    selections_frame.to_csv(results_dir / "model_selection.csv", index=False)
    importances = pd.DataFrame(importance_rows)
    importances.to_csv(results_dir / "feature_importance.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    config_specs: list[tuple[str, str, str, str]] = [
        ("same_target", "score_a", "mean", "pred_mean_same"),
        ("cross_trained", "score_b", "mean", "pred_mean_cross"),
        ("same_target", "score_a", "fixed_penalty", "pred_fixed_penalty"),
        ("frozen_cross", "score_b", "fixed_penalty", "pred_fixed_penalty"),
    ]
    for model_type in ("ridge", "catboost"):
        for feature_set in FEATURE_SETS:
            config_specs.extend([
                ("same_target", "score_a", model_type, f"pred_same_{model_type}_{feature_set}"),
                ("frozen_cross", "score_b", model_type, f"pred_same_{model_type}_{feature_set}"),
                ("cross_trained", "score_b", model_type, f"pred_cross_{model_type}_{feature_set}"),
            ])

    for scope in ("en_bho_cv", "pooled_holdout"):
        scope_frame = predictions[predictions["split_scope"] == scope]
        groups: list[tuple[str, pd.DataFrame]] = [("ALL_ESA" if scope == "pooled_holdout" else PRIMARY_LP, scope_frame)]
        if scope == "pooled_holdout":
            for language_pair, language_frame in scope_frame.groupby("language_pair", sort=True):
                if language_frame["translation_id"].nunique() >= 5:
                    groups.append((str(language_pair), language_frame))
        for language_pair, language_frame in groups:
            for regime, actual, model_type, prediction_column in config_specs:
                if prediction_column not in language_frame:
                    continue
                values = _metric_values(language_frame, actual, prediction_column, "pair_weight")
                low, high = _bootstrap_mae_ci(
                    language_frame, actual, prediction_column, "pair_weight", "doc_group_id",
                    SEED + len(metric_rows),
                )
                metric_rows.append({
                    "experiment_scope": scope, "language_pair": language_pair, "regime": regime,
                    "model": model_type, "feature_set": "none" if model_type in {"mean", "fixed_penalty"} else prediction_column.rsplit("_", 1)[-1],
                    "n_directed_pairs": int(len(language_frame)),
                    "n_translations": int(language_frame["translation_id"].nunique()),
                    "n_unique_documents": int(language_frame["doc_group_id"].nunique()),
                    "n_annotators": int(len(set(language_frame["annotator_a"]) | set(language_frame["annotator_b"]))),
                    **values, "mae_ci_low": low, "mae_ci_high": high,
                })
    metrics = pd.DataFrame(metric_rows)
    metrics = _append_human_input_baseline(metrics, predictions)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    ablations = metrics[
        metrics["model"].isin(["ridge", "catboost"])
        & metrics["feature_set"].isin(["F1", "F2", "F3"])
        & metrics["language_pair"].isin(["ALL_ESA", PRIMARY_LP])
    ].copy()
    ablations.to_csv(results_dir / "ablations.csv", index=False)

    gap_rows: list[dict[str, Any]] = []
    for scope, scope_frame in predictions.groupby("split_scope"):
        for model_type in ("ridge", "catboost"):
            for feature_set in FEATURE_SETS:
                prediction_column = f"pred_same_{model_type}_{feature_set}"
                same_mae = _metric_values(scope_frame, "score_a", prediction_column, "pair_weight")["mae"]
                frozen_mae = _metric_values(scope_frame, "score_b", prediction_column, "pair_weight")["mae"]
                cross_mae = _metric_values(scope_frame, "score_b", f"pred_cross_{model_type}_{feature_set}", "pair_weight")["mae"]
                low, high = _bootstrap_two_target_gap_ci(scope_frame, prediction_column, "pair_weight", "doc_group_id", SEED + len(gap_rows))
                gap_rows.append({
                    "experiment_scope": scope, "model": model_type, "feature_set": feature_set,
                    "same_target_mae": same_mae, "frozen_cross_target_mae": frozen_mae,
                    "frozen_generalization_gap": frozen_mae - same_mae,
                    "gap_ci_low": low, "gap_ci_high": high, "cross_trained_target_mae": cross_mae,
                    "n_translations": int(scope_frame["translation_id"].nunique()),
                    "n_documents": int(scope_frame["doc_group_id"].nunique()),
                })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(results_dir / "generalization_gap.csv", index=False)

    # Disagreement summaries from all paired translations; each translation has total weight one.
    pairs["absolute_human_difference"] = (pairs["score_a"] - pairs["score_b"]).abs()
    disagreement_rows: list[dict[str, Any]] = []
    for language_pair, part in list(pairs.groupby("language_pair", sort=True)) + [("ALL_ESA", pairs)]:
        diffs = part["absolute_human_difference"].to_numpy(float)
        weights = part["pair_weight"].to_numpy(float)
        disagreement_rows.append({
            "language_pair": language_pair,
            "translations": int(part["translation_id"].nunique()),
            "independent_annotator_pairs": int(part["pair_id"].nunique()),
            "directed_rows": int(len(part)),
            "weighted_mean_abs_difference": float(np.average(diffs, weights=weights)),
            "weighted_median_abs_difference": _weighted_median(diffs, weights),
            "weighted_p90_abs_difference": _weighted_quantile(diffs, weights, 0.9),
        })
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(results_dir / "human_disagreement.csv", index=False)

    failures = _failure_cases(predictions, results_dir / "failure_cases.csv")
    _plot_outputs(predictions, importances, annotations, results_dir / "plots")

    sample_path, local_path = _make_sample100(pairs, predictions, results_dir.parent)
    local_sample = pd.read_csv(local_path)
    local_metrics_path = results_dir.parent / "experiment3" / "metrics_local.csv"
    pd.DataFrame(_local_baseline_metric_rows(local_sample)).to_csv(local_metrics_path, index=False)

    _build_report(views, pairs, predictions, metrics, gaps, ablations, disagreement, failures, audit, results_dir / "report.md")
    summary = {
        "source_data_rows": int(len(annotations)),
        "annotator_views": int(len(views)),
        "paired_translations": int(pairs["translation_id"].nunique()),
        "directed_pairs": int(len(pairs)),
        "pair_weight_sum_per_translation_min": float(pairs.groupby("translation_id")["pair_weight"].sum().min()),
        "pair_weight_sum_per_translation_max": float(pairs.groupby("translation_id")["pair_weight"].sum().max()),
        "split_audit": audit,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "catboost_iterations_max": CATBOOST_ITERATIONS,
        "catboost_early_stopping_rounds": CATBOOST_EARLY_STOPPING,
        "feature_sets": FEATURE_SETS,
        "sample100_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = {
        "paired_annotations": pair_path,
        "metrics": results_dir / "metrics.csv",
        "ablations": results_dir / "ablations.csv",
        "generalization_gap": results_dir / "generalization_gap.csv",
        "human_disagreement": results_dir / "human_disagreement.csv",
        "failure_cases": results_dir / "failure_cases.csv",
        "report": results_dir / "report.md",
        "predictions": prediction_path,
        "sample100": sample_path,
        "sample100_local": local_path,
        "experiment3_local_metrics": local_metrics_path,
    }
    print(f"Paired translations: {pairs.translation_id.nunique():,}; directed rows: {len(pairs):,}")
    print(f"Metrics: {paths['metrics'].resolve()}")
    print(f"Report: {paths['report'].resolve()}")
    print(f"Frozen Experiment 3 sample: {sample_path.resolve()}")
    return paths


def postprocess_saved_run(
    data_path: Path = Path("data/processed/annotations.csv.gz"),
    results_dir: Path = Path("results/experiment2"),
) -> dict[str, Path]:
    """Finish plots, the frozen Experiment 3 sample and report from completed local fits."""
    annotations = pd.read_csv(data_path, compression="infer", low_memory=False)
    pairs = pd.read_csv(results_dir / "paired_annotations.csv", low_memory=False)
    predictions = pd.read_csv(results_dir / "predictions.csv.gz", compression="infer", low_memory=False)
    metrics = pd.read_csv(results_dir / "metrics.csv")
    metrics = _append_human_input_baseline(metrics, predictions)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    gaps = pd.read_csv(results_dir / "generalization_gap.csv")
    ablations = pd.read_csv(results_dir / "ablations.csv")
    disagreement = pd.read_csv(results_dir / "human_disagreement.csv")
    failures = _failure_cases(predictions, results_dir / "failure_cases.csv")
    importance = pd.read_csv(results_dir / "feature_importance.csv")
    audit = json.loads((results_dir / "split_audit.json").read_text(encoding="utf-8"))
    eligible_ids = set(pairs["translation_id"].astype(str))
    views = build_annotator_views(annotations)
    views = views[views["translation_id"].isin(eligible_ids)].copy()

    _plot_outputs(predictions, importance, annotations, results_dir / "plots")
    sample_path, local_path = _make_sample100(pairs, predictions, results_dir.parent)
    local_sample = pd.read_csv(local_path, low_memory=False)
    local_metrics_path = results_dir.parent / "experiment3" / "metrics_local.csv"
    pd.DataFrame(_local_baseline_metric_rows(local_sample)).to_csv(local_metrics_path, index=False)
    _build_report(views, pairs, predictions, metrics, gaps, ablations, disagreement, failures, audit, results_dir / "report.md")
    summary = {
        "source_data_rows": int(len(annotations)),
        "annotator_views": int(len(views)),
        "paired_translations": int(pairs["translation_id"].nunique()),
        "directed_pairs": int(len(pairs)),
        "pair_weight_sum_per_translation_min": float(pairs.groupby("translation_id")["pair_weight"].sum().min()),
        "pair_weight_sum_per_translation_max": float(pairs.groupby("translation_id")["pair_weight"].sum().max()),
        "split_audit": audit,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "catboost_iterations_max": CATBOOST_ITERATIONS,
        "catboost_early_stopping_rounds": CATBOOST_EARLY_STOPPING,
        "feature_sets": FEATURE_SETS,
        "sample100_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Plots: {(results_dir / 'plots').resolve()}")
    print(f"Report: {(results_dir / 'report.md').resolve()}")
    print(f"Frozen sample: {sample_path.resolve()}")
    return {"report": results_dir / "report.md", "sample100": sample_path, "local_metrics": local_metrics_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/annotations.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiment2"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/experiment2"))
    parser.add_argument("--postprocess-only", action="store_true", help="Finish outputs from completed saved model predictions without retraining")
    args = parser.parse_args()
    if args.postprocess_only:
        postprocess_saved_run(args.data, args.results_dir)
    else:
        run_experiment2(args.data, args.manifest, args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
