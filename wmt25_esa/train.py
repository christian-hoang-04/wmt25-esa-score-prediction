"""Fit shared-split baselines, Ridge, and CatBoost on WMT25 ESA features."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .prepare_data import FEATURE_SETS


PRIMARY_LANGUAGE_PAIR = "en-bho_IN"
PRIMARY_CV_DOCUMENT_THRESHOLD = 60
CV_FOLDS = 5
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CATBOOST_GRID = [
    {"depth": 3, "l2_leaf_reg": 3.0},
    {"depth": 4, "l2_leaf_reg": 3.0},
    {"depth": 4, "l2_leaf_reg": 8.0},
]
CATBOOST_ITERATIONS = 500
CATBOOST_EARLY_STOPPING = 50
CATBOOST_THREADS = 4
BOOTSTRAP_SEED = 42


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        lower, upper = sorted((root_a, root_b))
        self.parent[upper] = lower


def assign_leakage_groups(frame: pd.DataFrame) -> pd.Series:
    """Join documents that share any exact source text, preventing split leakage."""
    documents = sorted(frame["doc_group_id"].astype(str).unique())
    union_find = UnionFind(documents)
    first_document_for_source: dict[str, str] = {}
    unique_pairs = frame[["doc_group_id", "src_text"]].drop_duplicates()
    for document, source_text in unique_pairs.itertuples(index=False, name=None):
        document = str(document)
        source_text = str(source_text)
        previous = first_document_for_source.get(source_text)
        if previous is None:
            first_document_for_source[source_text] = document
        else:
            union_find.union(document, previous)
    roots = {document: union_find.find(document) for document in documents}
    return frame["doc_group_id"].astype(str).map(roots).rename("leakage_group_id")


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(actual, predicted)))


def _make_ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _make_catboost(params: dict[str, Any], iterations: int = CATBOOST_ITERATIONS) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=int(params["depth"]),
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=42,
        thread_count=CATBOOST_THREADS,
        allow_writing_files=False,
        verbose=False,
    )


def tune_and_fit_ridge(
    frame: pd.DataFrame,
    tune_train_idx: np.ndarray,
    validation_idx: np.ndarray,
    final_train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_set: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = FEATURE_SETS[feature_set]
    y = frame["score"].to_numpy(dtype=float)
    best: dict[str, Any] | None = None
    for alpha in RIDGE_ALPHAS:
        candidate = _make_ridge(alpha)
        candidate.fit(frame.iloc[tune_train_idx][columns], y[tune_train_idx])
        validation_pred = np.clip(
            candidate.predict(frame.iloc[validation_idx][columns]), 0, 100
        )
        validation_rmse = _rmse(y[validation_idx], validation_pred)
        candidate_info = {"alpha": alpha, "validation_rmse": validation_rmse}
        if best is None or validation_rmse < best["validation_rmse"]:
            best = candidate_info

    assert best is not None
    final_model = _make_ridge(float(best["alpha"]))
    final_model.fit(frame.iloc[final_train_idx][columns], y[final_train_idx])
    prediction = np.clip(final_model.predict(frame.iloc[test_idx][columns]), 0, 100)
    return prediction, {**best, "model": final_model}


def tune_and_fit_catboost(
    frame: pd.DataFrame,
    tune_train_idx: np.ndarray,
    validation_idx: np.ndarray,
    final_train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_set: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = FEATURE_SETS[feature_set]
    y = frame["score"].to_numpy(dtype=float)
    best: dict[str, Any] | None = None
    for params in CATBOOST_GRID:
        candidate = _make_catboost(params)
        candidate.fit(
            frame.iloc[tune_train_idx][columns],
            y[tune_train_idx],
            eval_set=(frame.iloc[validation_idx][columns], y[validation_idx]),
            early_stopping_rounds=CATBOOST_EARLY_STOPPING,
            use_best_model=True,
            verbose=False,
        )
        validation_pred = np.clip(
            candidate.predict(frame.iloc[validation_idx][columns]), 0, 100
        )
        validation_rmse = _rmse(y[validation_idx], validation_pred)
        best_iteration = candidate.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            best_iteration = CATBOOST_ITERATIONS - 1
        info = {
            **params,
            "validation_rmse": validation_rmse,
            "best_iteration": int(best_iteration) + 1,
        }
        if best is None or validation_rmse < best["validation_rmse"]:
            best = info

    assert best is not None
    final_params = {"depth": best["depth"], "l2_leaf_reg": best["l2_leaf_reg"]}
    final_model = _make_catboost(final_params, iterations=max(1, best["best_iteration"]))
    final_model.fit(
        frame.iloc[final_train_idx][columns],
        y[final_train_idx],
        verbose=False,
    )
    prediction = np.clip(final_model.predict(frame.iloc[test_idx][columns]), 0, 100)
    return prediction, {**best, "model": final_model}


def _grouped_validation_split(
    frame: pd.DataFrame, outer_train_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    groups = frame.iloc[outer_train_idx]["leakage_group_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    local_train, local_validation = next(
        splitter.split(np.zeros(len(outer_train_idx)), groups=groups)
    )
    return outer_train_idx[local_train], outer_train_idx[local_validation]


def _assert_no_group_or_source_overlap(
    frame: pd.DataFrame,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    left = frame.iloc[left_idx]
    right = frame.iloc[right_idx]
    shared_groups = set(left["leakage_group_id"]).intersection(right["leakage_group_id"])
    shared_sources = set(left["src_text"]).intersection(right["src_text"])
    if shared_groups or shared_sources:
        raise AssertionError(
            f"Leakage between {left_name} and {right_name}: "
            f"{len(shared_groups)} groups, {len(shared_sources)} exact source texts"
        )


def _holdout_score(frame: pd.DataFrame, assignments: dict[str, np.ndarray]) -> float:
    total = len(frame)
    totals_by_lp = frame.groupby("language_pair", observed=True).size().to_dict()
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    score = 0.0
    for name, indices in assignments.items():
        fraction = targets[name]
        row_fraction_error = abs(len(indices) / total - fraction)
        score += 4.0 * row_fraction_error
        counts = frame.iloc[indices].groupby("language_pair", observed=True).size().to_dict()
        for language_pair, all_count in totals_by_lp.items():
            expected = all_count * fraction
            observed = counts.get(language_pair, 0)
            score += abs(observed - expected) / max(expected, 1.0)
    return score


def stratified_group_holdout(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Choose a 70/15/15 grouped split with balanced per-language row counts."""
    grouped_counts = (
        frame.groupby(["leakage_group_id", "language_pair"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    group_ids = grouped_counts.index.to_numpy()
    group_labels = np.arange(len(group_ids))
    target_counts = frame.groupby("language_pair", observed=True).size().reindex(
        grouped_counts.columns, fill_value=0
    ).to_numpy(dtype=float)
    total_rows = int(target_counts.sum())

    def score_groups(assignments: dict[str, np.ndarray]) -> float:
        fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
        score = 0.0
        for name, group_indices in assignments.items():
            fraction = fractions[name]
            counts = grouped_counts.iloc[group_indices].sum(axis=0).to_numpy(dtype=float)
            row_fraction_error = abs(counts.sum() / total_rows - fraction)
            score += 4.0 * row_fraction_error
            expected = target_counts * fraction
            score += float(np.sum(np.abs(counts - expected) / np.maximum(expected, 1.0)))
        return score

    first_splitter = GroupShuffleSplit(n_splits=48, train_size=0.70, random_state=42)
    best_group_split: dict[str, np.ndarray] | None = None
    best_score = float("inf")
    for first_train, remainder in first_splitter.split(np.zeros(len(group_ids)), groups=group_labels):
        second_splitter = GroupShuffleSplit(n_splits=24, test_size=0.5, random_state=43)
        for local_val, local_test in second_splitter.split(
            np.zeros(len(remainder)), groups=group_labels[remainder]
        ):
            candidate = {
                "train": first_train,
                "validation": remainder[local_val],
                "test": remainder[local_test],
            }
            candidate_score = score_groups(candidate)
            if candidate_score < best_score:
                best_group_split, best_score = candidate, candidate_score
    if best_group_split is None:
        raise RuntimeError("Could not create grouped holdout split")
    best = {
        name: np.flatnonzero(
            frame["leakage_group_id"].isin(set(group_ids[group_indices])).to_numpy()
        )
        for name, group_indices in best_group_split.items()
    }
    _assert_no_group_or_source_overlap(frame, best["train"], best["validation"], "train", "validation")
    _assert_no_group_or_source_overlap(frame, best["train"], best["test"], "train", "test")
    _assert_no_group_or_source_overlap(frame, best["validation"], best["test"], "validation", "test")
    return best


def _write_split_rows(
    rows: list[dict[str, Any]],
    scope: str,
    frame: pd.DataFrame,
    split_by_index: dict[int, str],
    fold_name: str,
) -> None:
    unique_docs = frame[["doc_group_id", "leakage_group_id"]].drop_duplicates()
    for doc, leakage_group in unique_docs.itertuples(index=False, name=None):
        doc_indices = np.flatnonzero(frame["doc_group_id"].to_numpy() == doc)
        split_names = {split_by_index[int(i)] for i in doc_indices if int(i) in split_by_index}
        if len(split_names) > 1:
            raise AssertionError(f"Document {doc} was assigned to multiple splits")
        if split_names:
            rows.append(
                {
                    "experiment_scope": scope,
                    "doc_group_id": doc,
                    "leakage_group_id": leakage_group,
                    "fold": fold_name,
                    "split": next(iter(split_names)),
                }
            )


def _prediction_rows(
    frame: pd.DataFrame,
    indices: np.ndarray,
    scope: str,
    fold: str,
    predictions: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    annotation_ids = frame.iloc[indices]["annotation_id"].to_numpy()
    for (model, feature_set), values in predictions.items():
        for annotation_id, value in zip(annotation_ids, values):
            records.append(
                {
                    "experiment_scope": scope,
                    "fold": fold,
                    "annotation_id": annotation_id,
                    "model": model,
                    "feature_set": feature_set,
                    "predicted_score": float(value),
                }
            )
    return pd.DataFrame.from_records(records)


def _selection_row(
    scope: str, fold: str, model: str, feature_set: str, info: dict[str, Any]
) -> dict[str, Any]:
    return {
        "experiment_scope": scope,
        "fold": fold,
        "model": model,
        "feature_set": feature_set,
        "validation_rmse": info["validation_rmse"],
        "alpha": info.get("alpha"),
        "depth": info.get("depth"),
        "l2_leaf_reg": info.get("l2_leaf_reg"),
        "iterations": info.get("best_iteration"),
    }


def _final_catboost_selection(selection: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    rows = selection[(selection["model"] == "catboost") & (selection["feature_set"] == feature_set)]
    signatures = rows.groupby(["depth", "l2_leaf_reg"], dropna=True).agg(
        frequency=("validation_rmse", "size"), mean_validation_rmse=("validation_rmse", "mean")
    )
    signatures = signatures.sort_values(
        ["frequency", "mean_validation_rmse"], ascending=[False, True]
    )
    depth, l2 = signatures.index[0]
    selected = rows[(rows["depth"] == depth) & (rows["l2_leaf_reg"] == l2)]
    iterations = int(round(float(selected["iterations"].median())))
    return {"depth": int(depth), "l2_leaf_reg": float(l2), "iterations": max(1, iterations)}


def _final_ridge_selection(selection: pd.DataFrame, feature_set: str) -> float:
    rows = selection[(selection["model"] == "ridge") & (selection["feature_set"] == feature_set)]
    counts = rows["alpha"].value_counts()
    most_frequent = counts[counts == counts.max()].index
    candidates = rows[rows["alpha"].isin(most_frequent)]
    return float(candidates.groupby("alpha")["validation_rmse"].mean().sort_values().index[0])


def _save_final_cv_models(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
    models_dir: Path,
    results_dir: Path,
) -> None:
    y = frame["score"].to_numpy(dtype=float)
    importance_rows = []
    for feature_set, columns in FEATURE_SETS.items():
        ridge_alpha = _final_ridge_selection(selection, feature_set)
        ridge_model = _make_ridge(ridge_alpha)
        ridge_model.fit(frame[columns], y)
        joblib.dump(ridge_model, models_dir / f"en_bho_cv_ridge_{feature_set}.joblib")

        config = _final_catboost_selection(selection, feature_set)
        cat_model = _make_catboost(config, iterations=config["iterations"])
        cat_model.fit(frame[columns], y, verbose=False)
        cat_model.save_model(str(models_dir / f"en_bho_cv_catboost_{feature_set}.cbm"))
        if feature_set == "F3":
            importance_rows = [
                {
                    "feature": name,
                    "importance": float(value),
                    "feature_set": feature_set,
                    "training_scope": "en_bho_cv_full_data_after_oof_evaluation",
                }
                for name, value in zip(columns, cat_model.get_feature_importance())
            ]
    pd.DataFrame(importance_rows).sort_values("importance", ascending=False).to_csv(
        results_dir / "catboost_feature_importance.csv", index=False
    )


def _save_holdout_models(
    frame: pd.DataFrame,
    split: dict[str, np.ndarray],
    selected_models: dict[tuple[str, str], dict[str, Any]],
    models_dir: Path,
    results_dir: Path,
) -> None:
    train_idx = split["train"]
    y = frame["score"].to_numpy(dtype=float)
    importance_rows = []
    for feature_set, columns in FEATURE_SETS.items():
        ridge_info = selected_models[("ridge", feature_set)]
        ridge_model = _make_ridge(float(ridge_info["alpha"]))
        ridge_model.fit(frame.iloc[train_idx][columns], y[train_idx])
        joblib.dump(ridge_model, models_dir / f"pooled_ridge_{feature_set}.joblib")

        cat_info = selected_models[("catboost", feature_set)]
        cat_params = {
            "depth": int(cat_info["depth"]),
            "l2_leaf_reg": float(cat_info["l2_leaf_reg"]),
        }
        cat_model = _make_catboost(cat_params, iterations=int(cat_info["best_iteration"]))
        cat_model.fit(frame.iloc[train_idx][columns], y[train_idx], verbose=False)
        cat_model.save_model(str(models_dir / f"pooled_catboost_{feature_set}.cbm"))
        if feature_set == "F3":
            importance_rows = [
                {
                    "feature": name,
                    "importance": float(value),
                    "feature_set": feature_set,
                    "training_scope": "pooled_train_split",
                }
                for name, value in zip(columns, cat_model.get_feature_importance())
            ]
    pd.DataFrame(importance_rows).sort_values("importance", ascending=False).to_csv(
        results_dir / "pooled_catboost_feature_importance.csv", index=False
    )


def train_experiments(
    data_path: Path,
    results_dir: Path = Path("results"),
    models_dir: Path = Path("models"),
) -> tuple[Path, Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, compression="infer", low_memory=False)
    if frame.empty:
        raise ValueError("Prepared annotation table is empty")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    frame["leakage_group_id"] = assign_leakage_groups(frame).values
    feature_columns = sorted({column for columns in FEATURE_SETS.values() for column in columns})
    numeric_features = frame[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_features).all():
        raise ValueError("Nonfinite numeric features found after preparation")

    all_prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    run_metadata: dict[str, Any] = {
        "seed": 42,
        "feature_sets": FEATURE_SETS,
        "ridge_alphas": RIDGE_ALPHAS,
        "catboost_grid": CATBOOST_GRID,
        "catboost_iterations": CATBOOST_ITERATIONS,
        "catboost_early_stopping_rounds": CATBOOST_EARLY_STOPPING,
        "primary_language_pair": PRIMARY_LANGUAGE_PAIR,
        "primary_cv_document_threshold": PRIMARY_CV_DOCUMENT_THRESHOLD,
    }

    primary = frame[frame["language_pair"] == PRIMARY_LANGUAGE_PAIR].copy().reset_index(drop=True)
    if primary.empty:
        raise ValueError(f"Primary language pair {PRIMARY_LANGUAGE_PAIR} is absent")
    primary_docs = primary["doc_group_id"].nunique()
    primary_leakage_groups = primary["leakage_group_id"].nunique()
    run_metadata["primary_unique_documents"] = int(primary_docs)
    run_metadata["primary_unique_leakage_groups"] = int(primary_leakage_groups)

    if primary_docs < PRIMARY_CV_DOCUMENT_THRESHOLD:
        fold_count = min(CV_FOLDS, primary_leakage_groups)
        if fold_count < 3:
            raise ValueError("Too few primary document groups for grouped cross-validation")
        run_metadata["primary_split_strategy"] = f"{fold_count}-fold GroupKFold out-of-fold evaluation"
        splitter = GroupKFold(n_splits=fold_count)
        primary_groups = primary["leakage_group_id"].to_numpy()
        for fold_index, (outer_train, outer_test) in enumerate(
            splitter.split(np.zeros(len(primary)), groups=primary_groups), start=1
        ):
            _assert_no_group_or_source_overlap(
                primary, outer_train, outer_test, f"fold_{fold_index}_train", f"fold_{fold_index}_test"
            )
            tune_train, validation = _grouped_validation_split(
                primary, outer_train, seed=42 + fold_index
            )
            _assert_no_group_or_source_overlap(
                primary, tune_train, validation, f"fold_{fold_index}_tune_train", f"fold_{fold_index}_validation"
            )
            _assert_no_group_or_source_overlap(
                primary, np.concatenate([tune_train, validation]), outer_test,
                f"fold_{fold_index}_outer_train", f"fold_{fold_index}_test"
            )

            y = primary["score"].to_numpy(dtype=float)
            predictions: dict[tuple[str, str], np.ndarray] = {}
            mean_prediction = np.full(len(outer_test), float(np.mean(y[outer_train])))
            fixed_prediction = np.clip(
                100.0
                - 5.0 * primary.iloc[outer_test]["n_major"].to_numpy(dtype=float)
                - primary.iloc[outer_test]["n_minor"].to_numpy(dtype=float),
                0,
                100,
            )
            predictions[("mean", "F1")] = mean_prediction
            predictions[("fixed_penalty", "F1")] = fixed_prediction

            for feature_set in FEATURE_SETS:
                ridge_pred, ridge_info = tune_and_fit_ridge(
                    primary, tune_train, validation, outer_train, outer_test, feature_set
                )
                predictions[("ridge", feature_set)] = ridge_pred
                selection_rows.append(
                    _selection_row("en_bho_cv", f"fold_{fold_index}", "ridge", feature_set, ridge_info)
                )

                cat_pred, cat_info = tune_and_fit_catboost(
                    primary, tune_train, validation, outer_train, outer_test, feature_set
                )
                predictions[("catboost", feature_set)] = cat_pred
                selection_rows.append(
                    _selection_row("en_bho_cv", f"fold_{fold_index}", "catboost", feature_set, cat_info)
                )

            all_prediction_frames.append(
                _prediction_rows(primary, outer_test, "en_bho_cv", f"fold_{fold_index}", predictions)
            )
            split_map = {int(index): "outer_train" for index in outer_train}
            split_map.update({int(index): "outer_test" for index in outer_test})
            _write_split_rows(manifest_rows, "en_bho_cv", primary, split_map, f"fold_{fold_index}")
            print(
                f"Bhojpuri fold {fold_index}/{fold_count}: train={len(outer_train):,}, "
                f"test={len(outer_test):,}, tune={len(tune_train):,}, validation={len(validation):,}",
                flush=True,
            )

        selection_frame = pd.DataFrame(selection_rows)
        _save_final_cv_models(primary, selection_frame, models_dir, results_dir)
    else:
        run_metadata["primary_split_strategy"] = "grouped 70/15/15 holdout"
        primary_split = stratified_group_holdout(primary)
        primary_models: dict[tuple[str, str], dict[str, Any]] = {}
        _run_holdout_scope(
            primary,
            primary_split,
            "en_bho_holdout",
            all_prediction_frames,
            selection_rows,
            manifest_rows,
            primary_models,
        )
        _save_holdout_models(primary, primary_split, primary_models, models_dir, results_dir)

    pooled_split = stratified_group_holdout(frame)
    pooled_models: dict[tuple[str, str], dict[str, Any]] = {}
    _run_holdout_scope(
        frame,
        pooled_split,
        "pooled_holdout",
        all_prediction_frames,
        selection_rows,
        manifest_rows,
        pooled_models,
    )
    _save_holdout_models(frame, pooled_split, pooled_models, models_dir, results_dir)

    predictions_path = results_dir / "predictions.csv.gz"
    pd.concat(all_prediction_frames, ignore_index=True).to_csv(
        predictions_path, index=False, compression="gzip"
    )
    selection_path = results_dir / "model_selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    manifest_path = results_dir / "split_manifest.csv"
    pd.DataFrame(manifest_rows).drop_duplicates().to_csv(manifest_path, index=False)
    run_metadata["pooled_unique_documents"] = int(frame["doc_group_id"].nunique())
    run_metadata["pooled_unique_leakage_groups"] = int(frame["leakage_group_id"].nunique())
    run_metadata["pooled_split_rows"] = {
        name: int(len(indices)) for name, indices in pooled_split.items()
    }
    run_metadata["source_text_grouping"] = (
        "Documents sharing any exact source_text were unioned before splitting."
    )
    (results_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Predictions: {predictions_path.resolve()}")
    print(f"Split manifest: {manifest_path.resolve()}")
    return predictions_path, selection_path, manifest_path


def _run_holdout_scope(
    frame: pd.DataFrame,
    split: dict[str, np.ndarray],
    scope: str,
    prediction_frames: list[pd.DataFrame],
    selection_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    selected_models: dict[tuple[str, str], dict[str, Any]],
) -> None:
    train_idx, validation_idx, test_idx = split["train"], split["validation"], split["test"]
    _assert_no_group_or_source_overlap(frame, train_idx, validation_idx, "train", "validation")
    _assert_no_group_or_source_overlap(frame, train_idx, test_idx, "train", "test")
    _assert_no_group_or_source_overlap(frame, validation_idx, test_idx, "validation", "test")
    y = frame["score"].to_numpy(dtype=float)
    predictions: dict[tuple[str, str], np.ndarray] = {
        ("mean", "F1"): np.full(len(test_idx), float(np.mean(y[train_idx]))),
        (
            "fixed_penalty",
            "F1",
        ): np.clip(
            100.0
            - 5.0 * frame.iloc[test_idx]["n_major"].to_numpy(dtype=float)
            - frame.iloc[test_idx]["n_minor"].to_numpy(dtype=float),
            0,
            100,
        ),
    }
    for feature_set in FEATURE_SETS:
        ridge_pred, ridge_info = tune_and_fit_ridge(
            frame, train_idx, validation_idx, train_idx, test_idx, feature_set
        )
        predictions[("ridge", feature_set)] = ridge_pred
        selection_rows.append(_selection_row(scope, "holdout", "ridge", feature_set, ridge_info))
        selected_models[("ridge", feature_set)] = ridge_info

        cat_pred, cat_info = tune_and_fit_catboost(
            frame, train_idx, validation_idx, train_idx, test_idx, feature_set
        )
        predictions[("catboost", feature_set)] = cat_pred
        selection_rows.append(_selection_row(scope, "holdout", "catboost", feature_set, cat_info))
        selected_models[("catboost", feature_set)] = cat_info

    prediction_frames.append(_prediction_rows(frame, test_idx, scope, "holdout", predictions))
    split_by_index = {
        **{int(index): "train" for index in train_idx},
        **{int(index): "validation" for index in validation_idx},
        **{int(index): "test" for index in test_idx},
    }
    _write_split_rows(manifest_rows, scope, frame, split_by_index, "holdout")
    distribution = frame.iloc[test_idx].groupby("language_pair", observed=True).size().to_dict()
    print(
        f"{scope}: train={len(train_idx):,}, validation={len(validation_idx):,}, "
        f"test={len(test_idx):,}; test rows by pair={distribution}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/annotations.csv.gz"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    train_experiments(args.data, args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
