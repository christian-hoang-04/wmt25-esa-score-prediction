"""Test whether ESA error-to-score models generalize to held-out annotators.

All paths are resolved from the project root. This script only writes beneath
results/experiment4_unseen_evaluators/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "annotations.csv.gz"
SPLITS = ROOT / "results" / "split_manifest.csv"
FEATURES = [
    "n_minor", "n_major", "n_total", "target_char_length", "source_char_length",
    "target_word_length", "source_word_length", "minor_coverage", "major_coverage",
    "total_error_coverage", "max_error_span_length", "mean_error_span_length",
    "error_density", "major_fraction", "has_major_error", "has_zero_errors",
]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
SEED = 20261009
N_DOC_FOLDS = 5
N_ANNOTATOR_FOLDS = 5
BOOTSTRAPS = 1000


def equal_annotator_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every evaluator in this split the same total training/validation weight."""
    counts = frame["evaluator_key"].value_counts()
    return (len(frame) / len(counts) / frame["evaluator_key"].map(counts)).to_numpy(float)


def safe_corr(y: np.ndarray, pred: np.ndarray, method: str) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(y, pred).statistic)


def make_document_folds(frame: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Create five held-out document folds, grouping exact-source leakage groups."""
    doc_table = frame[["doc_group_id", "leakage_group_id"]].drop_duplicates()
    if doc_table["doc_group_id"].duplicated().any():
        raise ValueError("A document maps to more than one leakage group")
    groups = doc_table["leakage_group_id"].to_numpy()
    n_splits = min(N_DOC_FOLDS, len(np.unique(groups)))
    if n_splits < 3:
        raise ValueError(f"Not enough document leakage groups: {n_splits}")

    splits: list[dict] = []
    rows: list[dict] = []
    splitter = GroupKFold(n_splits=n_splits)
    for doc_fold, (dev_indices, test_indices) in enumerate(
        splitter.split(doc_table, groups=groups), start=1
    ):
        development = doc_table.iloc[dev_indices].reset_index(drop=True)
        test_docs = set(doc_table.iloc[test_indices]["doc_group_id"])
        # Test is 1/n folds. Validation is about 15% of all docs, split only
        # from the remaining development docs and kept source-group-disjoint.
        val_fraction_of_development = 0.15 / (1.0 - 1.0 / n_splits)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction_of_development,
            random_state=SEED + doc_fold,
        )
        train_ix, val_ix = next(
            gss.split(development, groups=development["leakage_group_id"])
        )
        train_docs = set(development.iloc[train_ix]["doc_group_id"])
        val_docs = set(development.iloc[val_ix]["doc_group_id"])
        train_groups = set(development.iloc[train_ix]["leakage_group_id"])
        val_groups = set(development.iloc[val_ix]["leakage_group_id"])
        test_groups = set(doc_table.iloc[test_indices]["leakage_group_id"])
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise AssertionError("Exact-source leakage groups cross document splits")
        split = {
            "doc_fold": doc_fold,
            "train_docs": train_docs,
            "validation_docs": val_docs,
            "test_docs": test_docs,
            "train_groups": train_groups,
            "validation_groups": val_groups,
            "test_groups": test_groups,
        }
        splits.append(split)
        for split_name, docs in (
            ("train", train_docs), ("validation", val_docs), ("test", test_docs)
        ):
            for doc_id in sorted(docs):
                rows.append({
                    "doc_fold": doc_fold,
                    "doc_group_id": doc_id,
                    "leakage_group_id": doc_table.loc[
                        doc_table.doc_group_id.eq(doc_id), "leakage_group_id"
                    ].iloc[0],
                    "document_split": split_name,
                })
    return splits, pd.DataFrame(rows)


def metric_row(scope: str, lp: str, model: str, data: pd.DataFrame) -> dict:
    y = data["score"].to_numpy(float)
    pred = data[model].to_numpy(float)
    per_evaluator = [
        mean_absolute_error(g.score, g[model])
        for _, g in data.groupby("evaluator_key", sort=False)
    ]
    return {
        "scope": scope,
        "language_pair": lp,
        "model": model,
        "n_annotations": len(data),
        "n_documents": data["doc_group_id"].nunique(),
        "n_unseen_evaluators": data["evaluator_key"].nunique(),
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "pearson": safe_corr(y, pred, "pearson"),
        "spearman": safe_corr(y, pred, "spearman"),
        "macro_evaluator_mae": float(np.mean(per_evaluator)) if per_evaluator else np.nan,
    }


def cluster_bootstrap(data: pd.DataFrame, scope: str, n_boot: int = BOOTSTRAPS) -> list[dict]:
    """Bootstrap paired MAE differences by exact-source leakage group."""
    rng = np.random.default_rng(SEED + (0 if scope == "ALL_ESA" else 1))
    grouped = data.groupby("leakage_group_id", sort=False)
    keys = list(grouped.groups)
    # Store cluster sums so resampling stays small and preserves document clusters.
    sums = {}
    for key, group in grouped:
        sums[key] = {
            model: float(np.abs(group.score.to_numpy(float) - group[model].to_numpy(float)).sum())
            for model in ["mean", "fixed_penalty", "ridge_f3", "catboost_f3"]
        }
        sums[key]["n"] = len(group)
    output: list[dict] = []
    for reference in ["mean", "fixed_penalty", "ridge_f3"]:
        deltas = []
        for _ in range(n_boot):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            n = sum(sums[k]["n"] for k in sampled)
            delta = (
                sum(sums[k]["catboost_f3"] - sums[k][reference] for k in sampled) / n
            )
            deltas.append(delta)
        output.append({
            "scope": scope,
            "comparison": f"catboost_f3_minus_{reference}_mae",
            "n_bootstrap": n_boot,
            "ci_low": float(np.quantile(deltas, 0.025)),
            "ci_high": float(np.quantile(deltas, 0.975)),
        })
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    manifest = pd.read_csv(SPLITS)
    pooled_manifest = manifest[manifest.experiment_scope.eq("pooled_holdout")]
    leak_map = pooled_manifest[["doc_group_id", "leakage_group_id"]].drop_duplicates()
    if leak_map.doc_group_id.duplicated().any():
        raise ValueError("Existing pooled split manifest maps a document more than once")
    df = df.merge(leak_map, on="doc_group_id", how="left", validate="many_to_one")
    if df.leakage_group_id.isna().any():
        raise ValueError("Some prepared documents are absent from the source-leakage manifest")

    df["annotator_id"] = df.annotator_id.fillna("").astype(str).str.strip()
    unknown_mask = df.annotator_id.str.lower().isin({"", "unknown", "nan", "none"})
    excluded_unknown = int(unknown_mask.sum())
    df = df.loc[~unknown_mask].copy()
    df["evaluator_key"] = df.language_pair.astype(str) + "::" + df.annotator_id
    if df.annotation_id.duplicated().any():
        raise ValueError("annotation_id is not unique")
    if df[FEATURES + ["score"]].isna().any().any():
        raise ValueError("A retained row has missing numeric features or score")

    all_predictions: list[pd.DataFrame] = []
    fold_audit: list[dict] = []
    evaluator_fold_rows: list[dict] = []
    document_fold_rows: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    fit_count = 0

    for lp in sorted(df.language_pair.unique()):
        lp_df = df[df.language_pair.eq(lp)].copy().sort_values("annotation_id").reset_index(drop=True)
        evaluator_counts = lp_df.evaluator_key.value_counts()
        n_eval_folds = min(N_ANNOTATOR_FOLDS, len(evaluator_counts))
        if n_eval_folds < 2:
            continue

        eval_splitter = GroupKFold(n_splits=n_eval_folds)
        evaluator_table = lp_df[["evaluator_key"]].drop_duplicates().sort_values("evaluator_key").reset_index(drop=True)
        # Split unique evaluator IDs, weighted by their annotation volume for balanced folds.
        eval_rows = lp_df[["evaluator_key"]].reset_index(drop=True)
        eval_fold_for_key: dict[str, int] = {}
        for fold_no, (_, held_ix) in enumerate(
            eval_splitter.split(eval_rows, groups=eval_rows["evaluator_key"]), start=1
        ):
            for key in eval_rows.iloc[held_ix].evaluator_key.unique():
                eval_fold_for_key[key] = fold_no
        for key, fold_no in sorted(eval_fold_for_key.items()):
            evaluator_fold_rows.append({
                "language_pair": lp,
                "evaluator_key": key,
                "annotator_id": key.split("::", 1)[1],
                "heldout_evaluator_fold": fold_no,
                "n_annotations": int(evaluator_counts[key]),
                "n_documents": int(lp_df.loc[lp_df.evaluator_key.eq(key), "doc_group_id"].nunique()),
            })

        doc_splits, doc_manifest = make_document_folds(lp_df)
        doc_manifest["language_pair"] = lp
        document_fold_rows.append(doc_manifest)

        for eval_fold in range(1, n_eval_folds + 1):
            held_evaluators = {
                key for key, fold_no in eval_fold_for_key.items() if fold_no == eval_fold
            }
            seen_evaluators = set(lp_df.evaluator_key.unique()) - held_evaluators
            for doc_split in doc_splits:
                doc_fold = doc_split["doc_fold"]
                train_docs = doc_split["train_docs"]
                validation_docs = doc_split["validation_docs"]
                test_docs = doc_split["test_docs"]
                train = lp_df[
                    lp_df.doc_group_id.isin(train_docs)
                    & lp_df.evaluator_key.isin(seen_evaluators)
                ].copy()
                train_eval_ids = set(train.evaluator_key.unique())
                validation = lp_df[
                    lp_df.doc_group_id.isin(validation_docs)
                    & lp_df.evaluator_key.isin(train_eval_ids)
                ].copy()
                test = lp_df[
                    lp_df.doc_group_id.isin(test_docs)
                    & lp_df.evaluator_key.isin(held_evaluators)
                ].copy()
                if train.empty or validation.empty:
                    raise ValueError(f"Empty train/validation for {lp}, evaluator fold {eval_fold}, document fold {doc_fold}")
                if test.empty:
                    # No held-out evaluator happened to annotate this doc fold.
                    continue
                if set(train.evaluator_key) & held_evaluators or set(validation.evaluator_key) & held_evaluators:
                    raise AssertionError("Held-out evaluator appears in training or validation")
                if not set(validation.evaluator_key).issubset(train_eval_ids):
                    raise AssertionError("Validation includes an evaluator with no training observations")
                if set(train.leakage_group_id) & set(test.leakage_group_id):
                    raise AssertionError("Test source leakage group overlaps train")
                if set(validation.leakage_group_id) & set(test.leakage_group_id):
                    raise AssertionError("Test source leakage group overlaps validation")
                if set(train.leakage_group_id) & set(validation.leakage_group_id):
                    raise AssertionError("Train source leakage group overlaps validation")

                X_train = train[FEATURES].to_numpy(float)
                y_train = train.score.to_numpy(float)
                X_val = validation[FEATURES].to_numpy(float)
                y_val = validation.score.to_numpy(float)
                X_test = test[FEATURES].to_numpy(float)
                y_test = test.score.to_numpy(float)
                w_train = equal_annotator_weights(train)
                w_val = equal_annotator_weights(validation)

                # Validation rows are from evaluators represented in training;
                # held-out evaluator IDs and scores are never used for selection.
                scaler = StandardScaler()
                scaler.fit(X_train, sample_weight=w_train)
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                best_ridge = None
                best_alpha = None
                best_val_mae = float("inf")
                for alpha in ALPHAS:
                    candidate = Ridge(alpha=alpha)
                    candidate.fit(X_train_scaled, y_train, sample_weight=w_train)
                    val_prediction = np.clip(candidate.predict(X_val_scaled), 0, 100)
                    val_mae = np.average(np.abs(y_val - val_prediction), weights=w_val)
                    if val_mae < best_val_mae:
                        best_val_mae, best_alpha, best_ridge = val_mae, alpha, candidate
                assert best_ridge is not None

                cat = CatBoostRegressor(
                    iterations=500,
                    depth=4,
                    learning_rate=0.03,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    l2_leaf_reg=3.0,
                    random_seed=SEED,
                    thread_count=4,
                    allow_writing_files=False,
                    verbose=False,
                )
                val_pool = Pool(X_val, y_val, weight=w_val)
                cat.fit(
                    X_train,
                    y_train,
                    sample_weight=w_train,
                    eval_set=val_pool,
                    early_stopping_rounds=60,
                    use_best_model=True,
                    verbose=False,
                )
                cat_prediction = np.clip(cat.predict(X_test), 0, 100)
                ridge_prediction = np.clip(best_ridge.predict(X_test_scaled), 0, 100)
                train_mean = float(np.average(y_train, weights=w_train))
                fixed_prediction = np.clip(
                    100 - 5 * test.n_major.to_numpy(float) - test.n_minor.to_numpy(float),
                    0,
                    100,
                )
                fold_pred = test[[
                    "annotation_id", "language_pair", "doc_group_id", "leakage_group_id",
                    "doc_id", "system_name", "annotator_id", "evaluator_key", "score",
                    "n_minor", "n_major", "n_total", "target_char_length", "source_char_length",
                    "target_word_length", "source_word_length", "minor_coverage", "major_coverage",
                    "total_error_coverage", "max_error_span_length", "mean_error_span_length",
                    "error_density", "major_fraction", "has_major_error", "has_zero_errors",
                ]].copy()
                fold_pred["mean"] = train_mean
                fold_pred["fixed_penalty"] = fixed_prediction
                fold_pred["ridge_f3"] = ridge_prediction
                fold_pred["catboost_f3"] = cat_prediction
                fold_pred["heldout_evaluator_fold"] = eval_fold
                fold_pred["heldout_document_fold"] = doc_fold
                all_predictions.append(fold_pred)
                fit_count += 1
                fold_audit.append({
                    "language_pair": lp,
                    "heldout_evaluator_fold": eval_fold,
                    "heldout_document_fold": doc_fold,
                    "n_train_annotations": len(train),
                    "n_validation_annotations": len(validation),
                    "n_test_annotations": len(test),
                    "n_train_documents": train.doc_group_id.nunique(),
                    "n_validation_documents": validation.doc_group_id.nunique(),
                    "n_test_documents": test.doc_group_id.nunique(),
                    "n_train_evaluators": train.evaluator_key.nunique(),
                    "n_heldout_evaluators": len(held_evaluators),
                    "n_validation_evaluators": validation.evaluator_key.nunique(),
                    "test_evaluators_not_in_train_or_validation": True,
                    "source_groups_disjoint": True,
                    "ridge_alpha_selected_on_validation": best_alpha,
                    "ridge_validation_weighted_mae": best_val_mae,
                    "catboost_best_iteration": int(cat.get_best_iteration()),
                    "catboost_validation_best_rmse": float(cat.get_best_score()["validation"]["RMSE"]),
                })
                importances = cat.get_feature_importance()
                for feature, importance in zip(FEATURES, importances):
                    importance_rows.append({
                        "language_pair": lp,
                        "heldout_evaluator_fold": eval_fold,
                        "heldout_document_fold": doc_fold,
                        "feature": feature,
                        "importance": float(importance),
                    })
                if fit_count % 25 == 0:
                    print(f"completed {fit_count} train/eval folds; current {lp}", flush=True)

    if not all_predictions:
        raise RuntimeError("No unseen-evaluator test predictions were generated")
    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions.annotation_id.duplicated().any():
        duplicates = int(predictions.annotation_id.duplicated().sum())
        raise AssertionError(f"Each annotation should have one test prediction; duplicates={duplicates}")
    expected_ids = set(df.annotation_id)
    predicted_ids = set(predictions.annotation_id)
    missing_ids = expected_ids - predicted_ids
    if missing_ids:
        raise AssertionError(f"{len(missing_ids)} identifiable annotations did not receive a test prediction")
    if predicted_ids - expected_ids:
        raise AssertionError("Test predictions contain unknown annotation IDs")

    predictions.to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_audit).to_csv(OUT / "fold_audit.csv", index=False)
    pd.DataFrame(evaluator_fold_rows).to_csv(OUT / "evaluator_fold_manifest.csv", index=False)
    pd.concat(document_fold_rows, ignore_index=True).to_csv(OUT / "document_fold_manifest.csv", index=False)

    rows = []
    for model in ["mean", "fixed_penalty", "ridge_f3", "catboost_f3"]:
        rows.append(metric_row("ALL_ESA", "ALL_ESA", model, predictions))
        for lp, group in predictions.groupby("language_pair", sort=True):
            rows.append(metric_row("per_language_pair", lp, model, group))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    evaluator_metrics = []
    for (lp, evaluator_key), group in predictions.groupby(["language_pair", "evaluator_key"], sort=True):
        for model in ["mean", "fixed_penalty", "ridge_f3", "catboost_f3"]:
            evaluator_metrics.append({
                "language_pair": lp,
                "evaluator_key": evaluator_key,
                "annotator_id": group.annotator_id.iloc[0],
                "model": model,
                "n_test_annotations": len(group),
                "n_test_documents": group.doc_group_id.nunique(),
                "human_score_mean": group.score.mean(),
                "mae": mean_absolute_error(group.score, group[model]),
            })
    evaluator_metrics_df = pd.DataFrame(evaluator_metrics)
    evaluator_metrics_df.to_csv(OUT / "unseen_evaluator_metrics.csv", index=False)

    boots = cluster_bootstrap(predictions, "ALL_ESA")
    bho = predictions[predictions.language_pair.eq("en-bho_IN")]
    boots += cluster_bootstrap(bho, "en-bho_IN")
    pd.DataFrame(boots).to_csv(OUT / "paired_bootstrap.csv", index=False)

    # Human disagreement on the same translation, averaging repeated ratings
    # from the same evaluator before comparing distinct evaluators.
    translation_cols = ["language_pair", "doc_id", "system_name"]
    rater_scores = (
        df.groupby(translation_cols + ["evaluator_key"], as_index=False)
        .score.mean()
    )
    disagreement_rows = []
    for lp, group in rater_scores.groupby("language_pair", sort=True):
        diffs: list[float] = []
        for _, trans in group.groupby(translation_cols[1:], sort=False):
            vals = trans.score.to_numpy(float)
            if len(vals) > 1:
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        diffs.append(abs(vals[i] - vals[j]))
        if diffs:
            disagreement_rows.append({
                "language_pair": lp,
                "translations_with_multiple_evaluators": int(
                    group.groupby(translation_cols[1:]).evaluator_key.nunique().gt(1).sum()
                ),
                "evaluator_pairs": len(diffs),
                "mean_abs_score_difference": float(np.mean(diffs)),
                "median_abs_score_difference": float(np.median(diffs)),
                "p90_abs_score_difference": float(np.quantile(diffs, 0.90)),
            })
    pd.DataFrame(disagreement_rows).to_csv(OUT / "human_disagreement.csv", index=False)

    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance.groupby("feature", as_index=False).importance.mean().sort_values(
            "importance", ascending=False
        ).to_csv(OUT / "catboost_feature_importance.csv", index=False)

    per_lp_counts = (
        df.groupby("language_pair").agg(
            n_annotations=("annotation_id", "size"),
            n_documents=("doc_group_id", "nunique"),
            n_systems=("system_name", "nunique"),
            n_evaluators=("evaluator_key", "nunique"),
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            no_error_pct=("has_zero_errors", lambda s: 100 * s.mean()),
            mean_major_errors=("n_major", "mean"),
            mean_minor_errors=("n_minor", "mean"),
        ).reset_index()
    )
    per_lp_counts.to_csv(OUT / "dataset_summary.csv", index=False)
    run_config = {
        "seed": SEED,
        "annotator_folds": N_ANNOTATOR_FOLDS,
        "document_folds": N_DOC_FOLDS,
        "catboost": {
            "iterations": 500, "depth": 4, "learning_rate": 0.03,
            "l2_leaf_reg": 3, "early_stopping_rounds": 60,
            "loss_function": "RMSE", "thread_count": 4,
        },
        "ridge_alphas": ALPHAS,
        "features": FEATURES,
        "excluded_unidentified_annotator_rows": excluded_unknown,
        "input_annotations_before_unidentified_exclusion": int(excluded_unknown + len(df)),
        "input_annotations_evaluated": int(len(df)),
        "training_and_validation_weights": "equal total weight per evaluator within each split",
        "test_design": "held-out evaluator group AND held-out exact-source document group",
        "train_eval_fits": fit_count,
    }
    (OUT / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"finished {fit_count} training/evaluation folds; predictions={len(predictions)}", flush=True)


if __name__ == "__main__":
    main()
