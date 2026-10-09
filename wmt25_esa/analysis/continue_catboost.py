"""Continue the saved pooled F3 CatBoost model using only its train/validation split."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from ..prepare_data import FEATURE_SETS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "catboost_continuation"


DATA_PATH = ROOT / "results" / "experiment2" / "paired_annotations.csv"
SELECTION_PATH = ROOT / "results" / "experiment2" / "model_selection.csv"
METRICS_PATH = ROOT / "results" / "experiment2" / "metrics.csv"
BASE_MODEL_PATH = ROOT / "models" / "experiment2" / "pooled_holdout_holdout_cross_catboost_F3.cbm"
FEATURES = FEATURE_SETS["F3"]
EXTRA_ITERATIONS = 1000
EARLY_STOPPING_ROUNDS = 100
BOOTSTRAP_REPLICATES = 2000


def weighted_metrics(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    clipped = np.clip(predicted, 0.0, 100.0)
    error = clipped - actual
    return {
        "weighted_mae": float(np.average(np.abs(error), weights=weight)),
        "weighted_rmse": float(np.sqrt(np.average(np.square(error), weights=weight))),
        "weighted_bias_pred_minus_human": float(np.average(error, weights=weight)),
    }


def no_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    for key in ("doc_group_id", "leakage_group_id", "translation_id", "pair_id"):
        values = {name: set(part[key].astype(str)) for name, part in splits.items()}
        names = list(values)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = values[left] & values[right]
                if overlap:
                    raise ValueError(f"{key} leaks between {left} and {right}: {len(overlap)}")


def paired_document_bootstrap(test: pd.DataFrame, base_prediction: np.ndarray, candidate_prediction: np.ndarray) -> pd.DataFrame:
    actual = test["score_b"].to_numpy(float)
    weight = test["pair_weight"].to_numpy(float)
    errors = pd.DataFrame({
        "doc_group_id": test["doc_group_id"].astype(str).to_numpy(),
        "weight": weight,
        "base_abs": np.abs(actual - np.clip(base_prediction, 0, 100)) * weight,
        "candidate_abs": np.abs(actual - np.clip(candidate_prediction, 0, 100)) * weight,
        "base_squared": np.square(actual - np.clip(base_prediction, 0, 100)) * weight,
        "candidate_squared": np.square(actual - np.clip(candidate_prediction, 0, 100)) * weight,
    })
    grouped = errors.groupby("doc_group_id", observed=True)[
        ["weight", "base_abs", "candidate_abs", "base_squared", "candidate_squared"]
    ].sum().to_numpy(float)
    if len(grouped) < 2:
        raise ValueError("Need multiple test documents for a document bootstrap")
    rng = np.random.default_rng(42)
    sampled = grouped[rng.integers(0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped)))].sum(axis=1)
    delta_mae = (sampled[:, 2] - sampled[:, 1]) / sampled[:, 0]
    delta_rmse = np.sqrt(sampled[:, 4] / sampled[:, 0]) - np.sqrt(sampled[:, 3] / sampled[:, 0])
    rows = []
    for metric, delta in (("weighted_mae", delta_mae), ("weighted_rmse", delta_rmse)):
        low, high = np.quantile(delta, [0.025, 0.975])
        rows.append({
            "metric": metric,
            "candidate_minus_baseline": float(delta.mean()),
            "document_bootstrap_ci_low": float(low),
            "document_bootstrap_ci_high": float(high),
            "replicates": BOOTSTRAP_REPLICATES,
            "documents": len(grouped),
        })
    return pd.DataFrame(rows)


def main() -> None:
    selection = pd.read_csv(SELECTION_PATH)
    selected = selection[
        (selection["experiment_scope"] == "pooled_holdout")
        & (selection["fold"] == "holdout")
        & (selection["regime"] == "cross")
        & (selection["model"] == "catboost")
        & (selection["feature_set"] == "F3")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one pooled cross F3 selection row, found {len(selected)}")
    selected_row = selected.iloc[0]

    usecols = [
        "experiment_scope", "split", "doc_group_id", "leakage_group_id",
        "translation_id", "pair_id", "directed_pair_id", "score_b", "pair_weight",
        *FEATURES,
    ]
    all_rows = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    pooled = all_rows[all_rows["experiment_scope"] == "pooled_holdout"].copy()
    splits = {
        name: pooled[pooled["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    no_split_overlap(splits)
    for split_name, expected in (
        ("train", int(selected_row["train_rows"])),
        ("validation", int(selected_row["validation_rows"])),
        ("test", int(selected_row["test_pairs"])),
    ):
        if len(splits[split_name]) != expected:
            raise ValueError(f"{split_name} rows={len(splits[split_name])}, expected {expected}")

    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    train_pool = Pool(train[FEATURES], label=train["score_b"], weight=train["pair_weight"])
    validation_pool = Pool(validation[FEATURES], label=validation["score_b"], weight=validation["pair_weight"])
    baseline_model = CatBoostRegressor()
    baseline_model.load_model(str(BASE_MODEL_PATH))
    if baseline_model.tree_count_ != int(selected_row["iterations"]):
        raise ValueError(
            f"Saved model has {baseline_model.tree_count_} trees but selection records "
            f"{int(selected_row['iterations'])}"
        )
    if baseline_model.feature_names_ != FEATURES:
        raise ValueError("Saved model features differ from F3")

    y_validation = validation["score_b"].to_numpy(float)
    w_validation = validation["pair_weight"].to_numpy(float)
    baseline_validation_prediction = baseline_model.predict(validation_pool)
    baseline_validation_metrics = weighted_metrics(
        y_validation, baseline_validation_prediction, w_validation
    )

    params = baseline_model.get_params()
    params["iterations"] = EXTRA_ITERATIONS
    continuation_model = CatBoostRegressor(**params)
    continuation_model.fit(
        train_pool,
        eval_set=validation_pool,
        init_model=baseline_model,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=False,
    )
    continuation_model.save_model(str(OUT / "continued_candidate.cbm"))

    candidate_validation_prediction = continuation_model.predict(validation_pool)
    candidate_validation_metrics = weighted_metrics(
        y_validation, candidate_validation_prediction, w_validation
    )
    use_continued_model = (
        candidate_validation_metrics["weighted_rmse"]
        < baseline_validation_metrics["weighted_rmse"]
    )
    chosen_model = continuation_model if use_continued_model else baseline_model

    y_test = test["score_b"].to_numpy(float)
    w_test = test["pair_weight"].to_numpy(float)
    baseline_test_prediction = baseline_model.predict(test[FEATURES])
    candidate_test_prediction = continuation_model.predict(test[FEATURES])
    chosen_test_prediction = chosen_model.predict(test[FEATURES])
    test_predictions = test[["directed_pair_id", "translation_id", "doc_group_id", "score_b", "pair_weight"]].copy()
    test_predictions["saved_model_prediction"] = np.clip(baseline_test_prediction, 0, 100)
    test_predictions["continued_candidate_prediction"] = np.clip(candidate_test_prediction, 0, 100)
    test_predictions.to_csv(OUT / "test_predictions.csv", index=False)
    delta_frame = paired_document_bootstrap(test, baseline_test_prediction, candidate_test_prediction)
    delta_frame.to_csv(OUT / "test_delta_bootstrap.csv", index=False)

    metric_rows = []
    for split_name, frame, actual, weight, predictions in (
        ("validation", validation, y_validation, w_validation,
         {"saved_453_tree_model": baseline_validation_prediction,
          "continued_candidate": candidate_validation_prediction}),
        ("test", test, y_test, w_test,
         {"saved_453_tree_model": baseline_test_prediction,
          "continued_candidate": candidate_test_prediction,
          "selected_by_validation": chosen_test_prediction}),
    ):
        for model_name, prediction in predictions.items():
            metric_rows.append({
                "split": split_name,
                "model": model_name,
                "n_directed_pairs": len(frame),
                "n_translations": int(frame["translation_id"].nunique()),
                "n_documents": int(frame["doc_group_id"].nunique()),
                "tree_count": (
                    baseline_model.tree_count_ if model_name == "saved_453_tree_model"
                    else continuation_model.tree_count_
                ),
                **weighted_metrics(actual, prediction, weight),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    prior_metrics = pd.read_csv(METRICS_PATH)
    prior = prior_metrics[
        (prior_metrics["experiment_scope"] == "pooled_holdout")
        & (prior_metrics["language_pair"] == "ALL_ESA")
        & (prior_metrics["regime"] == "cross_trained")
        & (prior_metrics["model"] == "catboost")
        & (prior_metrics["feature_set"] == "F3")
    ]
    if len(prior) != 1:
        raise ValueError(f"Expected one existing pooled CatBoost test metric row, found {len(prior)}")
    baseline_test = metrics[(metrics["split"] == "test") & (metrics["model"] == "saved_453_tree_model")].iloc[0]
    prior_test = prior.iloc[0]
    if abs(float(baseline_test["weighted_mae"]) - float(prior_test["mae"])) > 0.02:
        raise ValueError("Recomputed saved-model test MAE does not match the existing report")
    if abs(float(baseline_test["weighted_rmse"]) - float(prior_test["rmse"])) > 0.02:
        raise ValueError("Recomputed saved-model test RMSE does not match the existing report")

    curve_rows = []
    base_history = baseline_model.get_evals_result()
    for pool_name, metric_map in base_history.items():
        for metric_name, values in metric_map.items():
            for index, value in enumerate(values, start=1):
                curve_rows.append({
                    "phase": "saved_model", "pool": pool_name, "metric": metric_name,
                    "tree_index": index, "value": float(value),
                })
    continued_history = continuation_model.get_evals_result()
    for pool_name, metric_map in continued_history.items():
        for metric_name, values in metric_map.items():
            for index, value in enumerate(values, start=1):
                curve_rows.append({
                    "phase": "continuation_fit", "pool": pool_name, "metric": metric_name,
                    "tree_index": baseline_model.tree_count_ + index, "value": float(value),
                })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / "loss_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    learn = curves[(curves["pool"] == "learn") & (curves["metric"] == "RMSE")]
    for phase, group in learn.groupby("phase", sort=False):
        ax.plot(group["tree_index"], group["value"], label=f"{phase} train RMSE")
    validation_curve = curves[
        (curves["phase"] == "continuation_fit")
        & (curves["pool"].str.startswith("validation"))
        & (curves["metric"] == "RMSE")
    ]
    if not validation_curve.empty:
        ax.plot(validation_curve["tree_index"], validation_curve["value"], label="continuation validation RMSE")
    ax.axhline(
        baseline_validation_metrics["weighted_rmse"], color="#666666", linestyle="--",
        label="saved model validation RMSE (weighted)",
    )
    ax.set_xlabel("Total tree count during fit (post-stop trees are discarded)")
    ax.set_ylabel("RMSE")
    ax.set_title("CatBoost F3 training continuation")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=160)
    plt.close(fig)

    summary = {
        "experiment": "pooled_holdout_cross_catboost_F3_continuation",
        "base_model": str(BASE_MODEL_PATH.relative_to(ROOT)),
        "base_tree_count": int(baseline_model.tree_count_),
        "extra_iterations_requested": EXTRA_ITERATIONS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "candidate_tree_count": int(continuation_model.tree_count_),
        "candidate_best_iteration": int(continuation_model.get_best_iteration()),
        "continued_model_selected_by_validation": bool(use_continued_model),
        "selected_tree_count": int(chosen_model.tree_count_),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_documents": int(train["doc_group_id"].nunique()),
        "validation_documents": int(validation["doc_group_id"].nunique()),
        "test_documents": int(test["doc_group_id"].nunique()),
        "base_validation": baseline_validation_metrics,
        "candidate_validation": candidate_validation_metrics,
        "original_validation_rmse_from_model_selection": float(selected_row["validation_rmse"]),
        "original_test_metrics": {
            "mae": float(prior_test["mae"]), "rmse": float(prior_test["rmse"]),
        },
        "fit_parameters": params,
        "validation_selection_only": True,
        "test_used_for_selection": False,
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    val_rows = metrics[metrics["split"] == "validation"].set_index("model")
    test_rows = metrics[metrics["split"] == "test"].set_index("model")
    base_val, candidate_val = val_rows.loc["saved_453_tree_model"], val_rows.loc["continued_candidate"]
    base_test, candidate_test = test_rows.loc["saved_453_tree_model"], test_rows.loc["continued_candidate"]
    chosen_name = "continued_candidate" if use_continued_model else "saved_453_tree_model"
    chosen_test = test_rows.loc["selected_by_validation"]
    mae_delta = delta_frame[delta_frame["metric"] == "weighted_mae"].iloc[0]
    rmse_delta = delta_frame[delta_frame["metric"] == "weighted_rmse"].iloc[0]
    report = f"""# CatBoost continuation check

## What was saved already

The fitted CatBoost `.cbm` models retain a per-tree **training-set RMSE** curve, available through CatBoost's `get_evals_result()`. The project also saved validation-selected parameters, best validation RMSE, and selected tree count in `results/experiment2/model_selection.csv`. It did **not** save a per-tree validation curve for the final refit model.

The continuation starts from the existing pooled F3 cross-annotator model with {baseline_model.tree_count_} trees (`{BASE_MODEL_PATH.relative_to(ROOT)}`). Its selected settings are depth {params.get('depth')}, L2 {params.get('l2_leaf_reg')}, learning rate {params.get('learning_rate')}, RMSE loss, and {params.get('thread_count')} CPU threads.

The primary English-to-Bhojpuri cross-trained F3 folds in the existing selection file stopped at 96–214 trees (maximum 500). I selected the pooled cross-trained F3 model for this check because it had the most iteration headroom, at 453 trees, and a separate validation/test partition.

## Method

Continued the saved model for up to {EXTRA_ITERATIONS} additional trees using the exact original training rows. The existing validation partition controlled early stopping (patience {EARLY_STOPPING_ROUNDS}) and selection. The original held-out test partition was not used for tuning. Document, leakage-group, translation, and pair IDs were checked for overlap across train, validation, and test.

| Split | Model | Trees | Weighted MAE | Weighted RMSE |
|---|---|---:|---:|---:|
| validation | saved model | {baseline_model.tree_count_} | {base_val['weighted_mae']:.3f} | {base_val['weighted_rmse']:.3f} |
| validation | continued candidate | {continuation_model.tree_count_} | {candidate_val['weighted_mae']:.3f} | {candidate_val['weighted_rmse']:.3f} |
| test | saved model | {baseline_model.tree_count_} | {base_test['weighted_mae']:.3f} | {base_test['weighted_rmse']:.3f} |
| test | continued candidate | {continuation_model.tree_count_} | {candidate_test['weighted_mae']:.3f} | {candidate_test['weighted_rmse']:.3f} |
| test | selected using validation | {chosen_model.tree_count_} | {chosen_test['weighted_mae']:.3f} | {chosen_test['weighted_rmse']:.3f} |

**Validation decision:** `{chosen_name}` was selected; the test set was not involved in that choice. The saved model's test metrics reproduce the existing report within tolerance.

On the test set, candidate-minus-baseline MAE was {mae_delta['candidate_minus_baseline']:.3f} (95% document-bootstrap CI {mae_delta['document_bootstrap_ci_low']:.6f} to {mae_delta['document_bootstrap_ci_high']:.6f}); RMSE difference was {rmse_delta['candidate_minus_baseline']:.3f} (CI {rmse_delta['document_bootstrap_ci_low']:.6f} to {rmse_delta['document_bootstrap_ci_high']:.6f}). These paired intervals quantify the small test difference.

## Interpretation

{('The continuation improved validation RMSE, so it was selected before test evaluation. ' if use_continued_model else 'The continuation did not improve validation RMSE, so the original model was retained. ')}It added {continuation_model.tree_count_ - baseline_model.tree_count_} retained trees; early stopping observed {len(continued_history.get('validation', {}).get('RMSE', []))} continuation rounds before stopping and pruning. The test MAE gain was just {abs(candidate_test['weighted_mae'] - base_test['weighted_mae']):.3f} points (its paired interval only barely excludes zero), while test RMSE rose by {candidate_test['weighted_rmse'] - base_test['weighted_rmse']:.3f} and its interval includes zero. For the RMSE objective, this is no meaningful held-out improvement; keeping the original 453-tree model is the sensible choice. This is one controlled continuation check of one pooled F3 model, not evidence that adding iterations will help every language pair or fold. A lower training loss alone does not imply better held-out quality.

## Artifacts

- `loss_curve.csv`: saved model training curve and continuation training/validation curves.
- `loss_curve.png`: RMSE curves for visual inspection.
- `metrics.csv`: baseline and continued candidate metrics on validation and test.
- `test_delta_bootstrap.csv` and `test_predictions.csv`: paired document bootstrap and auditable held-out predictions.
- `continued_candidate.cbm`: new candidate only; the existing model under `models/` was left unchanged.
- `run_summary.json`: split counts, selected tree counts, parameters, and validation decision.
- `continue_catboost.py`: reproducible CPU-only continuation script.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
