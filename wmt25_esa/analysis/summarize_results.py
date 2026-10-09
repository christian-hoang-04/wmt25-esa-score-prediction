"""Create uncertainty summaries and plots from the unseen-evaluator run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[2] / "results" / "experiment4_unseen_evaluators"
SEED = 20261009
MODELS = ["mean", "fixed_penalty", "ridge_f3", "catboost_f3"]


def evaluator_bootstrap(evaluator_metrics: pd.DataFrame, n_boot: int = 5000) -> pd.DataFrame:
    """Paired bootstrap over evaluators, stratified by language pair."""
    pivot = evaluator_metrics.pivot_table(
        index=["language_pair", "evaluator_key"], columns="model", values="mae"
    ).reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    comparisons = ["ridge_f3", "mean", "fixed_penalty"]
    for scope in ["ALL_ESA", "en-bho_IN"]:
        d = pivot if scope == "ALL_ESA" else pivot[pivot.language_pair.eq(scope)]
        lp_groups = {lp: g.reset_index(drop=True) for lp, g in d.groupby("language_pair")}
        for reference in comparisons:
            observed = float((d.catboost_f3 - d[reference]).mean())
            samples = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                sampled_groups = []
                for g in lp_groups.values():
                    idx = rng.integers(0, len(g), size=len(g))
                    sampled_groups.append(g.iloc[idx])
                sample = pd.concat(sampled_groups, ignore_index=True)
                samples[b] = (sample.catboost_f3 - sample[reference]).mean()
            rows.append({
                "scope": scope,
                "comparison": f"catboost_f3_minus_{reference}_macro_evaluator_mae",
                "observed_delta": observed,
                "n_bootstrap": n_boot,
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    plot_dir = HERE / "plots"
    plot_dir.mkdir(exist_ok=True)
    predictions = pd.read_csv(HERE / "predictions.csv.gz")
    evaluator_metrics = pd.read_csv(HERE / "unseen_evaluator_metrics.csv")

    macro_ci = evaluator_bootstrap(evaluator_metrics)
    macro_ci.to_csv(HERE / "macro_evaluator_bootstrap.csv", index=False)

    predictions["catboost_absolute_error"] = (
        predictions.score - predictions.catboost_f3
    ).abs()
    predictions.nlargest(100, "catboost_absolute_error").to_csv(
        HERE / "largest_catboost_errors.csv", index=False
    )

    # Score scatter: hexbin keeps the dense pooled panel legible.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, lp, title in [
        (axes[0], "en-bho_IN", "English → Bhojpuri"),
        (axes[1], None, "All ESA directions, direction-specific models"),
    ]:
        d = predictions if lp is None else predictions[predictions.language_pair.eq(lp)]
        hb = ax.hexbin(
            d.score, d.catboost_f3, gridsize=40, mincnt=1, bins="log", cmap="Blues"
        )
        ax.plot([0, 100], [0, 100], color="#d62728", linewidth=1, linestyle="--")
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Human score", ylabel="CatBoost prediction", title=title)
        fig.colorbar(hb, ax=ax, label="log10 annotation count")
    fig.savefig(plot_dir / "catboost_unseen_evaluator_scatter.png", dpi=160)
    plt.close(fig)

    metrics = pd.read_csv(HERE / "metrics.csv")
    direction = metrics[
        metrics.language_pair.ne("ALL_ESA")
        & metrics.model.isin(["mean", "fixed_penalty", "ridge_f3", "catboost_f3"])
    ].pivot(index="language_pair", columns="model", values="mae")
    direction = direction.sort_index()
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    direction[["mean", "fixed_penalty", "ridge_f3", "catboost_f3"]].plot(
        kind="bar", ax=ax, color=["#9aa0a6", "#d99000", "#2878b5", "#2b8c56"]
    )
    ax.set_ylabel("MAE on unseen evaluator annotations")
    ax.set_xlabel("Language pair")
    ax.set_title("Unseen-evaluator MAE by language pair")
    ax.legend(["Training mean", "Fixed penalty", "Ridge F3", "CatBoost F3"], ncol=2)
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(plot_dir / "mae_by_language_pair.png", dpi=160)
    plt.close(fig)

    importance_path = HERE / "catboost_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(12).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.barh(importance.feature, importance.importance, color="#2b8c56")
        ax.set_xlabel("Mean CatBoost feature importance across folds")
        ax.set_title("Features used for unseen-evaluator predictions")
        fig.savefig(plot_dir / "catboost_feature_importance.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
