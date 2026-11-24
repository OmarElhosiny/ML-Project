"""Gaussian Mixture Model (GMM) clustering: baseline, tuning and final model.

Uses the shared clustering features (src/models/clustering/features.py) and the
shared evaluation (src/evaluation/clustering_metrics.py), so the results are
directly comparable with K-Means and Agglomerative clustering.

Run:  python -m src.models.clustering.gmm_model
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.evaluation.clustering_metrics import build_comparison_table, evaluate_clustering
from src.models.clustering.features import CLUSTERING_FEATURES, prepare_clustering_matrix

RANDOM_STATE = 42
N_INIT = 3  # restarts per configuration, keeps the best fit

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
ARTIFACTS_DIR = Path("artifacts")

# Baseline: a reasonable starting point, chosen before any tuning
BASELINE_PARAMS = {"n_components": 3, "covariance_type": "full"}

# Search space for tuning
N_COMPONENTS_RANGE = range(2, 9)
COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]

# Selection rules (see select_best_config)
MIN_COMPONENTS = 3          # k = 2 only separates bookings with / without children
MIN_CLUSTER_SHARE = 0.01    # every cluster must hold at least 1% of bookings


def fit_gmm(X, n_components, covariance_type):
    """Fit one GMM and return (model, labels)."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    labels = model.fit_predict(X)
    return model, labels


def score_config(X, model, labels):
    """All metrics for one fitted configuration."""
    metrics = evaluate_clustering(X, labels)
    sizes = np.bincount(labels)
    metrics["bic"] = model.bic(X)
    metrics["aic"] = model.aic(X)
    metrics["min_cluster_share"] = sizes.min() / len(labels)
    return metrics


def run_baseline(X):
    """Baseline GMM with fixed, untuned parameters."""
    model, labels = fit_gmm(X, **BASELINE_PARAMS)
    return {**BASELINE_PARAMS, **score_config(X, model, labels)}


def tune_gmm(X):
    """Try every (n_components, covariance_type) pair and score it."""
    rows = []
    for covariance_type in COVARIANCE_TYPES:
        for n_components in N_COMPONENTS_RANGE:
            model, labels = fit_gmm(X, n_components, covariance_type)
            row = {"n_components": n_components, "covariance_type": covariance_type}
            row.update(score_config(X, model, labels))
            rows.append(row)
            print(
                f"{covariance_type:>9}  k={n_components}  "
                f"Silhouette={row['silhouette']:.3f}  "
                f"Davies-Bouldin={row['davies_bouldin']:.3f}  "
                f"BIC={row['bic']:.0f}"
            )

    results = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "gmm_tuning_results.csv", index=False)
    return results


def select_best_config(results):
    """Choose the final configuration using Silhouette and Davies-Bouldin only.

    1. Keep configurations with at least MIN_COMPONENTS clusters and no cluster
       smaller than MIN_CLUSTER_SHARE of the bookings.
    2. Rank them by Silhouette (higher is better) and by Davies-Bouldin
       (lower is better), and pick the lowest sum of the two ranks, so both
       metrics count equally.

    BIC is reported but not used for selection: with integer features such as
    Adults and Children, full and diagonal GMMs place very narrow components on
    single values, so BIC keeps falling as components are added.
    """
    eligible = results[
        (results["n_components"] >= MIN_COMPONENTS)
        & (results["min_cluster_share"] >= MIN_CLUSTER_SHARE)
    ].copy()

    eligible["silhouette_rank"] = eligible["silhouette"].rank(ascending=False)
    eligible["davies_bouldin_rank"] = eligible["davies_bouldin"].rank(ascending=True)
    eligible["combined_rank"] = eligible["silhouette_rank"] + eligible["davies_bouldin_rank"]

    best = eligible.sort_values(["combined_rank", "silhouette"], ascending=[True, False]).iloc[0]
    return {
        "n_components": int(best["n_components"]),
        "covariance_type": best["covariance_type"],
    }


def plot_bic(results):
    """BIC against the number of components, one line per covariance type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for covariance_type, group in results.groupby("covariance_type"):
        ax.plot(group["n_components"], group["bic"], marker="o", label=covariance_type)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("GMM: BIC by number of components")
    ax.legend(title="Covariance type")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "gmm_bic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results):
    """Silhouette and Davies-Bouldin against the number of components."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for covariance_type, group in results.groupby("covariance_type"):
        axes[0].plot(group["n_components"], group["silhouette"], marker="o", label=covariance_type)
        axes[1].plot(group["n_components"], group["davies_bouldin"], marker="o", label=covariance_type)
    axes[0].set_title("Silhouette (higher is better)")
    axes[1].set_title("Davies-Bouldin (lower is better)")
    for ax in axes:
        ax.set_xlabel("Number of components")
        ax.legend(title="Covariance type")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "gmm_silhouette_dbi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clusters_pca(X, labels, title, filename):
    """2D view of the clusters using the first two principal components."""
    X_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def profile_clusters(df, labels):
    """Mean of each clustering feature per cluster, plus size.

    IsCanceled is added only after clustering, to describe the clusters.
    It was never used to build them.
    """
    profiled = df.assign(Cluster=labels)
    profile = profiled.groupby("Cluster")[CLUSTERING_FEATURES].mean()
    profile["Bookings"] = profiled.groupby("Cluster").size()
    profile["Share_%"] = profile["Bookings"] / len(profiled) * 100
    profile["CancelRate_%"] = profiled.groupby("Cluster")["IsCanceled"].mean() * 100

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    profile.round(2).to_csv(TABLES_DIR / "gmm_cluster_profiles.csv")
    return profile.round(2)


def save_final_gmm(model, scaler):
    """Save the final model so the app can load it without retraining."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "gmm_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "gmm_scaler.joblib")
    joblib.dump(CLUSTERING_FEATURES, ARTIFACTS_DIR / "gmm_features.joblib")


def main():
    X, scaler, df = prepare_clustering_matrix()
    print(f"Clustering {X.shape[0]} bookings on {X.shape[1]} features\n")

    print("Baseline GMM:", BASELINE_PARAMS)
    baseline = run_baseline(X)
    print(
        f"  Silhouette={baseline['silhouette']:.3f}  "
        f"Davies-Bouldin={baseline['davies_bouldin']:.3f}\n"
    )

    print("Tuning GMM...")
    results = tune_gmm(X)
    plot_bic(results)
    plot_metrics(results)

    best_params = select_best_config(results)
    print(f"\nSelected configuration: {best_params}")

    final_model, final_labels = fit_gmm(X, **best_params)
    optimized = {**best_params, **score_config(X, final_model, final_labels)}

    comparison = build_comparison_table(
        [
            {"algorithm": "GMM", "configuration": "baseline", **baseline},
            {"algorithm": "GMM", "configuration": "optimized", **optimized},
        ],
        save_path=TABLES_DIR / "gmm_baseline_vs_optimized.csv",
    )
    print("\nBaseline vs optimized:")
    print(comparison[["configuration", "n_components", "covariance_type",
                      "silhouette", "davies_bouldin"]].round(3).to_string(index=False))

    plot_clusters_pca(
        X,
        final_labels,
        f"GMM clusters (k={best_params['n_components']}, "
        f"{best_params['covariance_type']}) in PCA space",
        "gmm_clusters_pca.png",
    )

    print("\nCluster profiles:")
    print(profile_clusters(df, final_labels).to_string())

    save_final_gmm(final_model, scaler)
    print("\nSaved model to artifacts/ and results to outputs/")


if __name__ == "__main__":
    main()