"""Shared clustering evaluation used by every clustering algorithm.

All algorithms (K-Means, Agglomerative, GMM) must be scored with this module
so the comparison is fair: same metrics, same Silhouette sample, same seed.
Only internal metrics are used; IsCanceled is never used to score clusters.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

SILHOUETTE_SAMPLE_SIZE = 10000
RANDOM_STATE = 42


def evaluate_clustering(X, labels):
    """Return Silhouette (higher is better) and Davies-Bouldin (lower is better).

    Silhouette is computed on a fixed random sample of 10,000 rows because the
    full computation is too memory-heavy for ~53k rows. Noise points labelled
    -1 (DBSCAN) are excluded before scoring.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    mask = labels != -1
    X_valid, labels_valid = X[mask], labels[mask]
    n_clusters = len(np.unique(labels_valid))

    result = {
        "n_clusters": n_clusters,
        "n_rows": len(labels_valid),
        "noise_points": int((~mask).sum()),
        "silhouette": None,
        "davies_bouldin": None,
    }

    if n_clusters < 2:
        return result

    result["silhouette"] = silhouette_score(
        X_valid,
        labels_valid,
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X_valid)),
        random_state=RANDOM_STATE,
    )
    result["davies_bouldin"] = davies_bouldin_score(X_valid, labels_valid)
    return result


def cluster_sizes(labels):
    """Number of bookings in each cluster, to spot tiny or dominant clusters."""
    return pd.Series(labels).value_counts().sort_index()


def build_comparison_table(rows, save_path=None):
    """Combine results from all algorithms into one comparison table.

    rows: list of dicts, each with "algorithm" and "configuration"
          (baseline / optimized) plus the output of evaluate_clustering().
    """
    table = pd.DataFrame(rows)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path, index=False)
    return table 