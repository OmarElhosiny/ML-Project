
from src.models.clustering.features import (
    prepare_clustering_matrix,
    load_clustering_data,
    CLUSTERING_FEATURES
)
from src.evaluation.clustering_metrics import evaluate_clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import joblib


def prepare_kmeans_data(feature_set="experiment3"):
    if feature_set == "experiment3":
        X_scaled, scaler, df = prepare_clustering_matrix()
        features = CLUSTERING_FEATURES

    else:
        df = load_clustering_data()

        df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]

        if feature_set == "baseline":
            features = [
                "LeadTime",
                "ADR",
                "Adults",
                "Children",
                "Babies",
                "StaysInWeekendNights",
                "StaysInWeekNights"
            ]
        elif feature_set == "experiment2":
            features = [
                "LeadTime",
                "ADR",
                "TotalGuests",
                "TotalNights"
            ]
        else:
            raise ValueError("Unknown feature set")

        X = df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    print("\nFeature Set:", feature_set)
    print("\nScaled Data Shape:")
    print(X_scaled.shape)
    print("\nFirst 5 rows:")
    print(X_scaled[:5])

    print("Selected Features:")
    print(features)

    print("\nDataset Shape:")
    print(df[features].shape)

    print("\nMissing Values:")
    print(df[features].isnull().sum())

    return X_scaled


def run_Kmeans(feature_set="experiment3", k=3):
    X_scaled = prepare_kmeans_data(feature_set)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    print("\nKmeans Clustering:")
    print("\nNo. of clustering:", model.n_clusters)
    for i in range(model.n_clusters):
        print(f"Clusters {i}: {(labels == i).sum()} bookings")
    return model, labels


def elbow_method(feature_set="experiment3"):
    X_scaled = prepare_kmeans_data(feature_set)
    inertia_values = []
    k_values = range(1, 9)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertia_values.append(model.inertia_)
        print(f"K = {k}, Inertia = {model.inertia_:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(k_values))
    plt.tight_layout()

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"outputs/figures/kmeans_elbow_{feature_set}.png")
    plt.show()


def silhouette_method(feature_set="experiment3"):
    X_scaled = prepare_kmeans_data(feature_set)

    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        metrics = evaluate_clustering(X_scaled, labels)
        score = metrics["silhouette"]

        print(f"K = {k}, Silhouette Score = {score:.4f}")


def evaluate_kmeans(feature_set="experiment3"):
    X_scaled = prepare_kmeans_data(feature_set)
    results = []

    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        metrics = evaluate_clustering(X_scaled, labels)
        sil = metrics["silhouette"]
        dbi = metrics["davies_bouldin"]

        print(f"K = {k}, Silhouette = {sil:.4f}, Davies-Bouldin = {dbi:.4f}")

        results.append({
            "Feature_Set": feature_set,
            "K": k,
            "Silhouette": sil,
            "Davies_Bouldin": dbi
        })

    import pandas as pd
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"outputs/tables/kmeans_evaluation_{feature_set}.csv",
        index=False
    )

    print("\nEvaluation results saved successfully!")
    return results_df


def analyze_clusters(k=2, feature_set="experiment3"):
    if feature_set == "experiment3":
        X_scaled, scaler, df = prepare_clustering_matrix()
    else:
        df = load_clustering_data()
        X_scaled = prepare_kmeans_data(feature_set)

    df = df.copy()

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels
    df["TotalGuests"] = df["Adults"] + df["Children"] + df["Babies"]
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]

    print(f"\nCluster Analysis for K = {k}")
    print("\nNumber of Bookings in Each Cluster:")
    print(df["Cluster"].value_counts().sort_index())

    print("\nAverage Features by Cluster:")
    summary = df.groupby("Cluster")[
        ["LeadTime", "ADR", "TotalGuests", "TotalNights"]
    ].mean()

    print(summary.round(2))
    return df, summary


def check_kmeans_stability():
    X_scaled = prepare_kmeans_data(feature_set="experiment3")

    model = KMeans(n_clusters=6, random_state=42, n_init=10)
    original_labels = model.fit_predict(X_scaled)

    for seed in [10, 20, 30, 50]:
        model = KMeans(n_clusters=6, random_state=seed, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = adjusted_rand_score(original_labels, labels)
        print(f"Random State = {seed}, ARI = {score:.4f}")


def visualize_clusters(k=6, feature_set="experiment3"):
    df, summary = analyze_clusters(k=k, feature_set=feature_set)

    plt.figure(figsize=(9, 6))

    for i in range(k):
        cluster_data = df[df["Cluster"] == i]
        plt.scatter(
            cluster_data["LeadTime"],
            cluster_data["ADR"],
            label=f"Cluster {i}",
            alpha=0.3,
            s=8
        )

    plt.title(f"K-Means Clusters (K={k}) - {feature_set}")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Average Daily Rate (ADR)")
    plt.ylim(0, 350)
    plt.legend()
    plt.tight_layout()

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"outputs/figures/kmeans_clusters_{feature_set}_k{k}.png")
    plt.show()


def save_final_kmeans():
    X_scaled, scaler, df = prepare_clustering_matrix()
    features = CLUSTERING_FEATURES

    X = df[features].copy()

    model = KMeans(n_clusters=6, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "artifacts/kmeans_model.joblib")
    joblib.dump(scaler, "artifacts/kmeans_scaler.joblib")
    joblib.dump(features, "artifacts/kmeans_features.joblib")

    print("\nFinal K-Means Model Saved Successfully!")
    print("Feature Set: experiment3")
    print("Number of Clusters:", model.n_clusters)
    print("Dataset Shape:", X.shape)

    for i in range(model.n_clusters):
        print(f"Cluster {i}: {(labels == i).sum()} bookings")


def test_saved_kmeans():
    model = joblib.load("artifacts/kmeans_model.joblib")
    scaler = joblib.load("artifacts/kmeans_scaler.joblib")
    features = joblib.load("artifacts/kmeans_features.joblib")

    df = load_clustering_data()

    X = df[features].head(5).copy()
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    print("\nSaved K-Means Model Loaded Successfully!")
    print("\nFeatures:", features)
    print("\nNumber of Test Bookings:", len(X))

    print("\nPredicted Clusters:")
    for i in range(len(predictions)):
        print(f"Booking {i + 1}: Cluster {predictions[i]}")

if __name__ == "__main__":
    analyze_clusters(k=6, feature_set="experiment3")