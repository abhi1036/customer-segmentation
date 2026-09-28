import joblib
import pandas as pd
from pathlib import Path

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from preprocessing import preprocess_customer_data


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Prepare features
# --------------------------------------------------

X = preprocess_customer_data(df)

print(f"Feature matrix shape: {X.shape}")


# --------------------------------------------------
# 4. Load trained preprocessing artifacts
# --------------------------------------------------

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

pca = joblib.load(
    MODELS_DIR / "pca.pkl"
)


# --------------------------------------------------
# 5. Apply scaling and PCA
# --------------------------------------------------

X_scaled = scaler.transform(X)

X_pca = pca.transform(X_scaled)

print(f"PCA feature matrix shape: {X_pca.shape}")

# --------------------------------------------------
# 6. Load trained K-Means model
# --------------------------------------------------

kmeans = joblib.load(
    MODELS_DIR / "clustering_model.pkl"
)


# --------------------------------------------------
# 7. Get K-Means cluster labels
# --------------------------------------------------

kmeans_labels = kmeans.predict(X_pca)

print("\nK-Means cluster counts:")
print(
    pd.Series(kmeans_labels)
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 8. Evaluate K-Means
# --------------------------------------------------

kmeans_silhouette = silhouette_score(
    X_pca,
    kmeans_labels
)

kmeans_db = davies_bouldin_score(
    X_pca,
    kmeans_labels
)

kmeans_ch = calinski_harabasz_score(
    X_pca,
    kmeans_labels
)


# --------------------------------------------------
# 9. Print K-Means metrics
# --------------------------------------------------

print("\nK-Means Evaluation:")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Davies-Bouldin Index: {kmeans_db:.4f}")
print(f"Calinski-Harabasz Score: {kmeans_ch:.4f}")

# --------------------------------------------------
# 10. DBSCAN clustering
# --------------------------------------------------

dbscan = DBSCAN(
    eps=3.0,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_pca)


# --------------------------------------------------
# 11. DBSCAN cluster information
# --------------------------------------------------

unique_labels = set(dbscan_labels)

n_clusters = len(
    unique_labels - {-1}
)

n_noise = list(dbscan_labels).count(-1)

print("\nDBSCAN Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# --------------------------------------------------
# 12. Agglomerative Clustering
# --------------------------------------------------

agg = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

agg_labels = agg.fit_predict(X_pca)


# --------------------------------------------------
# 13. Agglomerative cluster information
# --------------------------------------------------

print("\nAgglomerative cluster counts:")
print(
    pd.Series(agg_labels)
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# 14. Evaluate Agglomerative Clustering
# --------------------------------------------------

agg_silhouette = silhouette_score(
    X_pca,
    agg_labels
)

agg_db = davies_bouldin_score(
    X_pca,
    agg_labels
)

agg_ch = calinski_harabasz_score(
    X_pca,
    agg_labels
)


# --------------------------------------------------
# 15. Print Agglomerative metrics
# --------------------------------------------------

print("\nAgglomerative Evaluation:")
print(f"Silhouette Score: {agg_silhouette:.4f}")
print(f"Davies-Bouldin Index: {agg_db:.4f}")
print(f"Calinski-Harabasz Score: {agg_ch:.4f}")

# --------------------------------------------------
# 16. Create evaluation results
# --------------------------------------------------

evaluation_results = pd.DataFrame({
    "algorithm": [
        "K-Means",
        "Agglomerative",
        "DBSCAN"
    ],
    "silhouette_score": [
        kmeans_silhouette,
        agg_silhouette,
        None
    ],
    "davies_bouldin_index": [
        kmeans_db,
        agg_db,
        None
    ],
    "calinski_harabasz_score": [
        kmeans_ch,
        agg_ch,
        None
    ],
    "number_of_clusters": [
        len(set(kmeans_labels)),
        len(set(agg_labels)),
        n_clusters
    ],
    "noise_points": [
        0,
        0,
        n_noise
    ]
})


# --------------------------------------------------
# 17. Save evaluation results
# --------------------------------------------------

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

evaluation_path = (
    OUTPUTS_DIR / "clustering_evaluation.csv"
)

evaluation_results.to_csv(
    evaluation_path,
    index=False
)


# --------------------------------------------------
# 18. Display final results
# --------------------------------------------------

print("\nFinal Clustering Evaluation:")
print(evaluation_results)

print(
    f"\nEvaluation results saved to: {evaluation_path}"
)