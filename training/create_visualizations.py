import joblib
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


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
# 4. Load scaler
# --------------------------------------------------

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

X_scaled = scaler.transform(X)

print(f"Scaled feature matrix shape: {X_scaled.shape}")


# --------------------------------------------------
# 5. Create 2D PCA
# --------------------------------------------------

pca_2d = PCA(
    n_components=2
)

X_pca_2d = pca_2d.fit_transform(X_scaled)

print(
    f"2D PCA shape: {X_pca_2d.shape}"
)

print(
    f"Explained variance: "
    f"{pca_2d.explained_variance_ratio_.sum():.4f}"
)

# --------------------------------------------------
# 6. Load trained K-Means model
# --------------------------------------------------

kmeans = joblib.load(
    MODELS_DIR / "clustering_model.pkl"
)


# --------------------------------------------------
# 7. Generate cluster labels
# --------------------------------------------------

cluster_labels = kmeans.predict(
    joblib.load(MODELS_DIR / "pca.pkl").transform(X_scaled)
)

print("\nCluster counts:")
print(
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# 8. Create visualization DataFrame
# --------------------------------------------------

pca_visualization = pd.DataFrame({
    "customer_id": df["customer_id"],
    "PC1": X_pca_2d[:, 0],
    "PC2": X_pca_2d[:, 1],
    "cluster": cluster_labels
})


# --------------------------------------------------
# 9. Save visualization data
# --------------------------------------------------

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

visualization_path = (
    OUTPUTS_DIR / "pca_visualization.csv"
)

pca_visualization.to_csv(
    visualization_path,
    index=False
)


print(
    f"\nPCA visualization data saved to: "
    f"{visualization_path}"
)

# --------------------------------------------------
# 10. Create hierarchical clustering linkage
# --------------------------------------------------

sample_size = 500

X_sample = X_scaled[:sample_size]

hierarchical_linkage = linkage(
    X_sample,
    method="ward"
)

print(
    f"\nHierarchical linkage shape: "
    f"{hierarchical_linkage.shape}"
)


# --------------------------------------------------
# 11. Save hierarchical linkage
# --------------------------------------------------

linkage_path = (
    MODELS_DIR / "hierarchical_linkage.pkl"
)

joblib.dump(
    hierarchical_linkage,
    linkage_path
)

print(
    f"Hierarchical linkage saved to: "
    f"{linkage_path}"
)

# --------------------------------------------------
# 12. Create dendrogram
# --------------------------------------------------

plt.figure(figsize=(14, 7))

dendrogram(
    hierarchical_linkage,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster / Sample Group")
plt.ylabel("Distance")

plt.tight_layout()


# --------------------------------------------------
# 13. Save dendrogram
# --------------------------------------------------

dendrogram_path = (
    OUTPUTS_DIR / "hierarchical_dendrogram.png"
)

plt.savefig(
    dendrogram_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"Dendrogram saved to: {dendrogram_path}"
)