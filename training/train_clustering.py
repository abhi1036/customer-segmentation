import joblib
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from preprocessing import preprocess_customer_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

X = preprocess_customer_data(df)

print(f"Processed feature shape: {X.shape}")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(f"Scaled feature shape: {X_scaled.shape}")

pca = PCA(n_components=8)

X_pca = pca.fit_transform(X_scaled)

print(f"PCA feature shape: {X_pca.shape}")
print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_pca)

print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Cluster counts:")
print(pd.Series(cluster_labels).value_counts().sort_index())

# Save trained preprocessing and clustering artifacts

MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.pkl"
)

joblib.dump(
    pca,
    MODELS_DIR / "pca.pkl"
)

joblib.dump(
    kmeans,
    MODELS_DIR / "clustering_model.pkl"
)

print("Training artifacts saved successfully.")
