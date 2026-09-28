import joblib
import pandas as pd

from pathlib import Path
from preprocessing import preprocess_customer_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "customer_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 2. Load trained clustering artifacts
# --------------------------------------------------

kmeans = joblib.load(
    MODELS_DIR / "clustering_model.pkl"
)

scaler = joblib.load(
    MODELS_DIR / "scaler.pkl"
)

pca = joblib.load(
    MODELS_DIR / "pca.pkl"
)


# --------------------------------------------------
# 3. Preprocess data
# --------------------------------------------------

X = preprocess_customer_data(df)


# --------------------------------------------------
# 4. Reproduce cluster assignments
# --------------------------------------------------

X_scaled = scaler.transform(X)

X_pca = pca.transform(X_scaled)

df["cluster"] = kmeans.predict(X_pca)

print("Cluster assignments created.")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 5. Feature engineering for cluster profiling
# --------------------------------------------------

df["recency"] = df["days_since_last_purchase"]

df["frequency"] = (
    df["orders"] / df["tenure"]
)

df["monetary_value"] = df["total_spend"]

df["discount_dependency"] = df["discount_usage"]


# --------------------------------------------------
# 6. Calculate engagement score
# --------------------------------------------------

engagement_scaler = joblib.load(
    MODELS_DIR / "engagement_scaler.pkl"
)

engagement_cols = [
    "website_visits",
    "app_sessions",
    "email_interaction",
    "wishlist_activity"
]


# Load the same training-time imputation values
imputation_values = joblib.load(
    MODELS_DIR / "imputation_values.pkl"
)


# Apply the same missing-value treatment
for col, value in imputation_values.items():
    df[col] = df[col].fillna(value)


# Scale engagement features
engagement_scaled = engagement_scaler.transform(
    df[engagement_cols]
)


df[engagement_cols] = engagement_scaled


# Create engagement score
df["engagement_score"] = (
    df[engagement_cols].mean(axis=1)
)


print("Profiling features created.")

# --------------------------------------------------
# 7. Create cluster profile
# --------------------------------------------------

profile_columns = [
    "age",
    "tenure",
    "frequency",
    "monetary_value",
    "avg_order_value",
    "recency",
    "returns",
    "product_categories",
    "engagement_score",
    "discount_dependency",
    "cart_additions",
    "cart_abandonment",
    "support_interactions"
]

cluster_profile = (
    df.groupby("cluster")[profile_columns]
    .mean()
    .round(4)
)

print("\nCluster Profile:")
print(cluster_profile)

# --------------------------------------------------
# 8. Calculate overall population profile
# --------------------------------------------------

overall_profile = (
    df[profile_columns]
    .mean()
)

print("\nOverall Profile:")
print(overall_profile.round(4))

cluster_profile_relative = (
    cluster_profile / overall_profile
).round(4)

print("\nRelative Cluster Profile:")
print(cluster_profile_relative)

# --------------------------------------------------
# 9. Generate automatic segment names
# --------------------------------------------------

def generate_segment_name(row):
    HIGH = 1.25
    LOW = 0.75

    # High-value and highly active
    if (
        row["monetary_value"] > HIGH
        and row["frequency"] > HIGH
        and row["recency"] < LOW
        and row["engagement_score"] > HIGH
    ):
        return "High-Value Active Customers"

    # Heavy discount usage + high cart abandonment
    elif (
        row["discount_dependency"] > HIGH
        and row["cart_abandonment"] > HIGH
    ):
        return "Discount-Dependent High-Abandonment Customers"

    # Inactive + low engagement + low frequency
    elif (
        row["recency"] > HIGH
        and row["engagement_score"] < LOW
        and row["frequency"] < LOW
    ):
        return "Low-Engagement Inactive Customers"

    else:
        return "General Customers"


cluster_profile_relative["segment_name"] = (
    cluster_profile_relative.apply(
        generate_segment_name,
        axis=1
    )
)

print("\nSegment Names:")
print(
    cluster_profile_relative["segment_name"]
)

# --------------------------------------------------
# 10. Save profiling artifacts
# --------------------------------------------------

cluster_name_map = (
    cluster_profile_relative["segment_name"]
    .to_dict()
)

joblib.dump(
    cluster_name_map,
    MODELS_DIR / "cluster_name_map.pkl"
)


cluster_info = cluster_profile.copy()

cluster_info["segment_name"] = (
    cluster_profile_relative["segment_name"]
)

joblib.dump(
    cluster_info.to_dict(orient="index"),
    MODELS_DIR / "cluster_info.pkl"
)


print("\nProfiling artifacts saved successfully.")