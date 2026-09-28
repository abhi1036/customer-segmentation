import joblib
import pandas as pd
from pathlib import Path


class CustomerSegmentPredictor:

    ENGAGEMENT_COLS = [
        "website_visits",
        "app_sessions",
        "email_interaction",
        "wishlist_activity"
    ]

    def __init__(self):
        # Project root = parent of backend/
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"

        self.kmeans = joblib.load(
            models_dir / "clustering_model.pkl"
        )

        self.scaler = joblib.load(
            models_dir / "scaler.pkl"
        )

        self.pca = joblib.load(
            models_dir / "pca.pkl"
        )

        self.feature_columns = joblib.load(
            models_dir / "feature_columns.pkl"
        )

        self.engagement_scaler = joblib.load(
            models_dir / "engagement_scaler.pkl"
        )

        self.imputation_values = joblib.load(
            models_dir / "imputation_values.pkl"
        )

        self.cluster_name_map = joblib.load(
            models_dir / "cluster_name_map.pkl"
        )

        self.cluster_info = joblib.load(
            models_dir / "cluster_info.pkl"
        )

    def preprocess(self, df):
        df = df.copy()

        # Apply training-time imputation values
        for col, value in self.imputation_values.items():
            df[col] = df[col].fillna(value)

        # Feature engineering
        df["recency"] = df["days_since_last_purchase"]

        df["frequency"] = (
            df["orders"] / df["tenure"]
        )

        df["monetary_value"] = df["total_spend"]

        # Engagement score
        engagement_scaled = self.engagement_scaler.transform(
            df[self.ENGAGEMENT_COLS]
        )

        df[self.ENGAGEMENT_COLS] = engagement_scaled

        df["engagement_score"] = (
            df[self.ENGAGEMENT_COLS].mean(axis=1)
        )

        df["discount_dependency"] = df["discount_usage"]

        # Exact feature order used during training
        X = df[self.feature_columns]

        return X

    def predict(self, df):
        X = self.preprocess(df)

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        clusters = self.kmeans.predict(X_pca)
        distances = self.kmeans.transform(X_pca)

        distance_from_centroid = [
        distances[i, cluster]
        for i, cluster in enumerate(clusters)
        ]

        segment_names = [
        self.cluster_name_map[int(cluster)]
        for cluster in clusters
        ]

        return clusters, segment_names, distance_from_centroid

    def get_cluster_info(self):
        return self.cluster_info