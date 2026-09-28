import joblib
import pandas as pd
from pathlib import Path


ENGAGEMENT_COLS = [
    "website_visits",
    "app_sessions",
    "email_interaction",
    "wishlist_activity"
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def preprocess_customer_data(df):
    df = df.copy()

    # Load training-time preprocessing artifacts
    imputation_values = joblib.load(
        MODELS_DIR / "imputation_values.pkl"
    )

    engagement_scaler = joblib.load(
        MODELS_DIR / "engagement_scaler.pkl"
    )

    feature_columns = joblib.load(
        MODELS_DIR / "feature_columns.pkl"
    )

    # Apply training medians
    for col, value in imputation_values.items():
        df[col] = df[col].fillna(value)

    # Feature engineering
    df["recency"] = df["days_since_last_purchase"]

    df["frequency"] = (
        df["orders"] / df["tenure"]
    )

    df["monetary_value"] = df["total_spend"]

    # Engagement score
    engagement_scaled = engagement_scaler.transform(
        df[ENGAGEMENT_COLS]
    )

    df[ENGAGEMENT_COLS] = engagement_scaled

    df["engagement_score"] = (
        df[ENGAGEMENT_COLS].mean(axis=1)
    )

    df["discount_dependency"] = df["discount_usage"]

    # Select exact training features
    X = df[feature_columns]

    return X