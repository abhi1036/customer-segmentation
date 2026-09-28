from fastapi import FastAPI
import pandas as pd

from backend.predictor import CustomerSegmentPredictor
from backend.schemas import (
    CustomerInput,
    SegmentResponse,
    BatchSegmentResponse,
    HealthResponse
)


app = FastAPI(
    title="Customer Segmentation API",
    description="ML-powered customer segmentation using K-Means clustering",
    version="1.0.0"
)


predictor = CustomerSegmentPredictor()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/segment", response_model=SegmentResponse)
def segment_customer(customer: CustomerInput):

    customer_data = pd.DataFrame(
        [customer.model_dump()]
    )

    clusters, segment_names, distances = predictor.predict(
        customer_data
    )

    return {
        "customer_id": customer.customer_id,
        "cluster": int(clusters[0]),
        "segment_name": segment_names[0],
        "distance_from_centroid": float(distances[0])
    }

@app.post("/batch-segment", response_model=BatchSegmentResponse)
def batch_segment(customers: list[CustomerInput]):

    customer_data = pd.DataFrame(
        [customer.model_dump() for customer in customers]
    )

    clusters, segment_names, distances = predictor.predict(
        customer_data
    )

    results = []

    for i, customer in enumerate(customers):
        results.append({
            "customer_id": customer.customer_id,
            "cluster": int(clusters[i]),
            "segment_name": segment_names[i],
            "distance_from_centroid": float(distances[i])
        })

    return {
        "results": results
    }


@app.get("/cluster-info")
def cluster_info():

    info = predictor.get_cluster_info()

    return {
        str(cluster): {
            key: value
            for key, value in profile.items()
        }
        for cluster, profile in info.items()
    }