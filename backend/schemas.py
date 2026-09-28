from typing import Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customer_id: str
    age: Optional[float] = None
    location: str
    tenure: Optional[float] = None
    orders: float
    total_spend: float
    avg_order_value: Optional[float] = None
    purchase_frequency: float
    days_since_last_purchase: float
    discount_usage: float
    returns: float
    product_categories: float
    website_visits: Optional[float] = None
    app_sessions: float
    email_interaction: Optional[float] = None
    cart_additions: float
    cart_abandonment: float
    wishlist_activity: float
    support_interactions: float


class SegmentResponse(BaseModel):
    customer_id: str
    cluster: int
    segment_name: str
    distance_from_centroid: float


class BatchSegmentResponse(BaseModel):
    results: list[SegmentResponse]


class ClusterInfo(BaseModel):
    cluster: int
    segment_name: str
    profile: dict


class HealthResponse(BaseModel):
    status: str