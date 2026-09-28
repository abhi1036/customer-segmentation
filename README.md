# Intelligent Customer Segmentation & Behavioral Analytics Platform

An end-to-end unsupervised learning platform that discovers customer segments from purchasing behavior, engagement, and interaction data.

The project uses multiple clustering algorithms, PCA-based visualization, automated cluster profiling, FastAPI for model serving, and Streamlit for interactive analytics.

---

## 1. Project Overview

This project automatically discovers meaningful customer groups without using predefined customer labels.

The platform analyzes:

- Purchasing behavior

- Customer engagement

- Discount usage

- Cart behavior

- Recency and frequency

- Monetary value

- Customer interactions

The discovered clusters are then profiled and assigned descriptive segment names based on their behavioral characteristics.

---

## 2. Problem Statement

Traditional customer segmentation often depends on manually defined categories.

This project uses unsupervised learning to allow customer groups to emerge directly from behavioral data.

The system identifies behavioral patterns such as:

- High-value active customers

- Discount-dependent customers with high cart abandonment

- Low-engagement inactive customers

The segment names are generated from the characteristics of the discovered clusters rather than being assigned before clustering.

---

## 3. Project Architecture

```text

Raw Customer Dataset

        |

        v

Data Cleaning

        |

        v

Exploratory Data Analysis

        |

        v

Feature Engineering

        |

        v

Feature Selection

        |

        v

StandardScaler

        |

        v

PCA

        |

        +--------------------+

        |                    |

        v                    v

   K-Means             Other Algorithms

        |              /          \

        |             /            \

        |        DBSCAN       Agglomerative

        |                          |

        +------------+-------------+

                     |

                     v

             Cluster Evaluation

                     |

                     v

             Cluster Profiling

                     |

                     v

          Automatic Segment Naming

                     |

                     v

              Model Artifacts

                     |

          +----------+----------+

          |                     |

          v                     v

       FastAPI              Streamlit

          |                     |

          +----------+----------+

                     |

                     v

              Customer Insights

```

## 4. Dataset

The dataset contains customer-level purchasing and engagement information.

### Customer Features

- `customer_id`

- `age`

- `location`

- `tenure`

- `orders`

- `total_spend`

- `avg_order_value`

- `purchase_frequency`

- `days_since_last_purchase`

- `discount_usage`

- `returns`

- `product_categories`

### Engagement Features

- `website_visits`

- `app_sessions`

- `email_interaction`

- `cart_additions`

- `cart_abandonment`

- `wishlist_activity`

- `support_interactions`

The dataset used in this project contains 10,000 customers.

Dataset note: The customer dataset is synthetic and was created for this project. It contains no predefined customer-segment labels; clustering is performed from the behavioral features described above.

## 5. Feature Engineering

Several behavioral features are derived from the raw customer data.

### Recency

Number of days since the customer's last purchase.

```text

recency = days_since_last_purchase

```

### Frequency

Purchase frequency is calculated using:

```text

frequency = orders / tenure

```

### Monetary Value

Total customer spending:

```text

monetary_value = total_spend

```

### Engagement Score

The engagement score is calculated using:

- Website visits

- App sessions

- Email interaction

- Wishlist activity

The four engagement variables are Min-Max scaled and then averaged.

### Discount Dependency

The dataset's discount usage field is used as the discount dependency measure.

## 6. Feature Selection

The final clustering feature set contains:

- `age`

- `tenure`

- `frequency`

- `monetary_value`

- `avg_order_value`

- `recency`

- `returns`

- `product_categories`

- `engagement_score`

- `discount_dependency`

- `cart_additions`

- `cart_abandonment`

- `support_interactions`

Highly redundant engineered/original features were removed.

For example:

`purchase_frequency`

was strongly correlated with:

`frequency`

so the engineered frequency feature was retained.

## 7. Dimensionality Reduction

StandardScaler is applied before PCA.

The clustering pipeline uses:

```text

13 selected features

        |

        v

StandardScaler

        |

        v

PCA

        |

        v

8 principal components

```

The 8 PCA components retain approximately 87.31% of the variance.

A separate 2-component PCA representation is used only for visualization.

## 8. Clustering Algorithms

Three clustering approaches were evaluated.

### K-Means

K-Means was used as the primary clustering model.

The number of clusters was investigated using:

- Elbow method

- Silhouette score

The selected configuration was:

Number of clusters: 3

### DBSCAN

DBSCAN was tested for discovering dense groups and identifying noise/outliers.

The tested configurations did not produce a meaningful multi-cluster solution for this dataset.

### Agglomerative Clustering

Hierarchical/Agglomerative Clustering was evaluated using:

```text

n_clusters = 3

linkage = ward

```

A dendrogram was also generated using a representative sample of the dataset.

## 9. Clustering Evaluation

### K-Means

```text

Silhouette Score:       0.2817

Davies-Bouldin Index:   1.3206

Calinski-Harabasz:      3292.45

```

### Agglomerative Clustering

```text

Silhouette Score:       0.2809

Davies-Bouldin Index:   1.3233

Calinski-Harabasz:      3266.14

```

DBSCAN did not produce a meaningful multi-cluster solution under the tested configurations, so comparable clustering metrics were not reported for it.

## 10. Discovered Customer Segments

The selected K-Means model produced three clusters.

### Cluster 0: High-Value Active Customers

Characteristics include:

- High purchase frequency

- High monetary value

- Low recency

- High engagement

- Lower discount dependency

### Cluster 1: Discount-Dependent High-Abandonment Customers

Characteristics include:

- Higher discount dependency

- Higher cart abandonment

- Moderate purchase frequency

- Lower monetary value compared with Cluster 0

### Cluster 2: Low-Engagement Inactive Customers

Characteristics include:

- High recency

- Low engagement

- Low purchase frequency

- Lower monetary value

- Lower cart activity

## 11. Model Artifacts

The trained model and preprocessing components are saved in the `models/` directory.

```text

models/

├── clustering_model.pkl

├── scaler.pkl

├── pca.pkl

├── feature_columns.pkl

├── engagement_scaler.pkl

├── imputation_values.pkl

├── cluster_name_map.pkl

├── cluster_info.pkl

└── hierarchical_linkage.pkl

```

The production application loads these artifacts instead of retraining the model. This keeps inference consistent with the preprocessing and clustering pipeline used during training.

## 12. FastAPI Backend

The FastAPI backend provides REST endpoints for model inference and cluster information.

### Health Check

`GET /health`

### Single Customer Segmentation

`POST /segment`

Returns:

```json

{

  "customer_id": "CUST_00001",

  "cluster": 0,

  "segment_name": "High-Value Active Customers",

  "distance_from_centroid": 3.0887

}

```

### Batch Segmentation

`POST /batch-segment`

Accepts multiple customers and returns segmentation results for each customer.

### Cluster Information

`GET /cluster-info`

Returns the discovered cluster profiles and generated segment names.

## 13. Validation & Testing

The platform was validated across the machine-learning pipeline, model artifacts, FastAPI backend, and Streamlit frontend.

API Endpoint Validation

Endpoint

Purpose

Status

GET /health

API health check

Passed

POST /segment

Single customer segmentation

Passed

POST /batch-segment

Batch customer segmentation

Passed

GET /cluster-info

Cluster profiles and segment names

Passed

Inference Validation

A known customer record was passed through the production predictor using the saved preprocessing and clustering artifacts. The prediction successfully returned:

Cluster ID

Automatically generated segment name

Distance from the assigned K-Means centroid

Additional test payloads, including inputs containing nullable fields, were also processed successfully.

Frontend Integration Validation

The Streamlit Customer Segmentation page was tested against the running FastAPI backend. The prediction displayed by Streamlit matched the corresponding API response, confirming the end-to-end integration:

Streamlit Input
      |
      v
FastAPI /segment
      |
      v
Saved preprocessing artifacts
      |
      v
K-Means prediction
      |
      v
Segment + centroid distance
      |
      v
Streamlit Result

## 14. Streamlit Dashboard

The Streamlit application provides an interactive analytics interface.

### Executive Dashboard

Displays:

- Total customers

- Number of customer segments

- Largest segment

- Total spend

- Customer distribution

- Segment business metrics

### Customer Segmentation

Allows users to enter customer behavioral information and receive a live prediction from the FastAPI model.

### Cluster Visualization

Provides:

- 2D PCA visualization

- Interactive customer exploration

- Hierarchical clustering dendrogram

### Segment Comparison

Allows comparison of segments across:

- Monetary value

- Frequency

- Average order value

- Recency

- Engagement

- Discount dependency

- Cart abandonment

- Returns

### Clustering Evaluation

Displays clustering evaluation metrics for the tested algorithms.

### Business Insights

Generates data-driven observations for each discovered segment.

## 15. Project Structure

The repository is organized as follows:

```text

customer-segmentation-platform/

├── backend/

│   ├── __init__.py

│   ├── app.py

│   ├── predictor.py

│   ├── preprocessing.py

│   └── schemas.py

│

├── data/

│   └── customer_data.csv

│

├── frontend/

│   └── streamlit_app.py

│

├── models/

│   ├── clustering_model.pkl

│   ├── scaler.pkl

│   ├── pca.pkl

│   ├── feature_columns.pkl

│   ├── engagement_scaler.pkl

│   ├── imputation_values.pkl

│   ├── cluster_name_map.pkl

│   ├── cluster_info.pkl

│   └── hierarchical_linkage.pkl

│

├── notebooks/

│   └── v1_EDA.ipynb

│

├── outputs/

│   ├── clustering_evaluation.csv

│   ├── hierarchical_dendrogram.png

│   ├── pca_visualization.csv

│   └── segmented_customers.csv

│

├── training/

│   ├── cluster_profiling.py

│   ├── create_visualizations.py

│   ├── evaluate_clusters.py

│   ├── preprocessing.py

│   └── train_clustering.py

│

├── utils/

│   ├── __init__.py

│   ├── clustering.py

│   ├── recommendations.py

│   └── visualization.py

│

├── .gitignore

├── requirements.txt

└── README.md

## 15. Installation

## 16. Installation

Create and activate a virtual environment:

```powershell

python -m venv .venv

```

Windows:

```powershell

.venv\Scripts\Activate.ps1

```

Install dependencies:

```powershell

python -m pip install -r requirements.txt

```

## 17. Running the FastAPI Backend

From the project root:

```powershell

uvicorn backend.app:app --reload

```

The API will run at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Deployed Swagger documentation:

https://customer-segmentation-k0gg.onrender.com/docs

## 18. Running the Streamlit Dashboard

Open another terminal with the virtual environment activated:

```powershell

streamlit run frontend/streamlit_app.py

```

Streamlit will provide a local URL, normally:

`http://localhost:8501`

Live dashboard:

https://customer-segmentation-abhi1036.streamlit.app/

Both FastAPI and Streamlit should be running for live customer segmentation.

## 19. Technologies

### Machine Learning and Data Processing

- Python

- Pandas

- NumPy

- Scikit-learn

- SciPy

### Clustering and Dimensionality Reduction

- K-Means

- DBSCAN

- Agglomerative Clustering

- PCA

### Backend

- FastAPI

- Uvicorn

- Joblib

### Frontend and Visualization

- Streamlit

- Plotly

- Matplotlib

- Seaborn

## 20. Project Goal

The final system provides an end-to-end workflow:

```text

Customer Data

        |

        v

Behavioral Feature Engineering

        |

        v

Unsupervised Learning

        |

        v

Customer Clusters

        |

        v

Cluster Profiling

        |

        v

Automatic Segment Names

        |

        v

FastAPI Model Serving

        |

        v

Interactive Streamlit Analytics

```

The platform can therefore be used to explore customer behavior, understand discovered customer groups, compare clustering approaches, and assign new customers to the learned segments through the deployed inference pipeline.