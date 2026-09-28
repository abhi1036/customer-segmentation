import os
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import requests
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

SEGMENTED_DATA_PATH = OUTPUTS_DIR / "segmented_customers.csv"
PCA_DATA_PATH = OUTPUTS_DIR / "pca_visualization.csv"
EVALUATION_DATA_PATH = OUTPUTS_DIR / "clustering_evaluation.csv"
LINKAGE_PATH = MODELS_DIR / "hierarchical_linkage.pkl"


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◐",
    layout="wide",
)


# ==================================================
# 3. DESIGN SYSTEM
#    Frosted panels over a soft, pale mesh. Ink-blue text,
#    one teal accent, a serif for headings and big numbers.
# ==================================================

INK = "#14213D"
INK_SOFT = "#4A5673"

# muted, earthy segment colours (used consistently in every chart)
PALETTE = [
    "#0F766E",  # deep teal
    "#D99A00",  # marigold
    "#3B5BA5",  # indigo blue
    "#B4415B",  # madder rose
    "#7A8F2A",  # moss
    "#6B7A90",  # slate
    "#8C5E9E",  # heather
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #14213D;
    --ink-soft: #4A5673;
    --line: rgba(20, 33, 61, 0.12);
    --glass: rgba(255, 255, 255, 0.52);
    --glass-edge: rgba(255, 255, 255, 0.85);
    --teal: #0F766E;
    --good: #0F766E;
    --bad: #B4415B;
    --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
    --sans: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

/* ---------- base ---------- */
html, body, .stApp, [class*="css"] {
    font-family: var(--sans) !important;
    color: var(--ink);
}
.stApp {
    background:
        radial-gradient(1100px 700px at 0% 0%, #CDE6DF 0%, transparent 60%),
        radial-gradient(900px 700px at 100% 8%, #F4E4C8 0%, transparent 58%),
        radial-gradient(1000px 800px at 65% 100%, #C9DBEE 0%, transparent 60%),
        #EEF2F1;
    background-attachment: fixed;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2.4rem 3rem 4rem 3rem; max-width: 1240px; }

h1, h2, h3, h4, p, li, label, span, div { color: var(--ink); }
hr { border-color: var(--line) !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.42);
    backdrop-filter: blur(26px) saturate(160%);
    -webkit-backdrop-filter: blur(26px) saturate(160%);
    border-right: 1px solid var(--glass-edge);
}
.brand { padding: 4px 4px 22px 4px; }
.brand-name { font-family: var(--serif); font-weight: 600; font-size: 1.35rem; letter-spacing: -0.01em; }
.brand-sub { font-size: .82rem; color: var(--ink-soft); margin-top: 2px; }

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 9px 14px; border-radius: 10px; width: 100%;
    border-left: 3px solid transparent; cursor: pointer;
    transition: background .15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,.55); }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-size: .95rem; font-weight: 500; color: var(--ink-soft);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255,255,255,.75);
    border-left-color: var(--teal);
    box-shadow: 0 1px 0 var(--glass-edge) inset, 0 4px 14px rgba(20,33,61,.06);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--ink); font-weight: 600;
}
.sidebar-foot {
    margin-top: 28px; padding: 14px 4px 0 4px; border-top: 1px solid var(--line);
    font-size: .82rem; color: var(--ink-soft); line-height: 1.55;
}
.sidebar-foot b { color: var(--ink); font-weight: 600; }

/* ---------- page header (plain, no banner) ---------- */
.page-head { margin: 0 0 26px 0; padding-bottom: 20px; border-bottom: 1px solid var(--line); }
.page-head h1 {
    font-family: var(--serif); font-weight: 600; font-size: 2.5rem;
    letter-spacing: -0.02em; line-height: 1.1; margin: 0 0 8px 0; padding: 0;
}
.page-head p { color: var(--ink-soft); font-size: 1.02rem; max-width: 620px; margin: 0; }

.section-title {
    font-family: var(--serif); font-weight: 600; font-size: 1.4rem;
    letter-spacing: -0.01em; margin: 34px 0 2px 0;
}
.section-sub { color: var(--ink-soft); font-size: .93rem; margin-bottom: 14px; }

/* ---------- glass surfaces ---------- */
.glass, [class*="st-key-panel"], [data-testid="stPlotlyChart"], [data-testid="stPyplot"], [data-testid="stDataFrame"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-edge) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 12px 34px rgba(20, 33, 61, 0.08);
}
.glass { padding: 22px 26px; }
[class*="st-key-panel"] { padding: 6px 10px 10px 10px; margin-bottom: 14px; }
[data-testid="stPlotlyChart"], [data-testid="stPyplot"] { padding: 10px; }
[data-testid="stDataFrame"] { overflow: hidden; }

.panel-title { font-family: var(--serif); font-weight: 600; font-size: 1.15rem; margin: 6px 0 0 0; }
.panel-sub { color: var(--ink-soft); font-size: .88rem; margin: 2px 0 10px 0; }

/* ---------- stat strip: one panel, hairline dividers ---------- */
.stat-strip {
    display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr;
    background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 20px;
    backdrop-filter: blur(20px) saturate(150%); -webkit-backdrop-filter: blur(20px) saturate(150%);
    box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 12px 34px rgba(20,33,61,.08);
}
.stat { padding: 22px 26px; }
.stat + .stat { border-left: 1px solid var(--line); }
.stat-label { font-size: .88rem; color: var(--ink-soft); font-weight: 500; }
.stat-value {
    font-family: var(--serif); font-weight: 600; font-size: 2rem;
    letter-spacing: -0.02em; line-height: 1.15; margin-top: 6px;
    font-variant-numeric: tabular-nums;
}
.stat.lead .stat-value { font-size: 3rem; }
.stat-note { font-size: .84rem; color: var(--ink-soft); margin-top: 6px; }

/* ---------- prediction result ---------- */
.result {
    padding: 24px 28px; border-radius: 18px; margin-top: 8px;
    background: rgba(255,255,255,.62); border: 1px solid var(--glass-edge);
    border-left: 5px solid var(--seg, #0F766E);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 12px 34px rgba(20,33,61,.08);
}
.result-label { font-size: .88rem; color: var(--ink-soft); }
.result-name {
    font-family: var(--serif); font-weight: 600; font-size: 2.3rem;
    letter-spacing: -0.02em; margin: 2px 0 12px 0;
}
.result-facts { display: flex; gap: 34px; flex-wrap: wrap; }
.fact-label { font-size: .8rem; color: var(--ink-soft); }
.fact-value { font-weight: 600; font-size: 1.05rem; font-variant-numeric: tabular-nums; }

/* ---------- insights ---------- */
.seg-panel {
    display: grid; grid-template-columns: 1fr 1.9fr; gap: 34px;
    padding: 26px 30px; margin-bottom: 16px; border-radius: 20px;
    background: var(--glass); border: 1px solid var(--glass-edge);
    backdrop-filter: blur(20px) saturate(150%); -webkit-backdrop-filter: blur(20px) saturate(150%);
    box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 12px 34px rgba(20,33,61,.08);
}
.seg-bar { width: 44px; height: 4px; border-radius: 2px; margin-bottom: 14px; }
.seg-name { font-family: var(--serif); font-weight: 600; font-size: 1.6rem; letter-spacing: -0.02em; line-height: 1.15; }
.seg-size { color: var(--ink-soft); font-size: .9rem; margin: 6px 0 14px 0; }
.seg-summary { font-size: .97rem; line-height: 1.6; }
.seg-summary + .seg-summary { margin-top: 6px; }

table.cmp { width: 100%; border-collapse: collapse; font-size: .93rem; }
table.cmp th {
    text-align: right; font-weight: 600; font-size: .8rem; color: var(--ink-soft);
    padding: 0 0 10px 0; border-bottom: 1px solid var(--line);
}
table.cmp th:first-child, table.cmp td:first-child { text-align: left; }
table.cmp td {
    text-align: right; padding: 11px 0; border-bottom: 1px solid rgba(20,33,61,.07);
    font-variant-numeric: tabular-nums;
}
table.cmp tr:last-child td { border-bottom: none; }
table.cmp td:first-child { color: var(--ink); font-weight: 500; }
table.cmp td.muted { color: var(--ink-soft); }
.d-good { color: var(--good); font-weight: 600; }
.d-bad { color: var(--bad); font-weight: 600; }
.d-flat { color: var(--ink-soft); }

/* ---------- native widgets ---------- */
.stButton > button {
    border-radius: 10px; padding: .65rem 1.5rem; font-weight: 600; color: #fff;
    background: var(--teal); border: 1px solid #0B5F58;
    box-shadow: 0 1px 0 rgba(255,255,255,.25) inset;
    transition: background .15s ease;
}
.stButton > button:hover { background: #0B5F58; color: #fff; border-color: #0B5F58; }
.stButton > button:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }

.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,.7) !important; color: var(--ink) !important;
    border: 1px solid var(--line) !important; border-radius: 10px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--teal) !important; box-shadow: 0 0 0 3px rgba(15,118,110,.18) !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,.7) !important; color: var(--ink) !important;
    border-color: var(--line) !important;
}
.stTextInput label p, .stNumberInput label p, .stSelectbox label p {
    color: var(--ink-soft) !important; font-size: .86rem; font-weight: 500;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,.7) !important; color: var(--ink) !important;
    border: 1px solid var(--line) !important; border-radius: 10px !important;
}
div[data-baseweb="select"] * { color: var(--ink) !important; }
div[data-baseweb="popover"] ul { background: #fff !important; }

[data-testid="stAlert"] {
    background: var(--glass) !important; border: 1px solid var(--glass-edge) !important;
    border-left: 4px solid var(--teal) !important; border-radius: 12px !important;
    backdrop-filter: blur(14px);
}
[data-testid="stAlert"] * { color: var(--ink) !important; }

@media (max-width: 900px) {
    .block-container { padding: 1.4rem 1rem 3rem 1rem; }
    .stat-strip { grid-template-columns: 1fr 1fr; }
    .stat.lead { grid-column: 1 / -1; }
    .stat + .stat { border-left: none; }
    .seg-panel { grid-template-columns: 1fr; gap: 18px; }
    .page-head h1 { font-size: 1.9rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------- UI helpers ----------

def page_head(title, subtitle):
    st.markdown(
        f"""
        <div class="page-head">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-title">{title}</div>{sub}',
        unsafe_allow_html=True,
    )


def panel_title(title, subtitle=""):
    sub = f'<div class="panel-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="panel-title">{title}</div>{sub}',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans, system-ui, sans-serif", color=INK, size=13),
        colorway=PALETTE,
        height=height,
        margin=dict(l=24, r=24, t=64, b=24),
        title=dict(
            font=dict(family="Fraunces, Georgia, serif", size=19, color=INK),
            x=0.02,
        ),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color=INK)),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="rgba(20,33,61,0.2)",
            font=dict(color=INK, family="Instrument Sans, sans-serif"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(20,33,61,0.07)",
        linecolor="rgba(20,33,61,0.2)",
        zeroline=False,
        tickfont=dict(color=INK_SOFT),
        title_font=dict(color=INK_SOFT),
    )
    fig.update_yaxes(
        gridcolor="rgba(20,33,61,0.07)",
        linecolor="rgba(20,33,61,0.2)",
        zeroline=False,
        tickfont=dict(color=INK_SOFT),
        title_font=dict(color=INK_SOFT),
    )
    return fig


def flat(html):
    """Collapse HTML onto one line so Markdown can't mistake indented
    lines for a code block."""
    return "".join(line.strip() for line in html.splitlines())


def join_labels(items):
    items = [i.lower() for i in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ==================================================
# 4. LOAD DATA
# ==================================================

df = pd.read_csv(SEGMENTED_DATA_PATH)
pca_df = pd.read_csv(PCA_DATA_PATH)
evaluation_df = pd.read_csv(EVALUATION_DATA_PATH)


# ==================================================
# 5. ADD SEGMENT NAMES TO PCA DATA
# ==================================================

cluster_names = df[["cluster", "segment_name"]].drop_duplicates()
pca_df = pca_df.merge(cluster_names, on="cluster", how="left")

segment_order = cluster_names.sort_values("cluster")["segment_name"].tolist()
SEGMENT_COLORS = {
    name: PALETTE[i % len(PALETTE)] for i, name in enumerate(segment_order)
}


# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-name">Customer Segmentation</div>
        <div class="brand-sub">Behavioural clustering for retail</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Dashboard",
        "Customer Segmentation",
        "Cluster Visualization",
        "Segment Comparison",
        "Clustering Evaluation",
        "Business Insights",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
    <div class="sidebar-foot">
        <b>{len(df):,}</b> customers in <b>{df['cluster'].nunique()}</b> segments.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# ==================================================

if page == "Executive Dashboard":

    page_head(
        "Executive dashboard",
        "How your customers split into segments, and what each segment is worth.",
    )

    total_customers = len(df)
    total_segments = df["cluster"].nunique()
    cluster_sizes = df["cluster"].value_counts()
    largest_cluster = cluster_sizes.max()
    largest_cluster_percentage = (largest_cluster / total_customers) * 100
    largest_name = df.loc[
        df["cluster"] == cluster_sizes.idxmax(), "segment_name"
    ].iloc[0]
    total_spend = df["monetary_value"].sum()
    spend_per_customer = total_spend / total_customers

    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat lead">
                <div class="stat-label">Total customers</div>
                <div class="stat-value">{total_customers:,}</div>
                <div class="stat-note">Grouped into {total_segments} segments</div>
            </div>
            <div class="stat">
                <div class="stat-label">Largest segment</div>
                <div class="stat-value">{largest_cluster_percentage:.1f}%</div>
                <div class="stat-note">{largest_name}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Total spend</div>
                <div class="stat-value">₹{total_spend:,.0f}</div>
                <div class="stat-note">Across all customers</div>
            </div>
            <div class="stat">
                <div class="stat-label">Spend per customer</div>
                <div class="stat-value">₹{spend_per_customer:,.0f}</div>
                <div class="stat-note">Overall average</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "Customers by segment",
        "Share of the customer base in each segment.",
    )

    segment_distribution = (
        df.groupby(["cluster", "segment_name"])
        .size()
        .reset_index(name="customers")
        .sort_values("cluster")
    )

    col1, col2 = st.columns([1.15, 1])

    with col1:
        fig = px.pie(
            segment_distribution,
            names="segment_name",
            values="customers",
            hole=0.62,
            title="Segment distribution",
            color="segment_name",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            insidetextfont=dict(color="#fff", size=13),
            marker=dict(line=dict(color="rgba(255,255,255,0.9)", width=2)),
            sort=False,
        )
        fig = style_fig(fig, 410)
        fig.add_annotation(
            text=f"<b>{total_customers:,}</b><br>customers",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Fraunces, Georgia, serif", size=20, color=INK),
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.dataframe(
            segment_distribution,
            width="stretch",
            hide_index=True,
            height=410,
        )

    section(
        "Segment metrics",
        "Average behaviour and value for each segment.",
    )

    segment_metrics = (
        df.groupby(["cluster", "segment_name"])
        .agg(
            customers=("customer_id", "count"),
            avg_frequency=("frequency", "mean"),
            avg_monetary_value=("monetary_value", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_recency=("recency", "mean"),
            avg_engagement=("engagement_score", "mean"),
            avg_discount_dependency=("discount_dependency", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    display_metrics = segment_metrics.rename(
        columns={
            "cluster": "Cluster",
            "segment_name": "Segment",
            "customers": "Customers",
            "avg_frequency": "Avg Frequency",
            "avg_monetary_value": "Avg Monetary Value",
            "avg_order_value": "Avg Order Value",
            "avg_recency": "Avg Recency",
            "avg_engagement": "Avg Engagement",
            "avg_discount_dependency": "Discount Dependency",
        }
    )

    display_metrics["Avg Frequency"] = display_metrics["Avg Frequency"].round(2)
    display_metrics["Avg Monetary Value"] = display_metrics["Avg Monetary Value"].round(2)
    display_metrics["Avg Order Value"] = display_metrics["Avg Order Value"].round(2)
    display_metrics["Avg Recency"] = display_metrics["Avg Recency"].round(2)
    display_metrics["Avg Engagement"] = display_metrics["Avg Engagement"].round(3)
    display_metrics["Discount Dependency"] = display_metrics["Discount Dependency"].round(3)

    st.dataframe(display_metrics, width="stretch", hide_index=True)


# ==================================================
# PAGE 2 — CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    page_head(
        "Find a customer's segment",
        "Fill in a customer's profile and the model will tell you which segment they belong to.",
    )

    # ----------------------------------------------
    # Customer Information
    # ----------------------------------------------

    with st.container(border=True, key="panel_customer"):

        panel_title("Customer", "Who they are.")

        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID", value="CUST_NEW_001")
        with col2:
            age = st.number_input("Age", min_value=1.0, max_value=100.0, value=35.0)
        with col3:
            location = st.text_input("Location", value="Mumbai")

    # ----------------------------------------------
    # Purchase Behavior
    # ----------------------------------------------

    with st.container(border=True, key="panel_purchase"):

        panel_title("Purchases", "How and how often they buy.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tenure = st.number_input("Tenure (years)", min_value=0.1, value=3.0)
        with col2:
            orders = st.number_input("Orders", min_value=0.0, value=20.0)
        with col3:
            total_spend = st.number_input("Total Spend", min_value=0.0, value=25000.0)
        with col4:
            avg_order_value = st.number_input("Average Order Value", min_value=0.0, value=1250.0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            days_since_last_purchase = st.number_input(
                "Days Since Last Purchase", min_value=0.0, value=30.0
            )
        with col2:
            discount_usage = st.number_input(
                "Discount Usage", min_value=0.0, max_value=1.0, value=0.4
            )
        with col3:
            returns = st.number_input("Returns", min_value=0.0, value=1.0)
        with col4:
            product_categories = st.number_input("Product Categories", min_value=0.0, value=3.0)

    # ----------------------------------------------
    # Engagement Behavior
    # ----------------------------------------------

    with st.container(border=True, key="panel_engagement"):

        panel_title("Engagement", "How they interact with your site, app and emails.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            website_visits = st.number_input("Website Visits", min_value=0.0, value=40.0)
        with col2:
            app_sessions = st.number_input("App Sessions", min_value=0.0, value=18.0)
        with col3:
            email_interaction = st.number_input("Email Interaction", min_value=0.0, value=11.0)
        with col4:
            wishlist_activity = st.number_input("Wishlist Activity", min_value=0.0, value=7.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            cart_additions = st.number_input("Cart Additions", min_value=0.0, value=10.0)
        with col2:
            cart_abandonment = st.number_input("Cart Abandonment", min_value=0.0, value=5.0)
        with col3:
            support_interactions = st.number_input("Support Interactions", min_value=0.0, value=2.0)

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    if st.button("Predict Customer Segment", type="primary"):

        frequency = orders / tenure

        customer_data = {
            "customer_id": customer_id,
            "age": age,
            "location": location,
            "tenure": tenure,
            "orders": orders,
            "total_spend": total_spend,
            "avg_order_value": avg_order_value,
            "purchase_frequency": frequency,
            "days_since_last_purchase": days_since_last_purchase,
            "discount_usage": discount_usage,
            "returns": returns,
            "product_categories": product_categories,
            "website_visits": website_visits,
            "app_sessions": app_sessions,
            "email_interaction": email_interaction,
            "cart_additions": cart_additions,
            "cart_abandonment": cart_abandonment,
            "wishlist_activity": wishlist_activity,
            "support_interactions": support_interactions,
        }

        try:

            response = requests.post(
                f"{API_URL}/segment",
                json=customer_data,
                timeout=10,
            )

            if response.status_code == 200:

                result = response.json()

                seg_color = SEGMENT_COLORS.get(result["segment_name"], "#0F766E")

                members = (df["cluster"] == result["cluster"]).sum()
                share = members / len(df) * 100

                st.markdown(
                    f"""
                    <div class="result" style="--seg:{seg_color};">
                        <div class="result-label">{customer_id} belongs to</div>
                        <div class="result-name">{result['segment_name']}</div>
                        <div class="result-facts">
                            <div>
                                <div class="fact-label">Cluster</div>
                                <div class="fact-value">{result['cluster']}</div>
                            </div>
                            <div>
                                <div class="fact-label">Distance from centroid</div>
                                <div class="fact-value">{result['distance_from_centroid']:.3f}</div>
                            </div>
                            <div>
                                <div class="fact-label">Segment size</div>
                                <div class="fact-value">{members:,} customers ({share:.1f}%)</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.error(f"API Error: {response.status_code}")
                st.json(response.json())

        except requests.exceptions.RequestException:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure Uvicorn is running on port 8000."
            )


# ==================================================
# PAGE 3 — CLUSTER VISUALIZATION
# ==================================================

elif page == "Cluster Visualization":

    page_head(
        "Cluster visualization",
        "How the segments separate in two dimensions, and how customers group together step by step.",
    )

    section(
        "Segments in PCA space",
        "Each dot is a customer. Hover to see their ID and cluster.",
    )

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["customer_id", "cluster"],
        title="Customer segments in 2D PCA space",
        opacity=0.8,
    )
    fig_pca.update_traces(
        marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.9)"))
    )
    fig_pca.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Segment",
    )

    st.plotly_chart(style_fig(fig_pca, 560), width="stretch")

    section(
        "Hierarchical dendrogram",
        "Shorter branches join more similar groups of customers.",
    )

    if LINKAGE_PATH.exists():

        Z = joblib.load(LINKAGE_PATH)

        set_link_color_palette(PALETTE)

        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        dendrogram(
            Z,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90,
            leaf_font_size=8,
            show_contracted=True,
            ax=ax,
            above_threshold_color="#6B7A90",
        )

        ax.set_title("Hierarchical clustering dendrogram", color=INK, fontsize=15, pad=14, loc="left")
        ax.set_xlabel("Cluster / Sample Group", color=INK_SOFT)
        ax.set_ylabel("Ward Distance", color=INK_SOFT)
        ax.tick_params(colors=INK_SOFT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color((0.08, 0.13, 0.24, 0.25))
        ax.grid(axis="y", color=(0.08, 0.13, 0.24, 0.07))

        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        set_link_color_palette(None)

    else:

        st.warning("Hierarchical linkage artifact was not found.")


# ==================================================
# PAGE 4 — SEGMENT COMPARISON
# ==================================================

elif page == "Segment Comparison":

    page_head(
        "Segment comparison",
        "Pick a metric and see how each segment compares.",
    )

    comparison_metric = st.selectbox(
        "Select a metric to compare",
        options=[
            "monetary_value",
            "frequency",
            "avg_order_value",
            "recency",
            "engagement_score",
            "discount_dependency",
            "cart_abandonment",
            "returns",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    comparison_df = (
        df.groupby(["cluster", "segment_name"])[comparison_metric]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )

    fig_comparison = px.bar(
        comparison_df,
        x="segment_name",
        y=comparison_metric,
        color="segment_name",
        color_discrete_map=SEGMENT_COLORS,
        title=f"{comparison_metric.replace('_', ' ').title()} by segment",
        labels={
            "segment_name": "Segment",
            comparison_metric: comparison_metric.replace("_", " ").title(),
        },
        text_auto=".3s",
    )
    fig_comparison.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK),
    )
    fig_comparison.update_layout(showlegend=False, xaxis_tickangle=-20)

    st.plotly_chart(style_fig(fig_comparison, 470), width="stretch")

    st.dataframe(comparison_df, width="stretch", hide_index=True)


# ==================================================
# PAGE 5 — CLUSTERING EVALUATION
# ==================================================

elif page == "Clustering Evaluation":

    page_head(
        "Clustering evaluation",
        "How the clustering algorithms score on standard unsupervised-learning metrics.",
    )

    display_evaluation = evaluation_df.rename(
        columns={
            "algorithm": "Algorithm",
            "silhouette_score": "Silhouette Score",
            "davies_bouldin_index": "Davies-Bouldin Index",
            "calinski_harabasz_score": "Calinski-Harabasz Score",
            "number_of_clusters": "Number of Clusters",
            "noise_points": "Noise Points",
        }
    )

    st.dataframe(display_evaluation, width="stretch", hide_index=True)

    metric_mapping = {
        "Silhouette Score": "silhouette_score",
        "Davies-Bouldin Index": "davies_bouldin_index",
        "Calinski-Harabasz Score": "calinski_harabasz_score",
    }

    section("Compare a metric", "Silhouette and Calinski-Harabasz: higher is better. Davies-Bouldin: lower is better.")

    selected_metric = st.selectbox(
        "Select evaluation metric",
        options=list(metric_mapping.keys()),
    )

    metric_column = metric_mapping[selected_metric]

    metric_df = evaluation_df[["algorithm", metric_column]].dropna()

    fig_evaluation = px.bar(
        metric_df,
        x="algorithm",
        y=metric_column,
        color="algorithm",
        color_discrete_sequence=PALETTE,
        title=f"{selected_metric} by algorithm",
        text_auto=".3f",
    )
    fig_evaluation.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=INK),
    )
    fig_evaluation.update_layout(
        xaxis_title="Algorithm",
        yaxis_title=selected_metric,
        showlegend=False,
    )

    st.plotly_chart(style_fig(fig_evaluation, 450), width="stretch")

    st.info(
        "DBSCAN produced one cluster and three noise points "
        "with the evaluated configuration, so the multi-cluster "
        "evaluation metrics are not reported for that result."
    )


# ==================================================
# PAGE 6 — BUSINESS INSIGHTS
# ==================================================

elif page == "Business Insights":

    page_head(
        "Business insights",
        "How each segment differs from the average customer, and where it is stronger or weaker.",
    )

    segment_metrics = (
        df.groupby(["cluster", "segment_name"])
        .agg(
            customers=("customer_id", "count"),
            avg_frequency=("frequency", "mean"),
            avg_monetary_value=("monetary_value", "mean"),
            avg_recency=("recency", "mean"),
            avg_engagement=("engagement_score", "mean"),
            avg_discount_dependency=("discount_dependency", "mean"),
            avg_cart_abandonment=("cart_abandonment", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    # (label, column, overall value, format, higher is better?)
    METRICS = [
        ("Purchase frequency", "avg_frequency", df["frequency"].mean(), "{:.2f}", True),
        ("Monetary value", "avg_monetary_value", df["monetary_value"].mean(), "₹{:,.0f}", True),
        ("Days since last purchase", "avg_recency", df["recency"].mean(), "{:.1f}", False),
        ("Engagement score", "avg_engagement", df["engagement_score"].mean(), "{:.3f}", True),
        ("Discount dependency", "avg_discount_dependency", df["discount_dependency"].mean(), "{:.3f}", False),
        ("Cart abandonment", "avg_cart_abandonment", df["cart_abandonment"].mean(), "{:.2f}", False),
    ]

    total_customers = len(df)

    for _, row in segment_metrics.iterrows():

        segment = row["segment_name"]
        color = SEGMENT_COLORS.get(segment, "#0F766E")

        strengths = []
        weaknesses = []
        table_rows = ""

        for label, col, overall, fmt, higher_better in METRICS:

            value = row[col]
            pct = ((value - overall) / overall * 100) if overall else 0.0

            if abs(pct) < 1:
                delta_html = '<span class="d-flat">On par</span>'
            else:
                is_good = (pct > 0) == higher_better
                arrow = "▲" if pct > 0 else "▼"
                css = "d-good" if is_good else "d-bad"
                delta_html = f'<span class="{css}">{arrow} {abs(pct):.0f}%</span>'
                (strengths if is_good else weaknesses).append((abs(pct), label))

            table_rows += f"""
                <tr>
                    <td>{label}</td>
                    <td>{fmt.format(value)}</td>
                    <td class="muted">{fmt.format(overall)}</td>
                    <td>{delta_html}</td>
                </tr>
            """

        strengths.sort(reverse=True)
        weaknesses.sort(reverse=True)

        summary = ""
        if strengths:
            summary += (
                f'<div class="seg-summary">Ahead of the average on '
                f'{join_labels([s[1] for s in strengths[:3]])}.</div>'
            )
        if weaknesses:
            summary += (
                f'<div class="seg-summary">Behind on '
                f'{join_labels([w[1] for w in weaknesses[:3]])}.</div>'
            )

        members = int(row["customers"])
        share = members / total_customers * 100

        st.markdown(
            flat(f"""
            <div class="seg-panel">
                <div>
                    <div class="seg-bar" style="background:{color};"></div>
                    <div class="seg-name">{segment}</div>
                    <div class="seg-size">{members:,} customers, {share:.1f}% of the base</div>
                    {summary}
                </div>
                <div>
                    <table class="cmp">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>This segment</th>
                                <th>All customers</th>
                                <th>Difference</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )