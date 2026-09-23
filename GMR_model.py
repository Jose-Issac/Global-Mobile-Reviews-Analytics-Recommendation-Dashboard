"""
Global Mobile Reviews — Streamlit Dashboard
Application Development & Visualization (Step 5)

Tabs:
  1. Overview & EDA        -> the 6 charts (sentiment, age, source, verified, brand price, brand performance)
  2. Cluster Insights       -> product segments (Budget / Mid-range / Premium / Luxury) using the saved KMeans model
  3. Recommendation System  -> similar-product lookup using the saved cosine similarity matrix

Data: GMR.xls already has nulls handled, so no cleaning pipeline is needed here.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Global Mobile Reviews Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# 1. DATA LOADING (already cleaned — nulls handled upstream, so just load it)
# ---------------------------------------------------------------------------
DATA_PATH = "GMR.xls"          # note: file is CSV-formatted despite the .xls extension
REC_MODEL_PATH = "Model_recommendation.pkl"
KMEANS_MODEL_PATH = "K_means_model.pkl"

RATING_COLS = ['rating', 'battery_life_rating', 'camera_rating',
               'performance_rating', 'design_rating', 'display_rating']
CLUSTER_FEATURES = ['avg_price', 'avg_rating', 'avg_battery', 'avg_camera', 'avg_performance']


@st.cache_data
def load_data(path):
    # GMR.xls is plain CSV text (not a binary Excel file), so read_csv reads it correctly.
    # All nulls are already handled upstream — no fillna/cleaning needed here.
    df = pd.read_csv(path)
    return df


@st.cache_data
def build_product_table(df):
    product_df = df.groupby(['brand', 'model']).agg(
        avg_price=('price_usd', 'mean'),
        avg_rating=('rating', 'mean'),
        avg_battery=('battery_life_rating', 'mean'),
        avg_camera=('camera_rating', 'mean'),
        avg_performance=('performance_rating', 'mean'),
    ).reset_index()
    product_df['product_label'] = product_df['brand'] + ' ' + product_df['model']
    return product_df


@st.cache_resource
def load_models():
    similarity_matrix = joblib.load(REC_MODEL_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    return similarity_matrix, kmeans_model


GMR = load_data(DATA_PATH)
product_df = build_product_table(GMR)
similarity_matrix, kmeans_model = load_models()

# similarity matrix rows/cols follow the same brand+model sort order as groupby -> product_df
sim_df = pd.DataFrame(similarity_matrix,
                       index=product_df['product_label'],
                       columns=product_df['product_label'])

# NOTE: the original StandardScaler used to train K_means_model.pkl was not saved alongside it,
# so we refit one here on the same product-level features. Since it's built from the same
# source data, this reproduces (very closely) the scale the model was trained on.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(product_df[CLUSTER_FEATURES])
product_df['cluster'] = kmeans_model.predict(X_scaled)

cluster_price_rank = product_df.groupby('cluster')['avg_price'].mean().sort_values()
tier_names = ['Budget', 'Mid-range', 'Premium', 'Luxury'][:len(cluster_price_rank)]
tier_labels = dict(zip(cluster_price_rank.index, tier_names))
product_df['segment'] = product_df['cluster'].map(tier_labels)

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("📱 Global Mobile Reviews")
page = st.sidebar.radio("Navigate", ["Overview & EDA", "Cluster Insights", "Recommendation System"])
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(GMR):,} reviews  •  {len(product_df)} unique products")

# ---------------------------------------------------------------------------
# PAGE 1: OVERVIEW & EDA
# ---------------------------------------------------------------------------
if page == "Overview & EDA":
    st.title("Overview & Exploratory Visualizations")

    col1, col2 = st.columns(2)

    # 1) Sentiment pie
    with col1:
        counts = GMR['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#27D6F5', '#F44336'], startangle=90)
        ax.set_title('Sentiment Analysis')
        ax.axis('equal')
        st.pyplot(fig)

    # 4) Verified purchase pie
    with col2:
        verified = GMR['verified_purchase'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(verified, labels=verified.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#27D6F5'], startangle=90)
        ax.set_title('Verified Customers')
        ax.axis('equal')
        st.pyplot(fig)

    # 2) Average purchase value by age
    age_price = GMR.groupby('age')['price_usd'].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(age_price.index, age_price.values, color='#27D6F5')
    ax.set_title('Average Purchase Value by Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Purchase Value')
    st.pyplot(fig)

    col3, col4 = st.columns(2)

    # 3) Average price by source
    with col3:
        source_value = GMR.groupby('source')['price_usd'].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(source_value.index, source_value.values, color='#27F5E0')
        ax.set_title('Average Price across Source')
        ax.set_xlabel('Source')
        ax.set_ylabel('Price Value')
        st.pyplot(fig)

    # 5) Total price by brand
    with col4:
        brand_total = GMR.groupby('brand')['price_usd'].sum().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(brand_total.index, brand_total.values, color='#27D6F5')
        ax.set_title('Total Price by Brand')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Purchase Value')
        st.pyplot(fig)

    # 6) Brand performance across all features (grouped bar)
    st.subheader("Brand Performance Across All Features")
    brand_perf = GMR.groupby('brand')[RATING_COLS].sum()
    x = np.arange(len(brand_perf.index))
    width = 0.13
    n_features = len(RATING_COLS)
    colors = ['#27D6F5', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#3F51B5']

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, col in enumerate(RATING_COLS):
        offset = (i - n_features / 2) * width + width / 2
        ax.bar(x + offset, brand_perf[col], width, label=col, color=colors[i])
    ax.set_xlabel('Brand')
    ax.set_ylabel('Summed Rating')
    ax.set_title('Brand Performance Across All Features')
    ax.set_xticks(x)
    ax.set_xticklabels(brand_perf.index, rotation=0)
    ax.legend(title='Feature', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# PAGE 2: CLUSTER INSIGHTS
# ---------------------------------------------------------------------------
elif page == "Cluster Insights":
    st.title("Cluster Insights — Product Segments")
    st.caption("Segments generated by the saved K-Means model (4 clusters), labeled by average price tier.")

    seg_summary = product_df.groupby('segment')[CLUSTER_FEATURES].mean().round(2)
    seg_summary['num_products'] = product_df.groupby('segment').size()
    st.dataframe(seg_summary.sort_values('avg_price'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(product_df.groupby('segment').size().reset_index(name='count'),
                       x='segment', y='count', color='segment',
                       title='Number of Products per Segment')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(product_df, x='avg_price', y='avg_rating', color='segment',
                           size='avg_performance', hover_name='product_label',
                           title='Price vs Rating by Segment')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Products by Segment")
    segment_filter = st.multiselect("Filter by segment", options=sorted(product_df['segment'].unique()),
                                     default=sorted(product_df['segment'].unique()))
    filtered = product_df[product_df['segment'].isin(segment_filter)]
    st.dataframe(
        filtered[['brand', 'model', 'segment'] + CLUSTER_FEATURES].sort_values('avg_price'),
        use_container_width=True
    )

# ---------------------------------------------------------------------------
# PAGE 3: RECOMMENDATION SYSTEM
# ---------------------------------------------------------------------------
else:
    st.title("Product Recommendation System")
    st.caption("Cosine-similarity based recommendations using the saved similarity model.")

    product_choice = st.selectbox("Select a product", options=product_df['product_label'].sort_values())
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    if product_choice:
        input_row = product_df[product_df['product_label'] == product_choice].iloc[0]

        st.markdown(f"### Selected: {product_choice}  ·  Segment: **{input_row['segment']}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Price", f"${input_row['avg_price']:.0f}")
        c2.metric("Avg Rating", f"{input_row['avg_rating']:.2f}")
        c3.metric("Battery", f"{input_row['avg_battery']:.2f}")
        c4.metric("Camera", f"{input_row['avg_camera']:.2f}")
        c5.metric("Performance", f"{input_row['avg_performance']:.2f}")

        scores = sim_df[product_choice].drop(product_choice)
        top_matches = scores.sort_values(ascending=False).head(top_n)

        rec_table = product_df.set_index('product_label').loc[top_matches.index,
                                                                CLUSTER_FEATURES + ['segment']].copy()
        rec_table.insert(0, 'similarity_score', top_matches.values.round(3))
        rec_table = rec_table.reset_index()

        st.subheader(f"Top {top_n} Similar Products")
        st.dataframe(rec_table, use_container_width=True)

        fig = px.bar(rec_table, x='product_label', y='similarity_score', color='segment',
                     title=f'Similarity Score vs {product_choice}')
        st.plotly_chart(fig, use_container_width=True)

        # simple relevance validation: average distance of recommendations from selected product
        input_vec = X_scaled[product_df['product_label'] == product_choice][0]
        rec_idx = product_df[product_df['product_label'].isin(top_matches.index)].index
        rec_vecs = X_scaled[rec_idx]
        avg_dist = np.mean(np.linalg.norm(rec_vecs - input_vec, axis=1))
        st.info(f"Validation — average feature distance of recommendations: **{avg_dist:.3f}** "
                f"(lower = more relevant; closer to 0 means the recommended products are very similar).")