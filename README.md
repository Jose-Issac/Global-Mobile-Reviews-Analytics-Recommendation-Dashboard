# Global-Mobile-Reviews-Analytics-Recommendation-Dashboard
An end-to-end data science project analyzing 50,000+ global mobile phone reviews — covering data cleaning, exploratory analysis, customer sentiment, K-Means segmentation, a similarity-based recommendation engine, and an interactive Streamlit dashboard to tie it all together.

Overview

This project takes a raw, messy dataset of mobile phone reviews (multi-currency pricing, null values, inconsistent formatting) and turns it into:

A cleaned, analysis-ready dataset
Exploratory visualizations on pricing, sentiment, ratings, and sources
Product segments (Budget / Mid-range / Premium / Luxury) via K-Means clustering
A cosine-similarity recommendation system for "customers who liked this also liked..." style suggestions
A live Streamlit app to explore all of the above interactively
Features
Data Cleaning — handles multi-currency price fields (₹, ,€,£,𝑅,€,£,R, د.إ, A,𝐶,C) with mixed formatting, thousands separators, and missing values; cross-fills price_usd/price_local using exchange rates where possible
Exploratory Data Analysis — sentiment distribution, purchase value by age, pricing by source and brand, verified purchase split, brand performance across all rating dimensions
Customer Segmentation — K-Means clustering (4 clusters) on product-level price and quality features, labeled into intuitive market tiers
Recommendation Engine — cosine similarity across product feature vectors (price, rating, battery, camera, performance) to recommend similar phones
Interactive Dashboard — Streamlit app with EDA, cluster insights, and a live product recommendation interface
Tech Stack
Python — pandas, numpy
Machine Learning — scikit-learn (KMeans, StandardScaler, cosine_similarity)
Visualization — Matplotlib, Plotly
App Framework — Streamlit
Model Persistence — joblib

Dashboard Preview

The app has three tabs:

Overview & EDA — sentiment breakdown, pricing trends by age/brand/source, verified purchase ratio, brand performance comparison
Cluster Insights — product segments with average price/rating/spec breakdowns, segment-size and price-vs-rating visualizations
Recommendation System — pick any phone, get the top-N most similar products with a similarity score and relevance validation metric
Future Improvements
Persist the original StandardScaler used during model training for exact reproducibility
Add filtering by country/language for region-specific insights
Expand the recommendation engine with collaborative filtering using individual review data, not just product-level averages
