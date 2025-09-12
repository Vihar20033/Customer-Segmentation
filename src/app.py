# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.utils import load_data, load_model

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/kmeans_model.pkl")
DEFAULT_DATA = os.environ.get("DATA_PATH", "data/Mall_Customers.csv")

st.title("Customer Segmentation — KMeans")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
data_path = DEFAULT_DATA
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"Dataset not found at {data_path}. Upload a CSV or place Mall_Customers.csv in data/ folder.")
        st.stop()

st.sidebar.markdown("### Features & Clusters")
available_features = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] or c in ["Annual Income (k$)","Spending Score (1-100)","Age"]]
# Ensure typical features present
default_feats = [f for f in ["Annual Income (k$)","Spending Score (1-100)","Age"] if f in df.columns]
features = st.sidebar.multiselect("Features for clustering", options=available_features, default=default_feats if default_feats else available_features[:2])
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=5)
recluster = st.sidebar.button("Run clustering")

st.subheader("Dataset preview")
st.dataframe(df.head(200))

# Prepare data
if len(features) < 2:
    st.warning("Select at least 2 numeric features for meaningful clustering.")
    st.stop()

X = df[features].copy().astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Decide to use saved model or compute new
use_saved_model = st.sidebar.checkbox("Use saved model (if exists)", value=True)

model_bundle = None
if use_saved_model and os.path.exists(MODEL_PATH):
    try:
        model_bundle = load_model(MODEL_PATH)
    except Exception:
        model_bundle = None

if recluster or model_bundle is None:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
else:
    # Try using the saved model but re-fit PCA/scaler for chosen features
    try:
        kmeans = model_bundle["kmeans"]
        # if features differ, we recreate scaler & transform
        labels = kmeans.predict(StandardScaler().fit_transform(X[model_bundle["features"]])) if set(model_bundle["features"]).issubset(set(features)) else kmeans.fit_predict(X_scaled)
        pca = model_bundle.get("pca", PCA(n_components=2))
        X_pca = pca.transform(X_scaled) if hasattr(pca, "transform") else PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    except Exception:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

df_seg = df.copy().reset_index(drop=True)
df_seg["Cluster"] = labels
df_seg["pc1"] = X_pca[:,0]
df_seg["pc2"] = X_pca[:,1]

# Cluster counts
st.subheader("Cluster counts")
counts = df_seg["Cluster"].value_counts().sort_index()
st.bar_chart(counts)

# PCA scatter plot (interactive)
st.subheader("Cluster visualization (PCA)")
fig = px.scatter(df_seg, x="pc1", y="pc2", color=df_seg["Cluster"].astype(str),
                 hover_data=df_seg.columns, title="PCA projection of clusters", width=900, height=600)
st.plotly_chart(fig, use_container_width=True)

# Feature vs Feature scatter (interactive)
st.subheader("Feature scatter (choose pair)")
f1 = st.selectbox("X axis", options=features, index=0)
f2 = st.selectbox("Y axis", options=features, index=1 if len(features)>1 else 0)
fig2 = px.scatter(df_seg, x=f1, y=f2, color=df_seg["Cluster"].astype(str), hover_data=df_seg.columns, title=f"{f1} vs {f2}")
st.plotly_chart(fig2, use_container_width=True)

# Cluster profiles
st.subheader("Cluster profile summary")
profile = df_seg.groupby("Cluster").agg({col: "mean" for col in features})
profile["Count"] = df_seg.groupby("Cluster").size()
cols = ["Count"] + [c for c in profile.columns if c != "Count"]
profile = profile[cols]
st.dataframe(profile.style.format("{:.2f}"))

# Download segmented CSV
st.subheader("Download segmented data")
csv = df_seg.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, file_name="segmented_customers.csv", mime="text/csv")

# Business insights (simple rules)
st.subheader("Automated business insights")
global_income_mean = profile[features[0]].mean() if features else 0
for idx, row in profile.iterrows():
    seg_label = f"Cluster {idx}"
    spend_mean = row[features[1]] if len(features) > 1 else row[features[0]]
    income_mean = row[features[0]]
    if spend_mean > profile[features[1]].mean() and income_mean > profile[features[0]].mean():
        st.write(f"**{seg_label}:** High value — target with premium offers & loyalty programs.")
    elif spend_mean > profile[features[1]].mean():
        st.write(f"**{seg_label}:** High spenders — cross-sell and promotional bundles.")
    else:
        st.write(f"**{seg_label}:** Low value — retention & discounts strategy.")

st.markdown("---")
st.caption("Tip: Use the sidebar to upload a new CSV, change features and number of clusters, then click 'Run clustering'.")

