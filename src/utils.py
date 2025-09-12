# src/utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

def load_data(path="data/Mall_Customers.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put Mall_Customers.csv in the data/ folder.")
    df = pd.read_csv(path)
    return df

def preprocess(df, features):
    # Drop rows with missing values for chosen features
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=features)
    X = df_clean[features].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df_clean.reset_index(drop=True), X_scaled, scaler

def run_elbow(X_scaled, kmax=10):
    wcss = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return wcss

def train_kmeans(X_scaled, k):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def pca_transform(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def silhouette(X_scaled, labels):
    if len(np.unique(labels)) <= 1:
        return -1.0
    return silhouette_score(X_scaled, labels)

def profile_clusters(df, labels, features):
    dfc = df.copy()
    dfc["Cluster"] = labels
    agg_map = {f: ["mean", "median"] for f in features}
    if "Age" in dfc.columns:
        agg_map["Age"] = ["mean", "median"]
    # Count customers
    profile = dfc.groupby("Cluster").agg(agg_map)
    # flatten columns
    profile.columns = ["_".join(col).strip() for col in profile.columns.values]
    profile["Count"] = dfc.groupby("Cluster").size()
    # reorder
    cols = ["Count"] + [c for c in profile.columns if c != "Count"]
    profile = profile[cols]
    return dfc, profile

def save_model(bundle, path="models/kmeans_model.pkl"):
    joblib.dump(bundle, path)

def load_model(path="models/kmeans_model.pkl"):
    return joblib.load(path)

def save_elbow_plot(wcss, out_path="outputs/figures/elbow.png"):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_cluster_plot(X_pca, labels, centers=None, out_path="outputs/figures/clusters.png"):
    plt.figure(figsize=(8,6))
    unique = np.unique(labels)
    for lab in unique:
        idx = labels == lab
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=50, label=f"Cluster {lab}")
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], s=250, marker="*", c="yellow", label="Centroids")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Customer Segments (PCA projection)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
