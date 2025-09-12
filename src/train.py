# src/train.py
import os
from src.utils import *
import pandas as pd

DATA_PATH = os.environ.get("DATA_PATH", "data/Mall_Customers.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/kmeans_model.pkl")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "outputs/segmented_customers.csv")

def main():
    ensure_dirs()
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    # Default features for segmentation
    features = ["Annual Income (k$)", "Spending Score (1-100)"]
    
    # If Age exists, you can include it if you prefer:
    if "Age" in df.columns:
        features = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]

    print("Preprocessing...")
    df_clean, X_scaled, scaler = preprocess(df, features)

    print("Running Elbow method...")
    wcss = run_elbow(X_scaled, kmax=10)
    save_elbow_plot(wcss, out_path="outputs/figures/elbow.png")
    print("Elbow plot saved to outputs/figures/elbow.png")

    # Choose k automatically (simple heuristic): elbow or default 5
    # For simplicity, default to 5 but you can inspect elbow.png and change
    k = 5
    print(f"Training KMeans with k={k} ...")
    kmeans, labels = train_kmeans(X_scaled, k)

    print("PCA transform for visualization...")
    X_pca, pca = pca_transform(X_scaled, n_components=2)

    print("Profiling clusters...")
    df_seg, profile = profile_clusters(df_clean, labels, features)
    df_seg["pc1"] = X_pca[:, 0]
    df_seg["pc2"] = X_pca[:, 1]

    print("Saving segmented CSV...")
    os.makedirs("outputs", exist_ok=True)
    df_seg.to_csv(OUTPUT_CSV, index=False)

    print("Saving cluster plot...")
    centers_pca = None
    try:
        # Transform centers to PCA space .
        centers = kmeans.cluster_centers_
        centers_pca = pca.transform(centers)
    except Exception:
        centers_pca = None
    save_cluster_plot(X_pca, labels, centers=centers_pca, out_path="outputs/figures/clusters.png")

    print("Saving profile CSV...")
    profile.to_csv("outputs/figures/cluster_profile.csv")

    print("Saving model bundle...")
    bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "features": features
    }
    save_model(bundle, path=MODEL_PATH)

    print("Training complete.")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Segmented CSV saved to {OUTPUT_CSV}")
    print("Cluster profile saved to outputs/figures/cluster_profile.csv")
    print("Cluster plots saved to outputs/figures/")

if __name__ == "__main__":
    main()
    

