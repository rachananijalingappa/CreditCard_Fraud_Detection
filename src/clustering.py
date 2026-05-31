"""Unsupervised clustering analysis: K-Means, Isolation Forest, DBSCAN."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score

from src.config import RANDOM_STATE


def kmeans_elbow_analysis(X_2d: np.ndarray, max_k: int = 10) -> int:
    """Find optimal K using elbow method and silhouette score.
    
    Args:
        X_2d: 2D PCA-projected data.
        max_k: Maximum K to evaluate.
        
    Returns:
        Best K based on silhouette score.
    """
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_2d)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_2d, labels, sample_size=10000,
                                           random_state=RANDOM_STATE))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    
    axes[1].plot(K_range, silhouettes, 'ro-')
    axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.show()
    
    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f'Best K: {best_k}')
    return best_k


def kmeans_clustering(X_2d: np.ndarray, df: pd.DataFrame, best_k: int) -> None:
    """Perform K-Means clustering and analyse fraud rates per cluster.
    
    Args:
        X_2d: 2D PCA-projected data.
        df: Full dataset with 'Class' column.
        best_k: Optimal number of clusters.
    """
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_2d)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cluster plot
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.1, s=3)
    axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    axes[0].set_title(f'K-Means (K={best_k})', fontsize=14, fontweight='bold')
    
    # Fraud rate per cluster
    cluster_df = pd.DataFrame({'Cluster': cluster_labels, 'Fraud': df['Class'].values})
    fraud_rate = cluster_df.groupby('Cluster')['Fraud'].mean() * 100
    axes[1].bar(fraud_rate.index, fraud_rate.values, color='#e74c3c', edgecolor='black')
    axes[1].set_title('Fraud Rate per Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Fraud Rate (%)')
    
    plt.tight_layout()
    plt.show()
    
    print('Fraud rate by cluster:')
    for c, rate in fraud_rate.items():
        print(f'  Cluster {c}: {rate:.3f}%')


def isolation_forest_analysis(X_2d: np.ndarray, df: pd.DataFrame) -> None:
    """Run Isolation Forest anomaly detection.
    
    Args:
        X_2d: 2D PCA-projected data.
        df: Full dataset with 'Class' column.
    """
    iso_forest = IsolationForest(n_estimators=200, contamination=0.00172,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    iso_pred = iso_forest.fit_predict(X_2d)
    iso_pred_binary = (iso_pred == -1).astype(int)
    
    print('Isolation Forest as fraud detector:')
    print(f'  Anomalies detected: {iso_pred_binary.sum():,}')
    print(f'  Actual frauds:      {df["Class"].sum():,}')
    print(f'  Precision: {precision_score(df["Class"], iso_pred_binary):.4f}')
    print(f'  Recall:    {recall_score(df["Class"], iso_pred_binary):.4f}')
    print(f'  F1-Score:  {f1_score(df["Class"], iso_pred_binary):.4f}')


def dbscan_analysis(X_2d: np.ndarray, df: pd.DataFrame,
                    sample_size: int = 20000) -> None:
    """Run DBSCAN density-based clustering on a sample.
    
    Args:
        X_2d: 2D PCA-projected data.
        df: Full dataset with 'Class' column.
        sample_size: Number of samples to use (DBSCAN is memory-intensive).
    """
    sample_idx = np.random.choice(len(X_2d), sample_size, replace=False)
    
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_2d[sample_idx])
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = (dbscan_labels == -1).sum()
    
    plt.figure(figsize=(10, 7))
    plt.scatter(X_2d[sample_idx, 0], X_2d[sample_idx, 1], c=dbscan_labels,
                cmap='tab20', alpha=0.5, s=5)
    plt.title(f'DBSCAN: {n_clusters} clusters, {n_noise:,} noise points',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Check if noise points contain more fraud
    noise_fraud = (df['Class'].values[sample_idx][dbscan_labels == -1].mean() * 100
                   if n_noise > 0 else 0)
    overall_fraud = df['Class'].mean() * 100
    print(f'Fraud rate in noise points: {noise_fraud:.2f}% (vs overall: {overall_fraud:.3f}%)')
