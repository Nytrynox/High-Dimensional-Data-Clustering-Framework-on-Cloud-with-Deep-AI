"""
Utility functions for data processing and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive clustering metrics"""
    metrics = {}
    
    # Basic metrics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1) if -1 in unique_labels else 0
    
    metrics['n_clusters'] = n_clusters
    metrics['n_samples'] = len(labels)
    metrics['n_noise'] = n_noise
    metrics['noise_ratio'] = n_noise / len(labels)
    
    # Only calculate advanced metrics if we have valid clusters
    valid_labels = labels[labels != -1] if -1 in labels else labels
    valid_X = X[labels != -1] if -1 in labels else X
    
    if len(np.unique(valid_labels)) > 1 and len(valid_X) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(valid_X, valid_labels)
        except:
            metrics['silhouette_score'] = -1
            
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_X, valid_labels)
        except:
            metrics['calinski_harabasz_score'] = 0
            
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(valid_X, valid_labels)
        except:
            metrics['davies_bouldin_score'] = float('inf')
    else:
        metrics['silhouette_score'] = -1
        metrics['calinski_harabasz_score'] = 0
        metrics['davies_bouldin_score'] = float('inf')
    
    return metrics


def plot_clustering_results(
    X: np.ndarray, 
    labels: np.ndarray, 
    title: str = "Clustering Results",
    method: str = "pca"
) -> go.Figure:
    """Create interactive visualization of clustering results"""
    
    # Apply dimensionality reduction for visualization
    if X.shape[1] > 2:
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            explained_var = reducer.explained_variance_ratio_
            subtitle = f"PCA (explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f})"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_2d = reducer.fit_transform(X[:1000])  # Limit for speed
            labels = labels[:1000]
            subtitle = "t-SNE (subset of 1000 points)"
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            X_2d = reducer.fit_transform(X)
            subtitle = "UMAP"
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        X_2d = X
        subtitle = "Original 2D data"
    
    # Create scatter plot
    fig = go.Figure()
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Set1 + px.colors.qualitative.Set2
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[i % len(colors)]
        
        if label == -1:  # Noise points
            name = "Noise"
            symbol = "x"
        else:
            name = f"Cluster {label}"
            symbol = "circle"
        
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode='markers',
            name=name,
            marker=dict(
                color=color,
                size=6,
                symbol=symbol,
                line=dict(width=1, color='white')
            ),
            text=[f"Point {idx}<br>Cluster: {label}" for idx in np.where(mask)[0]],
            hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))
    
    # Calculate metrics for subtitle
    metrics = calculate_clustering_metrics(X, labels)
    
    fig.update_layout(
        title=f"{title}<br><sub>{subtitle}</sub>",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        showlegend=True,
        width=800,
        height=600,
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"Clusters: {metrics['n_clusters']}<br>"
                     f"Silhouette: {metrics['silhouette_score']:.3f}<br>"
                     f"Samples: {metrics['n_samples']}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Any]], 
    metric: str = "silhouette_score"
) -> go.Figure:
    """Compare clustering metrics across different algorithms"""
    
    algorithms = list(results.keys())
    values = [results[alg]['metrics'].get(metric, 0) for alg in algorithms]
    
    fig = go.Figure(data=[
        go.Bar(
            x=algorithms,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title=f"Clustering Algorithm Comparison: {metric.replace('_', ' ').title()}",
        xaxis_title="Algorithm",
        yaxis_title=metric.replace('_', ' ').title(),
        showlegend=False
    )
    
    return fig


def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """Plot cluster size distribution"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Handle noise points
    cluster_names = []
    for label in unique_labels:
        if label == -1:
            cluster_names.append("Noise")
        else:
            cluster_names.append(f"Cluster {label}")
    
    fig = go.Figure(data=[
        go.Bar(
            x=cluster_names,
            y=counts,
            text=counts,
            textposition='auto',
            marker_color='lightgreen'
        )
    ])
    
    fig.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster",
        yaxis_title="Number of Points",
        showlegend=False
    )
    
    return fig


def create_feature_importance_plot(
    feature_names: List[str], 
    importances: np.ndarray, 
    title: str = "Feature Importance"
) -> go.Figure:
    """Create feature importance visualization"""
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    top_features = sorted_idx[:20]  # Show top 20 features
    
    fig = go.Figure(data=[
        go.Bar(
            y=[feature_names[i] for i in top_features],
            x=importances[top_features],
            orientation='h',
            marker_color='coral'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600,
        showlegend=False
    )
    
    return fig


def plot_elbow_curve(X: np.ndarray, max_k: int = 15) -> go.Figure:
    """Plot elbow curve for optimal number of clusters"""
    from sklearn.cluster import KMeans
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Elbow Method", "Silhouette Score"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow curve
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Silhouette scores
    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(
        title="Optimal Number of Clusters Analysis",
        showlegend=False,
        height=400
    )
    
    return fig


def generate_clustering_report(
    X: np.ndarray, 
    labels: np.ndarray, 
    algorithm: str,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate comprehensive clustering report"""
    
    metrics = calculate_clustering_metrics(X, labels)
    
    report = {
        "algorithm": algorithm,
        "metrics": metrics,
        "summary": {
            "total_samples": len(X),
            "features": X.shape[1],
            "clusters_found": metrics['n_clusters'],
            "noise_points": metrics['n_noise'],
            "quality_score": metrics['silhouette_score']
        }
    }
    
    # Cluster statistics
    unique_labels = np.unique(labels)
    cluster_stats = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = X[mask]
        
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{label}"
        
        cluster_stats[cluster_name] = {
            "size": int(np.sum(mask)),
            "percentage": float(np.sum(mask) / len(labels) * 100),
            "center": cluster_data.mean(axis=0).tolist(),
            "std": cluster_data.std(axis=0).tolist()
        }
    
    report["cluster_statistics"] = cluster_stats
    
    # Feature statistics if available
    if feature_names:
        report["feature_names"] = feature_names
    
    return report


class ClusteringVisualizer:
    """Class for advanced clustering visualizations"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3 + px.colors.qualitative.Set1
    
    def plot_3d_clusters(self, X: np.ndarray, labels: np.ndarray, title: str = "3D Clustering") -> go.Figure:
        """Create 3D visualization of clusters"""
        
        if X.shape[1] < 3:
            # Apply PCA to get 3 dimensions
            pca = PCA(n_components=3, random_state=42)
            X_3d = pca.fit_transform(X)
            subtitle = f"PCA 3D (explained variance: {pca.explained_variance_ratio_.sum():.3f})"
        else:
            X_3d = X[:, :3]  # Use first 3 dimensions
            subtitle = "First 3 dimensions"
        
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = self.colors[i % len(self.colors)]
            
            name = f"Cluster {label}" if label != -1 else "Noise"
            
            fig.add_trace(go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode='markers',
                name=name,
                marker=dict(
                    color=color,
                    size=4,
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title=f"{title}<br><sub>{subtitle}</sub>",
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_cluster_heatmap(self, X: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> go.Figure:
        """Create heatmap of cluster centroids"""
        
        unique_labels = np.unique(labels)
        centroids = []
        cluster_names = []
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            
            mask = labels == label
            centroid = X[mask].mean(axis=0)
            centroids.append(centroid)
            cluster_names.append(f"Cluster {label}")
        
        centroids = np.array(centroids)
        
        # Limit to top features if too many
        if len(feature_names) > 50:
            # Select features with highest variance across clusters
            variances = np.var(centroids, axis=0)
            top_features_idx = np.argsort(variances)[-50:]
            centroids = centroids[:, top_features_idx]
            feature_names = [feature_names[i] for i in top_features_idx]
        
        fig = go.Figure(data=go.Heatmap(
            z=centroids,
            x=feature_names,
            y=cluster_names,
            colorscale='RdBu',
            text=np.round(centroids, 3),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Cluster Centroids Heatmap",
            xaxis_title="Features",
            yaxis_title="Clusters",
            width=max(800, len(feature_names) * 15),
            height=max(400, len(cluster_names) * 50)
        )
        
        return fig
