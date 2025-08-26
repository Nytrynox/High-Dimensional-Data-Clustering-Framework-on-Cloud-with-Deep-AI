"""
Clustering pipeline for coordinating different algorithms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import asyncio
import pickle
import json
from datetime import datetime

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import hdbscan

from src.models.deep_embedding import DECClustering
from src.utils.logger import ClusteringLogger
from src.config import settings, ClusteringConfig


class ClusteringPipeline:
    """Main clustering pipeline"""
    
    def __init__(self):
        self.logger = ClusteringLogger("clustering_pipeline")
        self.scaler = StandardScaler()
        self.algorithms = {
            "kmeans": self._kmeans_clustering,
            "dbscan": self._dbscan_clustering,
            "spectral": self._spectral_clustering,
            "hdbscan": self._hdbscan_clustering,
            "deep_embedding": self._deep_embedding_clustering,
        }
    
    async def run(
        self,
        data_path: Path,
        algorithm: str,
        n_clusters: Optional[int] = None,
        output_dir: Path = Path("results"),
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete clustering pipeline"""
        
        self.logger.progress(f"Starting clustering pipeline with {algorithm}")
        
        try:
            # Load and preprocess data
            X, feature_names = await self._load_data(data_path)
            X_scaled = await self._preprocess_data(X)
            
            # Validate parameters
            if algorithm not in self.algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Auto-determine number of clusters if not provided
            if n_clusters is None and algorithm in ["kmeans", "spectral", "deep_embedding"]:
                n_clusters = await self._determine_optimal_clusters(X_scaled)
                self.logger.info(f"Auto-determined optimal clusters: {n_clusters}")
            
            # Run clustering
            clustering_func = self.algorithms[algorithm]
            labels, embeddings, metrics = await clustering_func(
                X_scaled, n_clusters=n_clusters, **kwargs
            )
            
            # Calculate additional metrics
            final_metrics = await self._calculate_metrics(X_scaled, labels, embeddings)
            final_metrics.update(metrics)
            
            # Save results
            results = {
                "algorithm": algorithm,
                "n_clusters": len(np.unique(labels)),
                "labels": labels.tolist(),
                "metrics": final_metrics,
                "feature_names": feature_names,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._save_results(results, output_dir, data_path.stem, algorithm)
            
            self.logger.success(f"Clustering completed successfully with {algorithm}")
            return results
            
        except Exception as e:
            self.logger.failure(f"Clustering pipeline failed: {str(e)}")
            raise
    
    async def _load_data(self, data_path: Path) -> Tuple[np.ndarray, list]:
        """Load data from various formats"""
        self.logger.progress(f"Loading data from {data_path}")
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            df = pd.read_json(data_path)
        elif data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.xlsx':
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Select numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            raise ValueError("No numeric columns found in the dataset")
        
        X = df[numeric_columns].values
        
        # Handle missing values
        if np.any(np.isnan(X)):
            self.logger.info("Handling missing values with mean imputation")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        self.logger.success(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, numeric_columns
    
    async def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for clustering"""
        self.logger.progress("Preprocessing data...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Dimensionality reduction for high-dimensional data
        if X_scaled.shape[1] > settings.max_dimensions:
            self.logger.info(f"Applying PCA to reduce from {X_scaled.shape[1]} to {settings.max_dimensions} dimensions")
            pca = PCA(n_components=settings.max_dimensions, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
        
        return X_scaled
    
    async def _determine_optimal_clusters(self, X: np.ndarray) -> int:
        """Determine optimal number of clusters using elbow method"""
        self.logger.progress("Determining optimal number of clusters...")
        
        max_k = min(settings.max_clusters, X.shape[0] // 2)
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow method (can be improved)
        deltas = np.diff(inertias)
        double_deltas = np.diff(deltas)
        optimal_k = np.argmax(double_deltas) + 2  # +2 because we started from k=2
        
        return min(max(optimal_k, 2), max_k)
    
    async def _kmeans_clustering(
        self, X: np.ndarray, n_clusters: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """K-means clustering"""
        self.logger.progress(f"Running K-means with {n_clusters} clusters")
        
        max_iter = kwargs.get("max_iter", 300)
        tol = kwargs.get("tol", 1e-4)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=42,
            n_init=10
        )
        
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        metrics = {
            "inertia": kmeans.inertia_,
            "n_iter": kmeans.n_iter_
        }
        
        return labels, centers, metrics
    
    async def _dbscan_clustering(
        self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """DBSCAN clustering"""
        self.logger.progress(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels)
        }
        
        # No explicit centers for DBSCAN
        embeddings = X
        
        return labels, embeddings, metrics
    
    async def _spectral_clustering(
        self, X: np.ndarray, n_clusters: int, gamma: float = 1.0, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Spectral clustering"""
        self.logger.progress(f"Running Spectral clustering with {n_clusters} clusters")
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            random_state=42,
            n_init=10
        )
        
        labels = spectral.fit_predict(X)
        
        metrics = {
            "gamma": gamma
        }
        
        return labels, X, metrics
    
    async def _hdbscan_clustering(
        self, X: np.ndarray, min_cluster_size: int = 5, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """HDBSCAN clustering"""
        self.logger.progress(f"Running HDBSCAN with min_cluster_size={min_cluster_size}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=kwargs.get("min_samples", None)
        )
        
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_persistence": clusterer.cluster_persistence_.tolist() if hasattr(clusterer, 'cluster_persistence_') else []
        }
        
        return labels, X, metrics
    
    async def _deep_embedding_clustering(
        self, X: np.ndarray, n_clusters: int, latent_dim: int = 10, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Deep embedding clustering"""
        self.logger.progress(f"Running Deep Embedding Clustering with {n_clusters} clusters")
        
        dec = DECClustering(
            n_clusters=n_clusters,
            latent_dim=latent_dim,
            alpha=kwargs.get("alpha", 1.0)
        )
        
        # Run in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        labels, embeddings, metrics = await loop.run_in_executor(
            None, dec.fit, X, kwargs.get("epochs", 100)
        )
        
        return labels, embeddings, metrics
    
    async def _calculate_metrics(
        self, X: np.ndarray, labels: np.ndarray, embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Calculate clustering evaluation metrics"""
        metrics = {}
        
        # Only calculate metrics if we have more than one cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters > 1 and -1 not in unique_labels:
            try:
                metrics["silhouette_score"] = silhouette_score(embeddings, labels)
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, labels)
                metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, labels)
            except Exception as e:
                self.logger.warning(f"Could not calculate some metrics: {e}")
        
        metrics["n_clusters"] = n_clusters
        metrics["n_samples"] = len(labels)
        
        return metrics
    
    async def _save_results(
        self, results: Dict[str, Any], output_dir: Path, 
        dataset_name: str, algorithm: str
    ):
        """Save clustering results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{algorithm}_{timestamp}"
        
        # Save as JSON
        json_path = output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save labels as CSV
        csv_path = output_dir / f"{filename}_labels.csv"
        pd.DataFrame({"cluster_label": results["labels"]}).to_csv(csv_path, index=False)
        
        self.logger.success(f"Results saved to {output_dir}")


class BatchClusteringPipeline:
    """Pipeline for batch processing multiple datasets"""
    
    def __init__(self):
        self.logger = ClusteringLogger("batch_clustering")
        self.pipeline = ClusteringPipeline()
    
    async def run_batch(
        self, 
        data_paths: list[Path], 
        algorithms: list[str], 
        output_dir: Path = Path("batch_results")
    ) -> Dict[str, Dict[str, Any]]:
        """Run clustering on multiple datasets with multiple algorithms"""
        
        self.logger.progress(f"Starting batch clustering on {len(data_paths)} datasets with {len(algorithms)} algorithms")
        
        results = {}
        
        for data_path in data_paths:
            dataset_name = data_path.stem
            results[dataset_name] = {}
            
            for algorithm in algorithms:
                try:
                    self.logger.info(f"Processing {dataset_name} with {algorithm}")
                    
                    result = await self.pipeline.run(
                        data_path=data_path,
                        algorithm=algorithm,
                        output_dir=output_dir / dataset_name
                    )
                    
                    results[dataset_name][algorithm] = result
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {dataset_name} with {algorithm}: {e}")
                    results[dataset_name][algorithm] = {"error": str(e)}
        
        # Save batch results summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.success("Batch clustering completed")
        return results
