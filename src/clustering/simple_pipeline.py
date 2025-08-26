"""
Simple Clustering Pipeline - Database-Free Implementation
Uses only local file storage and basic algorithms
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import json

from ..storage.local_storage import get_storage

logger = logging.getLogger(__name__)


class SimpleClustering:
    """Database-free clustering pipeline using only local storage"""
    
    def __init__(self):
        self.storage = get_storage()
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.algorithm = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded data with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, 
                       scale: bool = True, 
                       reduce_dims: bool = False, 
                       n_components: int = 50) -> np.ndarray:
        """Preprocess data with optional scaling and dimensionality reduction"""
        
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found in the data")
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Scale data if requested
        if scale:
            scaled_data = self.scaler.fit_transform(numeric_data)
            logger.info("Data scaled using StandardScaler")
        else:
            scaled_data = numeric_data.values
        
        # Reduce dimensions if requested and data is high-dimensional
        if reduce_dims and scaled_data.shape[1] > n_components:
            self.pca = PCA(n_components=n_components, random_state=42)
            processed_data = self.pca.fit_transform(scaled_data)
            logger.info(f"Reduced dimensions from {scaled_data.shape[1]} to {n_components}")
        else:
            processed_data = scaled_data
        
        return processed_data
    
    def cluster_kmeans(self, data: np.ndarray, n_clusters: int = 3, **kwargs) -> Dict[str, Any]:
        """Perform K-Means clustering"""
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = model.fit_predict(data)
        
        results = {
            'algorithm': 'kmeans',
            'labels': labels.tolist(),
            'cluster_centers': model.cluster_centers_.tolist(),
            'n_clusters': n_clusters,
            'parameters': {'n_clusters': n_clusters, **kwargs}
        }
        
        self.model = model
        self.algorithm = 'kmeans'
        
        return results
    
    def cluster_dbscan(self, data: np.ndarray, eps: float = 0.5, min_samples: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform DBSCAN clustering"""
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = model.fit_predict(data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        results = {
            'algorithm': 'dbscan',
            'labels': labels.tolist(),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'parameters': {'eps': eps, 'min_samples': min_samples, **kwargs}
        }
        
        self.model = model
        self.algorithm = 'dbscan'
        
        return results
    
    def cluster_hierarchical(self, data: np.ndarray, n_clusters: int = 3, linkage: str = 'ward', **kwargs) -> Dict[str, Any]:
        """Perform Hierarchical clustering"""
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)
        labels = model.fit_predict(data)
        
        results = {
            'algorithm': 'hierarchical',
            'labels': labels.tolist(),
            'n_clusters': n_clusters,
            'parameters': {'n_clusters': n_clusters, 'linkage': linkage, **kwargs}
        }
        
        self.model = model
        self.algorithm = 'hierarchical'
        
        return results
    
    def evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality"""
        metrics = {}
        
        # Only calculate metrics if we have more than one cluster
        unique_labels = set(labels)
        if len(unique_labels) > 1 and not all(label == -1 for label in unique_labels):
            try:
                # Silhouette Score
                metrics['silhouette_score'] = silhouette_score(data, labels)
            except:
                metrics['silhouette_score'] = 0.0
            
            try:
                # Calinski-Harabasz Score
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
        
        # Basic statistics
        metrics['n_clusters'] = len(unique_labels)
        metrics['n_noise'] = sum(1 for label in labels if label == -1)
        metrics['n_samples'] = len(labels)
        
        return metrics
    
    def run_clustering(self, filepath: str, algorithm: str = 'kmeans', 
                      preprocess: bool = True, **params) -> str:
        """Run complete clustering pipeline and save results"""
        
        # Start experiment
        experiment_data = {
            'algorithm': algorithm,
            'filepath': filepath,
            'parameters': params,
            'preprocess': preprocess,
            'started_at': datetime.now().isoformat()
        }
        
        experiment_id = self.storage.save_experiment(experiment_data)
        logger.info(f"Started experiment {experiment_id}")
        
        try:
            # Load data
            data = self.load_data(filepath)
            original_shape = data.shape
            
            # Preprocess data
            if preprocess:
                processed_data = self.preprocess_data(
                    data, 
                    scale=params.get('scale', True),
                    reduce_dims=params.get('reduce_dims', False),
                    n_components=params.get('n_components', 50)
                )
            else:
                # Just select numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                processed_data = numeric_data.fillna(numeric_data.mean()).values
            
            processed_shape = processed_data.shape
            
            # Run clustering
            if algorithm == 'kmeans':
                n_clusters = params.get('n_clusters', 3)
                results = self.cluster_kmeans(processed_data, n_clusters=n_clusters)
            elif algorithm == 'dbscan':
                eps = params.get('eps', 0.5)
                min_samples = params.get('min_samples', 5)
                results = self.cluster_dbscan(processed_data, eps=eps, min_samples=min_samples)
            elif algorithm == 'hierarchical':
                n_clusters = params.get('n_clusters', 3)
                linkage = params.get('linkage', 'ward')
                results = self.cluster_hierarchical(processed_data, n_clusters=n_clusters, linkage=linkage)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Evaluate results
            labels = np.array(results['labels'])
            metrics = self.evaluate_clustering(processed_data, labels)
            
            # Compile final results
            final_results = {
                **results,
                'metrics': metrics,
                'data_info': {
                    'original_shape': original_shape,
                    'processed_shape': processed_shape,
                    'preprocessing': {
                        'scaled': params.get('scale', True),
                        'pca_applied': processed_shape[1] != original_shape[1],
                        'n_components': processed_shape[1]
                    }
                },
                'experiment_id': experiment_id,
                'completed_at': datetime.now().isoformat()
            }
            
            # Save results
            results_id = self.storage.save_clustering_results(final_results, experiment_id)
            
            # Update experiment with completion
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'results_id': results_id,
                'status': 'completed',
                'metrics': metrics
            })
            self.storage.save_experiment(experiment_data)
            
            logger.info(f"Clustering completed successfully. Results ID: {results_id}")
            
            # Save cluster assignments with original data
            self.save_results_with_data(data, labels, results_id)
            
            return results_id
            
        except Exception as e:
            # Update experiment with error
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            self.storage.save_experiment(experiment_data)
            
            logger.error(f"Clustering failed: {e}")
            raise
    
    def save_results_with_data(self, original_data: pd.DataFrame, labels: np.ndarray, results_id: str):
        """Save original data with cluster labels"""
        try:
            # Add cluster labels to original data
            result_data = original_data.copy()
            result_data['cluster'] = labels
            
            # Save as CSV
            results_file = Path("results") / f"clustered_data_{results_id}.csv"
            result_data.to_csv(results_file, index=False)
            
            logger.info(f"Saved clustered data to {results_file}")
            
        except Exception as e:
            logger.warning(f"Could not save clustered data: {e}")
    
    def get_results_summary(self, results_id: str) -> Optional[Dict]:
        """Get a summary of clustering results"""
        results = self.storage.load_clustering_results(results_id)
        if not results:
            return None
        
        summary = {
            'results_id': results_id,
            'algorithm': results.get('algorithm'),
            'n_clusters': results.get('metrics', {}).get('n_clusters', 0),
            'silhouette_score': results.get('metrics', {}).get('silhouette_score', 0),
            'completed_at': results.get('completed_at'),
            'parameters': results.get('parameters', {}),
            'data_shape': results.get('data_info', {}).get('processed_shape', [0, 0])
        }
        
        return summary
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        return self.storage.list_experiments()
    
    def cleanup_old_results(self, days_old: int = 30):
        """Clean up old results and temporary files"""
        try:
            self.storage.cleanup_temp_files()
            self.storage.cleanup_expired_cache()
            logger.info("Cleaned up old files")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


# Convenience function for simple clustering
def quick_cluster(filepath: str, algorithm: str = 'kmeans', n_clusters: int = 3) -> str:
    """Quick clustering function for simple use cases"""
    clustering = SimpleClustering()
    
    params = {'n_clusters': n_clusters} if algorithm in ['kmeans', 'hierarchical'] else {}
    
    return clustering.run_clustering(filepath, algorithm=algorithm, **params)
