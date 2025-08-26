"""
Test cases for the clustering pipeline
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import asyncio

from src.clustering.pipeline import ClusteringPipeline
from src.config import settings


class TestClusteringPipeline:
    """Test the main clustering pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        
        # Generate blob data
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=100, centers=3, n_features=10, random_state=42)
        
        return X, y
    
    @pytest.fixture
    def sample_csv_file(self, sample_data):
        """Create a temporary CSV file with sample data"""
        X, y = sample_data
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["true_label"] = y
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return Path(f.name)
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return ClusteringPipeline()
    
    @pytest.mark.asyncio
    async def test_load_data(self, pipeline, sample_csv_file):
        """Test data loading functionality"""
        X, feature_names = await pipeline._load_data(sample_csv_file)
        
        assert X.shape[0] == 100  # 100 samples
        assert X.shape[1] == 10   # 10 features (excluding true_label)
        assert len(feature_names) == 10
        assert all(name.startswith("feature_") for name in feature_names)
    
    @pytest.mark.asyncio
    async def test_preprocess_data(self, pipeline, sample_data):
        """Test data preprocessing"""
        X, _ = sample_data
        X_processed = await pipeline._preprocess_data(X)
        
        # Check standardization
        assert np.allclose(X_processed.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_processed.std(axis=0), 1, atol=1e-10)
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering(self, pipeline, sample_data):
        """Test K-means clustering"""
        X, _ = sample_data
        X_scaled = pipeline.scaler.fit_transform(X)
        
        labels, centers, metrics = await pipeline._kmeans_clustering(X_scaled, n_clusters=3)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 3
        assert centers.shape == (3, 10)
        assert "inertia" in metrics
        assert "n_iter" in metrics
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering(self, pipeline, sample_data):
        """Test DBSCAN clustering"""
        X, _ = sample_data
        X_scaled = pipeline.scaler.fit_transform(X)
        
        labels, embeddings, metrics = await pipeline._dbscan_clustering(X_scaled, eps=0.5, min_samples=5)
        
        assert len(labels) == 100
        assert "n_clusters" in metrics
        assert "n_noise" in metrics
        assert "noise_ratio" in metrics
    
    @pytest.mark.asyncio
    async def test_determine_optimal_clusters(self, pipeline, sample_data):
        """Test optimal cluster determination"""
        X, _ = sample_data
        X_scaled = pipeline.scaler.fit_transform(X)
        
        optimal_k = await pipeline._determine_optimal_clusters(X_scaled)
        
        assert isinstance(optimal_k, int)
        assert 2 <= optimal_k <= settings.max_clusters
    
    @pytest.mark.asyncio
    async def test_calculate_metrics(self, pipeline, sample_data):
        """Test metrics calculation"""
        X, y = sample_data
        X_scaled = pipeline.scaler.fit_transform(X)
        
        metrics = await pipeline._calculate_metrics(X_scaled, y, X_scaled)
        
        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "n_clusters" in metrics
        assert "n_samples" in metrics
        
        assert metrics["n_clusters"] == len(np.unique(y))
        assert metrics["n_samples"] == 100
    
    @pytest.mark.asyncio
    async def test_full_pipeline_kmeans(self, pipeline, sample_csv_file):
        """Test the complete pipeline with K-means"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            results = await pipeline.run(
                data_path=sample_csv_file,
                algorithm="kmeans",
                n_clusters=3,
                output_dir=output_dir
            )
            
            # Check results structure
            assert "algorithm" in results
            assert "n_clusters" in results
            assert "labels" in results
            assert "metrics" in results
            assert "feature_names" in results
            assert "timestamp" in results
            
            assert results["algorithm"] == "kmeans"
            assert results["n_clusters"] == 3
            assert len(results["labels"]) == 100
            assert len(results["feature_names"]) == 10
            
            # Check that files were saved
            assert any(output_dir.glob("*.json"))
            assert any(output_dir.glob("*_labels.csv"))
    
    @pytest.mark.asyncio
    async def test_unsupported_algorithm(self, pipeline, sample_csv_file):
        """Test error handling for unsupported algorithm"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            with pytest.raises(ValueError, match="Unknown algorithm"):
                await pipeline.run(
                    data_path=sample_csv_file,
                    algorithm="unknown_algorithm",
                    output_dir=output_dir
                )
