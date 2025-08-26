"""
Configuration settings for the clustering framework
Database-free version using only local file storage
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings - Database-free configuration"""
    
    # Application
    app_name: str = "High-Dimensional Clustering Framework (Database-Free)"
    version: str = "1.0.0"
    debug: bool = False
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Local Storage Only (No Cloud/Database Dependencies)
    use_local_storage: bool = True
    use_database: bool = False
    use_cloud_storage: bool = False
    
    # Local File Storage Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    cache_dir: Path = Path("cache")
    logs_dir: Path = Path("logs")
    temp_dir: Path = Path("temp")
    
    # Local Database Alternative (JSON/CSV files)
    metadata_file: Path = Path("metadata.json")
    experiments_file: Path = Path("experiments.json")
    models_registry_file: Path = Path("models_registry.json")
    
    # Machine Learning
    default_algorithm: str = "kmeans"  # Start with simple algorithm
    max_dimensions: int = 1000  # Reduced for local processing
    batch_size: int = 256  # Smaller batches for local memory
    max_epochs: int = 50  # Fewer epochs for faster local training
    learning_rate: float = 0.001
    
    # Clustering Parameters
    min_cluster_size: int = 5
    max_clusters: int = 20  # Reduced for local processing
    similarity_threshold: float = 0.8
    
    # Performance (Local Machine Optimized)
    n_jobs: int = -1
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_port: int = 8080
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ClusteringConfig:
    """Configuration for different clustering algorithms"""
    
    ALGORITHMS = {
        "kmeans": {
            "name": "K-Means",
            "parameters": {
                "n_clusters": {"type": int, "default": 8, "min": 2, "max": 100},
                "max_iter": {"type": int, "default": 300, "min": 1, "max": 1000},
                "tol": {"type": float, "default": 1e-4, "min": 1e-6, "max": 1e-2}
            }
        },
        "dbscan": {
            "name": "DBSCAN",
            "parameters": {
                "eps": {"type": float, "default": 0.5, "min": 0.1, "max": 2.0},
                "min_samples": {"type": int, "default": 5, "min": 1, "max": 50}
            }
        },
        "spectral": {
            "name": "Spectral Clustering",
            "parameters": {
                "n_clusters": {"type": int, "default": 8, "min": 2, "max": 100},
                "gamma": {"type": float, "default": 1.0, "min": 0.1, "max": 10.0}
            }
        },
        "deep_embedding": {
            "name": "Deep Embedding Clustering",
            "parameters": {
                "n_clusters": {"type": int, "default": 8, "min": 2, "max": 100},
                "latent_dim": {"type": int, "default": 10, "min": 2, "max": 100},
                "alpha": {"type": float, "default": 1.0, "min": 0.1, "max": 10.0}
            }
        },
        "vade": {
            "name": "Variational Deep Embedding",
            "parameters": {
                "n_clusters": {"type": int, "default": 8, "min": 2, "max": 100},
                "latent_dim": {"type": int, "default": 10, "min": 2, "max": 100},
                "beta": {"type": float, "default": 1.0, "min": 0.1, "max": 10.0}
            }
        },
        "hdbscan": {
            "name": "HDBSCAN",
            "parameters": {
                "min_cluster_size": {"type": int, "default": 5, "min": 2, "max": 100},
                "min_samples": {"type": int, "default": None, "min": 1, "max": 50}
            }
        }
    }


class AzureConfig:
    """Azure-specific configuration"""
    
    SERVICES = {
        "ml_workspace": {
            "name": "ml-clustering",
            "sku": "Basic"
        },
        "storage_account": {
            "name": "stclusteringdata",
            "sku": "Standard_LRS",
            "kind": "StorageV2"
        },
        "cosmos_db": {
            "name": "cosmos-clustering",
            "api": "Core (SQL)"
        },
        "container_registry": {
            "name": "crclusteringmodels",
            "sku": "Basic"
        },
        "app_insights": {
            "name": "ai-clustering"
        }
    }


# Global settings instance
settings = Settings()
