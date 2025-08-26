"""
Local File Storage Manager - Database-Free Implementation
Replaces database with JSON/CSV files for metadata and results storage
"""

import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import pickle
import logging

logger = logging.getLogger(__name__)


class LocalStorageManager:
    """Manages local file storage as database replacement"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        dirs = ['data', 'models', 'results', 'cache', 'logs', 'temp', 'experiments']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
    
    # Experiment Management (replaces database tables)
    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Save experiment metadata to JSON file"""
        experiment_id = str(uuid.uuid4())
        experiment_data['id'] = experiment_id
        experiment_data['created_at'] = datetime.now().isoformat()
        
        experiments_file = self.base_dir / "experiments.json"
        
        # Load existing experiments
        experiments = self.load_experiments()
        experiments[experiment_id] = experiment_data
        
        # Save back to file
        with open(experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
        
        logger.info(f"Saved experiment {experiment_id}")
        return experiment_id
    
    def load_experiments(self) -> Dict[str, Dict]:
        """Load all experiments from JSON file"""
        experiments_file = self.base_dir / "experiments.json"
        
        if not experiments_file.exists():
            return {}
        
        try:
            with open(experiments_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted experiments file, starting fresh")
            return {}
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get specific experiment by ID"""
        experiments = self.load_experiments()
        return experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        experiments = self.load_experiments()
        return list(experiments.values())
    
    # Model Registry (replaces model database)
    def save_model_metadata(self, model_data: Dict[str, Any]) -> str:
        """Save model metadata"""
        model_id = str(uuid.uuid4())
        model_data['id'] = model_id
        model_data['created_at'] = datetime.now().isoformat()
        
        models_file = self.base_dir / "models_registry.json"
        
        # Load existing models
        models = self.load_models_registry()
        models[model_id] = model_data
        
        # Save back to file
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2, default=str)
        
        return model_id
    
    def load_models_registry(self) -> Dict[str, Dict]:
        """Load models registry"""
        models_file = self.base_dir / "models_registry.json"
        
        if not models_file.exists():
            return {}
        
        try:
            with open(models_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict]:
        """Get model metadata by ID"""
        models = self.load_models_registry()
        return models.get(model_id)
    
    # Data Management
    def save_dataset(self, data: pd.DataFrame, name: str, metadata: Dict = None) -> str:
        """Save dataset as CSV with metadata"""
        dataset_id = str(uuid.uuid4())
        filename = f"{name}_{dataset_id}.csv"
        filepath = self.base_dir / "data" / filename
        
        # Save data
        data.to_csv(filepath, index=False)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'id': dataset_id,
            'name': name,
            'filename': filename,
            'created_at': datetime.now().isoformat(),
            'shape': data.shape,
            'columns': list(data.columns)
        })
        
        self.save_dataset_metadata(dataset_id, metadata)
        
        logger.info(f"Saved dataset {name} with ID {dataset_id}")
        return dataset_id
    
    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset by ID"""
        metadata = self.get_dataset_metadata(dataset_id)
        if not metadata:
            return None
        
        filepath = self.base_dir / "data" / metadata['filename']
        if not filepath.exists():
            logger.error(f"Dataset file not found: {filepath}")
            return None
        
        return pd.read_csv(filepath)
    
    def save_dataset_metadata(self, dataset_id: str, metadata: Dict):
        """Save dataset metadata"""
        datasets_file = self.base_dir / "datasets_metadata.json"
        
        # Load existing datasets
        datasets = {}
        if datasets_file.exists():
            try:
                with open(datasets_file, 'r') as f:
                    datasets = json.load(f)
            except json.JSONDecodeError:
                pass
        
        datasets[dataset_id] = metadata
        
        # Save back to file
        with open(datasets_file, 'w') as f:
            json.dump(datasets, f, indent=2, default=str)
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset metadata by ID"""
        datasets_file = self.base_dir / "datasets_metadata.json"
        
        if not datasets_file.exists():
            return None
        
        try:
            with open(datasets_file, 'r') as f:
                datasets = json.load(f)
                return datasets.get(dataset_id)
        except json.JSONDecodeError:
            return None
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets"""
        datasets_file = self.base_dir / "datasets_metadata.json"
        
        if not datasets_file.exists():
            return []
        
        try:
            with open(datasets_file, 'r') as f:
                datasets = json.load(f)
                return list(datasets.values())
        except json.JSONDecodeError:
            return []
    
    # Results Storage
    def save_clustering_results(self, results: Dict[str, Any], experiment_id: str) -> str:
        """Save clustering results"""
        results_id = str(uuid.uuid4())
        results['id'] = results_id
        results['experiment_id'] = experiment_id
        results['created_at'] = datetime.now().isoformat()
        
        # Save main results as JSON
        results_file = self.base_dir / "results" / f"results_{results_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save cluster labels if present
        if 'labels' in results:
            labels_file = self.base_dir / "results" / f"labels_{results_id}.csv"
            pd.Series(results['labels']).to_csv(labels_file, index=False)
        
        # Save cluster centers if present
        if 'cluster_centers' in results:
            centers_file = self.base_dir / "results" / f"centers_{results_id}.csv"
            pd.DataFrame(results['cluster_centers']).to_csv(centers_file, index=False)
        
        logger.info(f"Saved clustering results {results_id}")
        return results_id
    
    def load_clustering_results(self, results_id: str) -> Optional[Dict]:
        """Load clustering results by ID"""
        results_file = self.base_dir / "results" / f"results_{results_id}.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Load additional files if they exist
            labels_file = self.base_dir / "results" / f"labels_{results_id}.csv"
            if labels_file.exists():
                results['labels'] = pd.read_csv(labels_file).iloc[:, 0].tolist()
            
            centers_file = self.base_dir / "results" / f"centers_{results_id}.csv"
            if centers_file.exists():
                results['cluster_centers'] = pd.read_csv(centers_file).values.tolist()
            
            return results
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading results {results_id}: {e}")
            return None
    
    # Model Storage
    def save_model(self, model: Any, model_id: str) -> bool:
        """Save trained model using pickle"""
        try:
            model_file = self.base_dir / "models" / f"model_{model_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load trained model"""
        try:
            model_file = self.base_dir / "models" / f"model_{model_id}.pkl"
            if not model_file.exists():
                return None
            
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    # Cache Management
    def save_cache(self, key: str, data: Any, expiry_hours: int = 24) -> bool:
        """Save data to cache"""
        try:
            cache_data = {
                'data': data,
                'created_at': datetime.now().isoformat(),
                'expiry_hours': expiry_hours
            }
            
            cache_file = self.base_dir / "cache" / f"cache_{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving cache {key}: {e}")
            return False
    
    def load_cache(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        try:
            cache_file = self.base_dir / "cache" / f"cache_{key}.pkl"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check expiry
            created_at = datetime.fromisoformat(cache_data['created_at'])
            expiry_hours = cache_data.get('expiry_hours', 24)
            
            if (datetime.now() - created_at).total_seconds() > expiry_hours * 3600:
                # Cache expired
                cache_file.unlink()
                return None
            
            return cache_data['data']
        except Exception as e:
            logger.error(f"Error loading cache {key}: {e}")
            return None
    
    # Cleanup Methods
    def cleanup_temp_files(self):
        """Clean temporary files"""
        temp_dir = self.base_dir / "temp"
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {file}: {e}")
    
    def cleanup_expired_cache(self):
        """Clean expired cache files"""
        cache_dir = self.base_dir / "cache"
        for cache_file in cache_dir.glob("cache_*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                created_at = datetime.fromisoformat(cache_data['created_at'])
                expiry_hours = cache_data.get('expiry_hours', 24)
                
                if (datetime.now() - created_at).total_seconds() > expiry_hours * 3600:
                    cache_file.unlink()
                    logger.info(f"Cleaned expired cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")


# Global storage manager instance
storage = LocalStorageManager()


def get_storage() -> LocalStorageManager:
    """Get the global storage manager instance"""
    return storage
