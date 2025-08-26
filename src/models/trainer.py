"""
Model trainer for deep clustering models
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
import mlflow.pytorch

from src.models.deep_embedding import DECModel, DECClustering
from src.utils.logger import ClusteringLogger
from src.config import settings


class ModelTrainer:
    """Trainer for deep clustering models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = ClusteringLogger("model_trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu")
        
        # Initialize MLflow
        if config.get("mlflow", {}).get("enabled", False):
            mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "file:./mlruns"))
            mlflow.set_experiment(config["mlflow"].get("experiment_name", "clustering_experiments"))
    
    @classmethod
    def from_config(cls, config_path: str) -> "ModelTrainer":
        """Create trainer from configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    async def train(self):
        """Train the model"""
        self.logger.progress("Starting model training...")
        
        model_config = self.config["model"]
        training_config = self.config["training"]
        data_config = self.config["data"]
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params({
                "algorithm": model_config["algorithm"],
                "n_clusters": model_config["n_clusters"],
                "latent_dim": model_config.get("latent_dim", 10),
                "learning_rate": training_config["learning_rate"],
                "epochs": training_config["epochs"],
                "batch_size": training_config["batch_size"]
            })
            
            try:
                # Load and prepare data
                X = await self._load_training_data(data_config)
                
                # Initialize model based on algorithm
                if model_config["algorithm"] == "deep_embedding":
                    model = await self._train_dec_model(X, model_config, training_config)
                else:
                    raise ValueError(f"Unsupported algorithm: {model_config['algorithm']}")
                
                # Save model
                model_path = await self._save_model(model, model_config)
                
                # Log model artifact
                mlflow.pytorch.log_model(model, "model")
                mlflow.log_artifact(str(model_path))
                
                self.logger.success(f"Model training completed. Saved to {model_path}")
                
            except Exception as e:
                self.logger.failure(f"Training failed: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise
    
    async def _load_training_data(self, data_config: Dict[str, Any]) -> np.ndarray:
        """Load training data"""
        data_path = Path(data_config["path"])
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        # Use the clustering pipeline's data loading logic
        from src.clustering.pipeline import ClusteringPipeline
        pipeline = ClusteringPipeline()
        
        X, _ = await pipeline._load_data(data_path)
        X_scaled = await pipeline._preprocess_data(X)
        
        return X_scaled
    
    async def _train_dec_model(
        self, X: np.ndarray, model_config: Dict[str, Any], training_config: Dict[str, Any]
    ) -> DECModel:
        """Train Deep Embedding Clustering model"""
        
        # Model parameters
        n_clusters = model_config["n_clusters"]
        latent_dim = model_config.get("latent_dim", 10)
        alpha = model_config.get("alpha", 1.0)
        
        # Training parameters
        epochs = training_config["epochs"]
        pretrain_epochs = training_config.get("pretrain_epochs", 100)
        batch_size = training_config["batch_size"]
        learning_rate = training_config["learning_rate"]
        
        # Initialize clustering wrapper
        dec = DECClustering(
            n_clusters=n_clusters,
            latent_dim=latent_dim,
            alpha=alpha
        )
        
        # Train model
        labels, embeddings, metrics = dec.fit(
            X, 
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            pretrain_epochs=pretrain_epochs
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        return dec.model
    
    async def _save_model(self, model: nn.Module, model_config: Dict[str, Any]) -> Path:
        """Save trained model"""
        
        # Create model directory
        models_dir = Path(settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm = model_config["algorithm"]
        model_name = f"{algorithm}_{timestamp}.pth"
        model_path = models_dir / model_name
        
        # Save model state
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "timestamp": timestamp
        }, model_path)
        
        # Save configuration
        config_path = models_dir / f"{algorithm}_{timestamp}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return model_path


class ModelEvaluator:
    """Evaluator for trained clustering models"""
    
    def __init__(self):
        self.logger = ClusteringLogger("model_evaluator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def evaluate_model(
        self, model_path: Path, test_data: np.ndarray, true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate a trained model"""
        
        self.logger.progress(f"Evaluating model from {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint["model_config"]
        
        # Initialize model
        if model_config["algorithm"] == "deep_embedding":
            model = DECModel(
                input_dim=test_data.shape[1],
                n_clusters=model_config["n_clusters"],
                latent_dim=model_config.get("latent_dim", 10),
                alpha=model_config.get("alpha", 1.0)
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
        else:
            raise ValueError(f"Unsupported algorithm: {model_config['algorithm']}")
        
        # Prepare data
        test_tensor = torch.FloatTensor(test_data).to(self.device)
        dataset = torch.utils.data.TensorDataset(test_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Get predictions
        labels = model.predict(data_loader)
        embeddings = model.get_embeddings(data_loader)
        
        # Calculate metrics
        from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
        
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        
        if true_labels is not None:
            metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, labels)
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(true_labels, labels)
        
        metrics["n_clusters"] = len(np.unique(labels))
        
        self.logger.success("Model evaluation completed")
        return metrics
