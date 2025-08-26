"""
Deep Embedding Clustering (DEC) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional

from src.utils.logger import ClusteringLogger


class AutoEncoder(nn.Module):
    """Autoencoder for feature learning"""
    
    def __init__(self, input_dim: int, latent_dim: int = 10):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DECModel(nn.Module):
    """Deep Embedding Clustering Model"""
    
    def __init__(self, input_dim: int, n_clusters: int, latent_dim: int = 10, alpha: float = 1.0):
        super(DECModel, self).__init__()
        
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        
        # Autoencoder
        self.autoencoder = AutoEncoder(input_dim, latent_dim)
        
        # Cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
        
        self.logger = ClusteringLogger("dec_model")
    
    def forward(self, x):
        """Forward pass"""
        encoded, decoded = self.autoencoder(x)
        
        # Compute soft assignments
        q = self._soft_assignment(encoded)
        
        return encoded, decoded, q
    
    def _soft_assignment(self, encoded):
        """Compute soft cluster assignments"""
        # Student's t-distribution
        distances = torch.sum((encoded.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def target_distribution(self, q):
        """Compute target distribution P"""
        weight = q ** 2 / torch.sum(q, dim=0)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p
    
    def pretrain(self, data_loader, epochs: int = 100, lr: float = 0.001):
        """Pretrain the autoencoder"""
        self.logger.progress("Pretraining autoencoder...")
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                
                _, decoded = self.autoencoder(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                self.logger.info(f"Pretrain Epoch {epoch}, Loss: {total_loss:.4f}")
        
        self.logger.success("Autoencoder pretraining completed")
    
    def initialize_clusters(self, data_loader):
        """Initialize cluster centers using K-means"""
        self.logger.progress("Initializing cluster centers...")
        
        self.autoencoder.eval()
        encoded_data = []
        
        with torch.no_grad():
            for batch in data_loader:
                encoded, _ = self.autoencoder(batch)
                encoded_data.append(encoded.cpu().numpy())
        
        encoded_data = np.vstack(encoded_data)
        
        # Use K-means to initialize cluster centers
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(encoded_data)
        
        self.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        )
        
        self.logger.success("Cluster centers initialized")
        return kmeans.labels_
    
    def fit(self, data_loader, epochs: int = 100, lr: float = 0.001, 
            pretrain_epochs: int = 100):
        """Train the DEC model"""
        self.logger.progress("Starting DEC training...")
        
        # Pretrain autoencoder
        self.pretrain(data_loader, pretrain_epochs, lr)
        
        # Initialize clusters
        initial_labels = self.initialize_clusters(data_loader)
        
        # Main training loop
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                encoded, decoded, q = self.forward(batch)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(decoded, batch)
                
                # Clustering loss (KL divergence)
                p = self.target_distribution(q)
                cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')
                
                # Total loss
                loss = recon_loss + cluster_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        self.logger.success("DEC training completed")
    
    def predict(self, data_loader):
        """Predict cluster assignments"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                _, _, q = self.forward(batch)
                pred = torch.argmax(q, dim=1)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def get_embeddings(self, data_loader):
        """Get learned embeddings"""
        self.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in data_loader:
                encoded, _, _ = self.forward(batch)
                embeddings.append(encoded.cpu().numpy())
        
        return np.vstack(embeddings)


class DECClustering:
    """Deep Embedding Clustering wrapper"""
    
    def __init__(self, n_clusters: int, latent_dim: int = 10, alpha: float = 1.0):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = ClusteringLogger("dec_clustering")
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 256,
            lr: float = 0.001, pretrain_epochs: int = 100):
        """Fit the DEC model"""
        self.logger.progress(f"Starting DEC clustering with {self.n_clusters} clusters")
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = DECModel(
            input_dim=input_dim,
            n_clusters=self.n_clusters,
            latent_dim=self.latent_dim,
            alpha=self.alpha
        ).to(self.device)
        
        # Train model
        self.model.fit(data_loader, epochs, lr, pretrain_epochs)
        
        # Get final predictions
        labels = self.model.predict(data_loader)
        embeddings = self.model.get_embeddings(data_loader)
        
        # Calculate metrics
        silhouette = silhouette_score(embeddings, labels) if len(np.unique(labels)) > 1 else -1
        
        self.logger.success(f"DEC clustering completed. Silhouette score: {silhouette:.3f}")
        
        return labels, embeddings, {"silhouette_score": silhouette}
    
    def predict(self, X: np.ndarray):
        """Predict cluster assignments for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        
        return self.model.predict(data_loader)
