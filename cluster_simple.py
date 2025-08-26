"""
Standalone Database-Free Clustering Script
Everything in one file - no imports needed!
"""

import click
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import uuid
import pickle
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalStorage:
    """Simple local storage manager"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories"""
        dirs = ['data', 'results', 'cache', 'logs', 'temp']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
    
    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Save experiment to JSON file"""
        experiment_id = str(uuid.uuid4())[:8]  # Short ID
        experiment_data['id'] = experiment_id
        experiment_data['created_at'] = datetime.now().isoformat()
        
        experiments_file = self.base_dir / "experiments.json"
        
        # Load existing experiments
        experiments = {}
        if experiments_file.exists():
            try:
                with open(experiments_file, 'r') as f:
                    experiments = json.load(f)
            except json.JSONDecodeError:
                pass
        
        experiments[experiment_id] = experiment_data
        
        # Save back to file
        with open(experiments_file, 'w') as f:
            json.dump(experiments, f, indent=2, default=str)
        
        return experiment_id
    
    def load_experiments(self) -> Dict[str, Dict]:
        """Load all experiments"""
        experiments_file = self.base_dir / "experiments.json"
        
        if not experiments_file.exists():
            return {}
        
        try:
            with open(experiments_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def save_results(self, results: Dict[str, Any], experiment_id: str) -> str:
        """Save clustering results"""
        results_id = str(uuid.uuid4())[:8]
        results['id'] = results_id
        results['experiment_id'] = experiment_id
        results['created_at'] = datetime.now().isoformat()
        
        # Save results as JSON
        results_file = self.base_dir / "results" / f"results_{results_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results_id
    
    def load_results(self, results_id: str) -> Optional[Dict]:
        """Load results by ID"""
        results_file = self.base_dir / "results" / f"results_{results_id}.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None


class SimpleClustering:
    """Simple clustering implementation"""
    
    def __init__(self):
        self.storage = LocalStorage()
        self.scaler = StandardScaler()
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Simple data preprocessing"""
        # Select numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found")
        
        # Fill missing values with mean
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        return scaled_data
    
    def run_clustering(self, filepath: str, algorithm: str = 'kmeans', **params) -> str:
        """Run clustering and save results"""
        
        # Start experiment
        experiment_data = {
            'algorithm': algorithm,
            'filepath': filepath,
            'parameters': params,
            'started_at': datetime.now().isoformat()
        }
        
        experiment_id = self.storage.save_experiment(experiment_data)
        click.echo(f"📊 Started experiment: {experiment_id}")
        
        try:
            # Load data
            data = pd.read_csv(filepath)
            click.echo(f"📈 Loaded data: {data.shape}")
            
            # Preprocess
            processed_data = self.preprocess_data(data)
            click.echo(f"🔧 Preprocessed data: {processed_data.shape}")
            
            # Run clustering
            if algorithm == 'kmeans':
                n_clusters = params.get('n_clusters', 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(processed_data)
                
                results = {
                    'algorithm': 'kmeans',
                    'labels': labels.tolist(),
                    'cluster_centers': model.cluster_centers_.tolist(),
                    'n_clusters': n_clusters,
                    'parameters': {'n_clusters': n_clusters}
                }
                
            elif algorithm == 'dbscan':
                eps = params.get('eps', 0.5)
                min_samples = params.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(processed_data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                results = {
                    'algorithm': 'dbscan',
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'parameters': {'eps': eps, 'min_samples': min_samples}
                }
                
            elif algorithm == 'hierarchical':
                n_clusters = params.get('n_clusters', 3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(processed_data)
                
                results = {
                    'algorithm': 'hierarchical',
                    'labels': labels.tolist(),
                    'n_clusters': n_clusters,
                    'parameters': {'n_clusters': n_clusters}
                }
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Evaluate clustering
            labels_array = np.array(results['labels'])
            unique_labels = set(labels_array)
            
            metrics = {}
            if len(unique_labels) > 1 and not all(label == -1 for label in unique_labels):
                try:
                    metrics['silhouette_score'] = silhouette_score(processed_data, labels_array)
                except:
                    metrics['silhouette_score'] = 0.0
                    
                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(processed_data, labels_array)
                except:
                    metrics['calinski_harabasz_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
            
            metrics['n_clusters'] = len(unique_labels)
            metrics['n_samples'] = len(labels_array)
            
            # Add metrics to results
            results['metrics'] = metrics
            results['data_shape'] = data.shape
            results['processed_shape'] = processed_data.shape
            
            # Save results
            results_id = self.storage.save_results(results, experiment_id)
            
            # Save clustered data as CSV
            clustered_data = data.copy()
            clustered_data['cluster'] = labels_array
            output_file = Path("results") / f"clustered_data_{results_id}.csv"
            clustered_data.to_csv(output_file, index=False)
            
            # Update experiment
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'results_id': results_id,
                'status': 'completed',
                'metrics': metrics
            })
            self.storage.save_experiment(experiment_data)
            
            click.echo(f"✅ Clustering completed!")
            click.echo(f"   Results ID: {results_id}")
            click.echo(f"   Clusters: {metrics['n_clusters']}")
            click.echo(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
            click.echo(f"   Output: {output_file}")
            
            return results_id
            
        except Exception as e:
            # Update experiment with error
            experiment_data.update({
                'completed_at': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            self.storage.save_experiment(experiment_data)
            
            raise


# CLI Commands
@click.group()
def cli():
    """Database-Free Clustering Framework"""
    pass


@cli.command()
def generate_sample():
    """Generate sample data for testing"""
    click.echo("🎲 Generating sample data...")
    
    try:
        # Generate synthetic data with 3 clear clusters
        np.random.seed(42)
        
        # Create 3 clusters in 2D space
        cluster1 = np.random.normal([2, 2], 0.8, (100, 2))
        cluster2 = np.random.normal([6, 6], 0.8, (100, 2))
        cluster3 = np.random.normal([2, 6], 0.8, (100, 2))
        
        # Combine clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        
        # Add some additional features
        feature3 = data[:, 0] + data[:, 1] + np.random.normal(0, 0.1, 300)
        feature4 = data[:, 0] * data[:, 1] + np.random.normal(0, 0.5, 300)
        feature5 = np.random.normal(0, 1, 300)  # Random noise feature
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature_1': data[:, 0],
            'feature_2': data[:, 1], 
            'feature_3': feature3,
            'feature_4': feature4,
            'feature_5': feature5
        })
        
        # Save to file
        Path("data").mkdir(exist_ok=True)
        filepath = Path("data") / "sample_data.csv"
        df.to_csv(filepath, index=False)
        
        click.echo(f"✅ Sample data generated: {filepath}")
        click.echo(f"   Shape: {df.shape}")
        click.echo(f"   Features: {list(df.columns)}")
        click.echo("   This data has 3 natural clusters")
        
    except Exception as e:
        click.echo(f"❌ Failed to generate sample data: {e}")
        raise


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--algorithm', default='kmeans', 
              type=click.Choice(['kmeans', 'dbscan', 'hierarchical']),
              help='Clustering algorithm')
@click.option('--n-clusters', default=3, help='Number of clusters (kmeans/hierarchical)')
@click.option('--eps', default=0.5, help='DBSCAN eps parameter') 
@click.option('--min-samples', default=5, help='DBSCAN min_samples parameter')
def cluster(filepath, algorithm, n_clusters, eps, min_samples):
    """Run clustering on a CSV file"""
    click.echo(f"🔬 Running {algorithm} clustering on {filepath}")
    
    try:
        clustering = SimpleClustering()
        
        # Prepare parameters
        if algorithm == 'kmeans':
            params = {'n_clusters': n_clusters}
        elif algorithm == 'dbscan':
            params = {'eps': eps, 'min_samples': min_samples}
        elif algorithm == 'hierarchical':
            params = {'n_clusters': n_clusters}
        else:
            params = {}
        
        # Run clustering
        results_id = clustering.run_clustering(filepath, algorithm=algorithm, **params)
        
    except Exception as e:
        click.echo(f"❌ Clustering failed: {e}")
        raise


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--n-clusters', default=3, help='Number of clusters')
def quick(filepath, n_clusters):
    """Quick K-means clustering with default settings"""
    click.echo(f"⚡ Quick clustering on {filepath}")
    
    try:
        clustering = SimpleClustering()
        results_id = clustering.run_clustering(filepath, algorithm='kmeans', n_clusters=n_clusters)
        
    except Exception as e:
        click.echo(f"❌ Quick clustering failed: {e}")
        raise


@cli.command()
def experiments():
    """List all experiments"""
    try:
        storage = LocalStorage()
        experiments = storage.load_experiments()
        
        if not experiments:
            click.echo("No experiments found.")
            return
        
        click.echo(f"📊 Found {len(experiments)} experiments:")
        click.echo()
        
        for exp_id, exp in experiments.items():
            status = exp.get('status', 'unknown')
            click.echo(f"Experiment: {exp_id}")
            click.echo(f"  Algorithm: {exp.get('algorithm', 'unknown')}")
            click.echo(f"  Status: {status}")
            click.echo(f"  Created: {exp.get('started_at', 'unknown')}")
            
            if status == 'completed':
                metrics = exp.get('metrics', {})
                click.echo(f"  Clusters: {metrics.get('n_clusters', 0)}")
                click.echo(f"  Silhouette: {metrics.get('silhouette_score', 0):.3f}")
                click.echo(f"  Results ID: {exp.get('results_id', 'unknown')}")
            
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Failed to list experiments: {e}")


@cli.command()
@click.argument('results_id')
def results(results_id):
    """Show detailed results for an experiment"""
    try:
        storage = LocalStorage()
        results = storage.load_results(results_id)
        
        if not results:
            click.echo(f"❌ Results not found: {results_id}")
            return
        
        click.echo(f"📋 Results for {results_id}:")
        click.echo(f"  Algorithm: {results.get('algorithm')}")
        click.echo(f"  Completed: {results.get('created_at')}")
        
        metrics = results.get('metrics', {})
        click.echo(f"  Clusters: {metrics.get('n_clusters')}")
        click.echo(f"  Silhouette Score: {metrics.get('silhouette_score', 0):.3f}")
        click.echo(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.3f}")
        click.echo(f"  Data shape: {results.get('data_shape')}")
        
        # Check if CSV file exists
        csv_file = Path("results") / f"clustered_data_{results_id}.csv"
        if csv_file.exists():
            click.echo(f"  CSV file: {csv_file}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load results: {e}")


@cli.command()
def status():
    """Show system status"""
    try:
        storage = LocalStorage()
        
        click.echo("🏥 System Status:")
        click.echo(f"  Base directory: {storage.base_dir}")
        
        # Check directories
        dirs = ['data', 'results', 'cache', 'logs', 'temp']
        for dir_name in dirs:
            dir_path = storage.base_dir / dir_name
            exists = "✅" if dir_path.exists() else "❌"
            count = len(list(dir_path.glob("*"))) if dir_path.exists() else 0
            click.echo(f"  {dir_name}/: {exists} ({count} files)")
        
        # Count experiments
        experiments = storage.load_experiments()
        click.echo(f"  Total experiments: {len(experiments)}")
        
        # Count completed experiments
        completed = sum(1 for exp in experiments.values() if exp.get('status') == 'completed')
        click.echo(f"  Completed: {completed}")
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")


@cli.command()
def cleanup():
    """Clean up temporary files"""
    try:
        # Clean temp directory
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
        
        # Clean cache directory
        cache_dir = Path("cache") 
        if cache_dir.exists():
            for file in cache_dir.glob("*"):
                file.unlink()
        
        click.echo("🧹 Cleanup completed")
        
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}")


if __name__ == '__main__':
    cli()
