"""
Simple Main Entry Point - Database-Free Version
"""

import click
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clustering.simple_pipeline import SimpleClustering, quick_cluster
from storage.local_storage import get_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """High-Dimensional Clustering Framework (Database-Free)"""
    pass


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to run the server on')
@click.option('--port', default=8000, help='Port to run the server on')
def serve(host, port):
    """Start the clustering API server"""
    try:
        # Try to import and start server
        import uvicorn
        from api.simple_app import app
        
        click.echo(f"🚀 Starting clustering API server on {host}:{port}")
        click.echo("📖 API documentation available at: http://127.0.0.1:8000/docs")
        
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        click.echo("❌ FastAPI/Uvicorn not installed. Install with:")
        click.echo("pip install fastapi uvicorn[standard]")
    except Exception as e:
        click.echo(f"❌ Server failed to start: {e}")


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--algorithm', default='kmeans', 
              type=click.Choice(['kmeans', 'dbscan', 'hierarchical']),
              help='Clustering algorithm to use')
@click.option('--n-clusters', default=3, help='Number of clusters (for kmeans/hierarchical)')
@click.option('--eps', default=0.5, help='DBSCAN eps parameter')
@click.option('--min-samples', default=5, help='DBSCAN min_samples parameter')
@click.option('--no-preprocess', is_flag=True, help='Skip data preprocessing')
@click.option('--reduce-dims', is_flag=True, help='Apply PCA for dimensionality reduction')
@click.option('--n-components', default=50, help='Number of PCA components')
def cluster(filepath, algorithm, n_clusters, eps, min_samples, no_preprocess, reduce_dims, n_components):
    """Run clustering on a CSV file"""
    click.echo(f"🔬 Running {algorithm} clustering on {filepath}")
    
    try:
        clustering = SimpleClustering()
        
        # Prepare parameters
        params = {
            'preprocess': not no_preprocess,
            'reduce_dims': reduce_dims,
            'n_components': n_components
        }
        
        if algorithm == 'kmeans':
            params['n_clusters'] = n_clusters
        elif algorithm == 'dbscan':
            params['eps'] = eps
            params['min_samples'] = min_samples
        elif algorithm == 'hierarchical':
            params['n_clusters'] = n_clusters
        
        # Run clustering
        results_id = clustering.run_clustering(filepath, algorithm=algorithm, **params)
        
        # Get and display summary
        summary = clustering.get_results_summary(results_id)
        
        click.echo(f"✅ Clustering completed!")
        click.echo(f"   Results ID: {results_id}")
        click.echo(f"   Algorithm: {summary['algorithm']}")
        click.echo(f"   Clusters found: {summary['n_clusters']}")
        click.echo(f"   Silhouette score: {summary['silhouette_score']:.3f}")
        click.echo(f"   Results saved to: results/clustered_data_{results_id}.csv")
        
    except Exception as e:
        click.echo(f"❌ Clustering failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--algorithm', default='kmeans', 
              type=click.Choice(['kmeans', 'dbscan', 'hierarchical']),
              help='Clustering algorithm to use')
@click.option('--n-clusters', default=3, help='Number of clusters')
def quick(filepath, algorithm, n_clusters):
    """Quick clustering with default parameters"""
    click.echo(f"⚡ Quick {algorithm} clustering on {filepath}")
    
    try:
        results_id = quick_cluster(filepath, algorithm=algorithm, n_clusters=n_clusters)
        
        clustering = SimpleClustering()
        summary = clustering.get_results_summary(results_id)
        
        click.echo(f"✅ Quick clustering completed!")
        click.echo(f"   Results ID: {results_id}")
        click.echo(f"   Clusters: {summary['n_clusters']}")
        click.echo(f"   Score: {summary['silhouette_score']:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Quick clustering failed: {e}")
        sys.exit(1)


@cli.command()
def experiments():
    """List all clustering experiments"""
    try:
        storage = get_storage()
        experiments = storage.list_experiments()
        
        if not experiments:
            click.echo("No experiments found.")
            return
        
        click.echo(f"📊 Found {len(experiments)} experiments:")
        click.echo()
        
        for exp in experiments:
            status = exp.get('status', 'unknown')
            click.echo(f"Experiment: {exp['id']}")
            click.echo(f"  Algorithm: {exp.get('algorithm', 'unknown')}")
            click.echo(f"  Status: {status}")
            click.echo(f"  Created: {exp.get('started_at', 'unknown')}")
            
            if status == 'completed':
                metrics = exp.get('metrics', {})
                click.echo(f"  Clusters: {metrics.get('n_clusters', 0)}")
                click.echo(f"  Silhouette: {metrics.get('silhouette_score', 0):.3f}")
            
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Failed to list experiments: {e}")


@cli.command()
@click.argument('results_id')
def results(results_id):
    """Show results for a specific experiment"""
    try:
        storage = get_storage()
        results = storage.load_clustering_results(results_id)
        
        if not results:
            click.echo(f"❌ Results not found: {results_id}")
            return
        
        click.echo(f"📋 Results for {results_id}:")
        click.echo(f"  Algorithm: {results.get('algorithm')}")
        click.echo(f"  Completed: {results.get('completed_at')}")
        
        metrics = results.get('metrics', {})
        click.echo(f"  Clusters: {metrics.get('n_clusters')}")
        click.echo(f"  Silhouette Score: {metrics.get('silhouette_score', 0):.3f}")
        click.echo(f"  Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 0):.3f}")
        
        data_info = results.get('data_info', {})
        if data_info:
            click.echo(f"  Data shape: {data_info.get('processed_shape')}")
        
        # Check if results file exists
        results_file = Path("results") / f"clustered_data_{results_id}.csv"
        if results_file.exists():
            click.echo(f"  Results file: {results_file}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load results: {e}")


@cli.command()
def generate_sample():
    """Generate sample data for testing"""
    try:
        import numpy as np
        import pandas as pd
        
        click.echo("🎲 Generating sample data...")
        
        # Generate synthetic data with clusters
        np.random.seed(42)
        
        # Create 3 clusters
        cluster1 = np.random.normal([2, 2], 0.5, (100, 2))
        cluster2 = np.random.normal([6, 6], 0.5, (100, 2))
        cluster3 = np.random.normal([4, 8], 0.5, (100, 2))
        
        # Combine clusters
        data = np.vstack([cluster1, cluster2, cluster3])
        
        # Add some noise dimensions
        noise = np.random.normal(0, 0.1, (300, 3))
        data = np.hstack([data, noise])
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns)
        
        # Save to file
        Path("data").mkdir(exist_ok=True)
        filepath = Path("data") / "sample_data.csv"
        df.to_csv(filepath, index=False)
        
        click.echo(f"✅ Sample data generated: {filepath}")
        click.echo(f"   Shape: {df.shape}")
        click.echo(f"   Try: python main.py cluster {filepath}")
        
    except ImportError:
        click.echo("❌ NumPy/Pandas not installed. Install with:")
        click.echo("pip install numpy pandas")
    except Exception as e:
        click.echo(f"❌ Failed to generate sample data: {e}")


@cli.command()
def status():
    """Show system status"""
    try:
        storage = get_storage()
        
        click.echo("🏥 System Status:")
        click.echo(f"  Storage type: Local files")
        click.echo(f"  Base directory: {storage.base_dir}")
        
        # Check directories
        dirs = ['data', 'models', 'results', 'cache', 'logs', 'temp', 'experiments']
        for dir_name in dirs:
            dir_path = storage.base_dir / dir_name
            exists = "✅" if dir_path.exists() else "❌"
            click.echo(f"  {dir_name}/: {exists}")
        
        # Count experiments
        experiments = storage.list_experiments()
        click.echo(f"  Experiments: {len(experiments)}")
        
        # Count data files
        data_dir = Path("data")
        data_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
        click.echo(f"  Data files: {len(data_files)}")
        
        # Count results
        results_dir = Path("results")
        result_files = list(results_dir.glob("*.json")) if results_dir.exists() else []
        click.echo(f"  Result files: {len(result_files)}")
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")


@cli.command()
def cleanup():
    """Clean up old files and cache"""
    try:
        storage = get_storage()
        storage.cleanup_temp_files()
        storage.cleanup_expired_cache()
        
        click.echo("🧹 Cleanup completed")
        
    except Exception as e:
        click.echo(f"❌ Cleanup failed: {e}")


if __name__ == '__main__':
    cli()
