"""
High-Dimensional Data Clustering Framework
Main application entry point
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler

from src.api.app import create_app
from src.config import settings
from src.utils.logger import setup_logging

console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--log-level", default="INFO", help="Set logging level")
def cli(debug: bool, log_level: str):
    """High-Dimensional Data Clustering Framework CLI"""
    setup_logging(log_level, debug)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server"""
    console.print(f"🚀 Starting clustering API server on {host}:{port}", style="green")
    
    app = create_app()
    uvicorn.run(
        "src.main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_config=None  # Use our custom logging
    )


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--algorithm", default="deep_embedding", help="Clustering algorithm")
@click.option("--n-clusters", default=None, type=int, help="Number of clusters")
@click.option("--output", default="results", help="Output directory")
def cluster(data_path: str, algorithm: str, n_clusters: Optional[int], output: str):
    """Run clustering on a dataset"""
    from src.clustering.pipeline import ClusteringPipeline
    
    console.print(f"🔍 Running {algorithm} clustering on {data_path}", style="blue")
    
    pipeline = ClusteringPipeline()
    results = asyncio.run(pipeline.run(
        data_path=Path(data_path),
        algorithm=algorithm,
        n_clusters=n_clusters,
        output_dir=Path(output)
    ))
    
    console.print(f"✅ Clustering completed! Results saved to {output}", style="green")
    console.print(f"📊 Found {results['n_clusters']} clusters", style="yellow")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str):
    """Train a deep clustering model"""
    from src.models.trainer import ModelTrainer
    
    console.print(f"🧠 Training model with config: {config_path}", style="blue")
    
    trainer = ModelTrainer.from_config(config_path)
    asyncio.run(trainer.train())
    
    console.print("✅ Model training completed!", style="green")


@cli.command()
def dashboard():
    """Launch the web dashboard"""
    import subprocess
    import sys
    
    console.print("🌐 Launching web dashboard...", style="blue")
    
    try:
        # Start the React development server
        subprocess.run([
            "npm", "start"
        ], cwd="web-dashboard", check=True)
    except subprocess.CalledProcessError:
        console.print("❌ Failed to start dashboard. Make sure Node.js is installed.", style="red")
        sys.exit(1)


@cli.command()
def setup():
    """Setup the development environment"""
    console.print("🛠️ Setting up development environment...", style="blue")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        console.print(f"✅ Created directory: {directory}", style="green")
    
    console.print("🎉 Setup completed!", style="green")


if __name__ == "__main__":
    cli()
