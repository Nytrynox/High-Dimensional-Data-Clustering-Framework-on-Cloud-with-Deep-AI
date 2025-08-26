"""
Database-Free API - Simple REST API using only local file storage
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import tempfile
import os
from pathlib import Path
import logging

from ..clustering.simple_pipeline import SimpleClustering, quick_cluster
from ..storage.local_storage import get_storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for requests
class ClusterRequest(BaseModel):
    filepath: str
    algorithm: str = 'kmeans'
    n_clusters: Optional[int] = 3
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5
    preprocess: bool = True
    scale: bool = True
    reduce_dims: bool = False
    n_components: int = 50

class QuickClusterRequest(BaseModel):
    algorithm: str = 'kmeans'
    n_clusters: int = 3


def create_simple_app() -> FastAPI:
    """Create FastAPI app without database dependencies"""
    
    app = FastAPI(
        title="High-Dimensional Clustering Framework (Database-Free)",
        description="A simple clustering API using only local file storage",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize storage and clustering
    storage = get_storage()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize directories on startup"""
        storage.ensure_directories()
        logger.info("Database-free clustering API started")
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "High-Dimensional Clustering Framework (Database-Free)",
            "version": "1.0.0",
            "endpoints": [
                "/docs - API documentation",
                "/health - Health check",
                "/cluster - Run clustering",
                "/upload - Upload data file",
                "/experiments - List experiments",
                "/results/{id} - Get results"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "storage_type": "local_files",
            "directories_exist": True
        }
    
    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a CSV file for clustering"""
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        try:
            # Save uploaded file
            upload_dir = Path("data")
            upload_dir.mkdir(exist_ok=True)
            
            filepath = upload_dir / file.filename
            
            with open(filepath, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Validate the file
            try:
                data = pd.read_csv(filepath)
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) == 0:
                    raise HTTPException(status_code=400, detail="No numeric columns found in the data")
                
                return {
                    "message": "File uploaded successfully",
                    "filename": file.filename,
                    "filepath": str(filepath),
                    "shape": data.shape,
                    "numeric_columns": numeric_cols,
                    "columns": data.columns.tolist()
                }
                
            except Exception as e:
                # Clean up invalid file
                filepath.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    @app.post("/cluster")
    async def run_clustering(request: ClusterRequest):
        """Run clustering on uploaded data"""
        try:
            filepath = Path(request.filepath)
            if not filepath.exists():
                raise HTTPException(status_code=404, detail="Data file not found")
            
            clustering = SimpleClustering()
            
            # Prepare parameters
            params = {
                'preprocess': request.preprocess,
                'scale': request.scale,
                'reduce_dims': request.reduce_dims,
                'n_components': request.n_components
            }
            
            # Algorithm-specific parameters
            if request.algorithm == 'kmeans':
                params['n_clusters'] = request.n_clusters
            elif request.algorithm == 'dbscan':
                params['eps'] = request.eps
                params['min_samples'] = request.min_samples
            elif request.algorithm == 'hierarchical':
                params['n_clusters'] = request.n_clusters
            
            # Run clustering
            results_id = clustering.run_clustering(
                str(filepath), 
                algorithm=request.algorithm,
                **params
            )
            
            # Get summary
            summary = clustering.get_results_summary(results_id)
            
            return {
                "message": "Clustering completed successfully",
                "results_id": results_id,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/quick-cluster/{filename}")
    async def quick_cluster_endpoint(filename: str, request: QuickClusterRequest):
        """Quick clustering for uploaded files"""
        try:
            filepath = Path("data") / filename
            if not filepath.exists():
                raise HTTPException(status_code=404, detail="Data file not found")
            
            results_id = quick_cluster(
                str(filepath), 
                algorithm=request.algorithm, 
                n_clusters=request.n_clusters
            )
            
            clustering = SimpleClustering()
            summary = clustering.get_results_summary(results_id)
            
            return {
                "message": "Quick clustering completed",
                "results_id": results_id,
                "summary": summary
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/experiments")
    async def list_experiments():
        """List all clustering experiments"""
        try:
            experiments = storage.list_experiments()
            return {"experiments": experiments}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/results/{results_id}")
    async def get_results(results_id: str):
        """Get clustering results by ID"""
        try:
            results = storage.load_clustering_results(results_id)
            if not results:
                raise HTTPException(status_code=404, detail="Results not found")
            
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/download-results/{results_id}")
    async def download_results(results_id: str):
        """Download clustered data as CSV"""
        try:
            results_file = Path("results") / f"clustered_data_{results_id}.csv"
            if not results_file.exists():
                raise HTTPException(status_code=404, detail="Results file not found")
            
            return FileResponse(
                path=str(results_file),
                filename=f"clustered_data_{results_id}.csv",
                media_type="text/csv"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/files")
    async def list_files():
        """List uploaded data files"""
        try:
            data_dir = Path("data")
            files = []
            
            if data_dir.exists():
                for file in data_dir.glob("*.csv"):
                    try:
                        data = pd.read_csv(file)
                        files.append({
                            "filename": file.name,
                            "filepath": str(file),
                            "size": file.stat().st_size,
                            "shape": data.shape,
                            "numeric_columns": len(data.select_dtypes(include=['number']).columns)
                        })
                    except:
                        files.append({
                            "filename": file.name,
                            "filepath": str(file),
                            "size": file.stat().st_size,
                            "error": "Could not read file"
                        })
            
            return {"files": files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/cleanup")
    async def cleanup():
        """Clean up old files and cache"""
        try:
            clustering = SimpleClustering()
            clustering.cleanup_old_results()
            return {"message": "Cleanup completed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create the app instance
app = create_simple_app()
