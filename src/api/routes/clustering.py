"""
Clustering API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import uuid
import pandas as pd
from io import StringIO

from src.utils.logger import get_logger

logger = get_logger("clustering_api")
router = APIRouter()


class ClusteringRequest(BaseModel):
    """Request model for clustering operations"""
    algorithm: str = Field(..., description="Clustering algorithm to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    n_clusters: Optional[int] = Field(None, description="Number of clusters (if applicable)")
    data_source: Optional[str] = Field(None, description="Data source identifier")


class ClusteringResponse(BaseModel):
    """Response model for clustering operations"""
    job_id: str
    status: str
    algorithm: str
    parameters: Dict[str, Any]
    created_at: datetime
    message: str


class ClusteringResult(BaseModel):
    """Result model for completed clustering"""
    job_id: str
    status: str
    algorithm: str
    n_clusters: int
    cluster_labels: List[int]
    cluster_centers: Optional[List[List[float]]]
    metrics: Dict[str, float]
    completed_at: datetime


# In-memory job storage (replace with database in production)
jobs_storage: Dict[str, Dict] = {}


@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available clustering algorithms"""
    from src.config import ClusteringConfig
    
    algorithms = []
    for key, config in ClusteringConfig.ALGORITHMS.items():
        algorithms.append({
            "id": key,
            "name": config["name"],
            "parameters": config["parameters"]
        })
    
    return {"algorithms": algorithms}


@router.post("/cluster", response_model=ClusteringResponse)
async def start_clustering(
    request: ClusteringRequest,
    background_tasks: BackgroundTasks
):
    """Start a clustering job"""
    job_id = str(uuid.uuid4())
    
    # Validate algorithm
    from src.config import ClusteringConfig
    if request.algorithm not in ClusteringConfig.ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {request.algorithm}"
        )
    
    # Create job entry
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "algorithm": request.algorithm,
        "parameters": request.parameters,
        "created_at": datetime.utcnow(),
        "progress": 0
    }
    
    jobs_storage[job_id] = job_data
    
    # Start background task
    background_tasks.add_task(
        run_clustering_job,
        job_id,
        request.algorithm,
        request.parameters,
        request.data_source
    )
    
    logger.info(f"Started clustering job {job_id} with algorithm {request.algorithm}")
    
    return ClusteringResponse(
        job_id=job_id,
        status="pending",
        algorithm=request.algorithm,
        parameters=request.parameters,
        created_at=datetime.utcnow(),
        message="Clustering job started successfully"
    )


@router.post("/cluster/upload")
async def cluster_uploaded_data(
    file: UploadFile = File(...),
    algorithm: str = "kmeans",
    n_clusters: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """Upload data and start clustering"""
    
    # Validate file type
    if not file.filename.endswith(('.csv', '.json', '.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Only CSV, JSON, and Parquet files are supported"
        )
    
    job_id = str(uuid.uuid4())
    
    try:
        # Read file content
        content = await file.read()
        
        # Store file data temporarily (in production, use proper storage)
        file_data = {
            "filename": file.filename,
            "content": content,
            "content_type": file.content_type
        }
        
        # Create job entry
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "algorithm": algorithm,
            "parameters": {"n_clusters": n_clusters} if n_clusters else {},
            "created_at": datetime.utcnow(),
            "file_data": file_data,
            "progress": 0
        }
        
        jobs_storage[job_id] = job_data
        
        # Start background task
        background_tasks.add_task(
            run_clustering_job_with_file,
            job_id,
            algorithm,
            {"n_clusters": n_clusters} if n_clusters else {},
            file_data
        )
        
        logger.info(f"Started clustering job {job_id} for uploaded file {file.filename}")
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"File {file.filename} uploaded and clustering started"
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster/{job_id}/status")
async def get_clustering_status(job_id: str):
    """Get the status of a clustering job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "progress": job_data.get("progress", 0),
        "created_at": job_data["created_at"],
        "message": job_data.get("message", "")
    }


@router.get("/cluster/{job_id}/result", response_model=ClusteringResult)
async def get_clustering_result(job_id: str):
    """Get the result of a completed clustering job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job_data['status']}"
        )
    
    return ClusteringResult(**job_data["result"])


@router.delete("/cluster/{job_id}")
async def cancel_clustering_job(job_id: str):
    """Cancel a clustering job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    
    if job_data["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job_data['status']}"
        )
    
    job_data["status"] = "cancelled"
    logger.info(f"Cancelled clustering job {job_id}")
    
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """List clustering jobs"""
    jobs = list(jobs_storage.values())
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    total = len(jobs)
    jobs_page = jobs[offset:offset + limit]
    
    return {
        "jobs": jobs_page,
        "total": total,
        "limit": limit,
        "offset": offset
    }


async def run_clustering_job(
    job_id: str,
    algorithm: str,
    parameters: Dict[str, Any],
    data_source: Optional[str]
):
    """Background task to run clustering job"""
    try:
        job_data = jobs_storage[job_id]
        job_data["status"] = "running"
        job_data["progress"] = 10
        
        logger.info(f"Running clustering job {job_id}")
        
        # Simulate clustering process (replace with actual implementation)
        await asyncio.sleep(2)  # Simulate processing time
        job_data["progress"] = 50
        
        await asyncio.sleep(2)  # More processing
        job_data["progress"] = 90
        
        # Generate mock results
        result = {
            "job_id": job_id,
            "status": "completed",
            "algorithm": algorithm,
            "n_clusters": parameters.get("n_clusters", 5),
            "cluster_labels": [0, 1, 2, 0, 1] * 10,  # Mock labels
            "cluster_centers": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Mock centers
            "metrics": {
                "silhouette_score": 0.75,
                "calinski_harabasz_score": 150.5,
                "davies_bouldin_score": 0.8
            },
            "completed_at": datetime.utcnow()
        }
        
        job_data["status"] = "completed"
        job_data["progress"] = 100
        job_data["result"] = result
        job_data["completed_at"] = datetime.utcnow()
        
        logger.info(f"Completed clustering job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in clustering job {job_id}: {e}")
        job_data["status"] = "failed"
        job_data["error"] = str(e)


async def run_clustering_job_with_file(
    job_id: str,
    algorithm: str,
    parameters: Dict[str, Any],
    file_data: Dict[str, Any]
):
    """Background task to run clustering job with uploaded file"""
    try:
        job_data = jobs_storage[job_id]
        job_data["status"] = "running"
        job_data["progress"] = 10
        
        logger.info(f"Processing file for clustering job {job_id}")
        
        # Process file data (simplified example)
        filename = file_data["filename"]
        content = file_data["content"]
        
        if filename.endswith('.csv'):
            # Parse CSV data
            df = pd.read_csv(StringIO(content.decode('utf-8')))
            job_data["progress"] = 30
            
            # Simulate clustering on real data
            n_samples = len(df)
            n_features = len(df.columns)
            
            logger.info(f"Processing {n_samples} samples with {n_features} features")
            
            await asyncio.sleep(3)  # Simulate processing
            job_data["progress"] = 80
            
            # Generate results based on data
            n_clusters = parameters.get("n_clusters", min(8, max(2, n_samples // 10)))
            
            result = {
                "job_id": job_id,
                "status": "completed",
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "cluster_labels": [i % n_clusters for i in range(n_samples)],
                "cluster_centers": None,
                "metrics": {
                    "silhouette_score": 0.65,
                    "n_samples": n_samples,
                    "n_features": n_features
                },
                "completed_at": datetime.utcnow()
            }
            
            job_data["status"] = "completed"
            job_data["progress"] = 100
            job_data["result"] = result
            job_data["completed_at"] = datetime.utcnow()
            
            logger.info(f"Completed clustering job {job_id} with uploaded file")
            
    except Exception as e:
        logger.error(f"Error processing file in job {job_id}: {e}")
        job_data["status"] = "failed"
        job_data["error"] = str(e)
