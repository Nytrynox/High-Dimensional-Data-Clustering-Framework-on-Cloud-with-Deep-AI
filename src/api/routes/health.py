"""
Health check endpoints
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
import psutil
import os

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger("health")
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    try:
        # Get system information
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            "python_version": f"{psutil.version_info[0]}.{psutil.version_info[1]}.{psutil.version_info[2]}"
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version=settings.version,
            uptime=0.0,  # TODO: Implement actual uptime tracking
            system=system_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version=settings.version,
            uptime=0.0,
            system={}
        )


@router.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    return {"status": "ready", "timestamp": datetime.utcnow()}


@router.get("/alive")
async def liveness_check():
    """Liveness probe for Kubernetes"""
    return {"status": "alive", "timestamp": datetime.utcnow()}
