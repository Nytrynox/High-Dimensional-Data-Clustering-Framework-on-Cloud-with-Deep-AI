"""
FastAPI application for clustering framework
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import Optional

from src.config import settings
from src.utils.logger import setup_logging, get_logger


# Initialize logging
setup_logging(settings.log_level, settings.debug)
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting clustering framework API...")
    
    # Initialize services
    await initialize_services()
    
    yield
    
    logger.info("Shutting down clustering framework API...")
    await cleanup_services()


async def initialize_services():
    """Initialize application services"""
    logger.info("Initializing services...")
    # TODO: Initialize ML models, database connections, etc.


async def cleanup_services():
    """Cleanup application services"""
    logger.info("Cleaning up services...")
    # TODO: Cleanup resources


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="High-Dimensional Data Clustering Framework API",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    from src.api.routes import clustering, health, models
    
    app.include_router(
        health.router,
        prefix=settings.api_prefix,
        tags=["health"]
    )
    
    app.include_router(
        clustering.router,
        prefix=settings.api_prefix,
        tags=["clustering"]
    )
    
    app.include_router(
        models.router,
        prefix=settings.api_prefix,
        tags=["models"]
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "High-Dimensional Data Clustering Framework API",
            "version": settings.version,
            "status": "running"
        }
    
    return app


# Create app instance
app = create_app()
