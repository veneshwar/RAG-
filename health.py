"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import asyncio

from app.api.dependencies import get_vector_store, get_llm_client


router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": asyncio.get_event_loop().time()
    }


@router.get("/detailed")
async def detailed_health_check(
    vector_store = Depends(get_vector_store),
    llm_client = Depends(get_llm_client)
) -> Dict[str, Any]:
    """
    Detailed health check including dependencies
    """
    health_status = {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": asyncio.get_event_loop().time(),
        "dependencies": {}
    }
    
    # Check vector store
    try:
        stats = await vector_store.get_stats()
        health_status["dependencies"]["vector_store"] = "healthy"
        health_status["dependencies"]["vector_store_stats"] = stats
    except Exception as e:
        health_status["dependencies"]["vector_store"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check LLM client
    try:
        is_healthy = await llm_client.health_check()
        health_status["dependencies"]["llm_client"] = "healthy" if is_healthy else "unhealthy"
        if not is_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["dependencies"]["llm_client"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for Kubernetes
    """
    # TODO: Check if system is ready to serve traffic
    return {
        "status": "ready",
        "timestamp": asyncio.get_event_loop().time()
    }


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for Kubernetes
    """
    return {
        "status": "alive",
        "timestamp": asyncio.get_event_loop().time()
    }
