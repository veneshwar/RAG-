"""
Metrics endpoints for monitoring and observability
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, List
import asyncio
import time
import random

from app.api.dependencies import get_document_store, get_vector_store


router = APIRouter()


# Simple in-memory metrics storage
_metrics_store = {
    "query_count": 0,
    "ingestion_count": 0,
    "total_query_latency": 0.0,
    "start_time": time.time()
}


@router.get("/")
async def get_metrics() -> Dict[str, Any]:
    """
    Get basic system metrics
    """
    uptime = time.time() - _metrics_store["start_time"]
    avg_latency = (_metrics_store["total_query_latency"] / max(1, _metrics_store["query_count"])) * 1000
    
    return {
        "uptime_seconds": uptime,
        "query_count": _metrics_store["query_count"],
        "ingestion_count": _metrics_store["ingestion_count"],
        "average_query_latency_ms": round(avg_latency, 2),
        "queries_per_second": round(_metrics_store["query_count"] / max(1, uptime), 2)
    }


@router.get("/query")
async def get_query_metrics() -> Dict[str, Any]:
    """
    Get query-specific metrics
    """
    return {
        "total_queries": _metrics_store["query_count"],
        "average_latency_ms": round((_metrics_store["total_query_latency"] / max(1, _metrics_store["query_count"])) * 1000, 2),
        "queries_per_minute": round(_metrics_store["query_count"] / max(1, (time.time() - _metrics_store["start_time"]) / 60), 2)
    }


@router.get("/ingestion")
async def get_ingestion_metrics(
    document_store = Depends(get_document_store)
) -> Dict[str, Any]:
    """
    Get ingestion-specific metrics
    """
    stats = await document_store.get_stats()
    return {
        "total_documents": stats.get("total_documents", 0),
        "ingestion_count": _metrics_store["ingestion_count"],
        "ingestion_rate": round(_metrics_store["ingestion_count"] / max(1, time.time() - _metrics_store["start_time"]), 2)
    }


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics
    """
    return {
        "query_latency_p50": round(random.uniform(100, 200), 2),  # Mock values
        "query_latency_p95": round(random.uniform(200, 400), 2),
        "query_latency_p99": round(random.uniform(400, 800), 2),
        "ingestion_throughput": round(random.uniform(50, 150), 2),
        "vector_store_size": random.randint(1000, 10000),
        "cache_hit_rate": round(random.uniform(0.7, 0.95), 2)
    }


@router.get("/alerts")
async def get_active_alerts() -> List[Dict[str, Any]]:
    """
    Get active alerts
    """
    # Mock alerts based on metrics
    alerts = []
    
    if _metrics_store["query_count"] > 0:
        avg_latency = (_metrics_store["total_query_latency"] / _metrics_store["query_count"]) * 1000
        if avg_latency > 500:
            alerts.append({
                "alert_id": "high_query_latency",
                "severity": "warning",
                "message": f"Query latency above threshold: {avg_latency:.2f}ms",
                "timestamp": time.time()
            })
    
    return alerts


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> Dict[str, str]:
    """
    Acknowledge an alert
    """
    return {"message": f"Alert {alert_id} acknowledged"}


def increment_query_count(latency: float):
    """Increment query metrics (called by query endpoint)"""
    _metrics_store["query_count"] += 1
    _metrics_store["total_query_latency"] += latency


def increment_ingestion_count():
    """Increment ingestion metrics (called by ingest endpoint)"""
    _metrics_store["ingestion_count"] += 1
