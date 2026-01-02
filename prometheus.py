"""
Prometheus metrics implementation
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time


logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self):
        # Define metrics
        self.query_count = Counter(
            'rag_queries_total',
            'Total number of RAG queries',
            ['status']
        )
        
        self.query_latency = Histogram(
            'rag_query_duration_seconds',
            'RAG query latency in seconds',
            ['endpoint']
        )
        
        self.ingestion_count = Counter(
            'rag_ingestion_total',
            'Total number of documents ingested',
            ['status']
        )
        
        self.vector_store_size = Gauge(
            'rag_vector_store_size',
            'Number of vectors in store'
        )
        
        self.llm_requests = Counter(
            'rag_llm_requests_total',
            'Total LLM requests',
            ['model', 'status']
        )
        
        self.embedding_requests = Counter(
            'rag_embedding_requests_total',
            'Total embedding requests',
            ['model', 'status']
        )
        
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'rag_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
    
    async def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest()
    
    async def get_query_metrics(self) -> Dict[str, Any]:
        """Get query-specific metrics"""
        return {
            "total_queries": self.query_count._value.get(),
            "avg_latency": self.query_latency._sum.get() / max(self.query_latency._count.get(), 1)
        }
    
    async def get_ingestion_metrics(self) -> Dict[str, Any]:
        """Get ingestion-specific metrics"""
        return {
            "total_ingested": self.ingestion_count._value.get(),
            "vector_store_size": self.vector_store_size._value.get()
        }
    
    async def get_histogram(self, name: str) -> float:
        """Get histogram value"""
        if name == "query_latency_p50":
            return self.query_latency.observe(0.5)
        elif name == "query_latency_p95":
            return self.query_latency.observe(0.95)
        elif name == "query_latency_p99":
            return self.query_latency.observe(0.99)
        return 0.0
    
    async def get_counter(self, name: str) -> float:
        """Get counter value"""
        if name == "ingestion_throughput":
            return self.ingestion_count._value.get()
        return 0.0
    
    async def get_gauge(self, name: str) -> float:
        """Get gauge value"""
        if name == "vector_store_size":
            return self.vector_store_size._value.get()
        return 0.0
    
    def increment_query_count(self, status: str = "success"):
        """Increment query counter"""
        self.query_count.labels(status=status).inc()
    
    def observe_query_latency(self, duration: float, endpoint: str = "query"):
        """Observe query latency"""
        self.query_latency.labels(endpoint=endpoint).observe(duration)
    
    def increment_ingestion_count(self, status: str = "success"):
        """Increment ingestion counter"""
        self.ingestion_count.labels(status=status).inc()
    
    def set_vector_store_size(self, size: int):
        """Set vector store size"""
        self.vector_store_size.set(size)
    
    def increment_llm_requests(self, model: str, status: str = "success"):
        """Increment LLM request counter"""
        self.llm_requests.labels(model=model, status=status).inc()
    
    def increment_embedding_requests(self, model: str, status: str = "success"):
        """Increment embedding request counter"""
        self.embedding_requests.labels(model=model, status=status).inc()
    
    def increment_cache_hits(self, cache_type: str):
        """Increment cache hits"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def increment_cache_misses(self, cache_type: str):
        """Increment cache misses"""
        self.cache_misses.labels(cache_type=cache_type).inc()
