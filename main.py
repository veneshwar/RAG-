"""
FastAPI main application for real-time RAG system
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import query, ingest, health, metrics
from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.middleware.tracing import TracingMiddleware
from app.observability.logging.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    setup_logging()
    yield
    # Cleanup


app = FastAPI(
    title="Real-time RAG API",
    description="Real-time Retrieval-Augmented Generation system",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)

# Include routers
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Real-time RAG API", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
