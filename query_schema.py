"""
Pydantic schemas for query endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="The query text")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    max_context: Optional[int] = Field(default=5, description="Maximum number of context documents")
    temperature: Optional[float] = Field(default=0.7, description="LLM temperature")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response")


class StreamQueryRequest(BaseModel):
    """Request model for streaming RAG query"""
    query: str = Field(..., description="The query text")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    max_context: Optional[int] = Field(default=5, description="Maximum number of context documents")
    temperature: Optional[float] = Field(default=0.7, description="LLM temperature")


class ContextDocument(BaseModel):
    """Model for retrieved context document"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    timestamp: datetime


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    query_id: str
    answer: str
    context: List[ContextDocument]
    metadata: Dict[str, Any]
    latency_ms: float
    timestamp: datetime


class StreamChunk(BaseModel):
    """Model for streaming response chunk"""
    query_id: str
    chunk: str
    done: bool
    timestamp: datetime


class QueryMetrics(BaseModel):
    """Model for query metrics"""
    query_id: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    context_count: int
    token_usage: Dict[str, int]
