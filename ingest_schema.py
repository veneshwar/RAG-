"""
Pydantic schemas for ingestion endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentSource(str, Enum):
    """Enum for document sources"""
    UPLOAD = "upload"
    WEB = "web"
    KAFKA = "kafka"
    RSS = "rss"
    WEBSOCKET = "websocket"


class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    source: DocumentSource = Field(default=DocumentSource.UPLOAD, description="Document source")
    priority: Optional[int] = Field(default=1, description="Processing priority")


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    document_id: str
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchIngestRequest(BaseModel):
    """Request model for batch ingestion"""
    documents: List[IngestRequest] = Field(..., description="List of documents to ingest")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier")


class StreamConfig(BaseModel):
    """Configuration for streaming ingestion"""
    topic: str
    consumer_group: str
    batch_size: int = Field(default=100)
    poll_timeout_ms: int = Field(default=1000)
    auto_commit: bool = Field(default=True)


class StreamIngestRequest(BaseModel):
    """Request model for streaming ingestion"""
    source: DocumentSource = Field(..., description="Stream source type")
    config: StreamConfig = Field(..., description="Stream configuration")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Content filters")


class StreamEvent(BaseModel):
    """Model for streaming events"""
    event_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    processed: bool = Field(default=False)


class IngestionStatus(BaseModel):
    """Model for ingestion status"""
    document_id: str
    status: str
    stage: str
    progress: float = Field(ge=0, le=1)
    error_message: Optional[str] = None
    timestamp: datetime


class IngestionMetrics(BaseModel):
    """Model for ingestion metrics"""
    documents_processed: int
    documents_failed: int
    processing_time_ms: float
    throughput_docs_per_second: float
    timestamp: datetime
