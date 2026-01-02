"""
Pydantic schemas for metadata management
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MetadataType(str, Enum):
    """Enum for metadata types"""
    DOCUMENT = "document"
    USER = "user"
    SESSION = "session"
    SYSTEM = "system"


class DocumentMetadata(BaseModel):
    """Model for document metadata"""
    document_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    language: Optional[str] = Field(default="en")
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class UserMetadata(BaseModel):
    """Model for user metadata"""
    user_id: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    query_history: List[str] = Field(default_factory=list)
    created_at: datetime
    last_active: datetime
    session_count: int = Field(default=0)


class SessionMetadata(BaseModel):
    """Model for session metadata"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    query_count: int = Field(default=0)
    total_tokens_used: int = Field(default=0)
    avg_response_time_ms: float = Field(default=0.0)


class SystemMetadata(BaseModel):
    """Model for system metadata"""
    key: str
    value: Any
    type: MetadataType
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class MetadataQuery(BaseModel):
    """Model for metadata queries"""
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="asc", regex="^(asc|desc)$")
    limit: Optional[int] = Field(default=100, ge=1, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)


class MetadataResponse(BaseModel):
    """Response model for metadata queries"""
    items: List[Dict[str, Any]]
    total_count: int
    has_more: bool
    query_time_ms: float
