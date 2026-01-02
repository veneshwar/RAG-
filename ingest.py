"""
Ingestion endpoints for real-time data processing
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import asyncio

from app.api.schemas.ingest_schema import (
    IngestRequest, 
    IngestResponse, 
    BatchIngestRequest,
    StreamIngestRequest
)
from app.api.dependencies import get_rag_orchestrator, get_document_store
from app.indexing.embeddings.openai_embedder import OpenAIEmbedder
from app.indexing.vectorstores.faiss_store import FAISSVectorStore
from app.api.routes.metrics import increment_ingestion_count


router = APIRouter()


@router.post("/", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    document_store = Depends(get_document_store)
):
    """
    Ingest a single document for real-time processing
    """
    try:
        # Store document directly (simplified version)
        success = await document_store.store_document(
            document_id=request.document_id,
            content=request.content,
            metadata=request.metadata
        )
        
        if success:
            increment_ingestion_count()
            return IngestResponse(
                document_id=request.document_id,
                status="completed",
                message="Document successfully ingested"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[IngestResponse])
async def batch_ingest(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    document_store = Depends(get_document_store)
):
    """
    Batch ingest multiple documents
    """
    responses = []
    
    try:
        for doc in request.documents:
            success = await document_store.store_document(
                document_id=doc.document_id,
                content=doc.content,
                metadata=doc.metadata
            )
            
            if success:
                increment_ingestion_count()
            
            responses.append(IngestResponse(
                document_id=doc.document_id,
                status="completed" if success else "failed",
                message="Document successfully ingested" if success else "Failed to store document"
            ))
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_ingest(
    request: StreamIngestRequest,
    document_store = Depends(get_document_store)
):
    """
    Stream ingestion for real-time data sources (simplified)
    """
    async def generate():
        try:
            # Mock streaming implementation
            yield f"data: {{'status': 'started', 'source': '{request.source}'}}\n\n"
            await asyncio.sleep(0.1)
            yield f"data: {{'status': 'completed', 'message': 'Stream processing completed'}}\n\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return asyncio.StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/status/{document_id}")
async def get_ingestion_status(document_id: str, document_store = Depends(get_document_store)):
    """
    Get ingestion status for a specific document
    """
    doc = await document_store.get_document(document_id)
    if doc:
        return {"document_id": document_id, "status": "completed", "found": True}
    else:
        return {"document_id": document_id, "status": "not_found", "found": False}


@router.delete("/{document_id}")
async def delete_document(document_id: str, document_store = Depends(get_document_store)):
    """
    Delete a document from the system
    """
    success = await document_store.delete_document(document_id)
    if success:
        return {"message": f"Document {document_id} deleted successfully"}
    else:
        return {"message": f"Document {document_id} not found"}
