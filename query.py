"""
Query endpoints for RAG system
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List
import asyncio

from app.api.schemas.query_schema import QueryRequest, QueryResponse, StreamQueryRequest
from app.api.dependencies import get_rag_orchestrator
from app.rag.context_manager import ContextManager
from app.api.dependencies import get_context_manager
from app.api.routes.metrics import increment_query_count


router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    orchestrator = Depends(get_rag_orchestrator)
):
    """
    Process a RAG query and return response
    """
    try:
        start_time = asyncio.get_event_loop().time()
        response = await orchestrator.process_query(
            query=request.query,
            filters=request.filters,
            max_context=request.max_context,
            temperature=request.temperature
        )
        
        # Track metrics
        latency = asyncio.get_event_loop().time() - start_time
        increment_query_count(latency)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_query(
    request: StreamQueryRequest,
    orchestrator = Depends(get_rag_orchestrator)
):
    """
    Stream a RAG query response
    """
    async def generate():
        try:
            async for chunk in orchestrator.stream_query(
                query=request.query,
                filters=request.filters,
                max_context=request.max_context,
                temperature=request.temperature
            ):
                yield f"data: {chunk.json()}\n\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    return asyncio.StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/context/{query_id}")
async def get_query_context(
    query_id: str,
    context_manager = Depends(get_context_manager)
):
    """
    Get context used for a specific query
    """
    try:
        context = await context_manager.get_context(query_id)
        return {"query_id": query_id, "context": context}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/context/{query_id}")
async def clear_query_context(
    query_id: str,
    context_manager = Depends(get_context_manager)
):
    """
    Clear context for a specific query
    """
    try:
        await context_manager.clear_context(query_id)
        return {"message": f"Context cleared for query {query_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
