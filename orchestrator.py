"""
RAG Orchestrator - Main coordinator for RAG pipeline
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
import time
import logging

from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.prompt_builder import PromptBuilder
from app.rag.context_manager import ContextManager
from app.api.schemas.query_schema import QueryResponse, ContextDocument, StreamChunk


logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """Main orchestrator for RAG pipeline"""
    
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        prompt_builder: PromptBuilder,
        context_manager: ContextManager
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.context_manager = context_manager
    
    async def process_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_context: int = 5,
        temperature: float = 0.7
    ) -> QueryResponse:
        """
        Process a RAG query end-to-end
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                filters=filters,
                top_k=max_context * 2  # Retrieve more for reranking
            )
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.3f}s")
            
            # Step 2: Rerank documents
            if len(retrieved_docs) > max_context:
                rerank_start = time.time()
                reranked_docs = await self.reranker.rerank(
                    query=query,
                    documents=retrieved_docs,
                    top_k=max_context
                )
                rerank_time = time.time() - rerank_start
                logger.info(f"Reranked to {len(reranked_docs)} documents in {rerank_time:.3f}s")
            else:
                reranked_docs = retrieved_docs
                rerank_time = 0
            
            # Step 3: Build prompt
            prompt_start = time.time()
            prompt = await self.prompt_builder.build_prompt(
                query=query,
                context=reranked_docs
            )
            prompt_time = time.time() - prompt_start
            
            # Step 4: Generate response
            generation_start = time.time()
            answer = await self.generator.generate(
                prompt=prompt,
                temperature=temperature
            )
            generation_time = time.time() - generation_start
            
            # Step 5: Store context for this query
            await self.context_manager.store_context(
                query_id=query_id,
                query=query,
                context=reranked_docs,
                answer=answer
            )
            
            # Convert to response format
            context_docs = [
                ContextDocument(
                    document_id=doc["id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    score=doc["score"],
                    timestamp=doc.get("timestamp", time.time())
                )
                for doc in reranked_docs
            ]
            
            total_time = time.time() - start_time
            
            return QueryResponse(
                query_id=query_id,
                answer=answer,
                context=context_docs,
                metadata={
                    "retrieval_time_ms": round(retrieval_time * 1000, 2),
                    "rerank_time_ms": round(rerank_time * 1000, 2),
                    "generation_time_ms": round(generation_time * 1000, 2),
                    "prompt_time_ms": round(prompt_time * 1000, 2),
                    "total_time_ms": round(total_time * 1000, 2),
                    "documents_retrieved": len(retrieved_docs),
                    "documents_used": len(reranked_docs)
                },
                latency_ms=round(total_time * 1000, 2),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            raise
    
    async def stream_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_context: int = 5,
        temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a RAG query response
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                filters=filters,
                top_k=max_context * 2
            )
            retrieval_time = time.time() - retrieval_start
            
            # Step 2: Rerank documents
            if len(retrieved_docs) > max_context:
                reranked_docs = await self.reranker.rerank(
                    query=query,
                    documents=retrieved_docs,
                    top_k=max_context
                )
            else:
                reranked_docs = retrieved_docs
            
            # Step 3: Build prompt
            prompt = await self.prompt_builder.build_prompt(
                query=query,
                context=reranked_docs
            )
            
            # Step 4: Stream generation
            full_answer = ""
            async for chunk in self.generator.stream_generate(
                prompt=prompt,
                temperature=temperature
            ):
                full_answer += chunk
                yield StreamChunk(
                    query_id=query_id,
                    chunk=chunk,
                    done=False,
                    timestamp=time.time()
                )
            
            # Step 5: Store context
            await self.context_manager.store_context(
                query_id=query_id,
                query=query,
                context=reranked_docs,
                answer=full_answer
            )
            
            # Send final chunk
            yield StreamChunk(
                query_id=query_id,
                chunk="",
                done=True,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error in stream query {query_id}: {str(e)}")
            yield StreamChunk(
                query_id=query_id,
                chunk=f"Error: {str(e)}",
                done=True,
                timestamp=time.time()
            )
    
    async def get_query_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get query history
        """
        return await self.context_manager.get_query_history(user_id, limit)
    
    async def clear_context(self, query_id: str) -> bool:
        """
        Clear context for a specific query
        """
        return await self.context_manager.clear_context(query_id)
