"""
Document retriever for RAG system
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from app.indexing.embeddings.embedder import BaseEmbedder
from app.indexing.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)


class Retriever:
    """Document retriever using vector similarity search"""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        similarity_threshold: float = 0.7,
        max_results: int = 100
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        """
        try:
            # Step 1: Embed the query
            query_embedding = await self.embedder.embed_text(query)
            
            # Step 2: Search vector store
            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=min(top_k, self.max_results),
                filters=filters,
                include_metadata=include_metadata
            )
            
            # Step 3: Filter by similarity threshold
            filtered_results = [
                result for result in search_results
                if result.get("score", 0) >= self.similarity_threshold
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} documents for query: {query[:50]}...")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    async def retrieve_with_expansion(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        expansion_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with query expansion
        """
        all_results = {}
        
        # Original query
        original_results = await self.retrieve(query, filters, top_k)
        for result in original_results:
            doc_id = result["id"]
            if doc_id not in all_results or result["score"] > all_results[doc_id]["score"]:
                all_results[doc_id] = result
        
        # Expansion queries
        if expansion_queries:
            for exp_query in expansion_queries:
                exp_results = await self.retrieve(exp_query, filters, top_k)
                for result in exp_results:
                    doc_id = result["id"]
                    # Average the scores
                    if doc_id in all_results:
                        all_results[doc_id]["score"] = (all_results[doc_id]["score"] + result["score"]) / 2
                    else:
                        all_results[doc_id] = result
        
        # Sort by score and return top_k
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]
    
    async def retrieve_by_ids(
        self,
        document_ids: List[str],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve specific documents by their IDs
        """
        try:
            results = await self.vector_store.get_by_ids(
                document_ids=document_ids,
                include_metadata=include_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {str(e)}")
            raise
    
    async def get_document_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Get total document count with optional filters
        """
        try:
            return await self.vector_store.count(filters=filters)
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for retriever components
        """
        health_status = {"status": "healthy", "components": {}}
        
        try:
            # Check embedder
            await self.embedder.health_check()
            health_status["components"]["embedder"] = "healthy"
        except Exception as e:
            health_status["components"]["embedder"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        try:
            # Check vector store
            await self.vector_store.health_check()
            health_status["components"]["vector_store"] = "healthy"
        except Exception as e:
            health_status["components"]["vector_store"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status


class HybridRetriever(Retriever):
    """Hybrid retriever combining vector and keyword search"""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        **kwargs
    ):
        super().__init__(embedder, vector_store, **kwargs)
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector and keyword search
        """
        # Vector search
        vector_results = await super().retrieve(query, filters, top_k * 2, include_metadata)
        
        # Keyword search (simplified - would integrate with full-text search)
        keyword_results = await self._keyword_search(query, filters, top_k * 2)
        
        # Combine and re-score
        combined_results = self._combine_results(vector_results, keyword_results)
        
        # Sort by combined score and return top_k
        sorted_results = sorted(combined_results, key=lambda x: x["combined_score"], reverse=True)
        return sorted_results[:top_k]
    
    async def _keyword_search(self, query: str, filters: Optional[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Simple keyword search implementation
        """
        # This would integrate with Elasticsearch or similar
        # For now, return empty list
        return []
    
    def _combine_results(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """
        Combine vector and keyword search results
        """
        combined = {}
        
        # Add vector results
        for result in vector_results:
            doc_id = result["id"]
            combined[doc_id] = {
                **result,
                "vector_score": result["score"],
                "keyword_score": 0.0,
                "combined_score": result["score"] * self.vector_weight
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["keyword_score"] = result["score"]
                combined[doc_id]["combined_score"] = (
                    combined[doc_id]["vector_score"] * self.vector_weight +
                    result["score"] * self.keyword_weight
                )
            else:
                combined[doc_id] = {
                    **result,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "combined_score": result["score"] * self.keyword_weight
                }
        
        return list(combined.values())
