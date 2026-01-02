"""
Context manager for RAG system
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
import time
import logging
from datetime import datetime, timedelta

from app.data.memory_store import MemoryDocumentStore


logger = logging.getLogger(__name__)


class ContextManager:
    """Manages query context and conversation history"""
    
    def __init__(
        self,
        document_store: MemoryDocumentStore,
        context_ttl: int = 3600,  # 1 hour
        max_contexts_per_user: int = 100
    ):
        self.document_store = document_store
        self.context_ttl = context_ttl
        self.max_contexts_per_user = max_contexts_per_user
        self._memory_cache = {}  # In-memory cache for recent contexts
    
    async def store_context(
        self,
        query_id: str,
        query: str,
        context: List[Dict[str, Any]],
        answer: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store query context for future reference
        """
        try:
            context_data = {
                "query_id": query_id,
                "query": query,
                "context": context,
                "answer": answer,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
                "timestamp": time.time(),
                "expires_at": time.time() + self.context_ttl
            }
            
            # Store in memory cache
            self._memory_cache[query_id] = context_data
            
            # Store in persistent storage
            await self.document_store.store_context(context_data)
            
            # Clean up old contexts for user
            if user_id:
                await self._cleanup_old_contexts(user_id)
            
            logger.info(f"Stored context for query {query_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing context: {str(e)}")
            return False
    
    async def get_context(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve context for a specific query
        """
        try:
            # Check memory cache first
            if query_id in self._memory_cache:
                context_data = self._memory_cache[query_id]
                
                # Check if expired
                if time.time() < context_data["expires_at"]:
                    return context_data
                else:
                    # Remove from cache if expired
                    del self._memory_cache[query_id]
            
            # Retrieve from persistent storage
            context_data = await self.document_store.get_context(query_id)
            
            if context_data:
                # Check if expired
                if time.time() < context_data.get("expires_at", time.time()):
                    # Add to memory cache
                    self._memory_cache[query_id] = context_data
                    return context_data
                else:
                    logger.info(f"Context {query_id} has expired")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return None
    
    async def clear_context(self, query_id: str) -> bool:
        """
        Clear context for a specific query
        """
        try:
            # Remove from memory cache
            if query_id in self._memory_cache:
                del self._memory_cache[query_id]
            
            # Remove from persistent storage
            await self.document_store.delete_context(query_id)
            
            logger.info(f"Cleared context for query {query_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return False
    
    async def get_user_contexts(
        self,
        user_id: str,
        limit: int = 50,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all contexts for a user
        """
        try:
            contexts = await self.document_store.get_user_contexts(
                user_id=user_id,
                limit=limit
            )
            
            if not include_expired:
                current_time = time.time()
                contexts = [
                    ctx for ctx in contexts
                    if ctx.get("expires_at", current_time) > current_time
                ]
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting user contexts: {str(e)}")
            return []
    
    async def get_session_contexts(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all contexts for a session
        """
        try:
            contexts = await self.document_store.get_session_contexts(
                session_id=session_id,
                limit=limit
            )
            
            # Filter out expired contexts
            current_time = time.time()
            contexts = [
                ctx for ctx in contexts
                if ctx.get("expires_at", current_time) > current_time
            ]
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting session contexts: {str(e)}")
            return []
    
    async def get_query_history(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get query history for user or session
        """
        try:
            if user_id:
                contexts = await self.get_user_contexts(user_id, limit)
            elif session_id:
                contexts = await self.get_session_contexts(session_id, limit)
            else:
                contexts = []
            
            # Return simplified query history
            history = []
            for ctx in contexts:
                history.append({
                    "query_id": ctx["query_id"],
                    "query": ctx["query"],
                    "answer": ctx["answer"],
                    "timestamp": ctx["timestamp"],
                    "metadata": ctx.get("metadata", {})
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting query history: {str(e)}")
            return []
    
    async def search_contexts(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search through stored contexts
        """
        try:
            results = await self.document_store.search_contexts(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching contexts: {str(e)}")
            return []
    
    async def get_context_analytics(
        self,
        user_id: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get analytics about stored contexts
        """
        try:
            analytics = await self.document_store.get_context_analytics(
                user_id=user_id,
                time_range=time_range
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting context analytics: {str(e)}")
            return {}
    
    async def _cleanup_old_contexts(self, user_id: str):
        """
        Clean up old contexts for a user to stay within limit
        """
        try:
            user_contexts = await self.get_user_contexts(
                user_id=user_id,
                limit=self.max_contexts_per_user + 10
            )
            
            if len(user_contexts) > self.max_contexts_per_user:
                # Sort by timestamp (oldest first)
                user_contexts.sort(key=lambda x: x["timestamp"])
                
                # Remove oldest contexts
                to_remove = len(user_contexts) - self.max_contexts_per_user
                for i in range(to_remove):
                    old_context = user_contexts[i]
                    await self.clear_context(old_context["query_id"])
                
                logger.info(f"Cleaned up {to_remove} old contexts for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old contexts: {str(e)}")
    
    async def extend_context_ttl(self, query_id: str, additional_seconds: int) -> bool:
        """
        Extend TTL for a specific context
        """
        try:
            context_data = await self.get_context(query_id)
            if not context_data:
                return False
            
            # Update expiration time
            new_expires_at = context_data["expires_at"] + additional_seconds
            context_data["expires_at"] = new_expires_at
            
            # Update in memory cache
            self._memory_cache[query_id] = context_data
            
            # Update in persistent storage
            await self.document_store.update_context(query_id, {
                "expires_at": new_expires_at
            })
            
            logger.info(f"Extended TTL for context {query_id} by {additional_seconds} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error extending context TTL: {str(e)}")
            return False
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics
        """
        try:
            cache_size = len(self._memory_cache)
            total_memory = sum(
                len(json.dumps(ctx, default=str)) 
                for ctx in self._memory_cache.values()
            )
            
            return {
                "cache_size": cache_size,
                "memory_usage_bytes": total_memory,
                "memory_usage_mb": round(total_memory / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {"cache_size": 0, "memory_usage_bytes": 0, "memory_usage_mb": 0}
