"""
In-memory document store implementation
"""

import asyncio
from typing import Dict, Any, List, Optional
import time
import logging


logger = logging.getLogger(__name__)


class MemoryDocumentStore:
    """In-memory document store for demo purposes"""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.query_history: List[Dict[str, Any]] = []
        self.context_store: Dict[str, Dict[str, Any]] = {}
    
    async def store_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a document"""
        try:
            self.documents[document_id] = {
                "id": document_id,
                "content": content,
                "metadata": metadata or {},
                "created_at": time.time(),
                "updated_at": time.time()
            }
            return True
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {e}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return self.documents.get(document_id)
    
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document"""
        if document_id not in self.documents:
            return False
        
        doc = self.documents[document_id]
        if content is not None:
            doc["content"] = content
        if metadata is not None:
            doc["metadata"].update(metadata)
        doc["updated_at"] = time.time()
        
        return True
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False
    
    async def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List documents with optional filters"""
        docs = list(self.documents.values())
        
        if filters:
            filtered_docs = []
            for doc in docs:
                match = True
                for key, value in filters.items():
                    if doc.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            docs = filtered_docs
        
        return docs[:limit]
    
    async def store_context(
        self,
        query_id: str,
        query: str,
        context: List[Dict[str, Any]],
        answer: str
    ) -> bool:
        """Store query context"""
        try:
            self.context_store[query_id] = {
                "query_id": query_id,
                "query": query,
                "context": context,
                "answer": answer,
                "timestamp": time.time()
            }
            return True
        except Exception as e:
            logger.error(f"Failed to store context for {query_id}: {e}")
            return False
    
    async def get_context(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get stored context"""
        return self.context_store.get(query_id)
    
    async def clear_context(self, query_id: str) -> bool:
        """Clear stored context"""
        if query_id in self.context_store:
            del self.context_store[query_id]
            return True
        return False
    
    async def store_query_history(
        self,
        user_id: Optional[str],
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store query in history"""
        try:
            history_entry = {
                "user_id": user_id,
                "query": query,
                "answer": answer,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            self.query_history.append(history_entry)
            
            # Keep only last 1000 entries
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            return True
        except Exception as e:
            logger.error(f"Failed to store query history: {e}")
            return False
    
    async def get_query_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get query history"""
        history = self.query_history
        
        if user_id:
            history = [entry for entry in history if entry.get("user_id") == user_id]
        
        # Sort by timestamp (most recent first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history[:limit]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "total_documents": len(self.documents),
            "total_contexts": len(self.context_store),
            "total_queries": len(self.query_history),
            "memory_usage": "N/A"  # Could implement actual memory tracking
        }
