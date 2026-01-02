"""
Document store for metadata and context management
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import uuid
import hashlib
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class BaseDocumentStore(ABC):
    """Base class for document storage"""
    
    @abstractmethod
    async def store_document(self, document: Dict[str, Any]) -> bool:
        """Store a document"""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        pass
    
    @abstractmethod
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents"""
        pass


class InMemoryDocumentStore(BaseDocumentStore):
    """In-memory document store for development and testing"""
    
    def __init__(self, max_documents: int = 10000):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.document_index: Dict[str, set] = {}  # For text search
        self.max_documents = max_documents
        self._lock = asyncio.Lock()
    
    async def store_document(self, document: Dict[str, Any]) -> bool:
        """
        Store a document in memory
        """
        try:
            document_id = document.get("document_id")
            if not document_id:
                document_id = str(uuid.uuid4())
                document["document_id"] = document_id
            
            # Add timestamp
            document["stored_at"] = datetime.utcnow().isoformat()
            document["updated_at"] = datetime.utcnow().isoformat()
            
            async with self._lock:
                # Check if we need to remove old documents
                if len(self.documents) >= self.max_documents and document_id not in self.documents:
                    await self._remove_oldest_document()
                
                # Store document
                self.documents[document_id] = document.copy()
                
                # Update search index
                await self._update_search_index(document_id, document)
            
            logger.debug(f"Stored document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID
        """
        async with self._lock:
            return self.documents.get(document_id, {}).copy()
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document
        """
        try:
            async with self._lock:
                if document_id not in self.documents:
                    return False
                
                # Update document
                self.documents[document_id].update(updates)
                self.documents[document_id]["updated_at"] = datetime.utcnow().isoformat()
                
                # Update search index
                await self._update_search_index(document_id, self.documents[document_id])
            
            logger.debug(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document
        """
        try:
            async with self._lock:
                if document_id not in self.documents:
                    return False
                
                # Remove from search index
                await self._remove_from_search_index(document_id)
                
                # Delete document
                del self.documents[document_id]
            
            logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents
        """
        try:
            async with self._lock:
                # Simple text search
                query_terms = query.lower().split()
                matching_docs = []
                
                for doc_id, doc in self.documents.items():
                    # Apply filters
                    if filters and not self._matches_filters(doc, filters):
                        continue
                    
                    # Text search
                    content = doc.get("content", "").lower()
                    if all(term in content for term in query_terms):
                        matching_docs.append(doc.copy())
                
                # Sort by relevance (simple: by number of query term matches)
                matching_docs.sort(
                    key=lambda doc: sum(
                        term in doc.get("content", "").lower() 
                        for term in query_terms
                    ),
                    reverse=True
                )
                
                return matching_docs[:limit]
                
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def store_context(self, context_data: Dict[str, Any]) -> bool:
        """
        Store query context
        """
        try:
            context_id = context_data.get("query_id")
            if not context_id:
                context_id = str(uuid.uuid4())
                context_data["query_id"] = context_id
            
            # Store as a special type of document
            context_document = {
                "document_id": f"context_{context_id}",
                "type": "context",
                "data": context_data,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            return await self.store_document(context_document)
            
        except Exception as e:
            logger.error(f"Error storing context: {str(e)}")
            return False
    
    async def get_context(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored context
        """
        try:
            context_doc = await self.get_document(f"context_{query_id}")
            if context_doc:
                return context_doc.get("data")
            return None
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return None
    
    async def delete_context(self, query_id: str) -> bool:
        """
        Delete stored context
        """
        return await self.delete_document(f"context_{query_id}")
    
    async def get_user_contexts(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get contexts for a specific user
        """
        try:
            async with self._lock:
                user_contexts = []
                
                for doc in self.documents.values():
                    if doc.get("type") == "context":
                        data = doc.get("data", {})
                        if data.get("user_id") == user_id:
                            user_contexts.append(data)
                
                # Sort by timestamp (most recent first)
                user_contexts.sort(
                    key=lambda x: x.get("timestamp", 0),
                    reverse=True
                )
                
                return user_contexts[:limit]
                
        except Exception as e:
            logger.error(f"Error getting user contexts: {str(e)}")
            return []
    
    async def get_session_contexts(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get contexts for a specific session
        """
        try:
            async with self._lock:
                session_contexts = []
                
                for doc in self.documents.values():
                    if doc.get("type") == "context":
                        data = doc.get("data", {})
                        if data.get("session_id") == session_id:
                            session_contexts.append(data)
                
                # Sort by timestamp (most recent first)
                session_contexts.sort(
                    key=lambda x: x.get("timestamp", 0),
                    reverse=True
                )
                
                return session_contexts[:limit]
                
        except Exception as e:
            logger.error(f"Error getting session contexts: {str(e)}")
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
            async with self._lock:
                matching_contexts = []
                query_terms = query.lower().split()
                
                for doc in self.documents.values():
                    if doc.get("type") != "context":
                        continue
                    
                    data = doc.get("data", {})
                    
                    # Filter by user if specified
                    if user_id and data.get("user_id") != user_id:
                        continue
                    
                    # Search in query and answer
                    query_text = data.get("query", "").lower()
                    answer_text = data.get("answer", "").lower()
                    combined_text = f"{query_text} {answer_text}"
                    
                    if all(term in combined_text for term in query_terms):
                        matching_contexts.append(data)
                
                # Sort by timestamp (most recent first)
                matching_contexts.sort(
                    key=lambda x: x.get("timestamp", 0),
                    reverse=True
                )
                
                return matching_contexts[:limit]
                
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
            async with self._lock:
                analytics = {
                    "total_contexts": 0,
                    "unique_users": set(),
                    "unique_sessions": set(),
                    "avg_query_length": 0.0,
                    "avg_answer_length": 0.0,
                    "time_range": time_range.total_seconds() if time_range else None
                }
                
                current_time = datetime.utcnow()
                cutoff_time = current_time - time_range if time_range else None
                
                total_query_length = 0
                total_answer_length = 0
                context_count = 0
                
                for doc in self.documents.values():
                    if doc.get("type") != "context":
                        continue
                    
                    data = doc.get("data", {})
                    
                    # Filter by user if specified
                    if user_id and data.get("user_id") != user_id:
                        continue
                    
                    # Filter by time range if specified
                    if cutoff_time:
                        timestamp_str = data.get("timestamp")
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if timestamp < cutoff_time:
                                    continue
                            except:
                                pass
                    
                    # Update analytics
                    analytics["total_contexts"] += 1
                    context_count += 1
                    
                    if data.get("user_id"):
                        analytics["unique_users"].add(data["user_id"])
                    
                    if data.get("session_id"):
                        analytics["unique_sessions"].add(data["session_id"])
                    
                    query_length = len(data.get("query", ""))
                    answer_length = len(data.get("answer", ""))
                    
                    total_query_length += query_length
                    total_answer_length += answer_length
                
                # Calculate averages
                if context_count > 0:
                    analytics["avg_query_length"] = total_query_length / context_count
                    analytics["avg_answer_length"] = total_answer_length / context_count
                
                # Convert sets to counts
                analytics["unique_users"] = len(analytics["unique_users"])
                analytics["unique_sessions"] = len(analytics["unique_sessions"])
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting context analytics: {str(e)}")
            return {}
    
    async def _update_search_index(self, document_id: str, document: Dict[str, Any]):
        """
        Update the search index for a document
        """
        content = document.get("content", "").lower()
        words = content.split()
        
        for word in words:
            if word not in self.document_index:
                self.document_index[word] = set()
            self.document_index[word].add(document_id)
    
    async def _remove_from_search_index(self, document_id: str):
        """
        Remove a document from the search index
        """
        for word, doc_set in self.document_index.items():
            doc_set.discard(document_id)
        
        # Remove empty word entries
        self.document_index = {
            word: doc_set for word, doc_set in self.document_index.items()
            if doc_set
        }
    
    async def _remove_oldest_document(self):
        """
        Remove the oldest document to make space
        """
        if not self.documents:
            return
        
        oldest_doc_id = min(
            self.documents.keys(),
            key=lambda doc_id: self.documents[doc_id].get("stored_at", "")
        )
        
        await self.delete_document(oldest_doc_id)
        logger.info(f"Removed oldest document {oldest_doc_id} to make space")
    
    def _matches_filters(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if a document matches the given filters
        """
        for key, value in filters.items():
            if key not in document:
                return False
            
            if isinstance(value, dict):
                # Handle nested filters
                if not self._matches_filters(document.get(key, {}), value):
                    return False
            elif document[key] != value:
                return False
        
        return True
    
    async def get_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document store
        """
        async with self._lock:
            total_docs = len(self.documents)
            context_docs = sum(1 for doc in self.documents.values() if doc.get("type") == "context")
            regular_docs = total_docs - context_docs
            
            return {
                "total_documents": total_docs,
                "regular_documents": regular_docs,
                "context_documents": context_docs,
                "max_documents": self.max_documents,
                "utilization": total_docs / self.max_documents,
                "search_index_size": len(self.document_index)
            }
    
    async def clear_all(self):
        """
        Clear all documents from the store
        """
        async with self._lock:
            self.documents.clear()
            self.document_index.clear()
        
        logger.info("Cleared all documents from store")


class DocumentStoreManager:
    """Manager for document store with multiple backends"""
    
    def __init__(self, document_store: BaseDocumentStore):
        self.document_store = document_store
    
    async def store_document(self, document: Dict[str, Any]) -> bool:
        """Store a document"""
        return await self.document_store.store_document(document)
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return await self.document_store.get_document(document_id)
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update a document"""
        return await self.document_store.update_document(document_id, updates)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        return await self.document_store.delete_document(document_id)
    
    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents"""
        return await self.document_store.search_documents(query, filters, limit)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the document store"""
        try:
            # Try to store and retrieve a test document
            test_doc = {
                "document_id": "health_check_test",
                "content": "Health check test document",
                "test": True
            }
            
            stored = await self.document_store.store_document(test_doc)
            if not stored:
                return {"status": "unhealthy", "error": "Failed to store test document"}
            
            retrieved = await self.document_store.get_document("health_check_test")
            if not retrieved:
                return {"status": "unhealthy", "error": "Failed to retrieve test document"}
            
            # Clean up
            await self.document_store.delete_document("health_check_test")
            
            return {"status": "healthy"}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
