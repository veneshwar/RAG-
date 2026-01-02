"""
Base vector store interface
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Base class for vector stores"""
    
    def __init__(
        self,
        dimension: int,
        metric_type: str = "cosine",
        **kwargs
    ):
        self.dimension = dimension
        self.metric_type = metric_type
    
    @abstractmethod
    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add vectors to the store"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    async def update(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a vector"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the vector store"""
        return {"status": "healthy"}
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        return {
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "type": self.__class__.__name__
        }


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, dimension: int, metric_type: str = "cosine", **kwargs):
        super().__init__(dimension, metric_type, **kwargs)
        self.index = None
        self.id_mapping = {}
        self.metadata_store = {}
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            if self.metric_type == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric_type == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"Initialized FAISS index with metric {self.metric_type}")
            
        except ImportError:
            logger.error("FAISS not installed")
            raise
    
    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add vectors to FAISS index"""
        try:
            import faiss
            
            if len(vectors) != len(ids):
                raise ValueError("Vectors and IDs must have same length")
            
            # Convert to numpy array
            vectors_np = np.array(vectors, dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.metric_type == "cosine":
                faiss.normalize_L2(vectors_np)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(vectors_np)
            
            # Update mappings
            for i, (vector_id, vector) in enumerate(zip(ids, vectors)):
                idx = start_idx + i
                self.id_mapping[vector_id] = idx
                if metadata and i < len(metadata):
                    self.metadata_store[vector_id] = metadata[i]
                else:
                    self.metadata_store[vector_id] = {}
            
            logger.debug(f"Added {len(vectors)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {str(e)}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            import faiss
            
            if self.index.ntotal == 0:
                return []
            
            # Convert query to numpy and normalize
            query_np = np.array([query_vector], dtype=np.float32)
            if self.metric_type == "cosine":
                faiss.normalize_L2(query_np)
            
            # Search
            scores, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Find ID from index mapping
                vector_id = None
                for vid, vidx in self.id_mapping.items():
                    if vidx == idx:
                        vector_id = vid
                        break
                
                if vector_id:
                    result = {
                        "id": vector_id,
                        "score": float(score),
                        "metadata": self.metadata_store.get(vector_id, {})
                    }
                    
                    # Apply filters
                    if not filters or self._matches_filters(result["metadata"], filters):
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            return []
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (not fully supported in FAISS)"""
        logger.warning("FAISS doesn't support efficient deletion")
        return False
    
    async def update(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a vector (not fully supported in FAISS)"""
        logger.warning("FAISS doesn't support efficient updates")
        return False
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for FAISS"""
        base_health = await super().health_check()
        
        return {
            **base_health,
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "metric_type": self.metric_type
        }


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for testing"""
    
    def __init__(self, dimension: int, metric_type: str = "cosine", **kwargs):
        super().__init__(dimension, metric_type, **kwargs)
        self.vectors = {}
        self.metadata = {}
    
    async def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add vectors to memory"""
        try:
            for i, (vector_id, vector) in enumerate(zip(ids, vectors)):
                self.vectors[vector_id] = vector
                if metadata and i < len(metadata):
                    self.metadata[vector_id] = metadata[i]
                else:
                    self.metadata[vector_id] = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to memory: {str(e)}")
            return False
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            if not self.vectors:
                return []
            
            similarities = []
            query_np = np.array(query_vector)
            
            for vector_id, vector in self.vectors.items():
                vector_np = np.array(vector)
                
                if self.metric_type == "cosine":
                    similarity = np.dot(query_np, vector_np) / (
                        np.linalg.norm(query_np) * np.linalg.norm(vector_np)
                    )
                else:  # l2
                    similarity = -np.linalg.norm(query_np - vector_np)
                
                # Apply filters
                metadata = self.metadata.get(vector_id, {})
                if not filters or self._matches_filters(metadata, filters):
                    similarities.append({
                        "id": vector_id,
                        "score": float(similarity),
                        "metadata": metadata
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["score"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching memory vectors: {str(e)}")
            return []
    
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        try:
            for vector_id in ids:
                self.vectors.pop(vector_id, None)
                self.metadata.pop(vector_id, None)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False
    
    async def update(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a vector"""
        try:
            self.vectors[id] = vector
            if metadata is not None:
                self.metadata[id] = metadata
            return True
            
        except Exception as e:
            logger.error(f"Error updating vector: {str(e)}")
            return False
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for in-memory store"""
        base_health = await super().health_check()
        
        return {
            **base_health,
            "vector_count": len(self.vectors),
            "dimension": self.dimension,
            "metric_type": self.metric_type
        }
