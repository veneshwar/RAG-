"""
FAISS vector store implementation
"""

import os
import asyncio
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

try:
    import faiss
except ImportError:
    faiss = None

from app.indexing.vectorstores.base import VectorStore


logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store"""
    
    def __init__(
        self,
        store_path: str,
        dimension: int = 1536,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        self.store_path = store_path
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.documents = []  # Store document metadata
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_path = os.path.join(self.store_path, "faiss.index")
        docs_path = os.path.join(self.store_path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_index()
        else:
            self._create_index()
    
    def _create_index(self):
        """Create new FAISS index"""
        if faiss is None:
            logger.warning("FAISS not installed, using mock store")
            self.index = None
            return
        
        if self.index_type == "flat":
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            nlist = min(100, max(1, len(self.documents) // 10))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.documents = []
        logger.info(f"Created new FAISS index with type {self.index_type}")
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add documents to the vector store"""
        if self.index is None:
            # Mock implementation
            doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                self.documents.append({
                    "id": doc_ids[i],
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "embedding": emb
                })
            return doc_ids
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings_array)
        
        # Generate document IDs
        doc_ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings_array)
        
        # Store document metadata
        for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            self.documents.append({
                "id": doc_id,
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "index": start_idx + i
            })
        
        # Save to disk
        await self._save()
        
        return doc_ids
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            # Mock implementation
            results = []
            for doc in self.documents[:top_k]:
                if self._passes_filters(doc, filters):
                    results.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": 0.8  # Mock score
                    })
            return results
        
        # Convert query to numpy array
        query_array = np.array([query_vector], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, min(top_k, len(self.documents)))
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            doc = self.documents[idx]
            if self._passes_filters(doc, filters):
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        
        return results
    
    def _passes_filters(self, doc: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if document passes filters"""
        if not filters:
            return True
        
        metadata = doc.get("metadata", {})
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        
        return True
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the store"""
        # FAISS doesn't support easy deletion, so we rebuild the index
        docs_to_keep = [doc for doc in self.documents if doc["id"] not in document_ids]
        
        if len(docs_to_keep) == len(self.documents):
            return True  # Nothing to delete
        
        # Rebuild index
        self.documents = docs_to_keep
        self._create_index()
        
        if docs_to_keep and self.index is not None:
            # Re-add remaining documents
            embeddings = [doc.get("embedding", [0.0] * self.dimension) for doc in docs_to_keep]
            if embeddings:
                await self.add_documents(docs_to_keep, embeddings)
        
        return True
    
    async def update_document(
        self,
        document_id: str,
        document: Dict[str, Any],
        embedding: List[float]
    ) -> bool:
        """Update a document in the store"""
        # Delete and re-add
        await self.delete_documents([document_id])
        await self.add_documents([document], [embedding])
        return True
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        for doc in self.documents:
            if doc["id"] == document_id:
                return doc
        return None
    
    async def _save(self):
        """Save index and documents to disk"""
        if self.index is None:
            return
        
        try:
            index_path = os.path.join(self.store_path, "faiss.index")
            docs_path = os.path.join(self.store_path, "documents.pkl")
            
            faiss.write_index(self.index, index_path)
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "index_size": self.index.ntotal if self.index else 0
        }
