"""
Base embedder interface and implementations
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np


logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for text embedders"""
    
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        max_sequence_length: int = 512,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch
        """
        pass
    
    async def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents
        """
        try:
            # Extract text content
            texts = [doc.get("content", "") for doc in documents]
            
            # Generate embeddings
            embeddings = await self.embed_batch(texts)
            
            # Add embeddings to documents
            embedded_documents = []
            for doc, embedding in zip(documents, embeddings):
                embedded_doc = doc.copy()
                embedded_doc["embedding"] = embedding
                embedded_doc["embedding_metadata"] = {
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "max_sequence_length": self.max_sequence_length
                }
                embedded_documents.append(embedded_doc)
            
            return embedded_documents
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find most similar documents to query embedding
        """
        try:
            if not document_embeddings:
                return []
            
            # Calculate cosine similarities
            similarities = []
            query_np = np.array(query_embedding)
            
            for i, doc_embedding in enumerate(document_embeddings):
                doc_np = np.array(doc_embedding)
                
                # Cosine similarity
                similarity = np.dot(query_np, doc_np) / (
                    np.linalg.norm(query_np) * np.linalg.norm(doc_np)
                )
                
                similarities.append({
                    "index": i,
                    "similarity": float(similarity)
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the embedder
        """
        try:
            # Test embedding with simple text
            test_embedding = await self.embed_text("Hello world")
            
            if len(test_embedding) == self.embedding_dim:
                return {
                    "status": "healthy",
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim,
                    "test_embedding_length": len(test_embedding)
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(test_embedding)}"
                }
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedder model
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size
        }


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing purposes"""
    
    def __init__(self, embedding_dim: int = 768, **kwargs):
        super().__init__(
            model_name="mock-embedder",
            embedding_dim=embedding_dim,
            **kwargs
        )
        np.random.seed(42)  # For reproducible results
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate mock embedding based on text hash
        """
        # Generate deterministic embedding based on text content
        text_hash = hash(text)
        np.random.seed(text_hash)
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for multiple texts
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        
        return embeddings


class CachedEmbedder(BaseEmbedder):
    """Embedder with caching functionality"""
    
    def __init__(
        self,
        base_embedder: BaseEmbedder,
        cache_size: int = 10000,
        cache_ttl: int = 3600  # 1 hour
    ):
        super().__init__(
            model_name=base_embedder.model_name,
            embedding_dim=base_embedder.embedding_dim,
            max_sequence_length=base_embedder.max_sequence_length,
            batch_size=base_embedder.batch_size
        )
        
        self.base_embedder = base_embedder
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamps = {}
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed text with caching
        """
        # Check cache
        cache_key = hash(text)
        current_time = asyncio.get_event_loop().time()
        
        if (cache_key in self._cache and 
            current_time - self._cache_timestamps[cache_key] < self.cache_ttl):
            return self._cache[cache_key]
        
        # Generate embedding
        embedding = await self.base_embedder.embed_text(text)
        
        # Update cache
        self._cache[cache_key] = embedding
        self._cache_timestamps[cache_key] = current_time
        
        # Clean up old cache entries
        if len(self._cache) > self.cache_size:
            await self._cleanup_cache()
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch with caching
        """
        # Check cache for each text
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            current_time = asyncio.get_event_loop().time()
            
            if (cache_key in self._cache and 
                current_time - self._cache_timestamps[cache_key] < self.cache_ttl):
                cached_embeddings[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self.base_embedder.embed_batch(uncached_texts)
            
            # Update cache
            for text, embedding, index in zip(uncached_texts, new_embeddings, uncached_indices):
                cache_key = hash(text)
                current_time = asyncio.get_event_loop().time()
                self._cache[cache_key] = embedding
                self._cache_timestamps[cache_key] = current_time
                cached_embeddings[index] = embedding
        
        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]
    
    async def _cleanup_cache(self):
        """
        Clean up old cache entries
        """
        current_time = asyncio.get_event_loop().time()
        
        # Remove expired entries
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]
        
        # If still too many entries, remove oldest ones
        if len(self._cache) > self.cache_size:
            sorted_items = sorted(
                self._cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            to_remove = len(self._cache) - self.cache_size
            for key, _ in sorted_items[:to_remove]:
                del self._cache[key]
                del self._cache_timestamps[key]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check including cache status
        """
        base_health = await self.base_embedder.health_check()
        
        return {
            **base_health,
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl
        }


class EnsembleEmbedder(BaseEmbedder):
    """
    Ensemble embedder that combines multiple embedders
    """
    
    def __init__(
        self,
        embedders: List[BaseEmbedder],
        weights: Optional[List[float]] = None,
        combination_method: str = "weighted_average"  # "weighted_average", "concatenate"
    ):
        if not embedders:
            raise ValueError("At least one embedder is required")
        
        # Determine embedding dimension based on combination method
        if combination_method == "concatenate":
            embedding_dim = sum(emb.embedding_dim for emb in embedders)
        else:  # weighted_average
            embedding_dim = embedders[0].embedding_dim
            # Ensure all embedders have same dimension
            for emb in embedders:
                if emb.embedding_dim != embedding_dim:
                    raise ValueError("All embedders must have same embedding dimension for weighted average")
        
        super().__init__(
            model_name=f"ensemble-{combination_method}",
            embedding_dim=embedding_dim,
            max_sequence_length=max(emb.max_sequence_length for emb in embedders),
            batch_size=min(emb.batch_size for emb in embedders)
        )
        
        self.embedders = embedders
        self.weights = weights or [1.0 / len(embedders)] * len(embedders)
        self.combination_method = combination_method
        
        if len(self.weights) != len(embedders):
            raise ValueError("Number of weights must match number of embedders")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed text using ensemble of embedders
        """
        # Get embeddings from all embedders
        embeddings = await asyncio.gather(
            *[emb.embed_text(text) for emb in self.embedders],
            return_exceptions=True
        )
        
        # Handle exceptions
        valid_embeddings = []
        valid_weights = []
        
        for emb, weight in zip(embeddings, self.weights):
            if isinstance(emb, Exception):
                logger.error(f"Error in ensemble embedder: {str(emb)}")
                continue
            valid_embeddings.append(emb)
            valid_weights.append(weight)
        
        if not valid_embeddings:
            raise RuntimeError("All embedders failed")
        
        # Normalize weights for valid embeddings
        total_weight = sum(valid_weights)
        valid_weights = [w / total_weight for w in valid_weights]
        
        # Combine embeddings
        if self.combination_method == "weighted_average":
            return self._weighted_average(valid_embeddings, valid_weights)
        elif self.combination_method == "concatenate":
            return self._concatenate(valid_embeddings)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch using ensemble of embedders
        """
        # Get embeddings from all embedders
        batch_results = await asyncio.gather(
            *[emb.embed_batch(texts) for emb in self.embedders],
            return_exceptions=True
        )
        
        # Handle exceptions and combine
        combined_embeddings = []
        
        for i in range(len(texts)):
            text_embeddings = []
            text_weights = []
            
            for j, batch_result in enumerate(batch_results):
                if isinstance(batch_result, Exception):
                    logger.error(f"Error in ensemble embedder {j} for text {i}: {str(batch_result)}")
                    continue
                
                text_embeddings.append(batch_result[i])
                text_weights.append(self.weights[j])
            
            if not text_embeddings:
                raise RuntimeError(f"All embedders failed for text {i}")
            
            # Normalize weights
            total_weight = sum(text_weights)
            text_weights = [w / total_weight for w in text_weights]
            
            # Combine embeddings
            if self.combination_method == "weighted_average":
                combined = self._weighted_average(text_embeddings, text_weights)
            elif self.combination_method == "concatenate":
                combined = self._concatenate(text_embeddings)
            else:
                raise ValueError(f"Unknown combination method: {self.combination_method}")
            
            combined_embeddings.append(combined)
        
        return combined_embeddings
    
    def _weighted_average(self, embeddings: List[List[float]], weights: List[float]) -> List[float]:
        """
        Compute weighted average of embeddings
        """
        if not embeddings:
            return []
        
        # Convert to numpy arrays
        emb_arrays = [np.array(emb) for emb in embeddings]
        
        # Compute weighted average
        weighted_sum = sum(w * emb for w, emb in zip(weights, emb_arrays))
        return weighted_sum.tolist()
    
    def _concatenate(self, embeddings: List[List[float]]) -> List[float]:
        """
        Concatenate embeddings
        """
        concatenated = []
        for emb in embeddings:
            concatenated.extend(emb)
        return concatenated
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for ensemble embedder
        """
        health_checks = await asyncio.gather(
            *[emb.health_check() for emb in self.embedders],
            return_exceptions=True
        )
        
        healthy_count = 0
        total_count = len(self.embedders)
        
        for i, health in enumerate(health_checks):
            if isinstance(health, Exception):
                logger.error(f"Health check failed for embedder {i}: {str(health)}")
            elif health.get("status") == "healthy":
                healthy_count += 1
        
        status = "healthy" if healthy_count == total_count else "degraded" if healthy_count > 0 else "unhealthy"
        
        return {
            "status": status,
            "healthy_embedders": healthy_count,
            "total_embedders": total_count,
            "combination_method": self.combination_method,
            "weights": self.weights
        }
