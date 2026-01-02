"""
Document reranker for RAG system
"""

import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import logging

from app.models.llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class Reranker:
    """Document reranker using various strategies"""
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        strategy: str = "simple_score",
        top_k: int = 10
    ):
        self.llm_client = llm_client
        self.strategy = strategy
        self.top_k = top_k
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        """
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        try:
            if self.strategy == "cross_encoder":
                return await self._cross_encoder_rerank(query, documents, top_k)
            elif self.strategy == "llm_rerank":
                return await self._llm_rerank(query, documents, top_k)
            elif self.strategy == "multi_criteria":
                return await self._multi_criteria_rerank(query, documents, top_k)
            else:
                # Default: return original order
                return documents[:top_k]
                
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
            # Fallback to original order
            return documents[:top_k]
    
    async def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using cross-encoder model (simplified implementation)
        """
        # This would integrate with a cross-encoder model like sentence-transformers
        # For now, implement a simple scoring based on keyword matching
        
        scored_docs = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words = set(content.split())
            
            # Simple relevance score based on word overlap
            overlap = len(query_words & doc_words)
            total_words = len(query_words | doc_words)
            jaccard_similarity = overlap / total_words if total_words > 0 else 0
            
            # Combine with original score
            original_score = doc.get("score", 0.0)
            combined_score = 0.7 * original_score + 0.3 * jaccard_similarity
            
            scored_docs.append({
                **doc,
                "rerank_score": combined_score
            })
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]
    
    async def _llm_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using LLM-based scoring
        """
        if not self.llm_client:
            logger.warning("LLM client not available, falling back to cross-encoder")
            return await self._cross_encoder_rerank(query, documents, top_k)
        
        scored_docs = []
        
        for doc in documents:
            try:
                # Create scoring prompt
                prompt = self._create_scoring_prompt(query, doc.get("content", ""))
                
                # Get relevance score from LLM
                response = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.1
                )
                
                # Extract score from response
                score = self._extract_score(response)
                
                scored_docs.append({
                    **doc,
                    "rerank_score": score
                })
                
            except Exception as e:
                logger.warning(f"Error scoring document with LLM: {str(e)}")
                # Use original score as fallback
                scored_docs.append({
                    **doc,
                    "rerank_score": doc.get("score", 0.0)
                })
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]
    
    async def _multi_criteria_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using multiple criteria
        """
        scored_docs = []
        
        for doc in documents:
            scores = {}
            
            # Original relevance score
            scores["relevance"] = doc.get("score", 0.0)
            
            # Recency score (if timestamp available)
            scores["recency"] = self._calculate_recency_score(doc)
            
            # Content quality score
            scores["quality"] = self._calculate_quality_score(doc)
            
            # Diversity score (to avoid similar content)
            scores["diversity"] = 0.5  # Placeholder
            
            # Weighted combination
            weights = {
                "relevance": 0.5,
                "recency": 0.2,
                "quality": 0.2,
                "diversity": 0.1
            }
            
            combined_score = sum(
                scores[criterion] * weights[criterion]
                for criterion in weights
            )
            
            scored_docs.append({
                **doc,
                "rerank_score": combined_score,
                "score_breakdown": scores
            })
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]
    
    def _create_scoring_prompt(self, query: str, content: str) -> str:
        """
        Create prompt for LLM-based scoring
        """
        return f"""Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.

Query: {query}

Document: {content[:500]}...

Respond with only the numerical score (e.g., 0.85):"""
    
    def _extract_score(self, response: str) -> float:
        """
        Extract numerical score from LLM response
        """
        try:
            # Try to extract float from response
            import re
            match = re.search(r'0\.\d+|1\.0|0\.0', response)
            if match:
                return float(match.group())
            else:
                # Fallback to default score
                return 0.5
        except:
            return 0.5
    
    def _calculate_recency_score(self, doc: Dict[str, Any]) -> float:
        """
        Calculate recency score based on document timestamp
        """
        import time
        
        timestamp = doc.get("timestamp")
        if not timestamp:
            return 0.5
        
        # Calculate age in days
        age_days = (time.time() - timestamp) / (24 * 60 * 60)
        
        # Exponential decay
        recency_score = np.exp(-age_days / 30)  # Half-life of 30 days
        return float(recency_score)
    
    def _calculate_quality_score(self, doc: Dict[str, Any]) -> float:
        """
        Calculate content quality score
        """
        content = doc.get("content", "")
        
        if not content:
            return 0.0
        
        # Simple quality metrics
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 0-1
        
        # Check for structured content
        has_structure = any(indicator in content.lower() for indicator in [
            "introduction", "conclusion", "summary", "therefore", "however"
        ])
        structure_score = 0.8 if has_structure else 0.5
        
        # Check for excessive repetition
        words = content.lower().split()
        diversity_score = len(set(words)) / len(words) if words else 0
        
        # Combine scores
        quality_score = (length_score + structure_score + diversity_score) / 3
        return quality_score
    
    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch rerank multiple query-document pairs
        """
        tasks = [
            self.rerank(query, docs, top_k)
            for query, docs in zip(queries, documents_list)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error reranking batch {i}: {str(result)}")
                processed_results.append(documents_list[i][:top_k or self.top_k])
            else:
                processed_results.append(result)
        
        return processed_results
