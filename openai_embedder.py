"""
OpenAI embedder implementation
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI

from app.indexing.embeddings.embedder import BaseEmbedder


logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder using their embedding API"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        embedding_dim: Optional[int] = None,
        max_sequence_length: int = 8192,
        batch_size: int = 100,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        # Set default embedding dimensions based on model
        if embedding_dim is None:
            if "3-small" in model_name:
                embedding_dim = 1536
            elif "3-large" in model_name:
                embedding_dim = 3072
            elif "ada-002" in model_name:
                embedding_dim = 1536
            else:
                embedding_dim = 1536  # Default
        
        super().__init__(
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size
        )
        
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries
        )
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string using OpenAI API
        """
        try:
            # Truncate text if too long
            if len(text) > self.max_sequence_length:
                text = text[:self.max_sequence_length]
                logger.warning(f"Text truncated to {self.max_sequence_length} characters")
            
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {len(embedding)}"
                )
            
            return embedding
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error embedding text with OpenAI: {str(e)}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch using OpenAI API
        """
        if not texts:
            return []
        
        try:
            # Process in batches to respect API limits
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Truncate texts if too long
                truncated_texts = []
                for text in batch_texts:
                    if len(text) > self.max_sequence_length:
                        truncated_texts.append(text[:self.max_sequence_length])
                    else:
                        truncated_texts.append(text)
                
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=truncated_texts,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Add small delay to avoid rate limiting
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error in batch embedding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error embedding batch with OpenAI: {str(e)}")
            raise
    
    async def embed_with_dimensions(
        self,
        text: str,
        dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Embed text with custom dimensions (for supported models)
        """
        try:
            # Only text-embedding-3 models support custom dimensions
            if "text-embedding-3" not in self.model_name and dimensions is not None:
                logger.warning(f"Model {self.model_name} does not support custom dimensions")
                dimensions = None
            
            # Truncate text if too long
            if len(text) > self.max_sequence_length:
                text = text[:self.max_sequence_length]
            
            kwargs = {"model": self.model_name, "input": text, "encoding_format": "float"}
            if dimensions:
                kwargs["dimensions"] = dimensions
            
            response = await self.client.embeddings.create(**kwargs)
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error embedding text with custom dimensions: {str(e)}")
            raise
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics (placeholder - would need to track API calls)
        """
        return {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the OpenAI model
        """
        info = super().get_model_info()
        
        # Add OpenAI-specific information
        model_info = {
            "provider": "OpenAI",
            "api_base": self.base_url or "https://api.openai.com/v1",
            "supports_custom_dimensions": "text-embedding-3" in self.model_name,
            "max_input_tokens": self.max_sequence_length // 4  # Rough estimate
        }
        
        info.update(model_info)
        return info


class AzureOpenAIEmbedder(OpenAIEmbedder):
    """Azure OpenAI embedder implementation"""
    
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "text-embedding-ada-002",
        **kwargs
    ):
        # Set base URL for Azure
        azure_base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment_name}"
        
        super().__init__(
            api_key=api_key,
            base_url=azure_base_url,
            model_name=deployment_name,
            **kwargs
        )
        
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        # Reinitialize client with Azure configuration
        self.client = AsyncOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed text using Azure OpenAI
        """
        try:
            # Truncate text if too long
            if len(text) > self.max_sequence_length:
                text = text[:self.max_sequence_length]
            
            response = await self.client.embeddings.create(
                model=self.deployment_name,  # Use deployment name for Azure
                input=text,
                encoding_format="float"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error embedding text with Azure OpenAI: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Azure OpenAI model information
        """
        info = super().get_model_info()
        
        azure_info = {
            "provider": "Azure OpenAI",
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name
        }
        
        info.update(azure_info)
        return info
