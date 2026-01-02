"""
Hugging Face embedder implementation
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

from app.indexing.embeddings.embedder import BaseEmbedder


logger = logging.getLogger(__name__)


class HFEmbedder(BaseEmbedder):
    """Hugging Face embedder using transformers"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        # Get model info
        try:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir
            )
            embedding_dim = config.hidden_size if hasattr(config, 'hidden_size') else 384
            max_sequence_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 512
        except:
            # Fallback defaults
            embedding_dim = 384
            max_sequence_length = 512
        
        super().__init__(
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            **kwargs
        )
        
        self.device = self._get_device(device)
        self.trust_remote_code = trust_remote_code
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def _load_model(self):
        """Load the model and tokenizer"""
        if self._model_loaded:
            return
        
        try:
            # Load tokenizer and model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            self.tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
            )
            
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                ).to(self.device)
            )
            
            self.model.eval()
            self._model_loaded = True
            
            logger.info(f"Loaded HF model {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading HF model: {str(e)}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        """
        await self._load_model()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_sequence_length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()[0].tolist()
            
        except Exception as e:
            logger.error(f"Error embedding text with HF: {str(e)}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch
        """
        await self._load_model()
        
        if not texts:
            return []
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_sequence_length
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch with HF: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the HF model
        """
        info = super().get_model_info()
        
        hf_info = {
            "provider": "Hugging Face",
            "device": self.device,
            "model_loaded": self._model_loaded,
            "trust_remote_code": self.trust_remote_code
        }
        
        info.update(hf_info)
        return info


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers embedder"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        trust_remote_code: bool = False,
        cache_folder: Optional[str] = None,
        **kwargs
    ):
        # Get model info
        try:
            model = SentenceTransformer(model_name, device=device)
            embedding_dim = model.get_sentence_embedding_dimension()
            max_sequence_length = model.max_seq_length
        except:
            # Fallback defaults
            embedding_dim = 384
            max_sequence_length = 512
        
        super().__init__(
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            **kwargs
        )
        
        self.device = self._get_device(device)
        self.trust_remote_code = trust_remote_code
        self.cache_folder = cache_folder
        
        self.model = None
        self._model_loaded = False
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def _load_model(self):
        """Load the sentence transformer model"""
        if self._model_loaded:
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                    cache_folder=self.cache_folder
                )
            )
            
            self._model_loaded = True
            
            logger.info(f"Loaded SentenceTransformer model {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        """
        await self._load_model()
        
        try:
            loop = asyncio.get_event_loop()
            
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding text with SentenceTransformer: {str(e)}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch
        """
        await self._load_model()
        
        if not texts:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error embedding batch with SentenceTransformer: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the SentenceTransformer model
        """
        info = super().get_model_info()
        
        st_info = {
            "provider": "Sentence Transformers",
            "device": self.device,
            "model_loaded": self._model_loaded,
            "trust_remote_code": self.trust_remote_code
        }
        
        info.update(st_info)
        return info


class MultilingualHFEmbedder(HFEmbedder):
    """Multilingual Hugging Face embedder"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        supported_languages: List[str] = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        
        self.supported_languages = supported_languages or [
            "en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko", "ar"
        ]
    
    async def detect_language(self, text: str) -> str:
        """
        Simple language detection (placeholder - would use proper language detection library)
        """
        # This is a very basic implementation
        # In production, use libraries like langdetect or fasttext
        
        # Check for common language indicators
        if any(char in text for char in "你好世界"):
            return "zh"
        elif any(char in text for char in "こんにちは"):
            return "ja"
        elif any(char in text for char in "안녕하세요"):
            return "ko"
        elif any(char in text for char in "مرحبا"):
            return "ar"
        elif any(char in text for char in "áéíóúñü¿¡"):
            return "es"
        elif any(char in text for char in "àâäçéèêëïîôöùûüÿç"):
            return "fr"
        elif any(char in text for char in "äöüß"):
            return "de"
        else:
            return "en"  # Default to English
    
    async def embed_text(self, text: str, language: Optional[str] = None) -> List[float]:
        """
        Embed text with optional language specification
        """
        detected_lang = language or await self.detect_language(text)
        
        if detected_lang not in self.supported_languages:
            logger.warning(f"Language {detected_lang} may not be well supported by model {self.model_name}")
        
        return await super().embed_text(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get multilingual model information
        """
        info = super().get_model_info()
        
        multilingual_info = {
            "supported_languages": self.supported_languages,
            "multilingual": True
        }
        
        info.update(multilingual_info)
        return info
