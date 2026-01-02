"""
Base LLM client interface
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
import time


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 30
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text completion
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for the LLM client
        """
        try:
            # Test with simple prompt
            test_response = await self.generate(
                "Respond with 'OK' if you can read this.",
                max_tokens=10,
                temperature=0.1
            )
            
            if "OK" in test_response:
                return {
                    "status": "healthy",
                    "model_name": self.model_name,
                    "test_response": test_response
                }
            else:
                return {
                    "status": "degraded",
                    "model_name": self.model_name,
                    "test_response": test_response
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_name": self.model_name,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout
        }


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="mock-llm", **kwargs)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate mock response
        """
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate simple response based on prompt
        if "?" in prompt:
            return "This is a mock response to your question."
        elif "summarize" in prompt.lower():
            return "This is a mock summary of the provided text."
        else:
            return "This is a mock response."
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream mock response
        """
        response = await self.generate(prompt, **kwargs)
        
        # Yield response word by word
        words = response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)


class CachedLLMClient(BaseLLMClient):
    """LLM client with caching functionality"""
    
    def __init__(
        self,
        base_client: BaseLLMClient,
        cache_size: int = 1000,
        cache_ttl: int = 3600  # 1 hour
    ):
        super().__init__(
            model_name=base_client.model_name,
            max_tokens=base_client.max_tokens,
            temperature=base_client.temperature,
            top_p=base_client.top_p,
            timeout=base_client.timeout
        )
        
        self.base_client = base_client
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_timestamps = {}
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate with caching
        """
        # Create cache key
        cache_key = self._create_cache_key(prompt, kwargs)
        current_time = time.time()
        
        # Check cache
        if (cache_key in self._cache and 
            current_time - self._cache_timestamps[cache_key] < self.cache_ttl):
            logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return self._cache[cache_key]
        
        # Generate response
        response = await self.base_client.generate(prompt, **kwargs)
        
        # Update cache
        self._cache[cache_key] = response
        self._cache_timestamps[cache_key] = current_time
        
        # Clean up old cache entries
        if len(self._cache) > self.cache_size:
            await self._cleanup_cache()
        
        return response
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream generation (no caching for streaming)
        """
        async for chunk in self.base_client.stream_generate(prompt, **kwargs):
            yield chunk
    
    def _create_cache_key(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """
        Create cache key from prompt and parameters
        """
        import hashlib
        
        # Create a deterministic string from prompt and relevant parameters
        key_parts = [prompt]
        
        # Include parameters that affect output
        for param in ["max_tokens", "temperature", "top_p"]:
            if param in kwargs:
                key_parts.append(f"{param}:{kwargs[param]}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _cleanup_cache(self):
        """
        Clean up old cache entries
        """
        current_time = time.time()
        
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
        base_health = await self.base_client.health_check()
        
        return {
            **base_health,
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl
        }


class RetryLLMClient(BaseLLMClient):
    """LLM client with retry functionality"""
    
    def __init__(
        self,
        base_client: BaseLLMClient,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        super().__init__(
            model_name=base_client.model_name,
            max_tokens=base_client.max_tokens,
            temperature=base_client.temperature,
            top_p=base_client.top_p,
            timeout=base_client.timeout
        )
        
        self.base_client = base_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate with retry logic
        """
        last_exception = None
        current_delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.base_client.generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"LLM generation attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= self.backoff_factor
                else:
                    logger.error(f"LLM generation failed after {self.max_retries + 1} attempts")
        
        raise last_exception
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream generation with retry logic
        """
        last_exception = None
        current_delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in self.base_client.stream_generate(prompt, **kwargs):
                    yield chunk
                return
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"LLM stream attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= self.backoff_factor
                else:
                    logger.error(f"LLM stream failed after {self.max_retries + 1} attempts")
        
        raise last_exception


class RateLimitedLLMClient(BaseLLMClient):
    """LLM client with rate limiting"""
    
    def __init__(
        self,
        base_client: BaseLLMClient,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 10000
    ):
        super().__init__(
            model_name=base_client.model_name,
            max_tokens=base_client.max_tokens,
            temperature=base_client.temperature,
            top_p=base_client.top_p,
            timeout=base_client.timeout
        )
        
        self.base_client = base_client
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self.request_times = []
        self.token_usage = []
        self._lock = asyncio.Lock()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate with rate limiting
        """
        async with self._lock:
            await self._check_rate_limits(len(prompt))
        
        return await self.base_client.generate(prompt, **kwargs)
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream generation with rate limiting
        """
        async with self._lock:
            await self._check_rate_limits(len(prompt))
        
        async for chunk in self.base_client.stream_generate(prompt, **kwargs):
            yield chunk
    
    async def _check_rate_limits(self, prompt_length: int):
        """
        Check and enforce rate limits
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if t > minute_ago]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
        
        # Check request rate limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + prompt_length >= self.tokens_per_minute:
            sleep_time = 60 - (current_time - self.token_usage[0][0])
            if sleep_time > 0:
                logger.info(f"Token rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(current_time)
        # Estimate token usage (rough approximation)
        estimated_tokens = max(prompt_length // 4, 1)
        self.token_usage.append((current_time, estimated_tokens))
