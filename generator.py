"""
Response generator for RAG system
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

from app.models.llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class Generator:
    """Response generator using LLM"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response for a given prompt
        """
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **kwargs
            )
            
            logger.info(f"Generated response of length {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream generate response for a given prompt
        """
        try:
            async for chunk in self.llm_client.stream_generate(
                prompt=prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            raise
    
    async def generate_with_fallback(
        self,
        prompt: str,
        fallback_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response with fallback prompt
        """
        try:
            return await self.generate(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary generation failed: {str(e)}, trying fallback")
            
            if fallback_prompt:
                return await self.generate(fallback_prompt, **kwargs)
            else:
                # Simple fallback
                return "I apologize, but I'm unable to generate a response at the moment. Please try again later."
    
    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts in parallel
        """
        try:
            tasks = [
                self.generate(prompt, **kwargs)
                for prompt in prompts
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error generating response for prompt {i}: {str(response)}")
                    processed_responses.append("Error generating response")
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            raise
    
    async def validate_response(
        self,
        prompt: str,
        response: str,
        validation_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate generated response against criteria
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "score": 1.0
        }
        
        # Basic validation checks
        if not response or len(response.strip()) < 10:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Response too short")
            validation_result["score"] = 0.0
        
        # Check for refusal patterns
        refusal_patterns = ["I cannot", "I'm unable to", "I apologize", "I don't have"]
        if any(pattern.lower() in response.lower() for pattern in refusal_patterns):
            validation_result["issues"].append("Possible refusal detected")
            validation_result["score"] *= 0.5
        
        # Check for repetition
        words = response.lower().split()
        if len(set(words)) / len(words) < 0.3 if words else 0:
            validation_result["issues"].append("High repetition detected")
            validation_result["score"] *= 0.7
        
        # Custom validation criteria
        if validation_criteria:
            if "min_length" in validation_criteria:
                if len(response) < validation_criteria["min_length"]:
                    validation_result["is_valid"] = False
                    validation_result["issues"].append("Response below minimum length")
            
            if "max_length" in validation_criteria:
                if len(response) > validation_criteria["max_length"]:
                    validation_result["issues"].append("Response exceeds maximum length")
                    validation_result["score"] *= 0.8
            
            if "required_keywords" in validation_criteria:
                required_keywords = validation_criteria["required_keywords"]
                missing_keywords = [
                    kw for kw in required_keywords
                    if kw.lower() not in response.lower()
                ]
                if missing_keywords:
                    validation_result["issues"].append(f"Missing keywords: {missing_keywords}")
                    validation_result["score"] *= 0.6
        
        return validation_result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for generator
        """
        try:
            # Test LLM client with simple prompt
            test_response = await self.llm_client.generate(
                prompt="Respond with 'OK' if you can read this.",
                max_tokens=10
            )
            
            if "OK" in test_response:
                return {"status": "healthy", "llm_client": "responsive"}
            else:
                return {"status": "degraded", "llm_client": "unexpected_response"}
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
