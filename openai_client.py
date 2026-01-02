"""
OpenAI LLM client implementation
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI

from app.models.llm_client import BaseLLMClient


logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout
        )
        
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
            max_retries=max_retries
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion using OpenAI API
        """
        try:
            messages = self._prepare_messages(prompt, kwargs)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                **{k: v for k, v in kwargs.items() if k not in ['messages']}
            )
            
            return response.choices[0].message.content
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion using OpenAI API
        """
        try:
            messages = self._prepare_messages(prompt, kwargs)
            
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=top_p or self.top_p,
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ['messages']}
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except openai.APIError as e:
            logger.error(f"OpenAI API error in stream: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error streaming text with OpenAI: {str(e)}")
            raise
    
    def _prepare_messages(self, prompt: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenAI chat completion
        """
        # Check if messages are already provided
        if "messages" in kwargs:
            return kwargs["messages"]
        
        # Check if system message is provided
        system_message = kwargs.get("system_message")
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate_with_system(
        self,
        prompt: str,
        system_message: str,
        **kwargs
    ) -> str:
        """
        Generate with system message
        """
        kwargs["system_message"] = system_message
        return await self.generate(prompt, **kwargs)
    
    async def stream_with_system(
        self,
        prompt: str,
        system_message: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream with system message
        """
        kwargs["system_message"] = system_message
        async for chunk in self.stream_generate(prompt, **kwargs):
            yield chunk
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the OpenAI model
        """
        info = super().get_model_info()
        
        openai_info = {
            "provider": "OpenAI",
            "api_base": self.base_url or "https://api.openai.com/v1",
            "supports_streaming": True,
            "supports_system_messages": True,
            "supports_chat_completion": True
        }
        
        # Add model-specific information
        if "gpt-4" in self.model_name:
            openai_info.update({
                "context_length": 8192 if "32k" not in self.model_name else 32768,
                "training_cutoff": "2021-09" if "gpt-4" in self.model_name else "2021-06"
            })
        elif "gpt-3.5" in self.model_name:
            openai_info.update({
                "context_length": 4096 if "16k" not in self.model_name else 16384,
                "training_cutoff": "2021-09"
            })
        
        info.update(openai_info)
        return info


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI client implementation"""
    
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "gpt-35-turbo",
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
    
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text using Azure OpenAI
        """
        try:
            messages = self._prepare_messages(prompt, kwargs)
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,  # Use deployment name for Azure
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                **{k: v for k, v in kwargs.items() if k not in ['messages']}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text with Azure OpenAI: {str(e)}")
            raise
    
    async def stream_generate(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text using Azure OpenAI
        """
        try:
            messages = self._prepare_messages(prompt, kwargs)
            
            stream = await self.client.chat.completions.create(
                model=self.deployment_name,  # Use deployment name for Azure
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ['messages']}
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming text with Azure OpenAI: {str(e)}")
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


class OpenAIClientWithFunctions(OpenAIClient):
    """OpenAI client with function calling support"""
    
    def __init__(
        self,
        functions: List[Dict[str, Any]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.functions = functions
    
    async def generate_with_functions(
        self,
        prompt: str,
        function_call: Optional[str] = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with function calling
        """
        try:
            messages = self._prepare_messages(prompt, kwargs)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                functions=self.functions,
                function_call=function_call,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p)
            )
            
            message = response.choices[0].message
            
            result = {
                "content": message.content,
                "function_call": None
            }
            
            if message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating with functions: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including function support
        """
        info = super().get_model_info()
        
        info.update({
            "supports_functions": True,
            "available_functions": len(self.functions)
        })
        
        return info
