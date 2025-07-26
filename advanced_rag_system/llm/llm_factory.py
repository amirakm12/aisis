"""
LLM Factory - Creates and manages different language model providers
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger

from ..utils.config import LLMConfig

class BaseLLM(ABC):
    """Base class for all LLM implementations"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the LLM"""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the LLM connection"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
    
    async def initialize(self):
        """Initialize OpenAI client"""
        if self.is_initialized:
            return
        
        try:
            import openai
            import os
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            self.is_initialized = True
            logger.info(f"OpenAI LLM '{self.config.model_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            temperature = temperature or self.config.temperature
            max_tokens = max_tokens or self.config.max_tokens
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.config.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise
    
    async def close(self):
        """Close OpenAI client"""
        try:
            # OpenAI client doesn't require explicit closing
            self.is_initialized = False
            logger.info("OpenAI LLM closed")
            
        except Exception as e:
            logger.error(f"Error closing OpenAI LLM: {str(e)}")

class LLMFactory:
    """Factory for creating LLM instances"""
    
    _llm_classes = {
        "openai": OpenAILLM,
    }
    
    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseLLM:
        """Create an LLM instance based on provider"""
        provider = config.provider.lower()
        
        if provider not in cls._llm_classes:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        llm_class = cls._llm_classes[provider]
        return llm_class(config)
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported LLM providers"""
        return list(cls._llm_classes.keys())