"""
LLM Factory - Manages different language model providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..utils.config import LLMConfig


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def generate_with_metadata(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with metadata"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the LLM"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        try:
            import openai
            
            self.client = openai.AsyncOpenAI(api_key=self.config.api_key)
            
        except ImportError:
            raise ImportError("OpenAI is not installed. Install with: pip install openai")
    
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def generate_with_metadata(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with metadata"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        # Merge config with kwargs
        generation_params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **kwargs
        }
        
        response = await self.client.chat.completions.create(**generation_params)
        
        return {
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }
    
    async def shutdown(self) -> None:
        """Shutdown the client"""
        self.client = None


class LLMFactory:
    """Factory for creating LLM instances"""
    
    @classmethod
    def create_llm(cls, config: LLMConfig) -> BaseLLM:
        """
        Create LLM instance based on configuration
        
        Args:
            config: LLM configuration
            
        Returns:
            LLM instance
        """
        if config.provider.lower() == "openai":
            return OpenAILLM(config)
        else:
            # Default to OpenAI
            return OpenAILLM(config)
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported LLM providers"""
        return ["openai", "anthropic", "cohere", "local"] 