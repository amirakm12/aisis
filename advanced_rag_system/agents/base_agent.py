"""
Base Agent class providing common functionality for all AI agents
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from ..utils.config import AgentConfig, LLMConfig
from ..llm.llm_factory import LLMFactory

class BaseAgent(ABC):
    """
    Base class for all AI agents in the RAG system
    
    Provides common functionality including:
    - LLM interaction
    - Configuration management
    - Error handling
    - Logging and metrics
    """
    
    def __init__(self, agent_config: AgentConfig, llm_config: LLMConfig):
        self.agent_config = agent_config
        self.llm_config = llm_config
        self.llm = None
        self.is_initialized = False
        
        # Agent state
        self.processing_count = 0
        self.error_count = 0
        self.last_error = None
        
        logger.info(f"{self.__class__.__name__} created with config")
    
    async def initialize(self):
        """Initialize the agent"""
        if self.is_initialized:
            return
        
        try:
            # Initialize LLM
            self.llm = LLMFactory.create_llm(self.llm_config)
            await self.llm.initialize()
            
            self.is_initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise
    
    async def process_with_retry(
        self,
        operation_func,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        **kwargs
    ):
        """Execute an operation with retry logic"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self.processing_count += 1
                result = await operation_func(**kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                self.error_count += 1
                self.last_error = str(e)
                
                if attempt < max_retries - 1:
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"{self.__class__.__name__} attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{self.__class__.__name__} failed after {max_retries} attempts: {str(e)}"
                    )
        
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "class_name": self.__class__.__name__,
            "initialized": self.is_initialized,
            "processing_count": self.processing_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "error_rate": self.error_count / max(self.processing_count, 1)
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Basic validation - can be overridden by subclasses
        return bool(input_data)
    
    async def preprocess_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data - can be overridden by subclasses"""
        return input_data
    
    async def postprocess_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess output data - can be overridden by subclasses"""
        return output_data
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Main processing method - must be implemented by subclasses"""
        pass
    
    async def shutdown(self):
        """Shutdown the agent"""
        try:
            if self.llm:
                await self.llm.close()
            
            self.is_initialized = False
            logger.info(f"{self.__class__.__name__} shutdown completed")
            
        except Exception as e:
            logger.error(f"Error shutting down {self.__class__.__name__}: {str(e)}")
            raise