"""
Base Agent class for AI agents in the RAG system
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from ..utils.config import RAGConfig
from ..llm.llm_factory import LLMFactory


@dataclass
class AgentStats:
    """Statistics for an agent"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Base class for all AI agents in the RAG system
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = None
        self.stats = AgentStats()
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        if self.initialized:
            return
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(self.config.llm)
        await self.llm.initialize()
        
        self.initialized = True
    
    async def process_with_retry(
        self,
        input_data: Any,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> Any:
        """
        Process input with retry logic
        
        Args:
            input_data: Input data to process
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Processed result
        """
        max_retries = max_retries or self.config.agent.max_retries
        retry_delay = retry_delay or self.config.agent.retry_delay
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Validate input
                validated_input = self.validate_input(input_data)
                
                # Preprocess input
                processed_input = self.preprocess_input(validated_input)
                
                # Process with LLM
                result = await self.process(processed_input)
                
                # Postprocess output
                final_result = self.postprocess_output(result)
                
                # Update stats
                self._update_stats(True, time.time() - start_time)
                
                return final_result
                
            except Exception as e:
                last_exception = e
                self._update_stats(False, time.time() - start_time, str(e))
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise last_exception
    
    def get_stats(self) -> AgentStats:
        """Get agent statistics"""
        return self.stats
    
    def validate_input(self, input_data: Any) -> Any:
        """
        Validate input data
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validated input data
        """
        # Default implementation - can be overridden by subclasses
        if input_data is None:
            raise ValueError("Input data cannot be None")
        return input_data
    
    def preprocess_input(self, input_data: Any) -> Any:
        """
        Preprocess input data
        
        Args:
            input_data: Input data to preprocess
            
        Returns:
            Preprocessed input data
        """
        # Default implementation - can be overridden by subclasses
        return input_data
    
    def postprocess_output(self, output: Any) -> Any:
        """
        Postprocess output data
        
        Args:
            output: Output data to postprocess
            
        Returns:
            Postprocessed output data
        """
        # Default implementation - can be overridden by subclasses
        return output
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        Main processing method to be implemented by subclasses
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed result
        """
        pass
    
    def _update_stats(
        self,
        success: bool,
        processing_time: float,
        error_message: Optional[str] = None
    ) -> None:
        """Update agent statistics"""
        self.stats.total_requests += 1
        self.stats.total_processing_time += processing_time
        
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
            if error_message:
                self.stats.errors.append(error_message)
        
        self.stats.average_response_time = (
            self.stats.total_processing_time / self.stats.total_requests
        )
        self.stats.last_request_time = time.time()
    
    async def shutdown(self) -> None:
        """Shutdown the agent"""
        if self.llm:
            await self.llm.shutdown()
        self.initialized = False 