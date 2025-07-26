"""
Base Forensic Agent
==================

Abstract base class for all forensic analysis agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio
from dataclasses import dataclass


@dataclass
class AgentCapabilities:
    """Defines what an agent can analyze"""
    evidence_types: List[str]
    analysis_types: List[str]
    required_tools: List[str]
    parallel_capable: bool = True
    gpu_accelerated: bool = False
    
    
class BaseAgent(ABC):
    """
    Abstract base class for forensic analysis agents
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.capabilities = self._define_capabilities()
        self._initialized = False
        
    @abstractmethod
    def _define_capabilities(self) -> AgentCapabilities:
        """Define what this agent can do"""
        pass
    
    async def initialize(self):
        """Initialize agent resources"""
        if self._initialized:
            return
            
        self.logger.info(f"Initializing {self.name} agent v{self.version}")
        await self._setup()
        self._initialized = True
        
    @abstractmethod
    async def _setup(self):
        """Agent-specific setup logic"""
        pass
    
    @abstractmethod
    async def analyze(self, evidence_data: Dict[str, Any], 
                     case_context: Any) -> Dict[str, Any]:
        """
        Perform forensic analysis on evidence
        
        Args:
            evidence_data: Evidence metadata and path
            case_context: Current case information
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    async def validate_evidence(self, evidence_data: Dict[str, Any]) -> bool:
        """Validate if this agent can process the evidence"""
        evidence_type = evidence_data.get('type', '')
        
        if evidence_type not in self.capabilities.evidence_types:
            self.logger.warning(
                f"{self.name} cannot process evidence type: {evidence_type}"
            )
            return False
            
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.name,
            'version': self.version,
            'capabilities': {
                'evidence_types': self.capabilities.evidence_types,
                'analysis_types': self.capabilities.analysis_types,
                'required_tools': self.capabilities.required_tools,
                'parallel_capable': self.capabilities.parallel_capable,
                'gpu_accelerated': self.capabilities.gpu_accelerated
            },
            'initialized': self._initialized
        }
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.logger.info(f"Cleaning up {self.name} agent")
        await self._cleanup()
        self._initialized = False
        
    @abstractmethod
    async def _cleanup(self):
        """Agent-specific cleanup logic"""
        pass