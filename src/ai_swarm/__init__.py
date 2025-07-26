"""
AI Agent Swarm System
Distributed cooperative AI agents for extreme performance optimization
"""

from .core.agent_base import BaseAgent
from .core.communication import SwarmCommunicator
from .core.orchestrator import SwarmOrchestrator
from .agents.compute_agent import ComputeAgent
from .agents.resource_agent import ResourceAgent

__version__ = "1.0.0"
__all__ = [
    "BaseAgent",
    "SwarmCommunicator", 
    "SwarmOrchestrator",
    "ComputeAgent",
    "ResourceAgent"
]