"""
Advanced Forensic Analysis System Core Module
============================================

This module provides the core infrastructure for advanced digital forensics,
including multi-agent coordination, evidence processing, and analysis pipelines.
"""

from .forensic_engine import ForensicEngine
from .evidence_manager import EvidenceManager
from .agent_coordinator import AgentCoordinator
from .forensic_timeline import ForensicTimeline

__all__ = [
    'ForensicEngine',
    'EvidenceManager', 
    'AgentCoordinator',
    'ForensicTimeline'
]

__version__ = '1.0.0'