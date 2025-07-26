"""
Athena - Agentic Research System

A comprehensive AI-powered research platform that combines multiple agents
to conduct autonomous research, analysis, and knowledge synthesis.
"""

__version__ = "1.0.0"
__author__ = "Athena Research Team"

from .core.athena_agent import AthenaAgent
from .core.research_orchestrator import ResearchOrchestrator
from .agents.web_researcher import WebResearchAgent
from .agents.academic_researcher import AcademicResearchAgent
from .agents.data_analyst import DataAnalystAgent
from .agents.synthesis_agent import SynthesisAgent

__all__ = [
    "AthenaAgent",
    "ResearchOrchestrator", 
    "WebResearchAgent",
    "AcademicResearchAgent",
    "DataAnalystAgent",
    "SynthesisAgent"
]