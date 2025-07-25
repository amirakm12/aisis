"""
Base Agent Implementation
Provides core functionality for all AISIS agents
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional
# If 'loguru' is not installed, run: pip install loguru
from loguru import logger


class AgentStatus:
    IDLE = 'IDLE'
    RUNNING = 'RUNNING'
    ERROR = 'ERROR'


class BaseAgent(ABC):
    """Base class for all AISIS agents"""
    version: str = "1.0.0"

    def __init__(self, name: str):
        self.name = name
        self.status = AgentStatus.IDLE
        self.error = None
        self.initialized = False
        self._results = {}
        self._feedback_history = []

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return a dictionary describing the agent's capabilities.
        Override in subclasses to specify supported tasks, modalities, etc.
        """
        return {"tasks": [], "modalities": [], "description": ""}

    async def initialize(self) -> None:
        """Initialize agent resources"""
        if self.initialized:
            return

        try:
            self.status = AgentStatus.RUNNING
            await self._initialize()
            self.initialized = True
            self.status = AgentStatus.IDLE
            logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise

    @abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization"""
        pass

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task

        Args:
            task: Dictionary containing task parameters

        Returns:
            Dictionary containing processing results
        """
        if not self.initialized:
            raise RuntimeError(f"Agent {self.name} not initialized")

        try:
            self.status = AgentStatus.RUNNING
            result = await self._process(task)
            self.status = AgentStatus.IDLE
            self._results[id(task)] = result
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            logger.error(f"Error in agent {self.name}: {e}")
            raise

    @abstractmethod
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific task processing"""
        pass

    def get_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get result for a specific task"""
        return self._results.get(task_id)

    def clear_results(self) -> None:
        """Clear stored results"""
        self._results.clear()

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            await self._cleanup()
            self.initialized = False
            self.clear_results()
            logger.info(f"Agent {self.name} cleanup complete")
        except Exception as e:
            logger.error(f"Error during {self.name} cleanup: {e}")
            raise

    @abstractmethod
    async def _cleanup(self) -> None:
        """Agent-specific cleanup"""
        pass

    def feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Accept feedback for local adaptation/self-improvement.
        Subclasses can override to update models, rules, or strategies.
        """
        self._feedback_history.append(feedback)
        # Optionally, implement local adaptation here

    @property
    def feedback_history(self) -> list:
        """Return the history of feedback received."""
        return self._feedback_history

    def __str__(self) -> str:
        return f"{self.name} ({self.status})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"status={self.status})"
        )


"""
How to use:
- Subclass BaseAgent and override 'capabilities' to describe agent features.
- Use 'version' to track agent updates.
- Use 'feedback' to accept user/system feedback for local learning.
- Register agents in a central registry for discovery and orchestration.
"""
