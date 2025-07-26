"""
Base Agent Class for Athena Research System

Provides common functionality and interfaces for all research agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

from pydantic import BaseModel, Field
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler


@dataclass
class ResearchTask:
    """Represents a research task with metadata"""
    id: str
    query: str
    context: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    assigned_agent: str = None
    status: str = "pending"  # pending, in_progress, completed, failed
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ResearchResult:
    """Represents the result of a research task"""
    task_id: str
    agent_id: str
    content: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for agent operations"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"athena.agent.{agent_id}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.logger.info(f"Agent {self.agent_id} starting LLM call")
    
    def on_llm_end(self, response, **kwargs) -> None:
        self.logger.info(f"Agent {self.agent_id} completed LLM call")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.logger.error(f"Agent {self.agent_id} LLM error: {error}")


class BaseAgent(ABC):
    """
    Abstract base class for all Athena research agents.
    
    Provides common functionality including:
    - Task management
    - Result handling
    - Logging and monitoring
    - Communication with other agents
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(f"athena.agent.{agent_id}")
        self.callback_handler = AgentCallbackHandler(agent_id)
        
        # Task management
        self.current_tasks: Dict[str, ResearchTask] = {}
        self.completed_tasks: Dict[str, ResearchResult] = {}
        
        # Agent state
        self.is_active = False
        self.last_activity = None
        
    @abstractmethod
    async def process_task(self, task: ResearchTask) -> ResearchResult:
        """
        Process a research task and return results.
        
        Args:
            task: The research task to process
            
        Returns:
            ResearchResult containing the findings
        """
        pass
    
    @abstractmethod
    def can_handle_task(self, task: ResearchTask) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            task: The research task to evaluate
            
        Returns:
            True if the agent can handle the task, False otherwise
        """
        pass
    
    async def execute_task(self, task: ResearchTask) -> ResearchResult:
        """
        Execute a research task with proper error handling and monitoring.
        
        Args:
            task: The research task to execute
            
        Returns:
            ResearchResult containing the findings
        """
        self.logger.info(f"Starting task {task.id}: {task.query}")
        
        try:
            # Mark task as in progress
            task.status = "in_progress"
            task.assigned_agent = self.agent_id
            self.current_tasks[task.id] = task
            self.is_active = True
            self.last_activity = datetime.now()
            
            # Process the task
            result = await self.process_task(task)
            
            # Mark task as completed
            task.status = "completed"
            self.completed_tasks[task.id] = result
            del self.current_tasks[task.id]
            
            self.logger.info(f"Completed task {task.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            task.status = "failed"
            
            # Create error result
            error_result = ResearchResult(
                task_id=task.id,
                agent_id=self.agent_id,
                content=f"Task failed with error: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
            
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
            
            return error_result
        
        finally:
            self.is_active = len(self.current_tasks) > 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "is_active": self.is_active,
            "current_tasks": len(self.current_tasks),
            "completed_tasks": len(self.completed_tasks),
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "capabilities": self.capabilities
        }
    
    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history"""
        recent_tasks = sorted(
            self.completed_tasks.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:limit]
        
        return [
            {
                "task_id": task.task_id,
                "content_preview": task.content[:200] + "..." if len(task.content) > 200 else task.content,
                "sources_count": len(task.sources),
                "confidence": task.confidence,
                "created_at": task.created_at.isoformat()
            }
            for task in recent_tasks
        ]
    
    async def cleanup(self):
        """Cleanup agent resources"""
        self.logger.info(f"Cleaning up agent {self.agent_id}")
        self.current_tasks.clear()
        self.is_active = False