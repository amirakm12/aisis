"""
Research Orchestrator for Athena Agentic Research System

Manages and coordinates multiple specialized research agents to execute
comprehensive research plans efficiently and intelligently.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

from .base_agent import BaseAgent, ResearchTask, ResearchResult
from ..agents.web_researcher import WebResearchAgent
from ..agents.academic_researcher import AcademicResearchAgent
from ..agents.data_analyst import DataAnalystAgent
from ..agents.synthesis_agent import SynthesisAgent


class ResearchOrchestrator:
    """
    Orchestrates multiple research agents to execute comprehensive research plans.
    
    Capabilities:
    - Agent lifecycle management
    - Task distribution and load balancing
    - Parallel execution coordination
    - Result aggregation and quality control
    - Agent performance monitoring
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.logger = logging.getLogger("athena.orchestrator")
        
        # Initialize available agents
        self.agents: Dict[str, BaseAgent] = {}
        self._initialize_agents(openai_api_key, anthropic_api_key)
        
        # Task management
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.completed_tasks: Dict[str, ResearchResult] = {}
        
        # Execution monitoring
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_duration": 0.0
        }
        
        self.logger.info("Research Orchestrator initialized with agents: " + 
                        ", ".join(self.agents.keys()))
    
    def _initialize_agents(self, openai_api_key: str, anthropic_api_key: str):
        """Initialize all available research agents"""
        
        # Web Research Agent
        self.agents["web_researcher"] = WebResearchAgent(
            openai_api_key=openai_api_key,
            config=self.config.get("web_researcher", {})
        )
        
        # Academic Research Agent
        self.agents["academic_researcher"] = AcademicResearchAgent(
            openai_api_key=openai_api_key,
            config=self.config.get("academic_researcher", {})
        )
        
        # Data Analyst Agent
        self.agents["data_analyst"] = DataAnalystAgent(
            openai_api_key=openai_api_key,
            config=self.config.get("data_analyst", {})
        )
        
        # Synthesis Agent
        self.agents["synthesis_agent"] = SynthesisAgent(
            openai_api_key=openai_api_key,
            config=self.config.get("synthesis_agent", {})
        )
    
    async def execute_research_plan(
        self,
        strategy: Dict[str, Any],
        max_agents: int = 4
    ) -> Dict[str, Any]:
        """
        Execute a comprehensive research plan using multiple agents.
        
        Args:
            strategy: Research strategy containing objectives and agent recommendations
            max_agents: Maximum number of agents to use concurrently
            
        Returns:
            Aggregated research results from all agents
        """
        self.logger.info("Executing research plan")
        
        # Extract research tasks from strategy
        tasks = self._create_research_tasks(strategy)
        
        # Select and assign agents
        selected_agents = self._select_agents(strategy, max_agents)
        
        # Execute tasks in parallel
        results = await self._execute_tasks_parallel(tasks, selected_agents)
        
        # Update execution statistics
        self._update_execution_stats(results)
        
        self.logger.info(f"Research plan completed with {len(results)} results")
        return results
    
    def _create_research_tasks(self, strategy: Dict[str, Any]) -> List[ResearchTask]:
        """Create specific research tasks from the strategy"""
        tasks = []
        
        # Main research task
        main_task = ResearchTask(
            id=str(uuid.uuid4()),
            query=strategy.get("query", ""),
            context={
                "strategy": strategy,
                "task_type": "primary_research",
                "objectives": strategy.get("research_objectives", [])
            },
            priority=1
        )
        tasks.append(main_task)
        
        # Create sub-tasks based on objectives
        for i, objective in enumerate(strategy.get("research_objectives", [])):
            sub_task = ResearchTask(
                id=str(uuid.uuid4()),
                query=objective,
                context={
                    "strategy": strategy,
                    "task_type": "sub_research",
                    "parent_task": main_task.id,
                    "objective_index": i
                },
                priority=2
            )
            tasks.append(sub_task)
        
        return tasks
    
    def _select_agents(
        self,
        strategy: Dict[str, Any],
        max_agents: int
    ) -> List[BaseAgent]:
        """Select the most appropriate agents for the research strategy"""
        recommended_agents = strategy.get("recommended_agents", [])
        selected = []
        
        for agent_name in recommended_agents[:max_agents]:
            if agent_name in self.agents:
                selected.append(self.agents[agent_name])
                self.logger.info(f"Selected agent: {agent_name}")
        
        # Ensure we have at least web research
        if not selected and "web_researcher" in self.agents:
            selected.append(self.agents["web_researcher"])
            self.logger.info("Fallback to web_researcher agent")
        
        return selected
    
    async def _execute_tasks_parallel(
        self,
        tasks: List[ResearchTask],
        agents: List[BaseAgent]
    ) -> Dict[str, Any]:
        """Execute research tasks in parallel across available agents"""
        
        if not agents:
            self.logger.error("No agents available for task execution")
            return {}
        
        # Assign tasks to agents based on capability matching
        agent_tasks = self._assign_tasks_to_agents(tasks, agents)
        
        # Execute tasks concurrently
        execution_futures = []
        for agent, agent_task_list in agent_tasks.items():
            for task in agent_task_list:
                future = asyncio.create_task(agent.execute_task(task))
                execution_futures.append((agent.agent_id, task.id, future))
        
        # Wait for all tasks to complete
        results = {}
        for agent_id, task_id, future in execution_futures:
            try:
                result = await future
                results[f"{agent_id}_{task_id}"] = {
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "content": result.content,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "metadata": result.metadata
                }
                self.logger.info(f"Completed task {task_id} with agent {agent_id}")
                
            except Exception as e:
                self.logger.error(f"Task {task_id} failed with agent {agent_id}: {e}")
                results[f"{agent_id}_{task_id}"] = {
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "content": f"Task failed: {str(e)}",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {"error": str(e)}
                }
        
        return results
    
    def _assign_tasks_to_agents(
        self,
        tasks: List[ResearchTask],
        agents: List[BaseAgent]
    ) -> Dict[BaseAgent, List[ResearchTask]]:
        """Intelligently assign tasks to agents based on capabilities"""
        
        agent_tasks = {agent: [] for agent in agents}
        
        for task in tasks:
            # Find the best agent for this task
            best_agent = None
            best_score = -1
            
            for agent in agents:
                if agent.can_handle_task(task):
                    # Simple scoring based on current load and capability match
                    current_load = len(agent_tasks[agent])
                    capability_score = self._calculate_capability_score(agent, task)
                    
                    # Lower load and higher capability = better score
                    score = capability_score - (current_load * 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_agent = agent
            
            # Assign to best agent or fallback to first available
            if best_agent:
                agent_tasks[best_agent].append(task)
            elif agents:
                agent_tasks[agents[0]].append(task)
        
        return agent_tasks
    
    def _calculate_capability_score(self, agent: BaseAgent, task: ResearchTask) -> float:
        """Calculate how well an agent's capabilities match a task"""
        # This would be enhanced with more sophisticated matching logic
        task_type = task.context.get("task_type", "general")
        
        capability_scores = {
            "web_researcher": {"primary_research": 0.9, "sub_research": 0.8, "general": 0.7},
            "academic_researcher": {"primary_research": 0.8, "sub_research": 0.9, "general": 0.6},
            "data_analyst": {"primary_research": 0.6, "sub_research": 0.7, "general": 0.5},
            "synthesis_agent": {"primary_research": 0.7, "sub_research": 0.6, "general": 0.8}
        }
        
        return capability_scores.get(agent.agent_id, {}).get(task_type, 0.5)
    
    def _update_execution_stats(self, results: Dict[str, Any]):
        """Update execution statistics"""
        self.execution_stats["total_tasks"] += len(results)
        
        completed = sum(1 for r in results.values() if r.get("confidence", 0) > 0)
        failed = len(results) - completed
        
        self.execution_stats["completed_tasks"] += completed
        self.execution_stats["failed_tasks"] += failed
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics"""
        return {
            "execution_stats": self.execution_stats,
            "active_agents": len([a for a in self.agents.values() if a.is_active]),
            "total_agents": len(self.agents),
            "agent_status": self.get_agent_status()
        }
    
    async def cleanup(self):
        """Cleanup all agents and resources"""
        self.logger.info("Cleaning up Research Orchestrator")
        
        cleanup_tasks = [agent.cleanup() for agent in self.agents.values()]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.agents.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()