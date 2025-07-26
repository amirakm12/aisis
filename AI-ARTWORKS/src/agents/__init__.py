"""
AI-ARTWORK Agents Package Initialization
"""

from .base_agent import BaseAgent
from typing import Optional

# Central registry for all agents
AGENT_REGISTRY = {}


def register_agent(name: str, agent: BaseAgent):
    """Register an agent with a unique name."""
    AGENT_REGISTRY[name] = agent


def get_agent(name: str) -> Optional[BaseAgent]:
    """Retrieve an agent by name."""
    return AGENT_REGISTRY.get(name)


"""
How to use:
- Import and call register_agent('my_agent', MyAgent()) in your agent module.
- Use get_agent('my_agent') to retrieve the agent for orchestration.
- AGENT_REGISTRY can be used for agent discovery, listing, and dynamic
  pipelines.
"""
