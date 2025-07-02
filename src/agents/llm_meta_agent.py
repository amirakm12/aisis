from typing import Any, Dict
from .base_agent import BaseAgent
from ..core.llm_manager import LLMManager


class LLMMetaAgent(BaseAgent):
    """
    Meta-agent powered by an offline LLM (Llama) for critique,
    planning, and negotiation.
    """
    def __init__(self, model_path: str = "models/llama-2-7b-chat.gguf"):
        super().__init__("LLMMetaAgent")
        self.llm_manager = LLMManager(model_path)

    async def _initialize(self) -> None:
        # LLMManager loads model on instantiation
        pass

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a meta-level task (critique, debate, planning, etc.)
        Args:
            task: Dict with keys like 'history', 'current', 'debate', etc.
        Returns:
            Dict with LLM-generated critique, plan, or negotiation result.
        """
        prompt = self._build_prompt(task)
        response = self.llm_manager.chat(prompt)
        return {"llm_response": response}

    def _build_prompt(self, task: Dict[str, Any]) -> str:
        # Simple prompt builder for demonstration; can be extended
        if "debate" in task:
            agents = ", ".join(task["debate"].get("agents", []))
            context = task["debate"].get("context", {})
            return (
                f"Agents: {agents}\nContext: {context}\n"
                "Debate the best approach and provide a reasoned consensus."
            )
        if "history" in task and "current" in task:
            return (
                f"Task history: {task['history']}\n"
                f"Current result: {task['current']}\n"
                "Critique the current result and suggest improvements."
            )
        return f"Meta task: {task}"

    async def _cleanup(self) -> None:
        # No resources to clean up for LLMManager stub
        pass 