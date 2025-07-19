"""
HyperOrchestrator Agent
LLM-powered orchestrator for agent selection, sequencing, and reasoning
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

# Placeholder for LLM integration (Mixtral, LLaMA, Phi-3)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

from .base_agent import BaseAgent


class HyperOrchestrator(BaseAgent):
    def __init__(self, llm_model_name: str = "mixtral-8x7b-instruct-v0.1"):
        super().__init__("HyperOrchestrator")
        self.llm_model_name = llm_model_name
        self.llm = None
        self.tokenizer = None
        self.base_orchestrator = OrchestratorAgent()
        self.is_initialized = False

    async def _initialize(self) -> None:
        """Initialize the LLM and base orchestrator"""
        try:
            logger.info(f"Initializing HyperOrchestrator with LLM: {self.llm_model_name}")
            await self.base_orchestrator.initialize()
            if AutoModelForCausalLM and AutoTokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                self.llm = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
                logger.info("LLM loaded successfully")
            else:
                logger.warning(
                    "Transformers not available or LLM model not installed. Using fallback mode."
                )
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize HyperOrchestrator: {e}")
            raise

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a high-level task using LLM-powered agent selection and sequencing"""
        try:
            if not self.is_initialized:
                await self._initialize()
            instruction = task.get("instruction") or task.get("description")
            image = task.get("image")
            logger.info(f"Received instruction: {instruction}")

            # Use LLM to select and sequence agents
            agent_sequence = await self._llm_select_agents(instruction)
            logger.info(f"Agent sequence selected: {agent_sequence}")

            # Run the selected agents in sequence
            current_image = image
            agent_reports = {}
            for agent_name in agent_sequence:
                agent_task = {"image": current_image, **task}
                result = await self.base_orchestrator.execute_single_agent(agent_name, agent_task)
                current_image = result.get("output_image", current_image)
                agent_reports[agent_name] = result

            return {
                "status": "success",
                "final_output": current_image,
                "agent_sequence": agent_sequence,
                "agent_reports": agent_reports,
            }
        except Exception as e:
            logger.error(f"HyperOrchestrator failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _llm_select_agents(self, instruction: str) -> List[str]:
        """Use the LLM to select and sequence agents based on the instruction"""
        if self.llm and self.tokenizer:
            prompt = (
                "You are an expert AI orchestrator. Given the following user instruction, "
                "select the most appropriate sequence of agents from the list: "
                f"{self.base_orchestrator.get_available_agents()}\n"
                f"Instruction: {instruction}\n"
                "Respond with a Python list of agent names in execution order."
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.llm.generate(**inputs, max_new_tokens=64)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract Python list from response (very basic parsing)
            import ast

            try:
                agent_sequence = ast.literal_eval(response.split("[")[-1].split("]")[0])
                if isinstance(agent_sequence, str):
                    agent_sequence = [agent_sequence]
                return agent_sequence
            except Exception:
                logger.warning(f"Failed to parse LLM response: {response}")
                return ["image_restoration"]
        else:
            # Fallback: simple keyword-based selection
            instruction = instruction.lower() if instruction else ""
            if "restore" in instruction:
                return [
                    "forensic_analysis",
                    "material_recognition",
                    "damage_classifier",
                    "image_restoration",
                    "meta_correction",
                ]
            if "style" in instruction or "aesthetic" in instruction:
                return ["style_aesthetic", "adaptive_enhancement"]
            if "text" in instruction:
                return ["text_recovery"]
            return ["image_restoration"]

    async def _cleanup(self) -> None:
        await self.base_orchestrator.cleanup()
        self.llm = None
        self.tokenizer = None
        self.is_initialized = False
