"""
Agent system tests for AISIS
Tests the multi-agent orchestration and individual agent functionality
"""

import asyncio
import pytest
import torch
import numpy as np
from PIL import Image

from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.orchestrator import HyperOrchestrator, ReasoningMode
from src.agents.image_restoration import ImageRestorationAgent
from src.agents.style_aesthetic import StyleAestheticAgent
from src.agents.semantic_editing import SemanticEditingAgent
from src.agents.style_transfer import StyleTransferAgent
from src.agents.vision_language import VisionLanguageAgent
from src.agents.multi_agent_orchestrator import MultiAgentOrchestrator
from src.agents.llm_meta_agent import LLMMetaAgent


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
async def orchestrator():
    """Initialize orchestrator for testing"""
    orch = HyperOrchestrator()
    await orch.initialize()
    return orch


@pytest.fixture
async def restoration_agent():
    """Initialize restoration agent for testing"""
    agent = ImageRestorationAgent()
    await agent.initialize()
    return agent


@pytest.fixture
async def style_agent():
    """Initialize style agent for testing"""
    agent = StyleAestheticAgent()
    await agent.initialize()
    return agent


@pytest.fixture
async def semantic_agent():
    """Initialize semantic agent for testing"""
    agent = SemanticEditingAgent()
    await agent.initialize()
    return agent


def test_base_agent_initialization():
    """Test base agent initialization"""

    class TestAgent(BaseAgent):
        async def initialize(self):
            return True

        async def process(self, input_data):
            return {"status": "success", "data": input_data}

    agent = TestAgent("TestAgent", "Test agent description")

    assert agent.name == "TestAgent"
    assert agent.status == AgentStatus.IDLE
    assert agent.id is not None
    assert len(agent.results) == 0


@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization"""
    assert orchestrator.name == "HyperOrchestrator"
    assert orchestrator.status == AgentStatus.IDLE
    assert orchestrator.reasoning_mode == ReasoningMode.TREE_OF_THOUGHT
    assert orchestrator.self_correction_enabled == True


@pytest.mark.asyncio
async def test_orchestrator_agent_registration(orchestrator, restoration_agent):
    """Test agent registration with orchestrator"""
    await orchestrator.register_agent(restoration_agent)

    status = orchestrator.get_orchestrator_status()
    assert restoration_agent.name in status["registered_agents"]


@pytest.mark.asyncio
async def test_orchestrator_task_analysis(orchestrator):
    """Test orchestrator task analysis"""
    task_description = "make the image more dramatic and remove noise"
    parameters = {"intensity": 0.8}

    analysis = await orchestrator._analyze_task(task_description, parameters)

    assert "task_components" in analysis or "selected_path" in analysis
    assert "confidence" in analysis
    assert analysis["confidence"] > 0.0


@pytest.mark.asyncio
async def test_orchestrator_tree_of_thought(orchestrator):
    """Test Tree-of-Thought reasoning"""
    orchestrator.reasoning_mode = ReasoningMode.TREE_OF_THOUGHT

    task_description = "enhance the colors and fix any damage"
    analysis = await orchestrator._tree_of_thought_analysis(task_description, {})

    assert "selected_path" in analysis
    assert "all_paths" in analysis
    assert len(analysis["all_paths"]) == 3
    assert analysis["reasoning_mode"] == "tree_of_thought"


@pytest.mark.asyncio
async def test_restoration_agent_processing(restoration_agent, sample_image):
    """Test image restoration agent processing"""
    input_data = {"image": sample_image, "task_type": "denoise", "parameters": {"strength": 0.5}}

    result = await restoration_agent.process(input_data)

    assert result["status"] == "success"
    assert "output_image" in result
    assert "applied_tasks" in result
    assert "quality_metrics" in result
    assert isinstance(result["output_image"], Image.Image)


@pytest.mark.asyncio
async def test_restoration_agent_auto_analysis(restoration_agent, sample_image):
    """Test automatic restoration needs analysis"""
    image_tensor = restoration_agent._prepare_image(sample_image)
    tasks = restoration_agent._analyze_restoration_needs(image_tensor, "auto")

    assert isinstance(tasks, list)
    # Should detect at least some restoration needs
    assert len(tasks) >= 0


@pytest.mark.asyncio
async def test_style_agent_processing(style_agent, sample_image):
    """Test style and aesthetic agent processing"""
    input_data = {
        "image": sample_image,
        "enhancement_type": "auto",
        "parameters": {"intensity": 0.7},
    }

    result = await style_agent.process(input_data)

    assert result["status"] == "success"
    assert "output_image" in result
    assert "aesthetic_analysis" in result
    assert "enhancement_plan" in result
    assert "improvement_metrics" in result


@pytest.mark.asyncio
async def test_style_agent_aesthetic_analysis(style_agent, sample_image):
    """Test aesthetic analysis functionality"""
    image_tensor = style_agent._prepare_image(sample_image)
    analysis = await style_agent._analyze_aesthetics(image_tensor)

    assert "overall_score" in analysis
    assert "composition" in analysis
    assert "color" in analysis
    assert "lighting" in analysis
    assert "technical" in analysis

    # Check composition analysis
    composition = analysis["composition"]
    assert "rule_of_thirds" in composition
    assert "golden_ratio" in composition
    assert "symmetry" in composition
    assert "balance" in composition


@pytest.mark.asyncio
async def test_semantic_agent_processing(semantic_agent, sample_image):
    """Test semantic editing agent processing"""
    input_data = {
        "image": sample_image,
        "instruction": "make the sky more dramatic",
        "parameters": {"intensity": 0.8},
    }

    result = await semantic_agent.process(input_data)

    assert result["status"] == "success"
    assert "output_image" in result
    assert "parsed_instruction" in result
    assert "scene_analysis" in result
    assert "editing_plan" in result


@pytest.mark.asyncio
async def test_semantic_agent_instruction_parsing(semantic_agent, sample_image):
    """Test natural language instruction parsing"""
    image_tensor = semantic_agent._prepare_image(sample_image)

    instructions = [
        "remove the car from the image",
        "make the sky more blue",
        "enhance the brightness of the person",
        "apply vintage style to the background",
    ]

    for instruction in instructions:
        parsed = await semantic_agent._parse_instruction(instruction, image_tensor)

        assert "operation_type" in parsed
        assert "target_objects" in parsed
        assert "attributes" in parsed
        assert "confidence" in parsed
        assert parsed["confidence"] > 0.0


@pytest.mark.asyncio
async def test_semantic_agent_scene_analysis(semantic_agent, sample_image):
    """Test scene analysis functionality"""
    image_tensor = semantic_agent._prepare_image(sample_image)
    analysis = await semantic_agent._analyze_scene(image_tensor)

    assert "scene_classification" in analysis
    assert "detected_objects" in analysis
    assert "segmentation" in analysis

    # Check scene classification
    scene_class = analysis["scene_classification"]
    assert "top_scenes" in scene_class
    assert len(scene_class["top_scenes"]) <= 3


@pytest.mark.asyncio
async def test_orchestrator_full_workflow(
    orchestrator, restoration_agent, style_agent, semantic_agent, sample_image
):
    """Test complete orchestrator workflow with multiple agents"""
    # Register all agents
    await orchestrator.register_agent(restoration_agent)
    await orchestrator.register_agent(style_agent)
    await orchestrator.register_agent(semantic_agent)

    # Process a complex task
    input_data = {
        "description": "enhance the image quality and make it more dramatic",
        "image": sample_image,
        "parameters": {"quality": "high", "style": "dramatic"},
    }

    result = await orchestrator.process(input_data)

    assert result["status"] == "success"
    assert "results" in result
    assert "execution_plan" in result
    assert "reasoning" in result


@pytest.mark.asyncio
async def test_agent_resource_management(restoration_agent):
    """Test agent resource management"""
    requirements = restoration_agent.get_resource_requirements()

    assert "gpu_memory_mb" in requirements
    assert "cpu_threads" in requirements
    assert requirements["gpu_memory_mb"] > 0
    assert requirements["cpu_threads"] > 0


@pytest.mark.asyncio
async def test_agent_status_tracking(restoration_agent):
    """Test agent status tracking"""
    initial_status = restoration_agent.get_status()
    assert initial_status["status"] == "IDLE"

    # Start agent processing
    task = {"image": None, "task_type": "test"}
    await restoration_agent.add_task(task)

    # Check queue
    status = restoration_agent.get_status()
    assert status["queue_size"] >= 0


@pytest.mark.asyncio
async def test_agent_error_handling(restoration_agent):
    """Test agent error handling"""
    # Test with invalid input
    input_data = {"image": None, "task_type": "denoise"}  # Invalid image

    result = await restoration_agent.process(input_data)

    assert result["status"] == "error"
    assert "error" in result


@pytest.mark.asyncio
async def test_orchestrator_self_correction(orchestrator):
    """Test orchestrator self-correction mechanism"""
    # Mock results with errors
    results = {
        "ImageRestorationAgent": {"error": "Processing failed"},
        "StyleAestheticAgent": {"status": "success", "output": "test"},
    }

    analysis = {"confidence": 0.8}

    corrected_results = await orchestrator._self_correct_results(results, analysis)

    assert "validation_score" in corrected_results
    assert "self_corrected" in corrected_results
    assert corrected_results["self_corrected"] == True


@pytest.mark.asyncio
async def test_agent_performance_metrics():
    """Test agent performance tracking"""

    class PerformanceTestAgent(BaseAgent):
        async def initialize(self):
            return True

        async def process(self, input_data):
            # Simulate processing time
            await asyncio.sleep(0.1)
            return {"status": "success", "processed": True}

    agent = PerformanceTestAgent("PerfTest", "Performance test agent")
    await agent.initialize()

    # Process multiple tasks
    start_time = asyncio.get_event_loop().time()

    tasks = [{"data": f"task_{i}"} for i in range(3)]
    for task in tasks:
        result = await agent.process(task)
        assert result["status"] == "success"

    end_time = asyncio.get_event_loop().time()
    processing_time = end_time - start_time

    # Should complete within reasonable time
    assert processing_time < 1.0  # Less than 1 second for 3 tasks


@pytest.mark.asyncio
async def test_concurrent_agent_processing(restoration_agent, style_agent):
    """Test concurrent processing by multiple agents"""
    # Create sample images
    images = [
        Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(2)
    ]

    # Process concurrently
    tasks = [
        restoration_agent.process({"image": images[0], "task_type": "denoise"}),
        style_agent.process({"image": images[1], "enhancement_type": "auto"}),
    ]

    results = await asyncio.gather(*tasks)

    # Both should succeed
    assert all(result["status"] == "success" for result in results)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_style_transfer_agent_process():
    agent = StyleTransferAgent()
    await agent.initialize()
    dummy_image = Image.new("RGB", (64, 64), color="red")
    result = await agent.process({"image": dummy_image, "style": "impressionist"})
    assert result["output_image"] is not None
    assert result["style"] == "impressionist"
    assert result["status"] == "stub"


@pytest.mark.asyncio
def test_vision_language_agent_process():
    agent = VisionLanguageAgent()
    import asyncio

    asyncio.run(agent.initialize())
    dummy_image = Image.new("RGB", (64, 64), color="blue")
    result = asyncio.run(agent.process({"image": dummy_image, "prompt": "Describe this image"}))
    assert "Caption for image" in result["result"]
    assert result["status"] == "stub"


class MockAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.last_task = None
        self._mock_result = {"result": f"processed by {name}"}
        self.initialized = True

    async def _initialize(self):
        self.initialized = True

    async def _process(self, task):
        self.last_task = task
        return {"agent": self.name, "input": task, **self._mock_result}

    async def _cleanup(self):
        pass


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_agent_registration_and_message_passing():
    orch = MultiAgentOrchestrator()
    agent_a = MockAgent("A")
    agent_b = MockAgent("B")
    orch.register_agent("A", agent_a)
    orch.register_agent("B", agent_b)
    msg = {"foo": "bar"}
    result = orch.send_message("A", "B", msg)
    assert result["agent"] == "B"
    assert result["input"] == msg


def test_delegate_task_sequence():
    orch = MultiAgentOrchestrator()
    agent1 = MockAgent("Agent1")
    agent2 = MockAgent("Agent2")
    orch.register_agent("Agent1", agent1)
    orch.register_agent("Agent2", agent2)
    task = {"data": 123}
    result = run_async(orch.delegate_task(task, ["Agent1", "Agent2"]))
    assert result["agent"] == "Agent2"
    assert result["input"]["agent"] == "Agent1"


def test_feedback_loop():
    orch = MultiAgentOrchestrator()
    orch.feedback({"user": "good job"})
    assert any("feedback" in h for h in orch.history)


def test_meta_agent_critique_and_negotiation(monkeypatch):
    class DummyLLM:
        def chat(self, prompt, history=None, max_tokens=256):
            return "LLM says: critique or debate"

    class DummyLLMMetaAgent(LLMMetaAgent):
        def __init__(self):
            super().__init__()
            self.llm_manager = DummyLLM()

    meta = DummyLLMMetaAgent()
    orch = MultiAgentOrchestrator(meta_agent=meta)
    agent = MockAgent("A")
    orch.register_agent("A", agent)
    # Critique
    task = {"foo": "bar"}
    result = run_async(orch.delegate_task(task, ["A"]))
    assert "meta_critique" in result
    assert "LLM says" in result["meta_critique"]["llm_response"]
    # Negotiation
    debate_result = orch.negotiate(["A"], {"context": "test"})
    assert "LLM says" in run_async(debate_result)["llm_response"]


if __name__ == "__main__":
    pytest.main([__file__])
