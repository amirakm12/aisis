"""
Integration tests for AISIS
Tests end-to-end workflows and cross-component interactions
"""

import asyncio
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.core.config import config
from src.core.gpu_utils import gpu_manager
from src.core.voice_manager import voice_manager
from src.agents.orchestrator import HyperOrchestrator
from src.agents.image_restoration import ImageRestorationAgent
from src.agents.style_aesthetic import StyleAestheticAgent
from src.agents.semantic_editing import SemanticEditingAgent
from src.ui.main_window import MainWindow

@pytest.fixture
async def initialized_system():
    """Initialize all system components"""
    # Initialize voice system
    await voice_manager.initialize()
    
    # Initialize orchestrator and agents
    orchestrator = HyperOrchestrator()
    await orchestrator.initialize()
    
    restoration_agent = ImageRestorationAgent()
    style_agent = StyleAestheticAgent()
    semantic_agent = SemanticEditingAgent()
    
    await restoration_agent.initialize()
    await style_agent.initialize()
    await semantic_agent.initialize()
    
    await orchestrator.register_agent(restoration_agent)
    await orchestrator.register_agent(style_agent)
    await orchestrator.register_agent(semantic_agent)
    
    yield {
        "orchestrator": orchestrator,
        "voice_manager": voice_manager,
        "restoration_agent": restoration_agent,
        "style_agent": style_agent,
        "semantic_agent": semantic_agent
    }
    
    # Cleanup
    voice_manager.cleanup()
    gpu_manager.clear_cache()

@pytest.mark.asyncio
async def test_voice_interaction_flow(initialized_system):
    """Test complete voice interaction flow"""
    voice_manager = initialized_system["voice_manager"]
    orchestrator = initialized_system["orchestrator"]
    
    # Test voice command processing
    test_command = "make the image more dramatic and remove noise"
    
    # Simulate voice input
    voice_manager.last_command = test_command
    
    # Process command through orchestrator
    result = await orchestrator.process({
        "description": test_command,
        "image": create_test_image(),
        "parameters": {}
    })
    
    assert result["status"] == "success"
    assert len(result["results"]) > 0
    assert "execution_plan" in result

@pytest.mark.asyncio
async def test_complex_multi_agent_scenario(initialized_system):
    """Test complex scenario involving multiple agents"""
    orchestrator = initialized_system["orchestrator"]
    
    # Create a complex task requiring multiple agents
    task = {
        "description": "enhance the image quality, apply vintage style, and remove the background",
        "image": create_test_image(),
        "parameters": {
            "quality": "high",
            "style": "vintage",
            "background_removal": True
        }
    }
    
    # Process task
    result = await orchestrator.process(task)
    
    assert result["status"] == "success"
    assert len(result["results"]) >= 3  # Should involve at least 3 agents
    
    # Verify agent coordination
    execution_plan = result["execution_plan"]
    assert len(execution_plan.execution_order) >= 3
    assert "ImageRestorationAgent" in execution_plan.required_agents
    assert "StyleAestheticAgent" in execution_plan.required_agents
    assert "SemanticEditingAgent" in execution_plan.required_agents

@pytest.mark.asyncio
async def test_gpu_memory_management(initialized_system):
    """Test GPU memory management under heavy load"""
    orchestrator = initialized_system["orchestrator"]
    
    # Create multiple large tasks
    tasks = []
    for _ in range(5):
        tasks.append({
            "description": "enhance and style the image",
            "image": create_large_test_image(),
            "parameters": {"quality": "ultra"}
        })
    
    # Process tasks concurrently
    results = await asyncio.gather(
        *[orchestrator.process(task) for task in tasks],
        return_exceptions=True
    )
    
    # Check results and memory usage
    successful_results = [r for r in results if not isinstance(r, Exception)]
    assert len(successful_results) > 0
    
    # Verify GPU memory was properly managed
    memory_info = gpu_manager.get_memory_info()
    assert memory_info["available_mb"] > 0  # Should not exhaust GPU memory

@pytest.mark.asyncio
async def test_ui_responsiveness(qtbot):
    """Test UI responsiveness during long operations"""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    
    # Start a long-running operation
    async def long_operation():
        await asyncio.sleep(2)  # Simulate long process
        return {"status": "success"}
    
    # Verify UI remains responsive
    with qtbot.waitSignal(window.operation_complete, timeout=3000):
        window.start_operation(long_operation())
        
        # Try UI interactions during operation
        qtbot.mouseClick(window.voice_button, Qt.LeftButton)
        assert window.voice_button.isEnabled()  # UI should remain responsive

@pytest.mark.asyncio
async def test_cross_agent_communication(initialized_system):
    """Test communication and data sharing between agents"""
    restoration_agent = initialized_system["restoration_agent"]
    style_agent = initialized_system["style_agent"]
    
    # Create shared data
    shared_image = create_test_image()
    shared_params = {"intensity": 0.8}
    
    # Process with first agent
    result1 = await restoration_agent.process({
        "image": shared_image,
        "task_type": "enhance",
        "parameters": shared_params
    })
    
    assert result1["status"] == "success"
    
    # Pass result to second agent
    result2 = await style_agent.process({
        "image": result1["output_image"],
        "enhancement_type": "artistic",
        "parameters": {
            **shared_params,
            "previous_results": result1
        }
    })
    
    assert result2["status"] == "success"
    assert "previous_results" in result2

@pytest.mark.asyncio
async def test_system_state_persistence(initialized_system, tmp_path):
    """Test system state persistence and recovery"""
    orchestrator = initialized_system["orchestrator"]
    
    # Create and process a task
    task = {
        "description": "enhance the image",
        "image": create_test_image(),
        "parameters": {"save_state": True}
    }
    
    result = await orchestrator.process(task)
    assert result["status"] == "success"
    
    # Save system state
    state_file = tmp_path / "system_state.json"
    orchestrator.save_state(state_file)
    
    # Create new orchestrator and restore state
    new_orchestrator = HyperOrchestrator()
    await new_orchestrator.initialize()
    new_orchestrator.load_state(state_file)
    
    # Verify state was restored
    assert len(new_orchestrator.registered_agents) == len(orchestrator.registered_agents)
    assert new_orchestrator.reasoning_mode == orchestrator.reasoning_mode

@pytest.mark.asyncio
async def test_performance_benchmarking(initialized_system):
    """Test system performance metrics"""
    orchestrator = initialized_system["orchestrator"]
    
    # Prepare benchmark tasks
    tasks = []
    image_sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    for size in image_sizes:
        tasks.append({
            "description": "benchmark test",
            "image": create_test_image(size),
            "parameters": {"benchmark": True}
        })
    
    # Measure processing time
    start_time = asyncio.get_event_loop().time()
    
    results = await asyncio.gather(
        *[orchestrator.process(task) for task in tasks]
    )
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    # Analyze results
    successful_tasks = [r for r in results if r["status"] == "success"]
    assert len(successful_tasks) == len(tasks)
    
    # Calculate metrics
    avg_time_per_task = total_time / len(tasks)
    assert avg_time_per_task < 5.0  # Should process each task within 5 seconds

def create_test_image(size=(256, 256)):
    """Create a test image"""
    return Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )

def create_large_test_image():
    """Create a large test image"""
    return create_test_image((2048, 2048))

if __name__ == "__main__":
    pytest.main([__file__])
