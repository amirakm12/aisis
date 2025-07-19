"""
Performance tests for AISIS
Tests system performance, resource usage, and scalability
"""

import asyncio
import time
import pytest
import torch
import numpy as np
from PIL import Image
import psutil
import os
from pathlib import Path

from src.core.config import config
from src.core.gpu_utils import gpu_manager
from src.agents.orchestrator import HyperOrchestrator
from src.agents.image_restoration import ImageRestorationAgent
from src.agents.style_aesthetic import StyleAestheticAgent
from src.agents.semantic_editing import SemanticEditingAgent


@pytest.fixture
def performance_metrics():
    """Fixture for tracking performance metrics"""

    class Metrics:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.peak_memory = None
            self.gpu_usage = []
            self.cpu_usage = []

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
            self.peak_memory = self.start_memory

        def update(self):
            current_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)

            if torch.cuda.is_available():
                self.gpu_usage.append(torch.cuda.memory_allocated())

            self.cpu_usage.append(psutil.cpu_percent())

        def stop(self):
            self.end_time = time.time()

        def get_results(self):
            return {
                "duration": self.end_time - self.start_time,
                "memory_increase": self.peak_memory - self.start_memory,
                "peak_memory": self.peak_memory,
                "avg_gpu_usage": np.mean(self.gpu_usage) if self.gpu_usage else 0,
                "avg_cpu_usage": np.mean(self.cpu_usage),
            }

    return Metrics()


@pytest.fixture
async def test_system():
    """Initialize test system components"""
    orchestrator = HyperOrchestrator()
    await orchestrator.initialize()

    agents = [ImageRestorationAgent(), StyleAestheticAgent(), SemanticEditingAgent()]

    for agent in agents:
        await agent.initialize()
        await orchestrator.register_agent(agent)

    return {"orchestrator": orchestrator, "agents": agents}


@pytest.mark.asyncio
async def test_concurrent_processing_scalability(test_system, performance_metrics):
    """Test system scalability with concurrent processing"""
    orchestrator = test_system["orchestrator"]
    performance_metrics.start()

    # Test with increasing concurrency levels
    concurrency_levels = [1, 2, 4, 8]
    results = {}

    for num_concurrent in concurrency_levels:
        tasks = [
            {
                "description": f"test task {i}",
                "image": create_test_image(),
                "parameters": {"quality": "high"},
            }
            for i in range(num_concurrent)
        ]

        start_time = time.time()
        task_results = await asyncio.gather(*[orchestrator.process(task) for task in tasks])
        duration = time.time() - start_time

        success_count = sum(1 for r in task_results if r["status"] == "success")

        results[num_concurrent] = {
            "duration": duration,
            "tasks_per_second": num_concurrent / duration,
            "success_rate": success_count / num_concurrent,
        }

        performance_metrics.update()

    performance_metrics.stop()
    metrics = performance_metrics.get_results()

    # Verify scalability
    # Should maintain reasonable tasks/second as concurrency increases
    tasks_per_second = [r["tasks_per_second"] for r in results.values()]
    assert all(tps > 0.5 for tps in tasks_per_second)  # At least 0.5 tasks/second

    # Memory usage should be reasonable
    memory_mb = metrics["peak_memory"] / (1024 * 1024)
    assert memory_mb < 4096  # Less than 4GB peak memory


@pytest.mark.asyncio
async def test_gpu_memory_efficiency(test_system, performance_metrics):
    """Test GPU memory usage efficiency"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    orchestrator = test_system["orchestrator"]
    performance_metrics.start()

    # Process increasingly large images
    image_sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    results = {}

    for size in image_sizes:
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        result = await orchestrator.process(
            {
                "description": "test gpu memory",
                "image": create_test_image(size),
                "parameters": {"quality": "ultra"},
            }
        )

        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - initial_memory

        results[f"{size[0]}x{size[1]}"] = {
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "memory_increase_mb": memory_increase / (1024 * 1024),
        }

        # Clear cache between tests
        torch.cuda.empty_cache()
        performance_metrics.update()

    performance_metrics.stop()

    # Verify memory efficiency
    # Memory usage should scale sub-linearly with image size
    memory_increases = [r["memory_increase_mb"] for r in results.values()]
    ratios = [m2 / m1 for m1, m2 in zip(memory_increases, memory_increases[1:])]
    assert all(r < 4 for r in ratios)  # Memory shouldn't grow too quickly


@pytest.mark.asyncio
async def test_long_running_stability(test_system, performance_metrics):
    """Test system stability during long-running operations"""
    orchestrator = test_system["orchestrator"]
    performance_metrics.start()

    num_iterations = 20
    results = []

    for i in range(num_iterations):
        start_time = time.time()

        result = await orchestrator.process(
            {
                "description": f"long running test {i}",
                "image": create_test_image(),
                "parameters": {"quality": "high", "iterations": 5},  # Make each task take longer
            }
        )

        duration = time.time() - start_time
        results.append({"iteration": i, "duration": duration, "status": result["status"]})

        performance_metrics.update()

    performance_metrics.stop()
    metrics = performance_metrics.get_results()

    # Verify stability
    success_rate = sum(1 for r in results if r["status"] == "success") / num_iterations
    assert success_rate > 0.95  # At least 95% success rate

    # Performance shouldn't degrade significantly
    durations = [r["duration"] for r in results]
    duration_increase = (durations[-1] - durations[0]) / durations[0]
    assert duration_increase < 0.5  # Less than 50% slowdown


@pytest.mark.asyncio
async def test_resource_cleanup(test_system, performance_metrics):
    """Test resource cleanup and memory release"""
    orchestrator = test_system["orchestrator"]
    performance_metrics.start()

    initial_memory = psutil.Process().memory_info().rss
    initial_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Run several tasks
    for _ in range(5):
        result = await orchestrator.process(
            {
                "description": "cleanup test",
                "image": create_test_image(),
                "parameters": {"quality": "high"},
            }
        )

        # Force cleanup
        gpu_manager.clear_cache()
        performance_metrics.update()

    # Trigger garbage collection
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    performance_metrics.stop()

    final_memory = psutil.Process().memory_info().rss
    final_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Verify cleanup
    memory_diff = final_memory - initial_memory
    gpu_memory_diff = final_gpu_memory - initial_gpu_memory

    # Allow for some memory overhead
    assert memory_diff < 100 * 1024 * 1024  # Less than 100MB increase
    if torch.cuda.is_available():
        assert gpu_memory_diff < 50 * 1024 * 1024  # Less than 50MB GPU memory increase


@pytest.mark.asyncio
async def test_cpu_gpu_load_balancing(test_system, performance_metrics):
    """Test CPU/GPU load balancing"""
    orchestrator = test_system["orchestrator"]
    performance_metrics.start()

    # Create mixed CPU/GPU tasks
    num_tasks = 10
    tasks = []

    for i in range(num_tasks):
        # Alternate between CPU and GPU intensive tasks
        is_gpu_task = i % 2 == 0
        tasks.append(
            {
                "description": f"{'gpu' if is_gpu_task else 'cpu'} intensive task",
                "image": create_test_image(),
                "parameters": {"use_gpu": is_gpu_task, "compute_intensive": True},
            }
        )

    # Process tasks concurrently
    results = await asyncio.gather(*[orchestrator.process(task) for task in tasks])

    performance_metrics.stop()
    metrics = performance_metrics.get_results()

    # Verify load balancing
    assert metrics["avg_cpu_usage"] > 20  # Should utilize CPU
    if torch.cuda.is_available():
        assert metrics["avg_gpu_usage"] > 0  # Should utilize GPU when available


def create_test_image(size=(512, 512)):
    """Create a test image of specified size"""
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))


if __name__ == "__main__":
    pytest.main([__file__])
