"""
Model Benchmarking System
Handles comprehensive model performance testing and profiling
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Optional dependencies
try:
    import torch
    import numpy as np
    import psutil
    from loguru import logger

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    if TYPE_CHECKING:
        import torch
        import numpy as np
        import psutil
        from loguru import logger


class BenchmarkType(Enum):
    """Types of benchmarks that can be performed"""

    INFERENCE_SPEED = "inference_speed"  # Basic inference speed test
    MEMORY_USAGE = "memory_usage"  # Memory consumption analysis
    THROUGHPUT = "throughput"  # Batch processing performance
    LATENCY = "latency"  # Response time analysis
    RESOURCE_USAGE = "resource_usage"  # CPU/GPU utilization


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""

    type: BenchmarkType
    num_runs: int = 100  # Number of iterations
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])  # For throughput testing
    input_shapes: List[Tuple[int, ...]] = field(
        default_factory=lambda: [(1, 3, 224, 224)]
    )  # Different input sizes
    warmup_runs: int = 10  # Number of warmup iterations
    max_memory: Optional[float] = None  # Maximum memory limit in GB
    device: str = "auto"  # Device to run on (auto/cpu/cuda)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""

    type: BenchmarkType
    metrics: Dict[str, float]
    system_info: Dict[str, Any]
    config: Dict[str, Any]
    timestamp: datetime


class ModelBenchmarker:
    """Handles comprehensive model benchmarking and profiling"""

    def __init__(self, results_dir: Path):
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "Required dependencies not found. Please install: torch, numpy, psutil, loguru"
            )

        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "benchmark_results.json"
        self.results: Dict[str, Dict[str, List[BenchmarkResult]]] = {}
        self._load_results()

    def _load_results(self) -> None:
        """Load benchmark results from disk"""
        if not self.results_file.exists():
            return

        try:
            with open(self.results_file, "r") as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.results[model_id] = {}
                    for version, results in versions.items():
                        self.results[model_id][version] = [
                            BenchmarkResult(
                                type=BenchmarkType(r["type"]),
                                metrics=r["metrics"],
                                system_info=r["system_info"],
                                config=r["config"],
                                timestamp=datetime.fromisoformat(r["timestamp"]),
                            )
                            for r in results
                        ]
        except Exception as e:
            if logger:
                logger.error(f"Error loading benchmark results: {e}")
            self.results = {}

    def _save_results(self) -> None:
        """Save benchmark results to disk"""
        data = {}
        for model_id, versions in self.results.items():
            data[model_id] = {}
            for version, results in versions.items():
                data[model_id][version] = [
                    {
                        "type": r.type.value,
                        "metrics": r.metrics,
                        "system_info": r.system_info,
                        "config": r.config,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in results
                ]

        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        if not psutil:
            return {}

        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "system": psutil.Process().name(),
            "python_version": ".".join(map(str, psutil.Process().version_info)),
        }

        if torch and torch.cuda.is_available():
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                }
            )

        return info

    async def run_benchmark(
        self, model: "torch.nn.Module", model_id: str, version: str, config: BenchmarkConfig
    ) -> BenchmarkResult:
        """
        Run benchmarks on a model

        Args:
            model: The model to benchmark
            model_id: Unique identifier for the model
            version: Version string
            config: Benchmark configuration

        Returns:
            BenchmarkResult containing performance metrics
        """
        if not torch:
            raise ImportError("PyTorch is required for benchmarking")

        # Set device
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()

        metrics = {}

        # Warmup runs
        for _ in range(config.warmup_runs):
            if config.type == BenchmarkType.INFERENCE_SPEED:
                self._run_inference(model, device)
            elif config.type == BenchmarkType.THROUGHPUT:
                self._run_batch_inference(model, device, config.batch_sizes[0])

        # Main benchmark
        if config.type == BenchmarkType.INFERENCE_SPEED:
            metrics = self._benchmark_inference_speed(model, device, config.num_runs)
        elif config.type == BenchmarkType.MEMORY_USAGE:
            metrics = self._benchmark_memory_usage(model, device)
        elif config.type == BenchmarkType.THROUGHPUT:
            metrics = self._benchmark_throughput(model, device, config.batch_sizes, config.num_runs)
        elif config.type == BenchmarkType.LATENCY:
            metrics = self._benchmark_latency(model, device, config.num_runs)
        elif config.type == BenchmarkType.RESOURCE_USAGE:
            metrics = self._benchmark_resource_usage(model, device, config.num_runs)

        # Create result
        result = BenchmarkResult(
            type=config.type,
            metrics=metrics,
            system_info=self._get_system_info(),
            config={"num_runs": config.num_runs, "device": device},
            timestamp=datetime.now(),
        )

        # Save result
        if model_id not in self.results:
            self.results[model_id] = {}
        if version not in self.results[model_id]:
            self.results[model_id][version] = []
        self.results[model_id][version].append(result)
        self._save_results()

        return result

    def _run_inference(self, model: "torch.nn.Module", device: str) -> None:
        """Run single inference pass"""
        if not torch:
            raise ImportError("PyTorch is required for inference")

        with torch.no_grad():
            input_tensor = torch.randn(1, 3, 224, 224).to(device)
            _ = model(input_tensor)

    def _run_batch_inference(self, model: "torch.nn.Module", device: str, batch_size: int) -> None:
        """Run batch inference pass"""
        if not torch:
            raise ImportError("PyTorch is required for inference")

        with torch.no_grad():
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            _ = model(input_tensor)

    def _benchmark_inference_speed(
        self, model: "torch.nn.Module", device: str, num_runs: int
    ) -> Dict[str, float]:
        """Benchmark inference speed"""
        if not torch or not np:
            raise ImportError("PyTorch and NumPy are required for benchmarking")

        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                self._run_inference(model, device)
                times.append(time.time() - start_time)

        return {
            "avg_inference_time": float(np.mean(times)),
            "std_inference_time": float(np.std(times)),
            "min_inference_time": float(np.min(times)),
            "max_inference_time": float(np.max(times)),
        }

    def _benchmark_memory_usage(self, model: "torch.nn.Module", device: str) -> Dict[str, float]:
        """Benchmark memory usage"""
        if not torch:
            raise ImportError("PyTorch is required for benchmarking")

        if device == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = psutil.Process().memory_info().rss if psutil else 0

        # Run inference to measure peak memory
        self._run_inference(model, device)

        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            peak_memory = psutil.Process().memory_info().rss if psutil else 0

        return {
            "initial_memory_mb": float(initial_memory / (1024 * 1024)),
            "peak_memory_mb": float(peak_memory / (1024 * 1024)),
            "memory_increase_mb": float((peak_memory - initial_memory) / (1024 * 1024)),
        }

    def _benchmark_throughput(
        self, model: "torch.nn.Module", device: str, batch_sizes: List[int], num_runs: int
    ) -> Dict[str, float]:
        """Benchmark throughput with different batch sizes"""
        if not torch or not np:
            raise ImportError("PyTorch and NumPy are required for benchmarking")

        results = {}

        with torch.no_grad():
            for batch_size in batch_sizes:
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    self._run_batch_inference(model, device, batch_size)
                    times.append(time.time() - start_time)

                avg_time = float(np.mean(times))
                throughput = float(batch_size / avg_time)
                results[f"throughput_batch_{batch_size}"] = throughput
                results[f"latency_batch_{batch_size}"] = avg_time

        return results

    def _benchmark_latency(
        self, model: "torch.nn.Module", device: str, num_runs: int
    ) -> Dict[str, float]:
        """Benchmark end-to-end latency"""
        if not torch or not np:
            raise ImportError("PyTorch and NumPy are required for benchmarking")

        latencies = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()

                # Include data transfer time
                input_tensor = torch.randn(1, 3, 224, 224)
                input_tensor = input_tensor.to(device)

                # Run inference
                output = model(input_tensor)

                # Transfer back to CPU if needed
                if device == "cuda":
                    output = output.cpu()

                latencies.append(time.time() - start_time)

        return {
            "avg_latency": float(np.mean(latencies)),
            "std_latency": float(np.std(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
        }

    def _benchmark_resource_usage(
        self, model: "torch.nn.Module", device: str, num_runs: int
    ) -> Dict[str, float]:
        """Benchmark CPU and GPU resource usage"""
        if not torch or not np:
            raise ImportError("PyTorch and NumPy are required for benchmarking")

        cpu_usage = []
        gpu_usage = []

        for _ in range(num_runs):
            start_cpu = psutil.cpu_percent() if psutil else 0

            if device == "cuda":
                start_gpu = torch.cuda.memory_allocated()

            self._run_inference(model, device)

            if psutil:
                cpu_usage.append(psutil.cpu_percent() - start_cpu)

            if device == "cuda":
                gpu_usage.append(torch.cuda.memory_allocated() - start_gpu)

        metrics = {
            "avg_cpu_usage": float(np.mean(cpu_usage)) if cpu_usage else 0.0,
            "max_cpu_usage": float(np.max(cpu_usage)) if cpu_usage else 0.0,
        }

        if device == "cuda":
            metrics.update(
                {
                    "avg_gpu_memory_usage": float(np.mean(gpu_usage) / (1024 * 1024)),
                    "max_gpu_memory_usage": float(np.max(gpu_usage) / (1024 * 1024)),
                }
            )

        return metrics

    def get_benchmark_history(
        self, model_id: str, version: str, benchmark_type: Optional[BenchmarkType] = None
    ) -> List[BenchmarkResult]:
        """Get benchmark history for a model version"""
        results = self.results.get(model_id, {}).get(version, [])
        if benchmark_type:
            results = [r for r in results if r.type == benchmark_type]
        return sorted(results, key=lambda x: x.timestamp)

    def get_latest_benchmark(
        self, model_id: str, version: str, benchmark_type: Optional[BenchmarkType] = None
    ) -> Optional[BenchmarkResult]:
        """Get the most recent benchmark result"""
        history = self.get_benchmark_history(model_id, version, benchmark_type)
        return history[-1] if history else None

    def compare_versions(
        self, model_id: str, version1: str, version2: str, benchmark_type: BenchmarkType
    ) -> Dict[str, Any]:
        """Compare benchmark results between two versions"""
        result1 = self.get_latest_benchmark(model_id, version1, benchmark_type)
        result2 = self.get_latest_benchmark(model_id, version2, benchmark_type)

        if not result1 or not result2:
            raise ValueError("Benchmark results not found for comparison")

        comparison = {}
        for metric in result1.metrics:
            if metric in result2.metrics:
                val1 = result1.metrics[metric]
                val2 = result2.metrics[metric]
                comparison[metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1,
                    "percent_change": ((val2 - val1) / val1) * 100,
                }

        return comparison

    def clear_benchmark_history(
        self, model_id: Optional[str] = None, version: Optional[str] = None
    ) -> None:
        """Clear benchmark history for specific model or all models"""
        if model_id is None:
            self.results.clear()
        elif version is None and model_id in self.results:
            del self.results[model_id]
        elif model_id in self.results and version in self.results[model_id]:
            del self.results[model_id][version]

        self._save_results()
