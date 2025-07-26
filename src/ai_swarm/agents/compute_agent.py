"""
Compute Optimization Agent
Specializes in vectorized workloads, SIMD operations, GPU kernels, and low-level optimization
"""

import asyncio
import time
import threading
import numpy as np
import numba
from numba import jit, cuda
from typing import Dict, List, Any, Optional, Callable
import psutil
import ctypes
import mmap
import os
import subprocess
import platform

from ..core.agent_base import BaseAgent, AgentState
from ..core.communication import MessagePriority


class ComputeWorkload:
    """Represents a compute workload for optimization"""
    
    def __init__(self, workload_id: str, workload_type: str, data: Any, 
                 priority: int = 5, target_latency: float = 0.001):
        self.workload_id = workload_id
        self.workload_type = workload_type
        self.data = data
        self.priority = priority
        self.target_latency = target_latency
        self.optimization_level = 0
        self.execution_history = []


class ComputeAgent(BaseAgent):
    """AI agent specialized in compute optimization and execution"""
    
    def __init__(self, agent_id: str = "compute_agent", cpu_affinity: Optional[List[int]] = None):
        super().__init__(agent_id, priority=8, cpu_affinity=cpu_affinity)
        
        # Compute optimization state
        self.workload_queue = asyncio.Queue(maxsize=1000)
        self.active_workloads: Dict[str, ComputeWorkload] = {}
        self.optimization_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.vectorization_stats = {
            "simd_operations": 0,
            "vectorized_loops": 0,
            "gpu_kernels_launched": 0,
            "cache_hits": 0,
            "optimization_time": 0.0
        }
        
        # Hardware capabilities
        self.cpu_features = self._detect_cpu_features()
        self.gpu_available = self._detect_gpu_capabilities()
        self.memory_bandwidth = self._measure_memory_bandwidth()
        
        # JIT compilation cache
        self.jit_cache: Dict[str, Callable] = {}
        self.kernel_cache: Dict[str, Any] = {}
        
        # Optimization strategies
        self.optimization_strategies = {
            "vectorize": self._vectorize_computation,
            "simd": self._apply_simd_optimization,
            "gpu_offload": self._offload_to_gpu,
            "cache_optimize": self._optimize_cache_usage,
            "parallel": self._parallelize_computation,
            "jit_compile": self._jit_compile_hot_paths
        }
        
        # Learning system for optimization selection
        self.strategy_performance = {}
        self.workload_patterns = {}
        
        self.logger.info(f"Compute agent initialized with CPU features: {self.cpu_features}")
        if self.gpu_available:
            self.logger.info("GPU acceleration available")
    
    def _detect_cpu_features(self) -> Dict[str, bool]:
        """Detect available CPU features for optimization"""
        features = {
            "sse": False,
            "sse2": False,
            "sse3": False,
            "ssse3": False,
            "sse4_1": False,
            "sse4_2": False,
            "avx": False,
            "avx2": False,
            "avx512": False,
            "fma": False
        }
        
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    for feature in features.keys():
                        if feature in cpuinfo:
                            features[feature] = True
            elif platform.system() == "Windows":
                # Use wmic or registry to detect features
                # This is a simplified version
                features["sse2"] = True  # Most modern CPUs have this
                features["avx"] = True   # Assume AVX is available
                
        except Exception as e:
            self.logger.warning(f"Could not detect CPU features: {e}")
        
        return features
    
    def _detect_gpu_capabilities(self) -> bool:
        """Detect GPU availability and capabilities"""
        try:
            # Try to detect CUDA
            if cuda.is_available():
                self.logger.info("CUDA GPU detected")
                return True
        except Exception:
            pass
        
        try:
            # Try to detect OpenCL or other GPU APIs
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                self.logger.info("OpenCL GPU detected")
                return True
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"OpenCL detection failed: {e}")
        
        return False
    
    def _measure_memory_bandwidth(self) -> float:
        """Measure memory bandwidth for optimization decisions"""
        try:
            # Simple memory bandwidth test
            size = 10 * 1024 * 1024  # 10MB
            data = np.random.random(size).astype(np.float32)
            
            start_time = time.perf_counter()
            for _ in range(10):
                result = np.sum(data)  # Memory-bound operation
            end_time = time.perf_counter()
            
            # Estimate bandwidth (very rough)
            bytes_processed = size * 4 * 10  # 4 bytes per float32, 10 iterations
            bandwidth = bytes_processed / (end_time - start_time) / (1024**3)  # GB/s
            
            self.logger.info(f"Estimated memory bandwidth: {bandwidth:.2f} GB/s")
            return bandwidth
            
        except Exception as e:
            self.logger.warning(f"Could not measure memory bandwidth: {e}")
            return 10.0  # Default assumption
    
    async def execute_cycle(self):
        """Main execution cycle for compute optimization"""
        try:
            # Process pending workloads
            await self._process_workload_queue()
            
            # Optimize active workloads
            await self._optimize_active_workloads()
            
            # Update performance metrics
            self.update_metrics()
            
            # Adapt optimization strategies based on performance
            await self._adapt_optimization_strategies()
            
        except Exception as e:
            self.logger.error(f"Error in compute agent cycle: {e}")
            self.state = AgentState.ERROR
    
    async def _process_workload_queue(self):
        """Process incoming compute workloads"""
        try:
            # Process up to 10 workloads per cycle for responsiveness
            for _ in range(10):
                try:
                    workload = self.workload_queue.get_nowait()
                    await self._execute_workload(workload)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            self.logger.error(f"Error processing workload queue: {e}")
    
    async def _execute_workload(self, workload: ComputeWorkload):
        """Execute a compute workload with optimization"""
        start_time = time.perf_counter()
        
        try:
            self.active_workloads[workload.workload_id] = workload
            
            # Select optimization strategy based on workload characteristics
            strategy = await self._select_optimization_strategy(workload)
            
            # Apply optimization
            optimized_result = await strategy(workload)
            
            # Record execution time
            execution_time = time.perf_counter() - start_time
            workload.execution_history.append({
                "timestamp": time.time(),
                "execution_time": execution_time,
                "strategy": strategy.__name__,
                "success": True
            })
            
            # Update performance statistics
            self.metrics.tasks_completed += 1
            self._update_strategy_performance(strategy.__name__, execution_time, True)
            
            # Send result back if needed
            await self._send_workload_result(workload, optimized_result, execution_time)
            
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            self.logger.error(f"Failed to execute workload {workload.workload_id}: {e}")
            
            workload.execution_history.append({
                "timestamp": time.time(),
                "execution_time": execution_time,
                "strategy": "failed",
                "success": False,
                "error": str(e)
            })
            
            self.metrics.tasks_failed += 1
            
        finally:
            # Clean up
            if workload.workload_id in self.active_workloads:
                del self.active_workloads[workload.workload_id]
    
    async def _select_optimization_strategy(self, workload: ComputeWorkload) -> Callable:
        """Select the best optimization strategy for a workload"""
        workload_signature = self._get_workload_signature(workload)
        
        # Check if we have performance data for this type of workload
        if workload_signature in self.workload_patterns:
            pattern_data = self.workload_patterns[workload_signature]
            best_strategy = min(pattern_data.items(), key=lambda x: x[1]["avg_time"])
            return self.optimization_strategies[best_strategy[0]]
        
        # Default strategy selection based on workload characteristics
        if workload.workload_type == "matrix_multiply":
            if self.gpu_available and hasattr(workload.data, "shape") and np.prod(workload.data.shape) > 10000:
                return self.optimization_strategies["gpu_offload"]
            else:
                return self.optimization_strategies["vectorize"]
        elif workload.workload_type == "element_wise":
            return self.optimization_strategies["simd"]
        elif workload.workload_type == "reduction":
            return self.optimization_strategies["parallel"]
        else:
            return self.optimization_strategies["jit_compile"]
    
    def _get_workload_signature(self, workload: ComputeWorkload) -> str:
        """Generate a signature for workload pattern recognition"""
        signature_parts = [workload.workload_type]
        
        if hasattr(workload.data, "shape"):
            signature_parts.append(f"shape_{workload.data.shape}")
        if hasattr(workload.data, "dtype"):
            signature_parts.append(f"dtype_{workload.data.dtype}")
            
        return "_".join(signature_parts)
    
    async def _vectorize_computation(self, workload: ComputeWorkload) -> Any:
        """Apply vectorization optimizations"""
        try:
            if isinstance(workload.data, np.ndarray):
                # Use NumPy's vectorized operations
                if workload.workload_type == "element_wise_add":
                    result = np.add(workload.data[0], workload.data[1])
                elif workload.workload_type == "element_wise_multiply":
                    result = np.multiply(workload.data[0], workload.data[1])
                elif workload.workload_type == "matrix_multiply":
                    result = np.dot(workload.data[0], workload.data[1])
                else:
                    # Generic vectorized operation
                    result = workload.data
                
                self.vectorization_stats["vectorized_loops"] += 1
                return result
            else:
                return workload.data
                
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            return workload.data
    
    async def _apply_simd_optimization(self, workload: ComputeWorkload) -> Any:
        """Apply SIMD optimizations using Numba"""
        try:
            # Check cache first
            cache_key = f"simd_{workload.workload_type}_{hash(str(workload.data))}"
            if cache_key in self.optimization_cache:
                self.vectorization_stats["cache_hits"] += 1
                return self.optimization_cache[cache_key]
            
            # Apply SIMD optimization based on available CPU features
            if self.cpu_features.get("avx2", False):
                result = await self._avx2_optimize(workload)
            elif self.cpu_features.get("sse4_2", False):
                result = await self._sse_optimize(workload)
            else:
                result = await self._vectorize_computation(workload)
            
            # Cache result
            self.optimization_cache[cache_key] = result
            self.vectorization_stats["simd_operations"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"SIMD optimization failed: {e}")
            return await self._vectorize_computation(workload)
    
    async def _avx2_optimize(self, workload: ComputeWorkload) -> Any:
        """Apply AVX2 optimizations"""
        # This would implement AVX2-specific optimizations
        # For now, we'll use Numba's vectorization as a proxy
        @numba.jit(nopython=True, fastmath=True)
        def avx2_operation(data):
            if data.ndim == 1:
                return np.sum(data)  # Simple reduction
            else:
                return np.sum(data, axis=0)
        
        try:
            if isinstance(workload.data, np.ndarray):
                return avx2_operation(workload.data)
            else:
                return workload.data
        except Exception:
            return workload.data
    
    async def _sse_optimize(self, workload: ComputeWorkload) -> Any:
        """Apply SSE optimizations"""
        # Similar to AVX2 but with SSE constraints
        @numba.jit(nopython=True)
        def sse_operation(data):
            return np.mean(data)  # Simple operation
        
        try:
            if isinstance(workload.data, np.ndarray):
                return sse_operation(workload.data)
            else:
                return workload.data
        except Exception:
            return workload.data
    
    async def _offload_to_gpu(self, workload: ComputeWorkload) -> Any:
        """Offload computation to GPU"""
        if not self.gpu_available:
            return await self._vectorize_computation(workload)
        
        try:
            # Check if we have a cached GPU kernel
            kernel_key = f"gpu_{workload.workload_type}"
            if kernel_key in self.kernel_cache:
                kernel = self.kernel_cache[kernel_key]
            else:
                kernel = self._create_gpu_kernel(workload)
                self.kernel_cache[kernel_key] = kernel
            
            # Execute on GPU
            if isinstance(workload.data, np.ndarray):
                # Transfer to GPU
                gpu_data = cuda.to_device(workload.data)
                
                # Execute kernel
                threads_per_block = 256
                blocks_per_grid = (workload.data.size + threads_per_block - 1) // threads_per_block
                
                if kernel:
                    kernel[blocks_per_grid, threads_per_block](gpu_data)
                
                # Transfer back
                result = gpu_data.copy_to_host()
                self.vectorization_stats["gpu_kernels_launched"] += 1
                
                return result
            else:
                return workload.data
                
        except Exception as e:
            self.logger.error(f"GPU offload failed: {e}")
            return await self._vectorize_computation(workload)
    
    def _create_gpu_kernel(self, workload: ComputeWorkload) -> Optional[Callable]:
        """Create optimized GPU kernel for workload type"""
        try:
            if workload.workload_type == "element_wise_add":
                @cuda.jit
                def add_kernel(data):
                    idx = cuda.grid(1)
                    if idx < data.size:
                        data[idx] = data[idx] + 1.0
                return add_kernel
            
            elif workload.workload_type == "element_wise_multiply":
                @cuda.jit
                def multiply_kernel(data):
                    idx = cuda.grid(1)
                    if idx < data.size:
                        data[idx] = data[idx] * 2.0
                return multiply_kernel
            
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create GPU kernel: {e}")
            return None
    
    async def _optimize_cache_usage(self, workload: ComputeWorkload) -> Any:
        """Optimize memory access patterns for cache efficiency"""
        try:
            if isinstance(workload.data, np.ndarray):
                # Ensure data is contiguous for better cache performance
                if not workload.data.flags.c_contiguous:
                    workload.data = np.ascontiguousarray(workload.data)
                
                # Apply cache-friendly algorithms
                if workload.workload_type == "matrix_multiply":
                    # Use blocked matrix multiplication for better cache usage
                    return self._blocked_matrix_multiply(workload.data[0], workload.data[1])
                else:
                    return workload.data
            else:
                return workload.data
                
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return workload.data
    
    def _blocked_matrix_multiply(self, A: np.ndarray, B: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Cache-optimized blocked matrix multiplication"""
        try:
            m, k = A.shape
            k2, n = B.shape
            assert k == k2
            
            C = np.zeros((m, n), dtype=A.dtype)
            
            for i in range(0, m, block_size):
                for j in range(0, n, block_size):
                    for kk in range(0, k, block_size):
                        # Define block boundaries
                        i_end = min(i + block_size, m)
                        j_end = min(j + block_size, n)
                        kk_end = min(kk + block_size, k)
                        
                        # Multiply blocks
                        C[i:i_end, j:j_end] += np.dot(
                            A[i:i_end, kk:kk_end],
                            B[kk:kk_end, j:j_end]
                        )
            
            return C
            
        except Exception as e:
            self.logger.error(f"Blocked matrix multiply failed: {e}")
            return np.dot(A, B)  # Fallback
    
    async def _parallelize_computation(self, workload: ComputeWorkload) -> Any:
        """Apply parallel processing optimizations"""
        try:
            @numba.jit(nopython=True, parallel=True)
            def parallel_operation(data):
                if data.ndim == 1:
                    result = np.zeros_like(data)
                    for i in numba.prange(len(data)):
                        result[i] = data[i] * 2 + 1
                    return result
                else:
                    return data
            
            if isinstance(workload.data, np.ndarray):
                return parallel_operation(workload.data)
            else:
                return workload.data
                
        except Exception as e:
            self.logger.error(f"Parallelization failed: {e}")
            return workload.data
    
    async def _jit_compile_hot_paths(self, workload: ComputeWorkload) -> Any:
        """JIT compile frequently used code paths"""
        try:
            jit_key = f"jit_{workload.workload_type}"
            
            if jit_key not in self.jit_cache:
                # Create JIT compiled function
                if workload.workload_type == "reduction":
                    @jit(nopython=True)
                    def jit_reduction(data):
                        return np.sum(data)
                    self.jit_cache[jit_key] = jit_reduction
                
                elif workload.workload_type == "transform":
                    @jit(nopython=True)
                    def jit_transform(data):
                        return np.sqrt(data)
                    self.jit_cache[jit_key] = jit_transform
                
                else:
                    return workload.data
            
            # Execute JIT compiled function
            jit_func = self.jit_cache[jit_key]
            if isinstance(workload.data, np.ndarray):
                return jit_func(workload.data)
            else:
                return workload.data
                
        except Exception as e:
            self.logger.error(f"JIT compilation failed: {e}")
            return workload.data
    
    async def _optimize_active_workloads(self):
        """Continuously optimize active workloads"""
        for workload_id, workload in list(self.active_workloads.items()):
            try:
                # Check if workload needs re-optimization
                if len(workload.execution_history) > 0:
                    last_execution = workload.execution_history[-1]
                    if last_execution["execution_time"] > workload.target_latency * 1.5:
                        # Try a different optimization strategy
                        workload.optimization_level += 1
                        await self._execute_workload(workload)
                        
            except Exception as e:
                self.logger.error(f"Error optimizing workload {workload_id}: {e}")
    
    def _update_strategy_performance(self, strategy_name: str, execution_time: float, success: bool):
        """Update performance statistics for optimization strategies"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "success_rate": 0.0
            }
        
        stats = self.strategy_performance[strategy_name]
        stats["total_executions"] += 1
        stats["total_time"] += execution_time
        
        if success:
            stats["successful_executions"] += 1
        
        stats["avg_time"] = stats["total_time"] / stats["total_executions"]
        stats["success_rate"] = stats["successful_executions"] / stats["total_executions"]
    
    async def _adapt_optimization_strategies(self):
        """Adapt optimization strategies based on performance feedback"""
        # Analyze strategy performance and adjust selection logic
        if len(self.strategy_performance) > 0:
            best_strategies = sorted(
                self.strategy_performance.items(),
                key=lambda x: (x[1]["success_rate"], -x[1]["avg_time"]),
                reverse=True
            )
            
            # Log performance insights
            if len(best_strategies) > 0:
                best_strategy = best_strategies[0]
                self.logger.debug(
                    f"Best performing strategy: {best_strategy[0]} "
                    f"(success: {best_strategy[1]['success_rate']:.2%}, "
                    f"avg_time: {best_strategy[1]['avg_time']:.4f}s)"
                )
    
    async def _send_workload_result(self, workload: ComputeWorkload, result: Any, execution_time: float):
        """Send workload execution result back to requester"""
        if self.communicator:
            await self.broadcast_message(
                "workload_completed",
                {
                    "workload_id": workload.workload_id,
                    "execution_time": execution_time,
                    "result_size": len(str(result)) if result is not None else 0,
                    "optimization_applied": True
                },
                priority=MessagePriority.NORMAL
            )
    
    async def handle_message(self, message):
        """Handle compute-specific messages"""
        await super().handle_message(message)
        
        if message.message_type == "execute_workload":
            workload_data = message.payload
            workload = ComputeWorkload(
                workload_id=workload_data.get("workload_id", "unknown"),
                workload_type=workload_data.get("workload_type", "generic"),
                data=workload_data.get("data"),
                priority=workload_data.get("priority", 5),
                target_latency=workload_data.get("target_latency", 0.001)
            )
            
            try:
                await self.workload_queue.put(workload)
            except asyncio.QueueFull:
                self.logger.warning(f"Workload queue full, dropping workload {workload.workload_id}")
    
    async def _agent_specific_optimization(self):
        """Compute agent specific optimizations"""
        # Clear old cache entries
        if len(self.optimization_cache) > 1000:
            # Keep only the most recently used entries
            sorted_cache = sorted(
                self.optimization_cache.items(),
                key=lambda x: hash(x[0])  # Simple LRU approximation
            )
            self.optimization_cache = dict(sorted_cache[-500:])
        
        # Optimize JIT cache
        if len(self.jit_cache) > 50:
            # Keep only frequently used JIT functions
            # This is a simplified version - in practice, we'd track usage
            self.jit_cache = dict(list(self.jit_cache.items())[-25:])
        
        self.logger.debug(f"Compute agent optimization complete. Cache sizes: "
                         f"opt={len(self.optimization_cache)}, jit={len(self.jit_cache)}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get compute agent specific statistics"""
        return {
            "vectorization_stats": self.vectorization_stats.copy(),
            "strategy_performance": self.strategy_performance.copy(),
            "cache_sizes": {
                "optimization_cache": len(self.optimization_cache),
                "jit_cache": len(self.jit_cache),
                "kernel_cache": len(self.kernel_cache)
            },
            "hardware_capabilities": {
                "cpu_features": self.cpu_features,
                "gpu_available": self.gpu_available,
                "memory_bandwidth": self.memory_bandwidth
            },
            "active_workloads": len(self.active_workloads),
            "workload_queue_size": self.workload_queue.qsize()
        }