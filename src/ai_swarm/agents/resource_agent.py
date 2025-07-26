"""
Resource Management Agent
Monitors CPU/GPU loads, memory bandwidth, I/O queues and dynamically reassigns tasks
"""

import asyncio
import time
import threading
import psutil
import os
import platform
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np

from ..core.agent_base import BaseAgent, AgentState
from ..core.communication import MessagePriority


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    memory_percent: float = 0.0
    memory_available: int = 0
    memory_bandwidth_utilization: float = 0.0
    disk_io_read: int = 0
    disk_io_write: int = 0
    network_io_sent: int = 0
    network_io_recv: int = 0
    gpu_utilization: float = 0.0
    gpu_memory_used: int = 0
    thermal_state: Dict[str, float] = field(default_factory=dict)
    power_consumption: float = 0.0


@dataclass
class ResourceAllocation:
    """Resource allocation for a task/agent"""
    agent_id: str
    cpu_cores: List[int]
    memory_mb: int
    gpu_memory_mb: int = 0
    priority: int = 5
    estimated_duration: float = 1.0
    actual_usage: Optional[ResourceMetrics] = None


class ResourceAgent(BaseAgent):
    """AI agent specialized in resource monitoring and dynamic allocation"""
    
    def __init__(self, agent_id: str = "resource_agent", cpu_affinity: Optional[List[int]] = None):
        super().__init__(agent_id, priority=9, cpu_affinity=cpu_affinity)
        
        # Resource monitoring
        self.current_metrics = ResourceMetrics()
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.monitoring_interval = 0.1  # 100ms monitoring interval
        
        # Resource allocation tracking
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history = deque(maxlen=500)
        
        # System capabilities
        self.total_cpu_cores = psutil.cpu_count(logical=True)
        self.physical_cpu_cores = psutil.cpu_count(logical=False)
        self.total_memory = psutil.virtual_memory().total
        self.gpu_info = self._detect_gpu_resources()
        
        # Resource optimization
        self.load_balancing_enabled = True
        self.dynamic_scaling_enabled = True
        self.bottleneck_detection_enabled = True
        
        # Performance tracking
        self.resource_stats = {
            "allocations_made": 0,
            "reallocations": 0,
            "bottlenecks_detected": 0,
            "load_balance_operations": 0,
            "memory_optimizations": 0
        }
        
        # Prediction models for resource usage
        self.usage_predictors = {}
        self.workload_patterns = defaultdict(list)
        
        # Critical thresholds
        self.thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 95.0,
            "memory_warning": 80.0,
            "gpu_critical": 95.0,
            "gpu_warning": 80.0,
            "thermal_critical": 85.0,
            "thermal_warning": 75.0
        }
        
        self.logger.info(f"Resource agent initialized: {self.total_cpu_cores} cores, "
                        f"{self.total_memory // (1024**3)} GB RAM")
        if self.gpu_info:
            self.logger.info(f"GPU resources detected: {len(self.gpu_info)} devices")
    
    def _detect_gpu_resources(self) -> List[Dict[str, Any]]:
        """Detect available GPU resources"""
        gpu_devices = []
        
        try:
            # Try NVIDIA GPUs first
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_devices.append({
                    "index": i,
                    "name": name,
                    "total_memory": memory_info.total,
                    "type": "nvidia"
                })
                
        except ImportError:
            self.logger.debug("pynvml not available, skipping NVIDIA GPU detection")
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU detection failed: {e}")
        
        try:
            # Try AMD GPUs or other OpenCL devices
            import pyopencl as cl
            platforms = cl.get_platforms()
            
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                for device in devices:
                    gpu_devices.append({
                        "index": len(gpu_devices),
                        "name": device.name,
                        "total_memory": device.global_mem_size,
                        "type": "opencl"
                    })
                    
        except ImportError:
            self.logger.debug("pyopencl not available, skipping OpenCL GPU detection")
        except Exception as e:
            self.logger.debug(f"OpenCL GPU detection failed: {e}")
        
        return gpu_devices
    
    async def execute_cycle(self):
        """Main execution cycle for resource monitoring and management"""
        try:
            # Collect current resource metrics
            await self._collect_resource_metrics()
            
            # Detect bottlenecks and performance issues
            await self._detect_bottlenecks()
            
            # Optimize resource allocations
            await self._optimize_resource_allocations()
            
            # Perform load balancing if needed
            await self._perform_load_balancing()
            
            # Update predictive models
            await self._update_prediction_models()
            
            # Update agent metrics
            self.update_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in resource agent cycle: {e}")
            self.state = AgentState.ERROR
    
    async def _collect_resource_metrics(self):
        """Collect comprehensive system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            
            # Network I/O metrics
            network_io = psutil.net_io_counters()
            
            # GPU metrics
            gpu_utilization, gpu_memory_used = await self._get_gpu_metrics()
            
            # Thermal metrics
            thermal_state = await self._get_thermal_metrics()
            
            # Power consumption (if available)
            power_consumption = await self._get_power_metrics()
            
            # Create metrics object
            self.current_metrics = ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_io_sent=network_io.bytes_sent if network_io else 0,
                network_io_recv=network_io.bytes_recv if network_io else 0,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory_used,
                thermal_state=thermal_state,
                power_consumption=power_consumption
            )
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
    
    async def _get_gpu_metrics(self) -> Tuple[float, int]:
        """Get GPU utilization and memory usage"""
        try:
            if not self.gpu_info:
                return 0.0, 0
            
            total_utilization = 0.0
            total_memory_used = 0
            
            # NVIDIA GPUs
            try:
                import pynvml
                for gpu in self.gpu_info:
                    if gpu["type"] == "nvidia":
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu["index"])
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        total_utilization += util.gpu
                        total_memory_used += memory_info.used
                        
            except Exception as e:
                self.logger.debug(f"NVIDIA GPU metrics collection failed: {e}")
            
            # Average utilization across all GPUs
            if len(self.gpu_info) > 0:
                total_utilization /= len(self.gpu_info)
            
            return total_utilization, total_memory_used
            
        except Exception as e:
            self.logger.debug(f"GPU metrics collection failed: {e}")
            return 0.0, 0
    
    async def _get_thermal_metrics(self) -> Dict[str, float]:
        """Get system thermal metrics"""
        thermal_state = {}
        
        try:
            # Try to get CPU temperature
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                for name, entries in temps.items():
                    for entry in entries:
                        thermal_state[f"{name}_{entry.label or 'temp'}"] = entry.current
            
            # Try platform-specific thermal monitoring
            if platform.system() == "Linux":
                # Try reading from /sys/class/thermal
                try:
                    for i in range(10):  # Check first 10 thermal zones
                        temp_path = f"/sys/class/thermal/thermal_zone{i}/temp"
                        if os.path.exists(temp_path):
                            with open(temp_path, 'r') as f:
                                temp_millicelsius = int(f.read().strip())
                                thermal_state[f"thermal_zone_{i}"] = temp_millicelsius / 1000.0
                except Exception as e:
                    self.logger.debug(f"Linux thermal reading failed: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Thermal metrics collection failed: {e}")
        
        return thermal_state
    
    async def _get_power_metrics(self) -> float:
        """Get system power consumption metrics"""
        try:
            # This is highly system-dependent
            # On Linux, we might read from /sys/class/power_supply/
            # On Windows, we might use WMI
            # For now, return 0 as placeholder
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Power metrics collection failed: {e}")
            return 0.0
    
    async def _detect_bottlenecks(self):
        """Detect system bottlenecks and performance issues"""
        try:
            bottlenecks_detected = []
            
            # CPU bottleneck detection
            if self.current_metrics.cpu_percent > self.thresholds["cpu_critical"]:
                bottlenecks_detected.append({
                    "type": "cpu_critical",
                    "value": self.current_metrics.cpu_percent,
                    "threshold": self.thresholds["cpu_critical"]
                })
            
            # Memory bottleneck detection
            if self.current_metrics.memory_percent > self.thresholds["memory_critical"]:
                bottlenecks_detected.append({
                    "type": "memory_critical",
                    "value": self.current_metrics.memory_percent,
                    "threshold": self.thresholds["memory_critical"]
                })
            
            # GPU bottleneck detection
            if self.current_metrics.gpu_utilization > self.thresholds["gpu_critical"]:
                bottlenecks_detected.append({
                    "type": "gpu_critical",
                    "value": self.current_metrics.gpu_utilization,
                    "threshold": self.thresholds["gpu_critical"]
                })
            
            # Thermal bottleneck detection
            for sensor, temp in self.current_metrics.thermal_state.items():
                if temp > self.thresholds["thermal_critical"]:
                    bottlenecks_detected.append({
                        "type": "thermal_critical",
                        "sensor": sensor,
                        "value": temp,
                        "threshold": self.thresholds["thermal_critical"]
                    })
            
            # Core imbalance detection
            if len(self.current_metrics.cpu_per_core) > 1:
                core_std = np.std(self.current_metrics.cpu_per_core)
                core_mean = np.mean(self.current_metrics.cpu_per_core)
                
                if core_std > 30.0 and core_mean > 50.0:  # High variance and load
                    bottlenecks_detected.append({
                        "type": "cpu_imbalance",
                        "std_dev": core_std,
                        "mean_load": core_mean
                    })
            
            # Send alerts for detected bottlenecks
            if bottlenecks_detected:
                self.resource_stats["bottlenecks_detected"] += len(bottlenecks_detected)
                await self._send_bottleneck_alerts(bottlenecks_detected)
                
        except Exception as e:
            self.logger.error(f"Bottleneck detection failed: {e}")
    
    async def _send_bottleneck_alerts(self, bottlenecks: List[Dict[str, Any]]):
        """Send bottleneck alerts to other agents"""
        for bottleneck in bottlenecks:
            await self.broadcast_message(
                "bottleneck_detected",
                {
                    "bottleneck": bottleneck,
                    "timestamp": time.time(),
                    "system_metrics": {
                        "cpu_percent": self.current_metrics.cpu_percent,
                        "memory_percent": self.current_metrics.memory_percent,
                        "gpu_utilization": self.current_metrics.gpu_utilization
                    }
                },
                priority=MessagePriority.HIGH
            )
    
    async def _optimize_resource_allocations(self):
        """Optimize current resource allocations"""
        try:
            # Review active allocations
            for agent_id, allocation in list(self.active_allocations.items()):
                # Check if allocation is still valid
                if not await self._is_allocation_valid(allocation):
                    await self._reallocate_resources(agent_id, allocation)
            
            # Suggest new allocations based on pending requests
            await self._suggest_optimal_allocations()
            
        except Exception as e:
            self.logger.error(f"Resource allocation optimization failed: {e}")
    
    async def _is_allocation_valid(self, allocation: ResourceAllocation) -> bool:
        """Check if a resource allocation is still valid and efficient"""
        try:
            # Check if allocated cores are overloaded
            if allocation.cpu_cores:
                core_loads = [self.current_metrics.cpu_per_core[i] 
                             for i in allocation.cpu_cores 
                             if i < len(self.current_metrics.cpu_per_core)]
                
                if core_loads and max(core_loads) > self.thresholds["cpu_warning"]:
                    return False
            
            # Check memory pressure
            if self.current_metrics.memory_percent > self.thresholds["memory_warning"]:
                return False
            
            # Check thermal constraints
            for temp in self.current_metrics.thermal_state.values():
                if temp > self.thresholds["thermal_warning"]:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Allocation validation failed: {e}")
            return True  # Conservative approach
    
    async def _reallocate_resources(self, agent_id: str, current_allocation: ResourceAllocation):
        """Reallocate resources for an agent"""
        try:
            # Find better resource allocation
            new_allocation = await self._find_optimal_allocation(
                agent_id, 
                current_allocation.memory_mb,
                current_allocation.gpu_memory_mb,
                current_allocation.priority
            )
            
            if new_allocation and new_allocation.cpu_cores != current_allocation.cpu_cores:
                # Update allocation
                self.active_allocations[agent_id] = new_allocation
                self.resource_stats["reallocations"] += 1
                
                # Notify agent of reallocation
                await self.send_message(
                    agent_id,
                    "resource_reallocation",
                    {
                        "old_allocation": {
                            "cpu_cores": current_allocation.cpu_cores,
                            "memory_mb": current_allocation.memory_mb
                        },
                        "new_allocation": {
                            "cpu_cores": new_allocation.cpu_cores,
                            "memory_mb": new_allocation.memory_mb
                        },
                        "reason": "performance_optimization"
                    },
                    priority=MessagePriority.HIGH
                )
                
        except Exception as e:
            self.logger.error(f"Resource reallocation failed for {agent_id}: {e}")
    
    async def _find_optimal_allocation(self, agent_id: str, memory_mb: int, 
                                     gpu_memory_mb: int = 0, priority: int = 5) -> Optional[ResourceAllocation]:
        """Find optimal resource allocation for an agent"""
        try:
            # Find least loaded CPU cores
            available_cores = []
            core_loads = self.current_metrics.cpu_per_core
            
            # Sort cores by load (ascending)
            sorted_cores = sorted(enumerate(core_loads), key=lambda x: x[1])
            
            # Select cores based on priority and availability
            num_cores_needed = max(1, min(4, priority))  # 1-4 cores based on priority
            
            for i, (core_idx, load) in enumerate(sorted_cores):
                if load < self.thresholds["cpu_warning"] and len(available_cores) < num_cores_needed:
                    # Check if core is not already heavily allocated
                    if not self._is_core_allocated(core_idx):
                        available_cores.append(core_idx)
            
            if not available_cores:
                return None
            
            return ResourceAllocation(
                agent_id=agent_id,
                cpu_cores=available_cores,
                memory_mb=memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                priority=priority
            )
            
        except Exception as e:
            self.logger.error(f"Optimal allocation search failed: {e}")
            return None
    
    def _is_core_allocated(self, core_idx: int) -> bool:
        """Check if a CPU core is already heavily allocated"""
        allocated_count = 0
        for allocation in self.active_allocations.values():
            if core_idx in allocation.cpu_cores:
                allocated_count += 1
        
        # Allow up to 2 agents per core for hyperthreading
        return allocated_count >= 2
    
    async def _suggest_optimal_allocations(self):
        """Suggest optimal resource allocations based on system state"""
        # This would analyze pending requests and suggest allocations
        # For now, we'll just log current system state
        if len(self.metrics_history) > 10:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = np.mean([m.memory_percent for m in recent_metrics])
            
            self.logger.debug(f"System state: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%")
    
    async def _perform_load_balancing(self):
        """Perform dynamic load balancing across agents"""
        if not self.load_balancing_enabled:
            return
        
        try:
            # Check if load balancing is needed
            if self._should_perform_load_balancing():
                await self._execute_load_balancing()
                
        except Exception as e:
            self.logger.error(f"Load balancing failed: {e}")
    
    def _should_perform_load_balancing(self) -> bool:
        """Determine if load balancing is needed"""
        try:
            # Check CPU core imbalance
            if len(self.current_metrics.cpu_per_core) > 1:
                core_loads = self.current_metrics.cpu_per_core
                load_std = np.std(core_loads)
                load_mean = np.mean(core_loads)
                
                # Perform load balancing if there's high variance and high load
                if load_std > 25.0 and load_mean > 60.0:
                    return True
            
            # Check overall system load
            if self.current_metrics.cpu_percent > self.thresholds["cpu_warning"]:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Load balancing decision failed: {e}")
            return False
    
    async def _execute_load_balancing(self):
        """Execute load balancing operations"""
        try:
            self.resource_stats["load_balance_operations"] += 1
            
            # Identify overloaded and underloaded cores
            core_loads = self.current_metrics.cpu_per_core
            overloaded_cores = [i for i, load in enumerate(core_loads) 
                              if load > self.thresholds["cpu_warning"]]
            underloaded_cores = [i for i, load in enumerate(core_loads) 
                               if load < 50.0]
            
            if not overloaded_cores or not underloaded_cores:
                return
            
            # Move some allocations from overloaded to underloaded cores
            for overloaded_core in overloaded_cores:
                # Find allocations using this core
                agents_to_move = []
                for agent_id, allocation in self.active_allocations.items():
                    if overloaded_core in allocation.cpu_cores and len(allocation.cpu_cores) > 1:
                        agents_to_move.append(agent_id)
                
                # Move one agent to underloaded core
                if agents_to_move and underloaded_cores:
                    agent_to_move = agents_to_move[0]
                    target_core = underloaded_cores.pop(0)
                    
                    allocation = self.active_allocations[agent_to_move]
                    new_cores = [c for c in allocation.cpu_cores if c != overloaded_core]
                    new_cores.append(target_core)
                    
                    allocation.cpu_cores = new_cores
                    
                    # Notify agent of core reassignment
                    await self.send_message(
                        agent_to_move,
                        "cpu_core_reassignment",
                        {
                            "old_cores": allocation.cpu_cores,
                            "new_cores": new_cores,
                            "reason": "load_balancing"
                        },
                        priority=MessagePriority.NORMAL
                    )
                    
                    self.logger.info(f"Load balanced: moved {agent_to_move} from core {overloaded_core} to {target_core}")
                    
        except Exception as e:
            self.logger.error(f"Load balancing execution failed: {e}")
    
    async def _update_prediction_models(self):
        """Update predictive models for resource usage"""
        try:
            if len(self.metrics_history) < 10:
                return
            
            # Simple trend analysis for now
            recent_metrics = list(self.metrics_history)[-10:]
            
            # CPU trend
            cpu_values = [m.cpu_percent for m in recent_metrics]
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
            
            # Memory trend
            memory_values = [m.memory_percent for m in recent_metrics]
            memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
            # Store predictions
            self.usage_predictors["cpu_trend"] = cpu_trend
            self.usage_predictors["memory_trend"] = memory_trend
            
            # Predict potential issues
            if cpu_trend > 2.0:  # Rapidly increasing CPU usage
                await self.broadcast_message(
                    "resource_prediction",
                    {
                        "type": "cpu_usage_increasing",
                        "trend": cpu_trend,
                        "current_usage": self.current_metrics.cpu_percent,
                        "prediction": "high_cpu_load_expected"
                    },
                    priority=MessagePriority.NORMAL
                )
            
        except Exception as e:
            self.logger.error(f"Prediction model update failed: {e}")
    
    async def handle_message(self, message):
        """Handle resource management specific messages"""
        await super().handle_message(message)
        
        if message.message_type == "request_resources":
            await self._handle_resource_request(message)
        elif message.message_type == "release_resources":
            await self._handle_resource_release(message)
        elif message.message_type == "get_system_metrics":
            await self._handle_metrics_request(message)
    
    async def _handle_resource_request(self, message):
        """Handle resource allocation requests"""
        try:
            request_data = message.payload
            agent_id = message.sender_id
            
            memory_mb = request_data.get("memory_mb", 1024)
            gpu_memory_mb = request_data.get("gpu_memory_mb", 0)
            priority = request_data.get("priority", 5)
            
            # Find optimal allocation
            allocation = await self._find_optimal_allocation(agent_id, memory_mb, gpu_memory_mb, priority)
            
            if allocation:
                self.active_allocations[agent_id] = allocation
                self.resource_stats["allocations_made"] += 1
                
                # Send allocation response
                await self.send_message(
                    agent_id,
                    "resource_allocation",
                    {
                        "cpu_cores": allocation.cpu_cores,
                        "memory_mb": allocation.memory_mb,
                        "gpu_memory_mb": allocation.gpu_memory_mb,
                        "allocation_id": f"{agent_id}_{int(time.time())}"
                    },
                    priority=MessagePriority.HIGH
                )
            else:
                # No resources available
                await self.send_message(
                    agent_id,
                    "resource_allocation_failed",
                    {
                        "reason": "insufficient_resources",
                        "current_load": {
                            "cpu_percent": self.current_metrics.cpu_percent,
                            "memory_percent": self.current_metrics.memory_percent
                        }
                    },
                    priority=MessagePriority.HIGH
                )
                
        except Exception as e:
            self.logger.error(f"Resource request handling failed: {e}")
    
    async def _handle_resource_release(self, message):
        """Handle resource release notifications"""
        try:
            agent_id = message.sender_id
            
            if agent_id in self.active_allocations:
                del self.active_allocations[agent_id]
                self.logger.info(f"Released resources for agent {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Resource release handling failed: {e}")
    
    async def _handle_metrics_request(self, message):
        """Handle system metrics requests"""
        try:
            await self.send_message(
                message.sender_id,
                "system_metrics_response",
                {
                    "current_metrics": {
                        "cpu_percent": self.current_metrics.cpu_percent,
                        "cpu_per_core": self.current_metrics.cpu_per_core,
                        "memory_percent": self.current_metrics.memory_percent,
                        "memory_available": self.current_metrics.memory_available,
                        "gpu_utilization": self.current_metrics.gpu_utilization,
                        "thermal_state": self.current_metrics.thermal_state
                    },
                    "system_capabilities": {
                        "total_cpu_cores": self.total_cpu_cores,
                        "physical_cpu_cores": self.physical_cpu_cores,
                        "total_memory": self.total_memory,
                        "gpu_devices": len(self.gpu_info)
                    },
                    "active_allocations": len(self.active_allocations)
                },
                priority=MessagePriority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Metrics request handling failed: {e}")
    
    async def _agent_specific_optimization(self):
        """Resource agent specific optimizations"""
        # Clean up old metrics history if it gets too large
        if len(self.metrics_history) > 1000:
            # Keep only recent entries
            self.metrics_history = deque(list(self.metrics_history)[-500:], maxlen=1000)
        
        # Clean up old allocation history
        if len(self.allocation_history) > 500:
            self.allocation_history = deque(list(self.allocation_history)[-250:], maxlen=500)
        
        # Optimize prediction models
        if len(self.usage_predictors) > 20:
            # Keep only recent predictions
            recent_keys = list(self.usage_predictors.keys())[-10:]
            self.usage_predictors = {k: self.usage_predictors[k] for k in recent_keys}
        
        self.logger.debug(f"Resource agent optimization complete. "
                         f"Metrics history: {len(self.metrics_history)}, "
                         f"Active allocations: {len(self.active_allocations)}")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get resource agent specific statistics"""
        return {
            "resource_stats": self.resource_stats.copy(),
            "current_metrics": {
                "cpu_percent": self.current_metrics.cpu_percent,
                "memory_percent": self.current_metrics.memory_percent,
                "gpu_utilization": self.current_metrics.gpu_utilization,
                "thermal_max": max(self.current_metrics.thermal_state.values()) if self.current_metrics.thermal_state else 0.0
            },
            "system_capabilities": {
                "total_cpu_cores": self.total_cpu_cores,
                "physical_cpu_cores": self.physical_cpu_cores,
                "total_memory_gb": self.total_memory // (1024**3),
                "gpu_devices": len(self.gpu_info)
            },
            "active_allocations": len(self.active_allocations),
            "metrics_history_size": len(self.metrics_history),
            "prediction_models": len(self.usage_predictors),
            "thresholds": self.thresholds.copy()
        }