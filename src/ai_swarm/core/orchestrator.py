"""
Swarm Orchestrator
Coordinates all AI agents and manages the overall distributed system
"""

import asyncio
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import uuid

from .agent_base import BaseAgent, AgentState, AgentMetrics
from .communication import SwarmCommunicator, MessagePriority
from ..agents.compute_agent import ComputeAgent
from ..agents.resource_agent import ResourceAgent


@dataclass
class SwarmConfiguration:
    """Configuration for the AI agent swarm"""
    max_agents: int = 32
    communication_channel_size: int = 1024 * 1024  # 1MB
    monitoring_interval: float = 1.0
    optimization_interval: float = 5.0
    fault_tolerance_enabled: bool = True
    load_balancing_enabled: bool = True
    auto_scaling_enabled: bool = True
    telemetry_retention_hours: int = 24


@dataclass
class SwarmMetrics:
    """Overall swarm performance metrics"""
    timestamp: float = field(default_factory=time.time)
    total_agents: int = 0
    active_agents: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_response_time: float = 0.0
    system_throughput: float = 0.0
    resource_utilization: float = 0.0
    communication_latency: float = 0.0
    fault_recovery_time: float = 0.0


class SwarmOrchestrator:
    """Main orchestrator for the AI agent swarm system"""
    
    def __init__(self, config: Optional[SwarmConfiguration] = None):
        self.config = config or SwarmConfiguration()
        
        # Core components
        self.communicator = SwarmCommunicator(
            max_agents=self.config.max_agents,
            channel_size=self.config.communication_channel_size
        )
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        # System state
        self.swarm_metrics = SwarmMetrics()
        self.metrics_history = deque(maxlen=1000)
        self.system_state = "initializing"
        
        # Performance monitoring
        self.performance_monitor = None
        self.optimization_scheduler = None
        self.fault_detector = None
        
        # Event handling
        self.event_handlers: Dict[str, List[callable]] = defaultdict(list)
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.orchestrator_stats = {
            "agents_created": 0,
            "agents_destroyed": 0,
            "failovers_executed": 0,
            "optimizations_performed": 0,
            "load_balancing_operations": 0
        }
        
        # Logging
        self.logger = logging.getLogger("SwarmOrchestrator")
        self.logger.setLevel(logging.INFO)
        
        # Initialize default agent types
        self._register_default_agent_types()
        
        self.logger.info("Swarm orchestrator initialized")
    
    def _register_default_agent_types(self):
        """Register default agent types"""
        self.agent_types.update({
            "compute": ComputeAgent,
            "resource": ResourceAgent,
            # Additional agent types will be registered as they're created
        })
    
    async def start(self):
        """Start the swarm orchestrator and all subsystems"""
        try:
            self.logger.info("Starting AI agent swarm...")
            self.system_state = "starting"
            
            # Start core subsystems
            await self._start_core_subsystems()
            
            # Create initial agent configuration
            await self._create_initial_agents()
            
            # Start monitoring and optimization loops
            await self._start_monitoring_loops()
            
            self.system_state = "active"
            self.logger.info("AI agent swarm started successfully")
            
            # Main orchestration loop
            await self._main_orchestration_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start swarm orchestrator: {e}")
            self.system_state = "error"
            raise
    
    async def _start_core_subsystems(self):
        """Start core subsystems"""
        # Communication system is already initialized
        # Additional subsystems can be started here
        pass
    
    async def _create_initial_agents(self):
        """Create the initial set of agents"""
        try:
            # Create resource monitoring agent (highest priority)
            await self.create_agent("resource", "resource_monitor", priority=10, cpu_affinity=[0])
            
            # Create compute optimization agents
            num_compute_agents = min(4, max(1, self.config.max_agents // 8))
            for i in range(num_compute_agents):
                agent_id = f"compute_agent_{i}"
                cpu_cores = [i + 1] if i + 1 < 16 else None  # Distribute across cores
                await self.create_agent("compute", agent_id, priority=8, cpu_affinity=cpu_cores)
            
            self.logger.info(f"Created {len(self.agents)} initial agents")
            
        except Exception as e:
            self.logger.error(f"Failed to create initial agents: {e}")
            raise
    
    async def create_agent(self, agent_type: str, agent_id: str, 
                          priority: int = 5, cpu_affinity: Optional[List[int]] = None,
                          **kwargs) -> bool:
        """Create and register a new agent"""
        try:
            if agent_type not in self.agent_types:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return False
            
            if agent_id in self.agents:
                self.logger.warning(f"Agent {agent_id} already exists")
                return False
            
            # Create agent instance
            agent_class = self.agent_types[agent_type]
            agent = agent_class(agent_id=agent_id, cpu_affinity=cpu_affinity, **kwargs)
            agent.priority = priority
            
            # Register with communicator
            self.communicator.register_agent(agent_id, agent)
            agent.set_communicator(self.communicator)
            
            # Store agent
            self.agents[agent_id] = agent
            
            # Start agent in its own task
            agent_task = asyncio.create_task(agent.start())
            self.agent_tasks[agent_id] = agent_task
            
            self.orchestrator_stats["agents_created"] += 1
            self.logger.info(f"Created {agent_type} agent: {agent_id}")
            
            # Trigger agent creation event
            await self._trigger_event("agent_created", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "priority": priority
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {e}")
            return False
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent and clean up resources"""
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Agent {agent_id} not found")
                return False
            
            agent = self.agents[agent_id]
            
            # Shutdown agent
            await agent.shutdown()
            
            # Cancel agent task
            if agent_id in self.agent_tasks:
                task = self.agent_tasks[agent_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.agent_tasks[agent_id]
            
            # Unregister from communicator
            self.communicator.unregister_agent(agent_id)
            
            # Remove from registry
            del self.agents[agent_id]
            
            self.orchestrator_stats["agents_destroyed"] += 1
            self.logger.info(f"Destroyed agent: {agent_id}")
            
            # Trigger agent destruction event
            await self._trigger_event("agent_destroyed", {
                "agent_id": agent_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent {agent_id}: {e}")
            return False
    
    async def _start_monitoring_loops(self):
        """Start monitoring and optimization loops"""
        try:
            # Performance monitoring loop
            self.performance_monitor = asyncio.create_task(self._performance_monitoring_loop())
            
            # Optimization scheduler loop
            self.optimization_scheduler = asyncio.create_task(self._optimization_loop())
            
            # Fault detection loop
            self.fault_detector = asyncio.create_task(self._fault_detection_loop())
            
            self.logger.info("Started monitoring and optimization loops")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring loops: {e}")
            raise
    
    async def _main_orchestration_loop(self):
        """Main orchestration loop"""
        while not self.shutdown_event.is_set():
            try:
                # Update swarm metrics
                await self._update_swarm_metrics()
                
                # Handle system events
                await self._process_system_events()
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Auto-scaling if enabled
                if self.config.auto_scaling_enabled:
                    await self._perform_auto_scaling()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main orchestration loop: {e}")
                await asyncio.sleep(5.0)  # Recovery delay
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics from all agents
                agent_metrics = {}
                for agent_id, agent in self.agents.items():
                    try:
                        if hasattr(agent, 'get_agent_statistics'):
                            agent_metrics[agent_id] = agent.get_agent_statistics()
                        else:
                            agent_metrics[agent_id] = {
                                "state": agent.state.value,
                                "tasks_completed": agent.metrics.tasks_completed,
                                "tasks_failed": agent.metrics.tasks_failed,
                                "cpu_usage": agent.metrics.cpu_usage,
                                "memory_usage": agent.metrics.memory_usage
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to collect metrics from {agent_id}: {e}")
                
                # Store metrics
                await self._store_performance_metrics(agent_metrics)
                
                # Analyze performance trends
                await self._analyze_performance_trends()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _optimization_loop(self):
        """System optimization loop"""
        while not self.shutdown_event.is_set():
            try:
                # Trigger optimization across all agents
                await self._trigger_system_optimization()
                
                # Optimize communication patterns
                await self._optimize_communication()
                
                # Optimize resource allocation
                await self._optimize_resource_allocation()
                
                self.orchestrator_stats["optimizations_performed"] += 1
                
                await asyncio.sleep(self.config.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _fault_detection_loop(self):
        """Fault detection and recovery loop"""
        while not self.shutdown_event.is_set():
            try:
                # Check agent health
                failed_agents = await self._detect_failed_agents()
                
                # Handle failures
                for agent_id in failed_agents:
                    await self._handle_agent_failure(agent_id)
                
                # Check system-wide issues
                await self._detect_system_issues()
                
                await asyncio.sleep(2.0)  # More frequent fault detection
                
            except Exception as e:
                self.logger.error(f"Error in fault detection loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_swarm_metrics(self):
        """Update overall swarm performance metrics"""
        try:
            total_agents = len(self.agents)
            active_agents = sum(1 for agent in self.agents.values() 
                              if agent.state == AgentState.ACTIVE)
            
            total_tasks_completed = sum(agent.metrics.tasks_completed 
                                      for agent in self.agents.values())
            total_tasks_failed = sum(agent.metrics.tasks_failed 
                                   for agent in self.agents.values())
            
            # Calculate average response time
            response_times = [agent.metrics.avg_response_time 
                            for agent in self.agents.values() 
                            if agent.metrics.avg_response_time > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Calculate system throughput (tasks per second)
            if len(self.metrics_history) > 0:
                prev_metrics = self.metrics_history[-1]
                time_delta = time.time() - prev_metrics.timestamp
                task_delta = total_tasks_completed - prev_metrics.total_tasks_completed
                system_throughput = task_delta / time_delta if time_delta > 0 else 0.0
            else:
                system_throughput = 0.0
            
            # Get communication statistics
            comm_stats = self.communicator.get_statistics()
            communication_latency = comm_stats["message_stats"]["avg_latency"]
            
            # Update metrics
            self.swarm_metrics = SwarmMetrics(
                timestamp=time.time(),
                total_agents=total_agents,
                active_agents=active_agents,
                total_tasks_completed=total_tasks_completed,
                total_tasks_failed=total_tasks_failed,
                average_response_time=avg_response_time,
                system_throughput=system_throughput,
                communication_latency=communication_latency
            )
            
            # Store in history
            self.metrics_history.append(self.swarm_metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to update swarm metrics: {e}")
    
    async def _process_system_events(self):
        """Process system-wide events"""
        # This would handle various system events like:
        # - Resource allocation requests
        # - Agent coordination requests
        # - Performance alerts
        # - Configuration changes
        pass
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents"""
        try:
            unhealthy_agents = []
            
            for agent_id, agent in self.agents.items():
                # Check if agent is responsive
                last_heartbeat = agent.metrics.last_heartbeat
                if time.time() - last_heartbeat > 30.0:  # 30 seconds timeout
                    unhealthy_agents.append(agent_id)
                    continue
                
                # Check agent state
                if agent.state == AgentState.ERROR:
                    unhealthy_agents.append(agent_id)
                    continue
                
                # Check task completion rate
                if (agent.metrics.tasks_failed > 0 and 
                    agent.metrics.tasks_completed > 0):
                    failure_rate = (agent.metrics.tasks_failed / 
                                  (agent.metrics.tasks_completed + agent.metrics.tasks_failed))
                    if failure_rate > 0.5:  # 50% failure rate
                        unhealthy_agents.append(agent_id)
            
            # Handle unhealthy agents
            for agent_id in unhealthy_agents:
                await self._handle_unhealthy_agent(agent_id)
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    async def _perform_auto_scaling(self):
        """Perform automatic scaling based on system load"""
        try:
            # Simple auto-scaling logic
            if self.swarm_metrics.active_agents == 0:
                return
            
            # Calculate system load
            avg_cpu_usage = sum(agent.metrics.cpu_usage for agent in self.agents.values()) / len(self.agents)
            
            # Scale up if high load and few agents
            if (avg_cpu_usage > 80.0 and 
                self.swarm_metrics.active_agents < self.config.max_agents // 2):
                
                await self._scale_up()
                
            # Scale down if low load and many agents
            elif (avg_cpu_usage < 20.0 and 
                  self.swarm_metrics.active_agents > 2):  # Keep minimum agents
                
                await self._scale_down()
                
        except Exception as e:
            self.logger.error(f"Auto-scaling failed: {e}")
    
    async def _scale_up(self):
        """Scale up the system by adding more agents"""
        try:
            # Add a compute agent
            agent_id = f"compute_agent_{uuid.uuid4().hex[:8]}"
            success = await self.create_agent("compute", agent_id, priority=7)
            
            if success:
                self.logger.info(f"Scaled up: added {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Scale up failed: {e}")
    
    async def _scale_down(self):
        """Scale down the system by removing agents"""
        try:
            # Find least utilized compute agent
            compute_agents = {aid: agent for aid, agent in self.agents.items() 
                            if aid.startswith("compute_agent_")}
            
            if len(compute_agents) > 1:  # Keep at least one
                least_utilized = min(compute_agents.items(), 
                                   key=lambda x: x[1].metrics.tasks_completed)
                
                agent_id = least_utilized[0]
                success = await self.destroy_agent(agent_id)
                
                if success:
                    self.logger.info(f"Scaled down: removed {agent_id}")
                    
        except Exception as e:
            self.logger.error(f"Scale down failed: {e}")
    
    async def _detect_failed_agents(self) -> List[str]:
        """Detect failed agents"""
        failed_agents = []
        
        for agent_id, task in self.agent_tasks.items():
            if task.done():
                try:
                    # Check if task completed with exception
                    task.result()
                except Exception as e:
                    self.logger.error(f"Agent {agent_id} failed: {e}")
                    failed_agents.append(agent_id)
        
        return failed_agents
    
    async def _handle_agent_failure(self, agent_id: str):
        """Handle agent failure with recovery"""
        try:
            self.logger.warning(f"Handling failure of agent {agent_id}")
            
            # Get agent type for recreation
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent_type = type(agent).__name__.lower().replace("agent", "")
                priority = agent.priority
                cpu_affinity = agent.cpu_affinity
                
                # Destroy failed agent
                await self.destroy_agent(agent_id)
                
                # Recreate agent if fault tolerance is enabled
                if self.config.fault_tolerance_enabled:
                    new_agent_id = f"{agent_id}_recovery_{int(time.time())}"
                    success = await self.create_agent(
                        agent_type, new_agent_id, 
                        priority=priority, 
                        cpu_affinity=cpu_affinity
                    )
                    
                    if success:
                        self.orchestrator_stats["failovers_executed"] += 1
                        self.logger.info(f"Successfully recovered agent as {new_agent_id}")
                    else:
                        self.logger.error(f"Failed to recover agent {agent_id}")
                        
        except Exception as e:
            self.logger.error(f"Agent failure handling failed for {agent_id}: {e}")
    
    async def _handle_unhealthy_agent(self, agent_id: str):
        """Handle unhealthy agent"""
        try:
            self.logger.warning(f"Agent {agent_id} is unhealthy, attempting recovery")
            
            # Try to restart the agent
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Send optimization message to try recovery
                await self.communicator.send_message({
                    "sender_id": "orchestrator",
                    "recipient_id": agent_id,
                    "message_type": "optimize",
                    "payload": {"recovery_attempt": True},
                    "priority": MessagePriority.HIGH
                })
                
                # If still unhealthy after some time, treat as failure
                # This would be implemented with a delayed check
                
        except Exception as e:
            self.logger.error(f"Unhealthy agent handling failed for {agent_id}: {e}")
    
    async def _detect_system_issues(self):
        """Detect system-wide issues"""
        try:
            # Check communication system health
            comm_stats = self.communicator.get_statistics()
            if comm_stats["message_stats"]["dropped"] > 100:  # Too many dropped messages
                self.logger.warning("High message drop rate detected")
            
            # Check overall system performance
            if (self.swarm_metrics.active_agents > 0 and 
                self.swarm_metrics.system_throughput < 0.1):  # Very low throughput
                self.logger.warning("System throughput is critically low")
            
        except Exception as e:
            self.logger.error(f"System issue detection failed: {e}")
    
    async def _trigger_system_optimization(self):
        """Trigger optimization across all agents"""
        try:
            optimization_message = {
                "type": "system_optimization",
                "timestamp": time.time(),
                "swarm_metrics": {
                    "total_agents": self.swarm_metrics.total_agents,
                    "system_throughput": self.swarm_metrics.system_throughput,
                    "avg_response_time": self.swarm_metrics.average_response_time
                }
            }
            
            await self.communicator.broadcast(optimization_message, MessagePriority.NORMAL)
            
        except Exception as e:
            self.logger.error(f"System optimization trigger failed: {e}")
    
    async def _optimize_communication(self):
        """Optimize communication patterns"""
        # This would analyze message patterns and optimize routing
        # For now, we'll just log current communication statistics
        comm_stats = self.communicator.get_statistics()
        self.logger.debug(f"Communication stats: {comm_stats}")
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation across agents"""
        # This would work with the resource agent to optimize allocations
        if "resource_monitor" in self.agents:
            try:
                await self.communicator.send_message({
                    "sender_id": "orchestrator",
                    "recipient_id": "resource_monitor",
                    "message_type": "optimize_allocations",
                    "payload": {
                        "system_load": self.swarm_metrics.resource_utilization,
                        "agent_count": self.swarm_metrics.active_agents
                    },
                    "priority": MessagePriority.NORMAL
                })
            except Exception as e:
                self.logger.error(f"Resource allocation optimization failed: {e}")
    
    async def _store_performance_metrics(self, agent_metrics: Dict[str, Any]):
        """Store performance metrics for analysis"""
        # This would store metrics in a database or file system
        # For now, we'll just keep them in memory
        pass
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and patterns"""
        if len(self.metrics_history) < 10:
            return
        
        try:
            # Simple trend analysis
            recent_metrics = list(self.metrics_history)[-10:]
            
            # Throughput trend
            throughputs = [m.system_throughput for m in recent_metrics]
            if len(throughputs) > 1:
                throughput_trend = (throughputs[-1] - throughputs[0]) / len(throughputs)
                
                if throughput_trend < -0.1:  # Declining throughput
                    self.logger.warning("System throughput is declining")
                    
        except Exception as e:
            self.logger.error(f"Performance trend analysis failed: {e}")
    
    def add_event_handler(self, event_type: str, handler: callable):
        """Add event handler"""
        self.event_handlers[event_type].append(handler)
    
    async def _trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger event handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event_type}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_state": self.system_state,
            "swarm_metrics": {
                "total_agents": self.swarm_metrics.total_agents,
                "active_agents": self.swarm_metrics.active_agents,
                "total_tasks_completed": self.swarm_metrics.total_tasks_completed,
                "total_tasks_failed": self.swarm_metrics.total_tasks_failed,
                "system_throughput": self.swarm_metrics.system_throughput,
                "average_response_time": self.swarm_metrics.average_response_time,
                "communication_latency": self.swarm_metrics.communication_latency
            },
            "orchestrator_stats": self.orchestrator_stats.copy(),
            "communication_stats": self.communicator.get_statistics(),
            "agent_states": {
                agent_id: agent.state.value 
                for agent_id, agent in self.agents.items()
            },
            "configuration": {
                "max_agents": self.config.max_agents,
                "fault_tolerance_enabled": self.config.fault_tolerance_enabled,
                "auto_scaling_enabled": self.config.auto_scaling_enabled,
                "load_balancing_enabled": self.config.load_balancing_enabled
            }
        }
    
    async def shutdown(self):
        """Shutdown the entire swarm system"""
        try:
            self.logger.info("Shutting down AI agent swarm...")
            self.system_state = "shutting_down"
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop all agents
            shutdown_tasks = []
            for agent_id in list(self.agents.keys()):
                task = asyncio.create_task(self.destroy_agent(agent_id))
                shutdown_tasks.append(task)
            
            # Wait for all agents to shutdown
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Stop monitoring loops
            for task in [self.performance_monitor, self.optimization_scheduler, self.fault_detector]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown communication system
            await self.communicator.shutdown()
            
            self.system_state = "shutdown"
            self.logger.info("AI agent swarm shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            raise