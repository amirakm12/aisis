"""
Base Agent Class
Foundation for all specialized AI agents in the swarm
"""

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import multiprocessing as mp
import psutil
import logging


class AgentState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMetrics:
    """Performance metrics for agent monitoring"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    

@dataclass
class AgentMessage:
    """Inter-agent communication message"""
    sender_id: str
    recipient_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more priority


class BaseAgent(ABC):
    """Base class for all AI agents in the swarm"""
    
    def __init__(self, agent_id: str, priority: int = 0, cpu_affinity: Optional[List[int]] = None):
        self.agent_id = agent_id
        self.priority = priority
        self.cpu_affinity = cpu_affinity or []
        self.state = AgentState.INITIALIZING
        self.metrics = AgentMetrics()
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.communicator = None
        
        # Threading and process management
        self.process = None
        self.thread = None
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.performance_history = []
        self.last_optimization = time.time()
        
        # Learning and adaptation
        self.learning_data = {}
        self.adaptation_callbacks = []
        
        # Logging
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        self.logger.setLevel(logging.INFO)
        
    def set_communicator(self, communicator):
        """Set the swarm communicator"""
        self.communicator = communicator
        
    def set_cpu_affinity(self):
        """Pin agent to specific CPU cores for optimal performance"""
        if self.cpu_affinity:
            try:
                p = psutil.Process()
                p.cpu_affinity(self.cpu_affinity)
                self.logger.info(f"Agent {self.agent_id} pinned to CPUs: {self.cpu_affinity}")
            except Exception as e:
                self.logger.warning(f"Failed to set CPU affinity: {e}")
    
    async def start(self):
        """Start the agent in its own process/thread"""
        self.state = AgentState.ACTIVE
        self.set_cpu_affinity()
        
        # Start main agent loop
        await asyncio.gather(
            self._main_loop(),
            self._heartbeat_loop(),
            self._message_handler_loop()
        )
    
    async def _main_loop(self):
        """Main agent execution loop"""
        while not self.shutdown_event.is_set():
            try:
                if self.state == AgentState.ACTIVE:
                    await self.execute_cycle()
                    await asyncio.sleep(0.001)  # Ultra-low latency
                elif self.state == AgentState.IDLE:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.state = AgentState.ERROR
                await asyncio.sleep(0.1)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat signals"""
        while not self.shutdown_event.is_set():
            self.metrics.last_heartbeat = time.time()
            if self.communicator:
                await self.communicator.broadcast({
                    "type": "heartbeat",
                    "agent_id": self.agent_id,
                    "state": self.state.value,
                    "metrics": self.metrics.__dict__
                })
            await asyncio.sleep(1.0)
    
    async def _message_handler_loop(self):
        """Handle incoming messages"""
        while not self.shutdown_event.is_set():
            try:
                # Check for messages with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=0.1
                    )
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
    
    @abstractmethod
    async def execute_cycle(self):
        """Execute one cycle of agent-specific logic"""
        pass
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming inter-agent messages"""
        self.logger.debug(f"Received message: {message.message_type} from {message.sender_id}")
        
        if message.message_type == "shutdown":
            await self.shutdown()
        elif message.message_type == "optimize":
            await self.optimize_performance()
        elif message.message_type == "status_request":
            await self.send_status_response(message.sender_id)
    
    async def send_message(self, recipient_id: str, message_type: str, payload: Dict[str, Any], priority: int = 0):
        """Send message to another agent"""
        if self.communicator:
            message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
                priority=priority
            )
            await self.communicator.send_message(message)
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any], priority: int = 0):
        """Broadcast message to all agents"""
        if self.communicator:
            message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id="*",
                message_type=message_type,
                payload=payload,
                priority=priority
            )
            await self.communicator.broadcast_message(message)
    
    async def send_status_response(self, requester_id: str):
        """Send status information to requesting agent"""
        status = {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "metrics": self.metrics.__dict__,
            "performance_history": self.performance_history[-10:],  # Last 10 entries
            "learning_data": self.learning_data
        }
        await self.send_message(requester_id, "status_response", status)
    
    async def optimize_performance(self):
        """Optimize agent performance based on current metrics"""
        current_time = time.time()
        
        # Only optimize if enough time has passed
        if current_time - self.last_optimization < 5.0:
            return
            
        self.last_optimization = current_time
        
        # Record performance metrics
        self.performance_history.append({
            "timestamp": current_time,
            "cpu_usage": self.metrics.cpu_usage,
            "memory_usage": self.metrics.memory_usage,
            "tasks_completed": self.metrics.tasks_completed,
            "avg_response_time": self.metrics.avg_response_time
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Agent-specific optimization
        await self._agent_specific_optimization()
    
    @abstractmethod
    async def _agent_specific_optimization(self):
        """Agent-specific performance optimization logic"""
        pass
    
    def update_metrics(self):
        """Update agent performance metrics"""
        try:
            process = psutil.Process()
            self.metrics.cpu_usage = process.cpu_percent()
            self.metrics.memory_usage = process.memory_percent()
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {e}")
    
    def add_adaptation_callback(self, callback: Callable):
        """Add callback for adaptive behavior"""
        self.adaptation_callbacks.append(callback)
    
    async def adapt_behavior(self, context: Dict[str, Any]):
        """Adapt agent behavior based on context"""
        for callback in self.adaptation_callbacks:
            try:
                await callback(self, context)
            except Exception as e:
                self.logger.error(f"Adaptation callback failed: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.state = AgentState.SHUTDOWN
        self.shutdown_event.set()
        
        # Clean up resources
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, state={self.state.value})>"