"""
Ultra-Low-Latency Inter-Agent Communication System
Implements shared memory channels, priority queues, and message routing
"""

import asyncio
import mmap
import struct
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import multiprocessing as mp
from multiprocessing import shared_memory
import pickle
import logging
import uuid

from .agent_base import AgentMessage


class MessagePriority:
    CRITICAL = 10
    HIGH = 7
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


@dataclass
class SharedMemoryChannel:
    """Shared memory channel for ultra-fast communication"""
    name: str
    size: int
    read_offset: int = 0
    write_offset: int = 0
    lock: Optional[threading.Lock] = None


class SwarmCommunicator:
    """High-performance communication system for AI agent swarm"""
    
    def __init__(self, max_agents: int = 32, channel_size: int = 1024 * 1024):  # 1MB per channel
        self.max_agents = max_agents
        self.channel_size = channel_size
        
        # Agent registry
        self.agents: Dict[str, Any] = {}
        self.agent_queues: Dict[str, asyncio.Queue] = {}
        
        # Shared memory channels
        self.shared_channels: Dict[str, SharedMemoryChannel] = {}
        self.memory_pools: Dict[str, shared_memory.SharedMemory] = {}
        
        # Message routing
        self.message_routers: Dict[str, Callable] = {}
        self.broadcast_subscribers: List[str] = []
        
        # Performance optimization
        self.priority_queues: Dict[str, deque] = defaultdict(deque)
        self.message_cache: Dict[str, Any] = {}
        self.routing_table: Dict[str, str] = {}
        
        # Statistics
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "dropped": 0,
            "avg_latency": 0.0
        }
        
        # Threading
        self.router_thread = None
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger("SwarmCommunicator")
        self.logger.setLevel(logging.INFO)
        
        # Initialize communication infrastructure
        self._initialize_infrastructure()
    
    def _initialize_infrastructure(self):
        """Initialize shared memory and communication infrastructure"""
        try:
            # Create shared memory pools for each priority level
            for priority in ["critical", "high", "normal", "low", "background"]:
                pool_name = f"swarm_pool_{priority}_{uuid.uuid4().hex[:8]}"
                try:
                    shm = shared_memory.SharedMemory(
                        create=True, 
                        size=self.channel_size,
                        name=pool_name
                    )
                    self.memory_pools[priority] = shm
                    self.logger.info(f"Created shared memory pool: {pool_name}")
                except Exception as e:
                    self.logger.error(f"Failed to create shared memory pool {priority}: {e}")
            
            # Start background threads
            self.router_thread = threading.Thread(target=self._message_router_loop, daemon=True)
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            
            self.router_thread.start()
            self.cleanup_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication infrastructure: {e}")
    
    def register_agent(self, agent_id: str, agent_ref: Any):
        """Register an agent with the communication system"""
        self.agents[agent_id] = agent_ref
        self.agent_queues[agent_id] = asyncio.Queue(maxsize=1000)  # High-capacity queue
        self.broadcast_subscribers.append(agent_id)
        
        # Create dedicated shared memory channel for high-priority messages
        channel_name = f"channel_{agent_id}_{uuid.uuid4().hex[:8]}"
        try:
            shm = shared_memory.SharedMemory(
                create=True,
                size=self.channel_size // 4,  # Smaller dedicated channels
                name=channel_name
            )
            
            channel = SharedMemoryChannel(
                name=channel_name,
                size=self.channel_size // 4,
                lock=threading.Lock()
            )
            
            self.shared_channels[agent_id] = channel
            self.memory_pools[f"agent_{agent_id}"] = shm
            
            self.logger.info(f"Registered agent {agent_id} with dedicated channel")
            
        except Exception as e:
            self.logger.error(f"Failed to create dedicated channel for {agent_id}: {e}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent and clean up resources"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
        if agent_id in self.agent_queues:
            del self.agent_queues[agent_id]
            
        if agent_id in self.broadcast_subscribers:
            self.broadcast_subscribers.remove(agent_id)
            
        # Clean up shared memory
        if agent_id in self.shared_channels:
            del self.shared_channels[agent_id]
            
        pool_key = f"agent_{agent_id}"
        if pool_key in self.memory_pools:
            try:
                self.memory_pools[pool_key].close()
                self.memory_pools[pool_key].unlink()
                del self.memory_pools[pool_key]
            except Exception as e:
                self.logger.warning(f"Failed to clean up memory pool for {agent_id}: {e}")
        
        self.logger.info(f"Unregistered agent {agent_id}")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to specific agent with ultra-low latency"""
        start_time = time.perf_counter()
        
        try:
            recipient_id = message.recipient_id
            
            # Check if recipient exists
            if recipient_id not in self.agents:
                self.logger.warning(f"Recipient {recipient_id} not found")
                self.message_stats["dropped"] += 1
                return False
            
            # Route message based on priority and type
            if message.priority >= MessagePriority.CRITICAL:
                success = await self._send_via_shared_memory(message)
            else:
                success = await self._send_via_queue(message)
            
            # Update statistics
            if success:
                self.message_stats["sent"] += 1
                latency = time.perf_counter() - start_time
                self._update_latency_stats(latency)
            else:
                self.message_stats["dropped"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            self.message_stats["dropped"] += 1
            return False
    
    async def _send_via_shared_memory(self, message: AgentMessage) -> bool:
        """Send high-priority message via shared memory"""
        try:
            recipient_id = message.recipient_id
            
            if recipient_id not in self.shared_channels:
                # Fallback to queue
                return await self._send_via_queue(message)
            
            channel = self.shared_channels[recipient_id]
            pool_key = f"agent_{recipient_id}"
            
            if pool_key not in self.memory_pools:
                return await self._send_via_queue(message)
            
            # Serialize message
            serialized = pickle.dumps(message)
            message_size = len(serialized)
            
            # Check if message fits in channel
            if message_size > channel.size - 16:  # Reserve space for metadata
                self.logger.warning(f"Message too large for shared memory: {message_size} bytes")
                return await self._send_via_queue(message)
            
            # Write to shared memory with lock
            with channel.lock:
                shm = self.memory_pools[pool_key]
                
                # Write message size and data
                struct.pack_into('I', shm.buf, channel.write_offset, message_size)
                shm.buf[channel.write_offset + 4:channel.write_offset + 4 + message_size] = serialized
                
                # Update write offset (circular buffer)
                channel.write_offset = (channel.write_offset + message_size + 4) % channel.size
            
            # Notify recipient (if possible)
            if recipient_id in self.agent_queues:
                notification = AgentMessage(
                    sender_id="communicator",
                    recipient_id=recipient_id,
                    message_type="shared_memory_notification",
                    payload={"channel": channel.name}
                )
                await self.agent_queues[recipient_id].put(notification)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send via shared memory: {e}")
            return False
    
    async def _send_via_queue(self, message: AgentMessage) -> bool:
        """Send message via asyncio queue"""
        try:
            recipient_id = message.recipient_id
            
            if recipient_id not in self.agent_queues:
                return False
            
            queue = self.agent_queues[recipient_id]
            
            # Try to put message without blocking
            try:
                queue.put_nowait(message)
                return True
            except asyncio.QueueFull:
                # Queue is full, try to make space by removing low-priority messages
                if message.priority >= MessagePriority.HIGH:
                    # Remove one low-priority message if possible
                    try:
                        old_message = queue.get_nowait()
                        if old_message.priority < MessagePriority.HIGH:
                            queue.put_nowait(message)
                            return True
                        else:
                            # Put the old message back and fail
                            queue.put_nowait(old_message)
                            return False
                    except asyncio.QueueEmpty:
                        return False
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send via queue: {e}")
            return False
    
    async def broadcast_message(self, message: AgentMessage) -> int:
        """Broadcast message to all agents"""
        success_count = 0
        
        for agent_id in self.broadcast_subscribers:
            if agent_id != message.sender_id:  # Don't send to sender
                broadcast_msg = AgentMessage(
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    message_type=message.message_type,
                    payload=message.payload,
                    priority=message.priority
                )
                
                if await self.send_message(broadcast_msg):
                    success_count += 1
        
        return success_count
    
    async def broadcast(self, payload: Dict[str, Any], priority: int = MessagePriority.NORMAL) -> int:
        """Convenience method for broadcasting"""
        message = AgentMessage(
            sender_id="communicator",
            recipient_id="*",
            message_type="broadcast",
            payload=payload,
            priority=priority
        )
        return await self.broadcast_message(message)
    
    def _message_router_loop(self):
        """Background thread for message routing optimization"""
        while not self.shutdown_event.is_set():
            try:
                # Optimize routing table based on message patterns
                self._optimize_routing_table()
                
                # Clean up old cached messages
                self._cleanup_message_cache()
                
                time.sleep(1.0)  # Run every second
                
            except Exception as e:
                self.logger.error(f"Error in message router loop: {e}")
                time.sleep(5.0)
    
    def _cleanup_loop(self):
        """Background thread for resource cleanup"""
        while not self.shutdown_event.is_set():
            try:
                # Clean up disconnected agents
                self._cleanup_disconnected_agents()
                
                # Optimize shared memory usage
                self._optimize_shared_memory()
                
                time.sleep(10.0)  # Run every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(30.0)
    
    def _optimize_routing_table(self):
        """Optimize message routing based on patterns"""
        # Analyze message patterns and update routing table
        # This could include caching frequently used routes,
        # load balancing, etc.
        pass
    
    def _cleanup_message_cache(self):
        """Clean up old cached messages"""
        current_time = time.time()
        expired_keys = []
        
        for key, (timestamp, _) in self.message_cache.items():
            if current_time - timestamp > 300:  # 5 minutes
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.message_cache[key]
    
    def _cleanup_disconnected_agents(self):
        """Remove agents that haven't sent heartbeat recently"""
        current_time = time.time()
        disconnected_agents = []
        
        for agent_id, agent_ref in self.agents.items():
            try:
                # Check if agent is still responsive
                if hasattr(agent_ref, 'metrics') and hasattr(agent_ref.metrics, 'last_heartbeat'):
                    if current_time - agent_ref.metrics.last_heartbeat > 30:  # 30 seconds
                        disconnected_agents.append(agent_id)
            except Exception:
                disconnected_agents.append(agent_id)
        
        for agent_id in disconnected_agents:
            self.logger.warning(f"Cleaning up disconnected agent: {agent_id}")
            self.unregister_agent(agent_id)
    
    def _optimize_shared_memory(self):
        """Optimize shared memory usage"""
        # Analyze usage patterns and resize channels if needed
        # Defragment memory pools
        # This is a placeholder for advanced memory management
        pass
    
    def _update_latency_stats(self, latency: float):
        """Update average latency statistics"""
        if self.message_stats["avg_latency"] == 0.0:
            self.message_stats["avg_latency"] = latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.message_stats["avg_latency"] = (
                alpha * latency + (1 - alpha) * self.message_stats["avg_latency"]
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        return {
            "message_stats": self.message_stats.copy(),
            "active_agents": len(self.agents),
            "shared_channels": len(self.shared_channels),
            "memory_pools": len(self.memory_pools),
            "queue_sizes": {
                agent_id: queue.qsize() 
                for agent_id, queue in self.agent_queues.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown communication system and clean up resources"""
        self.logger.info("Shutting down swarm communicator")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for background threads
        if self.router_thread and self.router_thread.is_alive():
            self.router_thread.join(timeout=5)
            
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Clean up shared memory
        for pool_name, shm in self.memory_pools.items():
            try:
                shm.close()
                shm.unlink()
                self.logger.info(f"Cleaned up shared memory pool: {pool_name}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {pool_name}: {e}")
        
        # Clear data structures
        self.agents.clear()
        self.agent_queues.clear()
        self.shared_channels.clear()
        self.memory_pools.clear()
        
        self.logger.info("Swarm communicator shutdown complete")