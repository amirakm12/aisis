"""
Base Agent Class with Real Model Loading and Advanced Features
Provides comprehensive foundation for all AI agents with memory management and error recovery
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
from loguru import logger

from ..core.memory_manager import memory_manager
from ..core.model_manager import model_manager, ModelStatus
from ..core.error_recovery import error_recovery, ErrorSeverity, with_error_recovery


class AgentState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskInfo:
    """Task information container"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class AgentCapabilities:
    """Agent capabilities definition"""
    tasks: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    required_models: List[str] = field(default_factory=list)
    memory_requirements_gb: float = 1.0
    gpu_required: bool = False
    async_capable: bool = True


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_processed: int = 0
    tasks_successful: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    memory_usage_peak: float = 0.0
    last_activity: Optional[float] = None


class BaseAgent(ABC):
    """
    Advanced base agent class with real model loading and comprehensive features
    """
    
    def __init__(self, 
                 name: str,
                 capabilities: Optional[AgentCapabilities] = None,
                 max_concurrent_tasks: int = 1,
                 task_timeout: float = 300.0):
        """
        Initialize base agent
        
        Args:
            name: Agent name/identifier
            capabilities: Agent capabilities
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Task timeout in seconds
        """
        self.name = name
        self.capabilities = capabilities or AgentCapabilities()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        
        # State management
        self.state = AgentState.UNINITIALIZED
        self.state_lock = threading.Lock()
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: List[TaskInfo] = []
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Models and resources
        self.loaded_models: Dict[str, Any] = {}
        self.model_loading_lock = asyncio.Lock()
        
        # Metrics and monitoring
        self.metrics = AgentMetrics()
        self.performance_callbacks: List[Callable] = []
        
        # Error handling
        self.error_handlers: Dict[str, Callable] = {}
        
        # Lifecycle hooks
        self.initialization_hooks: List[Callable] = []
        self.cleanup_hooks: List[Callable] = []
        
        logger.info(f"Agent {self.name} initialized")
    
    @property
    def is_ready(self) -> bool:
        """Check if agent is ready to process tasks"""
        return self.state == AgentState.READY
    
    @property
    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks"""
        return len(self.active_tasks) > 0
    
    async def initialize(self) -> bool:
        """
        Initialize the agent with model loading and setup
        
        Returns:
            Success status
        """
        with self.state_lock:
            if self.state != AgentState.UNINITIALIZED:
                logger.warning(f"Agent {self.name} already initialized")
                return True
            
            self.state = AgentState.INITIALIZING
        
        try:
            logger.info(f"Initializing agent {self.name}")
            
            # Check memory requirements
            memory_check = memory_manager.can_load_model(self.capabilities.memory_requirements_gb)
            if not memory_check['can_load']:
                logger.error(f"Insufficient memory for {self.name}: {memory_check}")
                self.state = AgentState.ERROR
                return False
            
            # Load required models
            await self._load_required_models()
            
            # Execute custom initialization
            await self._initialize_agent()
            
            # Run initialization hooks
            for hook in self.initialization_hooks:
                try:
                    await hook(self)
                except Exception as e:
                    logger.error(f"Initialization hook failed: {e}")
            
            # Register cleanup callback with memory manager
            memory_manager.add_cleanup_callback(self._memory_cleanup)
            
            # Start task processing
            asyncio.create_task(self._task_processor())
            
            self.state = AgentState.READY
            logger.info(f"Agent {self.name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {e}")
            await error_recovery.handle_error(
                e, 
                {"agent": self.name, "operation": "initialize"},
                ErrorSeverity.HIGH
            )
            self.state = AgentState.ERROR
            return False
    
    async def _load_required_models(self):
        """Load all required models for this agent"""
        if not self.capabilities.required_models:
            return
        
        async with self.model_loading_lock:
            logger.info(f"Loading {len(self.capabilities.required_models)} models for {self.name}")
            
            for model_name in self.capabilities.required_models:
                try:
                    # Check if model is registered
                    model_info = model_manager.get_model_info(model_name)
                    if not model_info:
                        logger.warning(f"Model {model_name} not registered, skipping")
                        continue
                    
                    # Download model if needed
                    if model_info.status != ModelStatus.DOWNLOADED:
                        logger.info(f"Downloading model {model_name}")
                        success = await model_manager.download_model(model_name)
                        if not success:
                            raise RuntimeError(f"Failed to download model {model_name}")
                    
                    # Load model
                    logger.info(f"Loading model {model_name}")
                    device = "cuda" if self.capabilities.gpu_required else "auto"
                    model = await model_manager.load_model(model_name, device=device)
                    
                    self.loaded_models[model_name] = model
                    logger.info(f"Successfully loaded model {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    raise
    
    @abstractmethod
    async def _initialize_agent(self):
        """
        Custom agent initialization logic
        Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    async def process_task(self, task: TaskInfo) -> Any:
        """
        Process a single task
        Must be implemented by subclasses
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        pass
    
    async def submit_task(self, 
                         task_type: str, 
                         data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         task_id: Optional[str] = None) -> TaskInfo:
        """
        Submit a task for processing
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Task priority
            task_id: Optional task ID
            
        Returns:
            Task info object
        """
        if not self.is_ready:
            raise RuntimeError(f"Agent {self.name} is not ready")
        
        if task_type not in self.capabilities.tasks:
            raise ValueError(f"Agent {self.name} does not support task type: {task_type}")
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"{self.name}_{task_type}_{int(time.time() * 1000)}"
        
        # Create task info
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data
        )
        
        # Add to queue
        await self.task_queue.put(task)
        logger.debug(f"Task {task_id} submitted to {self.name}")
        
        return task
    
    async def _task_processor(self):
        """Main task processing loop"""
        logger.info(f"Task processor started for {self.name}")
        
        while self.state in [AgentState.READY, AgentState.PROCESSING]:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process task with concurrency control
                async with self.task_semaphore:
                    await self._process_single_task(task)
                
            except Exception as e:
                logger.error(f"Task processor error in {self.name}: {e}")
                await error_recovery.handle_error(
                    e,
                    {"agent": self.name, "operation": "task_processing"},
                    ErrorSeverity.MEDIUM
                )
        
        logger.info(f"Task processor stopped for {self.name}")
    
    @with_error_recovery(ErrorSeverity.MEDIUM)
    async def _process_single_task(self, task: TaskInfo):
        """Process a single task with error handling and metrics"""
        task.started_at = time.time()
        self.active_tasks[task.task_id] = task
        
        try:
            logger.debug(f"Processing task {task.task_id} in {self.name}")
            
            # Update state
            self.state = AgentState.PROCESSING
            
            # Process task with timeout
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=self.task_timeout
            )
            
            # Update task info
            task.completed_at = time.time()
            task.result = result
            
            # Update metrics
            self._update_metrics(task, success=True)
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            task.error = f"Task timeout after {self.task_timeout}s"
            self._update_metrics(task, success=False)
            logger.error(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.error = str(e)
            self._update_metrics(task, success=False)
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Handle specific error types
            error_type = type(e).__name__
            if error_type in self.error_handlers:
                try:
                    await self.error_handlers[error_type](task, e)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
        
        finally:
            # Cleanup
            self.active_tasks.pop(task.task_id, None)
            self.completed_tasks.append(task)
            
            # Keep only recent tasks
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-500:]
            
            # Update state
            if not self.active_tasks:
                self.state = AgentState.READY
    
    def _update_metrics(self, task: TaskInfo, success: bool):
        """Update agent performance metrics"""
        self.metrics.tasks_processed += 1
        self.metrics.last_activity = time.time()
        
        if success:
            self.metrics.tasks_successful += 1
        else:
            self.metrics.tasks_failed += 1
        
        if task.started_at and task.completed_at:
            processing_time = task.completed_at - task.started_at
            self.metrics.total_processing_time += processing_time
            self.metrics.average_processing_time = (
                self.metrics.total_processing_time / self.metrics.tasks_processed
            )
        
        # Update memory usage
        memory_stats = memory_manager.get_memory_stats()
        self.metrics.memory_usage_peak = max(
            self.metrics.memory_usage_peak,
            memory_stats.used_ram
        )
        
        # Notify performance callbacks
        for callback in self.performance_callbacks:
            try:
                callback(self, task, success)
            except Exception as e:
                logger.error(f"Performance callback failed: {e}")
    
    def _memory_cleanup(self):
        """Memory cleanup callback"""
        logger.debug(f"Memory cleanup triggered for {self.name}")
        
        # Unload non-essential models if memory pressure
        memory_stats = memory_manager.get_memory_stats()
        if memory_stats.ram_percent > 0.8:
            logger.warning(f"High memory usage, considering model unloading for {self.name}")
            # Could implement selective model unloading here
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "is_ready": self.is_ready,
            "is_busy": self.is_busy,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "loaded_models": list(self.loaded_models.keys()),
            "capabilities": {
                "tasks": self.capabilities.tasks,
                "input_types": self.capabilities.input_types,
                "output_types": self.capabilities.output_types,
                "memory_requirements_gb": self.capabilities.memory_requirements_gb,
                "gpu_required": self.capabilities.gpu_required
            },
            "metrics": {
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_successful": self.metrics.tasks_successful,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": (
                    self.metrics.tasks_successful / max(1, self.metrics.tasks_processed)
                ) * 100,
                "average_processing_time": self.metrics.average_processing_time,
                "memory_usage_peak": self.metrics.memory_usage_peak,
                "last_activity": self.metrics.last_activity
            }
        }
    
    def add_error_handler(self, error_type: str, handler: Callable):
        """Add custom error handler"""
        self.error_handlers[error_type] = handler
        logger.debug(f"Added error handler for {error_type} in {self.name}")
    
    def add_performance_callback(self, callback: Callable):
        """Add performance monitoring callback"""
        self.performance_callbacks.append(callback)
    
    def add_initialization_hook(self, hook: Callable):
        """Add initialization hook"""
        self.initialization_hooks.append(hook)
    
    def add_cleanup_hook(self, hook: Callable):
        """Add cleanup hook"""
        self.cleanup_hooks.append(hook)
    
    async def shutdown(self):
        """Graceful agent shutdown"""
        logger.info(f"Shutting down agent {self.name}")
        
        self.state = AgentState.SHUTDOWN
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            while self.active_tasks and len(self.active_tasks) > 0:
                await asyncio.sleep(0.1)
        
        # Run cleanup hooks
        for hook in self.cleanup_hooks:
            try:
                await hook(self)
            except Exception as e:
                logger.error(f"Cleanup hook failed: {e}")
        
        # Unload models
        for model_name in list(self.loaded_models.keys()):
            try:
                await model_manager.unload_model(model_name)
                del self.loaded_models[model_name]
            except Exception as e:
                logger.error(f"Failed to unload model {model_name}: {e}")
        
        # Custom cleanup
        try:
            await self._cleanup_agent()
        except Exception as e:
            logger.error(f"Agent cleanup failed: {e}")
        
        logger.info(f"Agent {self.name} shutdown completed")
    
    async def _cleanup_agent(self):
        """
        Custom agent cleanup logic
        Can be overridden by subclasses
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', state='{self.state.value}')>"