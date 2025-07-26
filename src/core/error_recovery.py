"""
Advanced Error Recovery and Crash Recovery System
Provides comprehensive error handling, recovery mechanisms, and system resilience
"""

import asyncio
import traceback
import sys
import signal
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import threading
from loguru import logger

from .memory_manager import memory_manager


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    SHUTDOWN = "shutdown"
    IGNORE = "ignore"


@dataclass
class ErrorInfo:
    """Error information container"""
    error_type: str
    message: str
    traceback: str
    timestamp: float
    severity: ErrorSeverity
    context: Dict[str, Any]
    recovery_action: Optional[RecoveryAction] = None
    retry_count: int = 0
    resolved: bool = False


@dataclass
class SystemState:
    """System state snapshot for recovery"""
    timestamp: float
    memory_stats: Dict[str, Any]
    loaded_models: List[str]
    active_tasks: List[str]
    configuration: Dict[str, Any]
    error_count: int


class ErrorRecoverySystem:
    """
    Comprehensive error recovery and crash handling system
    """
    
    def __init__(self, 
                 state_dir: str = "recovery_state",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 checkpoint_interval: float = 60.0):
        """
        Initialize error recovery system
        
        Args:
            state_dir: Directory for storing recovery state
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            checkpoint_interval: State checkpoint interval (seconds)
        """
        self.state_dir = Path(state_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checkpoint_interval = checkpoint_interval
        
        # Create state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.errors: List[ErrorInfo] = []
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # System state management
        self.last_checkpoint = None
        self.checkpoint_thread = None
        self.checkpointing = False
        
        # Graceful shutdown handling
        self.shutdown_handlers: List[Callable] = []
        self.shutdown_in_progress = False
        
        # Recovery callbacks
        self.recovery_callbacks: List[Callable] = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("Error Recovery System initialized")
    
    def register_error_handler(self, 
                              exception_type: Type[Exception], 
                              handler: Callable[[Exception, Dict], RecoveryAction]):
        """
        Register a custom error handler
        
        Args:
            exception_type: Exception type to handle
            handler: Handler function that returns recovery action
        """
        self.error_handlers[exception_type] = handler
        logger.debug(f"Registered error handler for {exception_type.__name__}")
    
    def register_recovery_strategy(self, 
                                  strategy_name: str, 
                                  strategy: Callable[[ErrorInfo], bool]):
        """
        Register a recovery strategy
        
        Args:
            strategy_name: Name of the strategy
            strategy: Strategy function that returns success status
        """
        self.recovery_strategies[strategy_name] = strategy
        logger.debug(f"Registered recovery strategy: {strategy_name}")
    
    def add_shutdown_handler(self, handler: Callable):
        """Add a graceful shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    def add_recovery_callback(self, callback: Callable):
        """Add a recovery callback"""
        self.recovery_callbacks.append(callback)
    
    async def handle_error(self, 
                          error: Exception, 
                          context: Optional[Dict[str, Any]] = None,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> RecoveryAction:
        """
        Handle an error with appropriate recovery action
        
        Args:
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            Recovery action to take
        """
        context = context or {}
        
        # Create error info
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            timestamp=time.time(),
            severity=severity,
            context=context
        )
        
        # Log the error
        log_func = {
            ErrorSeverity.LOW: logger.debug,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }[severity]
        
        log_func(f"Error occurred: {error_info.error_type} - {error_info.message}")
        
        # Store error
        self.errors.append(error_info)
        
        # Determine recovery action
        recovery_action = await self._determine_recovery_action(error, error_info)
        error_info.recovery_action = recovery_action
        
        # Execute recovery action
        success = await self._execute_recovery_action(recovery_action, error_info)
        
        if success:
            error_info.resolved = True
            logger.info(f"Error recovered using action: {recovery_action.value}")
        else:
            logger.error(f"Recovery action failed: {recovery_action.value}")
        
        # Notify callbacks
        for callback in self.recovery_callbacks:
            try:
                await callback(error_info, recovery_action, success)
            except Exception as e:
                logger.error(f"Recovery callback failed: {e}")
        
        return recovery_action
    
    async def _determine_recovery_action(self, 
                                       error: Exception, 
                                       error_info: ErrorInfo) -> RecoveryAction:
        """Determine the appropriate recovery action"""
        
        # Check for custom error handler
        error_type = type(error)
        if error_type in self.error_handlers:
            try:
                return self.error_handlers[error_type](error, error_info.context)
            except Exception as e:
                logger.error(f"Custom error handler failed: {e}")
        
        # Default recovery logic based on error type and severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.RESTART
        
        if isinstance(error, MemoryError):
            return RecoveryAction.RESTART
        
        if isinstance(error, (ConnectionError, TimeoutError)):
            return RecoveryAction.RETRY
        
        if isinstance(error, FileNotFoundError):
            return RecoveryAction.FALLBACK
        
        if isinstance(error, (ValueError, TypeError)):
            return RecoveryAction.FALLBACK
        
        # Default action based on severity
        if error_info.severity == ErrorSeverity.HIGH:
            return RecoveryAction.RESTART
        elif error_info.severity == ErrorSeverity.MEDIUM:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.IGNORE
    
    async def _execute_recovery_action(self, 
                                     action: RecoveryAction, 
                                     error_info: ErrorInfo) -> bool:
        """Execute the recovery action"""
        
        try:
            if action == RecoveryAction.RETRY:
                return await self._retry_operation(error_info)
            
            elif action == RecoveryAction.FALLBACK:
                return await self._execute_fallback(error_info)
            
            elif action == RecoveryAction.RESTART:
                return await self._restart_system(error_info)
            
            elif action == RecoveryAction.SHUTDOWN:
                await self._graceful_shutdown(error_info)
                return True
            
            elif action == RecoveryAction.IGNORE:
                logger.debug("Ignoring error as per recovery action")
                return True
            
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery action execution failed: {e}")
            return False
    
    async def _retry_operation(self, error_info: ErrorInfo) -> bool:
        """Retry the failed operation"""
        if error_info.retry_count >= self.max_retries:
            logger.error(f"Max retries exceeded for {error_info.error_type}")
            return False
        
        error_info.retry_count += 1
        logger.info(f"Retrying operation (attempt {error_info.retry_count}/{self.max_retries})")
        
        # Wait before retry
        await asyncio.sleep(self.retry_delay * error_info.retry_count)
        
        # Execute retry strategy if available
        if "retry" in self.recovery_strategies:
            return self.recovery_strategies["retry"](error_info)
        
        return True  # Assume retry will be handled by calling code
    
    async def _execute_fallback(self, error_info: ErrorInfo) -> bool:
        """Execute fallback strategy"""
        logger.info("Executing fallback strategy")
        
        if "fallback" in self.recovery_strategies:
            return self.recovery_strategies["fallback"](error_info)
        
        # Default fallback: reduce system load
        memory_manager.cleanup_memory(force=True)
        return True
    
    async def _restart_system(self, error_info: ErrorInfo) -> bool:
        """Restart system components"""
        logger.warning("Restarting system components")
        
        try:
            # Save current state
            await self._save_checkpoint()
            
            # Execute restart strategy
            if "restart" in self.recovery_strategies:
                return self.recovery_strategies["restart"](error_info)
            
            # Default restart: cleanup and reinitialize
            memory_manager.cleanup_memory(force=True)
            
            # Restart would typically be handled by the main application
            logger.info("System restart initiated")
            return True
            
        except Exception as e:
            logger.error(f"System restart failed: {e}")
            return False
    
    async def _graceful_shutdown(self, error_info: ErrorInfo):
        """Perform graceful shutdown"""
        if self.shutdown_in_progress:
            return
        
        self.shutdown_in_progress = True
        logger.critical("Initiating graceful shutdown")
        
        try:
            # Save final checkpoint
            await self._save_checkpoint()
            
            # Execute shutdown handlers
            for handler in self.shutdown_handlers:
                try:
                    await handler()
                except Exception as e:
                    logger.error(f"Shutdown handler failed: {e}")
            
            # Cleanup memory
            memory_manager.cleanup_memory(force=True)
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
    
    def start_checkpointing(self):
        """Start automatic state checkpointing"""
        if self.checkpointing:
            return
        
        self.checkpointing = True
        self.checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
        self.checkpoint_thread.start()
        logger.info("State checkpointing started")
    
    def stop_checkpointing(self):
        """Stop automatic state checkpointing"""
        self.checkpointing = False
        if self.checkpoint_thread:
            self.checkpoint_thread.join(timeout=5.0)
        logger.info("State checkpointing stopped")
    
    def _checkpoint_loop(self):
        """Checkpoint loop for saving system state"""
        while self.checkpointing:
            try:
                asyncio.run(self._save_checkpoint())
                time.sleep(self.checkpoint_interval)
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")
    
    async def _save_checkpoint(self):
        """Save system state checkpoint"""
        try:
            # Gather system state
            state = SystemState(
                timestamp=time.time(),
                memory_stats=memory_manager.get_memory_stats().__dict__,
                loaded_models=[],  # Would be populated by model manager
                active_tasks=[],   # Would be populated by task manager
                configuration={},  # Would be populated by config manager
                error_count=len(self.errors)
            )
            
            # Save to file
            checkpoint_file = self.state_dir / f"checkpoint_{int(state.timestamp)}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(asdict(state), f, indent=2)
            
            # Keep only recent checkpoints
            self._cleanup_old_checkpoints()
            
            self.last_checkpoint = state
            logger.debug(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, keep_count: int = 10):
        """Remove old checkpoint files"""
        try:
            checkpoint_files = list(self.state_dir.glob("checkpoint_*.json"))
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_file in checkpoint_files[keep_count:]:
                old_file.unlink()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def load_latest_checkpoint(self) -> Optional[SystemState]:
        """Load the latest system state checkpoint"""
        try:
            checkpoint_files = list(self.state_dir.glob("checkpoint_*.json"))
            if not checkpoint_files:
                return None
            
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            state = SystemState(**data)
            logger.info(f"Loaded checkpoint from {latest_file}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self._graceful_shutdown(None))
        
        # Setup handlers for common termination signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            try:
                signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Signal handling might not be available in all environments
                pass
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.errors:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.errors),
            "resolved_errors": sum(1 for e in self.errors if e.resolved),
            "unresolved_errors": sum(1 for e in self.errors if not e.resolved),
            "errors_by_type": {},
            "errors_by_severity": {},
            "recent_errors": len([e for e in self.errors if time.time() - e.timestamp < 3600])
        }
        
        for error in self.errors:
            stats["errors_by_type"][error.error_type] = stats["errors_by_type"].get(error.error_type, 0) + 1
            stats["errors_by_severity"][error.severity.value] = stats["errors_by_severity"].get(error.severity.value, 0) + 1
        
        return stats


# Decorators for error handling
def with_error_recovery(severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                       context: Optional[Dict] = None):
    """Decorator for automatic error recovery"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_action = await error_recovery.handle_error(e, context, severity)
                if recovery_action == RecoveryAction.RETRY:
                    # Retry once automatically
                    try:
                        return await func(*args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        raise
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't easily do async error handling
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def error_context(context_name: str, **context_data):
    """Context manager for error tracking"""
    context = {"context_name": context_name, **context_data}
    try:
        yield context
    except Exception as e:
        asyncio.create_task(error_recovery.handle_error(e, context))
        raise


# Global error recovery instance
error_recovery = ErrorRecoverySystem()


def setup_error_recovery(state_dir: str = "recovery_state") -> ErrorRecoverySystem:
    """
    Setup global error recovery system
    
    Args:
        state_dir: Directory for recovery state
        
    Returns:
        Configured error recovery system
    """
    global error_recovery
    error_recovery = ErrorRecoverySystem(state_dir=state_dir)
    error_recovery.start_checkpointing()
    return error_recovery