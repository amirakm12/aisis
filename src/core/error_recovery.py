"""
Error Recovery System
Handles error recovery, crash reporting, and logging
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from loguru import logger

class ErrorRecovery:
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Crash reports directory
        self.crash_dir = self.logs_dir / "crashes"
        self.crash_dir.mkdir(exist_ok=True)
        
        # Recovery state
        self.recovery_handlers: Dict[str, List[Callable]] = {}
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 3

    def _setup_logging(self) -> None:
        """Configure logging system"""
        # Remove default handler
        logger.remove()
        
        # Add file handler for debug logs
        debug_log = self.logs_dir / "debug.log"
        logger.add(
            debug_log,
            rotation="1 day",
            retention="7 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # Add file handler for error logs
        error_log = self.logs_dir / "error.log"
        logger.add(
            error_log,
            rotation="1 day",
            retention="30 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # Add console handler
        logger.add(
            sys.stderr,
            format="{time:HH:mm:ss} | {level} | {message}",
            level="INFO"
        )

    def register_recovery_handler(self, error_type: str, handler: Callable) -> None:
        """Register a recovery handler for a specific error type"""
        if error_type not in self.recovery_handlers:
            self.recovery_handlers[error_type] = []
        self.recovery_handlers[error_type].append(handler)

    def _save_crash_report(self, error: Exception, context: Dict[str, Any]) -> str:
        """Save crash report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.crash_dir / f"crash_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
            
        return str(report_file)

    def recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        Attempt to recover from an error
        
        Args:
            error: The exception that occurred
            context: Dictionary containing error context
            
        Returns:
            bool: True if recovery was successful
        """
        error_type = error.__class__.__name__
        
        # Save crash report
        report_file = self._save_crash_report(error, context)
        logger.error(f"Crash report saved to: {report_file}")
        
        # Check retry count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        if self.error_counts[error_type] > self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded for {error_type}")
            return False
            
        # Try recovery handlers
        if error_type in self.recovery_handlers:
            for handler in self.recovery_handlers[error_type]:
                try:
                    logger.info(f"Attempting recovery for {error_type}")
                    if handler(error, context):
                        logger.info(f"Recovery successful for {error_type}")
                        return True
                except Exception as e:
                    logger.error(f"Recovery handler failed: {e}")
                    
        logger.error(f"No successful recovery for {error_type}")
        return False

    def with_recovery(self, context: Dict[str, Any] = None):
        """
        Decorator for functions that need error recovery
        
        Args:
            context: Additional context for error handling
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_context = context or {}
                    error_context.update({
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    })
                    
                    if self.recover(e, error_context):
                        # Retry the function once if recovery was successful
                        return func(*args, **kwargs)
                    else:
                        raise  # Re-raise if recovery failed
            return wrapper
        return decorator

    def get_crash_reports(
        self,
        error_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get crash reports with optional filtering"""
        reports = []
        
        for report_file in self.crash_dir.glob("crash_*.json"):
            with open(report_file, "r") as f:
                report = json.load(f)
                
            # Apply filters
            if error_type and report["error_type"] != error_type:
                continue
                
            timestamp = datetime.fromisoformat(report["timestamp"])
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
                
            reports.append(report)
            
        return reports

    def clear_error_counts(self, error_type: Optional[str] = None) -> None:
        """Clear error counts for recovery attempts"""
        if error_type:
            self.error_counts.pop(error_type, None)
        else:
            self.error_counts.clear()

# Example recovery handlers
def memory_cleanup_handler(error: Exception, context: Dict[str, Any]) -> bool:
    """Handle out of memory errors"""
    if isinstance(error, MemoryError):
        import gc
        gc.collect()
        return True
    return False

def network_retry_handler(error: Exception, context: Dict[str, Any]) -> bool:
    """Handle network-related errors"""
    if isinstance(error, (ConnectionError, TimeoutError)):
        import time
        time.sleep(1)  # Wait before retry
        return True
    return False

# Register common recovery handlers
error_recovery = ErrorRecovery()
error_recovery.register_recovery_handler("MemoryError", memory_cleanup_handler)
error_recovery.register_recovery_handler("ConnectionError", network_retry_handler)
error_recovery.register_recovery_handler("TimeoutError", network_retry_handler)

# Example usage:
"""
@error_recovery.with_recovery({"module": "image_processing"})
def process_image(image_path: str) -> None:
    # Processing code that might fail
    pass
""" 