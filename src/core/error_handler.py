"""
Comprehensive Error Handling and Recovery System for AISIS
"""

import sys
import traceback
import logging
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime
import json
from enum import Enum
from dataclasses import dataclass, asdict

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # Application cannot continue
    ERROR = "error"       # Feature broken but app can continue
    WARNING = "warning"   # Potential issue
    INFO = "info"        # Informational message

class ErrorCategory(Enum):
    """Error categories for better classification"""
    SYSTEM = "system"
    NETWORK = "network"
    MODEL = "model"
    UI = "ui"
    PLUGIN = "plugin"
    CONFIG = "config"
    PERMISSION = "permission"
    MEMORY = "memory"
    GPU = "gpu"
    FILE_IO = "file_io"
    USER_INPUT = "user_input"

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_code: str
    message: str
    details: Optional[str] = None
    stack_trace: Optional[str] = None
    user_action: Optional[str] = None
    recovery_suggestions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []
        if self.metadata is None:
            self.metadata = {}

class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_history: List[ErrorContext] = []
        self.max_history = 100
        self.log_file = Path.home() / ".aisis" / "logs" / "errors.log"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup error logging"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def register_recovery_strategy(self, error_code: str, strategy: Callable):
        """Register a recovery strategy for a specific error code"""
        self.recovery_strategies[error_code] = strategy
    
    def handle_error(self, 
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    error_code: Optional[str] = None,
                    user_action: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """
        Handle an error with comprehensive logging and recovery
        
        Args:
            error: The exception that occurred
            severity: Severity level of the error
            category: Category of the error
            error_code: Unique error code
            user_action: What the user was doing when error occurred
            metadata: Additional context information
            
        Returns:
            ErrorContext: Detailed error information
        """
        
        # Generate error code if not provided
        if error_code is None:
            error_code = f"{category.value}_{type(error).__name__}".upper()
        
        # Create error context
        error_context = ErrorContext(
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category=category,
            error_code=error_code,
            message=str(error),
            details=self._get_error_details(error),
            stack_trace=traceback.format_exc(),
            user_action=user_action,
            recovery_suggestions=self._get_recovery_suggestions(error_code, category),
            metadata=metadata or {}
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Add to history
        self._add_to_history(error_context)
        
        # Attempt recovery
        self._attempt_recovery(error_context)
        
        return error_context
    
    def _get_error_details(self, error: Exception) -> str:
        """Get detailed error information"""
        details = []
        
        # Add error type and message
        details.append(f"Type: {type(error).__name__}")
        details.append(f"Message: {str(error)}")
        
        # Add specific details based on error type
        if hasattr(error, 'errno'):
            details.append(f"Error Number: {error.errno}")
        
        if hasattr(error, 'filename'):
            details.append(f"File: {error.filename}")
        
        if hasattr(error, 'lineno'):
            details.append(f"Line: {error.lineno}")
        
        return "\n".join(details)
    
    def _get_recovery_suggestions(self, error_code: str, category: ErrorCategory) -> List[str]:
        """Get recovery suggestions based on error code and category"""
        suggestions = []
        
        # Category-specific suggestions
        if category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check your internet connection",
                "Verify firewall settings",
                "Try again in a few moments",
                "Check if the service is available"
            ])
        elif category == ErrorCategory.MODEL:
            suggestions.extend([
                "Verify model files are downloaded",
                "Check available disk space",
                "Restart the application",
                "Re-download the model"
            ])
        elif category == ErrorCategory.GPU:
            suggestions.extend([
                "Check GPU drivers are installed",
                "Verify CUDA installation",
                "Try running in CPU mode",
                "Close other GPU-intensive applications"
            ])
        elif category == ErrorCategory.MEMORY:
            suggestions.extend([
                "Close unnecessary applications",
                "Restart AISIS",
                "Check available system memory",
                "Try processing smaller files"
            ])
        elif category == ErrorCategory.FILE_IO:
            suggestions.extend([
                "Check file permissions",
                "Verify file exists and is accessible",
                "Check available disk space",
                "Try a different file location"
            ])
        elif category == ErrorCategory.CONFIG:
            suggestions.extend([
                "Check configuration file syntax",
                "Reset to default configuration",
                "Verify all required settings are present",
                "Check file permissions"
            ])
        
        # Add generic suggestions
        suggestions.extend([
            "Restart the application",
            "Check the logs for more details",
            "Contact support if the problem persists"
        ])
        
        return suggestions
    
    def _log_error(self, error_context: ErrorContext):
        """Log error to file and console"""
        log_message = f"[{error_context.error_code}] {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            print(f"🚨 CRITICAL ERROR: {log_message}")
        elif error_context.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
            print(f"❌ ERROR: {log_message}")
        elif error_context.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
            print(f"⚠️  WARNING: {log_message}")
        else:
            self.logger.info(log_message)
            print(f"ℹ️  INFO: {log_message}")
        
        # Log detailed information to file
        detailed_info = {
            "error_context": asdict(error_context),
            "stack_trace": error_context.stack_trace
        }
        
        self.logger.error(f"Detailed error info: {json.dumps(detailed_info, indent=2)}")
    
    def _add_to_history(self, error_context: ErrorContext):
        """Add error to history"""
        self.error_history.append(error_context)
        
        # Keep history size manageable
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt to recover from the error"""
        error_code = error_context.error_code
        
        if error_code in self.recovery_strategies:
            try:
                print(f"🔧 Attempting recovery for {error_code}...")
                self.recovery_strategies[error_code](error_context)
                print("✅ Recovery successful")
            except Exception as recovery_error:
                print(f"❌ Recovery failed: {recovery_error}")
                # Log recovery failure
                self.logger.error(f"Recovery failed for {error_code}: {recovery_error}")
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        recent_errors = self.error_history[-10:] if self.error_history else []
        
        error_counts = {}
        for error in self.error_history:
            category = error.category.value
            error_counts[category] = error_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": [asdict(error) for error in recent_errors],
            "error_counts_by_category": error_counts,
            "log_file": str(self.log_file),
            "generated_at": datetime.now().isoformat()
        }
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        print("Error history cleared")
    
    def export_error_report(self, file_path: Optional[Path] = None) -> Path:
        """Export error report to file"""
        if file_path is None:
            file_path = Path.home() / ".aisis" / "error_report.json"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.get_error_report()
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Error report exported to: {file_path}")
        return file_path

# Global error manager instance
error_manager = ErrorRecoveryManager()

def handle_error(error: Exception, **kwargs) -> ErrorContext:
    """Convenience function to handle errors"""
    return error_manager.handle_error(error, **kwargs)

def register_recovery_strategy(error_code: str, strategy: Callable):
    """Convenience function to register recovery strategies"""
    error_manager.register_recovery_strategy(error_code, strategy)

# Decorator for automatic error handling
def error_handler(severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 error_code: Optional[str] = None):
    """Decorator for automatic error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(
                    e,
                    severity=severity,
                    category=category,
                    error_code=error_code,
                    user_action=f"Calling function: {func.__name__}",
                    metadata={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                )
                raise
        return wrapper
    return decorator

# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in a block of code"""
    
    def __init__(self, user_action: str, **kwargs):
        self.user_action = user_action
        self.kwargs = kwargs
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            handle_error(
                exc_val,
                user_action=self.user_action,
                **self.kwargs
            )
        return False  # Don't suppress the exception