"""
Logging setup for Al-artworks

This module provides centralized logging configuration for the Al-artworks system,
including file logging, console output, and structured logging capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

def setup_logging(log_level="INFO", log_file="logs/al_artworks.log"):
    """
    Set up centralized logging for Al-artworks using loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    
    # Remove default loguru handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Log startup message
    logger.info("Al-artworks logging system initialized")
    logger.info(f"Log level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

def get_logger(name: str) -> logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)

def set_log_level(level: str):
    """
    Set the logging level dynamically.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()
    setup_logging(level)

def add_file_handler(log_file: str, level: str = "INFO"):
    """
    Add a file handler to the logging system.
    
    Args:
        log_file: Path to log file
        level: Logging level for this handler
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info(f"Added file handler: {log_file}")

def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.debug(f"{func.__name__} completed in {duration:.4f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    return wrapper

def log_async_performance(func):
    """
    Decorator to log async function performance.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.debug(f"{func.__name__} completed in {duration:.4f}s")
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    
    return wrapper

# Initialize logging on module import
setup_logging() 