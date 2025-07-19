from loguru import logger
import sys


def setup_logging(log_file: str = "aisis.log") -> None:
    """
    Set up centralized logging for AISIS using loguru.
    Logs to both console and file with rotation and formatting.
    Usage:
        from src.core.logging_setup import setup_logging
        setup_logging()
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time}</green> <level>{message}</level>",
    )
    logger.add(
        log_file,
        rotation="10 MB",
        level="DEBUG",
        format="{time} {level} {message}",
    )
