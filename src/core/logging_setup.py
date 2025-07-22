import logging
from loguru import logger
import sys


def setup_logging(log_level="INFO", log_file="logs/aisis.log"):
    """
    Set up centralized logging for AISIS using loguru.
    Logs to both console and file with rotation and formatting.
    Usage:
        from src.core.logging_setup import setup_logging
        setup_logging()
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(log_file, rotation="1 week", retention="4 weeks", level=log_level)
    logging.basicConfig(level=log_level)
