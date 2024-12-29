import sys

from loguru import logger

from praice.core.config import settings
from praice.core.constants import PATHS


def setup_logging():
    """
    Set up logging configuration using loguru.
    """
    # Remove the default handler
    logger.remove()

    # Add a new handler with the configured log level
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # Add a file handler for persistent logging
    logger.add(
        PATHS["logs"] / "praice.log",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        level=settings.LOG_LEVEL,
    )


def get_scheduler_logger():
    """
    Create and return a scheduler-specific logger.
    """
    # Create a scheduler-specific logger
    scheduler_logger = logger.bind(name="scheduler")

    # Add a file handler for scheduler logs
    log_file = PATHS["logs"] / "scheduler.log"
    scheduler_logger.add(
        log_file,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "scheduler:{function}:{line} - {message}"
        ),
        level=settings.LOG_LEVEL,
    )

    return scheduler_logger
