"""
Logging configuration for the Data Science Sandbox project.
Provides structured logging with different levels and formatters.
"""

import logging
import logging.config
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Constants for logging handlers
ROTATING_FILE_HANDLER = "logging.handlers.RotatingFileHandler"
MAX_BYTES = 10485760  # 10MB
BACKUP_COUNT = 5

# Try to import pythonjsonlogger for JSON formatting
try:
    import pythonjsonlogger.jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

# Logging configuration dictionary
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(levelname)s - %(message)s"},
        "json": {
            "()": (
                "pythonjsonlogger.jsonlogger.JsonFormatter"
                if HAS_JSON_LOGGER
                else "logging.Formatter"
            ),
            "format": (
                "%(asctime)s %(name)s %(levelname)s %(message)s"
                if HAS_JSON_LOGGER
                else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": ROTATING_FILE_HANDLER,
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "sandbox.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
        },
        "error_file": {
            "class": ROTATING_FILE_HANDLER,
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "errors.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
        },
        "challenge_file": {
            "class": ROTATING_FILE_HANDLER,
            "level": "INFO",
            "formatter": "json",
            "filename": str(LOGS_DIR / "challenges.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
        },
    },
    "loggers": {
        "sandbox": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False,
        },
        "sandbox.challenges": {
            "level": "INFO",
            "handlers": ["challenge_file"],
            "propagate": True,
        },
        "sandbox.core": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": True,
        },
        "sandbox.integrations": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": True,
        },
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
}


def setup_logging(config_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Set up logging configuration for the application.

    Args:
        config_dict: Optional custom logging configuration dictionary
    """
    if config_dict is None:
        config_dict = LOGGING_CONFIG

    logging.config.dictConfig(config_dict)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Performance logging decorator
def log_performance(
    logger: Optional[logging.Logger] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{func.__name__} completed in {duration:.4f} seconds")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"{func.__name__} failed after {duration:.4f} seconds: {str(e)}"
                )
                raise

        return wrapper

    return decorator
