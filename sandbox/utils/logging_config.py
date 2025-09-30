"""
Logging configuration for the Data Science Sandbox project.
Provides structured logging with different levels and formatters.
"""

import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

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
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
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
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "sandbox.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "challenge_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": str(LOGS_DIR / "challenges.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
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


def setup_logging(config_dict: Dict[str, Any] = None) -> None:
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
def log_performance(logger: logging.Logger = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
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
