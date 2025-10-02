"""
Unit tests for logging configuration utilities
"""

import logging
from unittest.mock import Mock, patch

import pytest

from sandbox.utils.logging_config import (
    LOGGING_CONFIG,
    get_logger,
    log_performance,
    setup_logging,
)


class TestLoggingConfig:
    """Test suite for logging configuration utilities."""

    def test_logging_config_structure(self):
        """Test that logging configuration has all required components."""
        assert "version" in LOGGING_CONFIG
        assert "formatters" in LOGGING_CONFIG
        assert "handlers" in LOGGING_CONFIG
        assert "loggers" in LOGGING_CONFIG
        assert "root" in LOGGING_CONFIG

        # Check formatters
        formatters = LOGGING_CONFIG["formatters"]
        assert "detailed" in formatters
        assert "simple" in formatters
        assert "json" in formatters

        # Check handlers
        handlers = LOGGING_CONFIG["handlers"]
        assert "console" in handlers
        assert "file" in handlers
        assert "error_file" in handlers

    def test_setup_logging_basic(self):
        """Test basic logging setup without errors."""
        with patch("logging.config.dictConfig") as mock_config:
            setup_logging()

            mock_config.assert_called_once_with(LOGGING_CONFIG)

    def test_setup_logging_with_custom_config(self):
        """Test logging setup with custom configuration."""
        custom_config = {"version": 1, "disable_existing_loggers": False}

        with patch("logging.config.dictConfig") as mock_config:
            setup_logging(config_dict=custom_config)

            mock_config.assert_called_once_with(custom_config)

    def test_get_logger(self):
        """Test getting logger instance."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = get_logger("test_module")

            mock_get_logger.assert_called_once_with("test_module")
            assert logger == mock_logger

    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = get_logger("test_name")

            mock_get_logger.assert_called_once_with("test_name")
            assert logger == mock_logger

    def test_log_performance_decorator(self):
        """Test performance logging decorator."""
        mock_logger = Mock()

        @log_performance(logger=mock_logger)
        def test_function(x, y):
            return x + y

        result = test_function(1, 2)

        assert result == 3
        assert mock_logger.info.called

        # Check that performance info was logged
        call_args = mock_logger.info.call_args[0][0]
        assert "test_function" in call_args
        assert "completed in" in call_args

    def test_log_performance_with_exception(self):
        """Test performance logging when function raises exception."""
        mock_logger = Mock()

        @log_performance(logger=mock_logger)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        # Should log the exception
        assert mock_logger.error.called

    def test_logging_config_constants(self):
        """Test that logging configuration constants are properly defined."""
        # Test that the constants exist and have expected values
        from sandbox.utils.logging_config import LOGS_DIR, MAX_BYTES, BACKUP_COUNT

        assert LOGS_DIR.name == "logs"
        assert MAX_BYTES == 10485760  # 10MB
        assert BACKUP_COUNT == 5

    def test_logging_config_json_fallback(self):
        """Test JSON formatter fallback when pythonjsonlogger not available."""
        # The config should handle missing pythonjsonlogger gracefully
        json_formatter = LOGGING_CONFIG["formatters"]["json"]

        # Should have a fallback format
        assert "format" in json_formatter
        assert isinstance(json_formatter["format"], str)

    def test_error_file_handler_config(self):
        """Test error file handler configuration."""
        error_handler = LOGGING_CONFIG["handlers"]["error_file"]

        assert error_handler["level"] == "ERROR"
        assert error_handler["class"] == "logging.handlers.RotatingFileHandler"
        assert "errors.log" in error_handler["filename"]

    def test_console_handler_config(self):
        """Test console handler configuration."""
        console_handler = LOGGING_CONFIG["handlers"]["console"]

        assert console_handler["level"] == "INFO"
        assert console_handler["class"] == "logging.StreamHandler"
        assert console_handler["formatter"] == "simple"

    def test_file_handler_rotation_config(self):
        """Test file handler rotation configuration."""
        file_handler = LOGGING_CONFIG["handlers"]["file"]

        assert "maxBytes" in file_handler
        assert "backupCount" in file_handler
        assert file_handler["maxBytes"] == 10485760  # 10MB
        assert file_handler["backupCount"] == 5

    def test_challenge_file_handler_config(self):
        """Test challenge-specific file handler configuration."""
        challenge_handler = LOGGING_CONFIG["handlers"]["challenge_file"]

        assert challenge_handler["level"] == "INFO"
        assert challenge_handler["formatter"] == "json"
        assert "challenges.log" in challenge_handler["filename"]

    def test_logger_hierarchy_config(self):
        """Test logger hierarchy configuration."""
        loggers = LOGGING_CONFIG["loggers"]

        # Should have sandbox logger configured
        assert "sandbox" in loggers
        sandbox_logger = loggers["sandbox"]
        assert sandbox_logger["level"] == "DEBUG"
        assert "console" in sandbox_logger["handlers"]
        assert "file" in sandbox_logger["handlers"]

    def test_root_logger_config(self):
        """Test root logger configuration."""
        root_config = LOGGING_CONFIG["root"]

        assert root_config["level"] == "WARNING"
        assert "console" in root_config["handlers"]
