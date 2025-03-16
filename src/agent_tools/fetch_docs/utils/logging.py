"""
Logging utilities for the Cursor Rules package.

This module provides consistent logging configuration and functions across
the Cursor Rules package using the loguru library.
"""

import sys
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any
from loguru import logger

# Default log format
DEFAULT_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
DEFAULT_LOG_LEVEL = "INFO"

def configure_logger(
    console_level: str = DEFAULT_LOG_LEVEL,
    file_level: str = "DEBUG",
    log_file: Optional[Union[str, Path]] = None,
    format_string: str = DEFAULT_FORMAT,
    rotation: str = "10 MB",
    retention: str = "1 week",
    env_level_var: str = "CURSOR_RULES_LOG_LEVEL"
) -> None:
    """
    Configure the logger for the Cursor Rules package.

    Args:
        console_level: Log level for console output
        file_level: Log level for file output
        log_file: Path to log file, default is ~/.cursor/logs/cursor_rules.log
        format_string: Log message format
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 week", "10 days")
        env_level_var: Environment variable name to override console_level
    """
    # Remove any existing handlers
    logger.remove()
    
    # Override console level from environment if set
    if env_level_var in os.environ:
        console_level = os.environ[env_level_var]
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=console_level,
        format=format_string
    )
    
    # Add file handler if requested
    if log_file is None:
        log_file = Path.home() / ".cursor" / "logs" / "cursor_rules.log"
    
    if isinstance(log_file, str):
        log_file = Path(log_file)
    
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=file_level,
        rotation=rotation,
        retention=retention,
        format=format_string.replace("<green>", "").replace("</green>", "")
                               .replace("<level>", "").replace("</level>", "")
                               .replace("<cyan>", "").replace("</cyan>", "")
    )

def get_module_logger(name: str) -> logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)

def log_db_operation(
    operation: str,
    collection: str,
    result: Any,
    error: Optional[Exception] = None,
    **kwargs
) -> None:
    """
    Log a database operation with consistent format.
    
    Args:
        operation: Operation type (e.g., "insert", "query")
        collection: Collection name
        result: Operation result
        error: Exception if operation failed
        **kwargs: Additional context to log
    """
    context = {
        "operation": operation,
        "collection": collection,
        **kwargs
    }
    
    if error:
        logger.error(f"{operation.upper()} failed on {collection}: {error}", extra=context)
    else:
        logger.debug(f"{operation.upper()} on {collection}: {result}", extra=context)

# Configure logger on module import
configure_logger()

__all__ = [
    'logger',
    'configure_logger',
    'get_module_logger',
    'log_db_operation'
] 