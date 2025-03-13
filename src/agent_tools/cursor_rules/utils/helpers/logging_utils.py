#!/usr/bin/env python3
"""
Logging utility functions for the fetch_page package.

This module contains utility functions for setting up logging.

OFFICIAL DOCUMENTATION:
- Loguru: https://loguru.readthedocs.io/
"""

import sys
from loguru import logger

def setup_logging(level="INFO"):
    """
    Set up logging with the specified level.
    
    Args:
        level: The logging level to use (default: "INFO")
               Can be "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    """
    logger.remove()
    logger.add(sys.stderr, level=level) 