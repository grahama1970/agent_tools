"""
report_server module

This module provides utilities for serving HTML reports using Docker or Python's built-in HTTP server.
It's designed to be used by various components in the agent_tools package for displaying test results
and other HTML-based reports.
"""

from .docker_server import serve_with_docker, stop_docker_server
from .utils import get_available_port, create_terminal_link

__all__ = [
    'serve_with_docker',
    'stop_docker_server',
    'get_available_port',
    'create_terminal_link',
]