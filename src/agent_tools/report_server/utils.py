"""
Utility functions for the report server module
"""

import os
import socket
import random
from typing import Optional

# Port range to try for the server
DEFAULT_PORT_RANGE = (8700, 8900)

def get_available_port(start_port: int = None, end_port: int = None) -> int:
    """
    Find an available port in the specified range.
    
    Args:
        start_port: Starting port number (default: 8700)
        end_port: Ending port number (default: 8900)
    
    Returns:
        An available port number, or 0 if none is available
    """
    # Use default range if not specified
    start_port = start_port or DEFAULT_PORT_RANGE[0]
    end_port = end_port or DEFAULT_PORT_RANGE[1]
    
    # Create a list of ports and shuffle it to avoid conflicts
    ports = list(range(start_port, end_port))
    random.shuffle(ports)
    
    # Try each port
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except:
            continue
    
    return 0


def create_terminal_link(file_path: str, text: Optional[str] = None) -> str:
    """
    Creates a clickable terminal link that works in most modern terminals.
    
    Args:
        file_path: Path to the file
        text: Text to display for the link (defaults to file_path)
        
    Returns:
        Formatted string with terminal hyperlink
    """
    if text is None:
        text = file_path
    
    # Convert to absolute path if it's not already
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    
    # Format as a terminal hyperlink
    # This works in many modern terminals like iTerm2, GNOME Terminal, etc.
    return f"\033]8;;file://{file_path}\033\\{text}\033]8;;\033\\"


def create_url_link(url: str, text: Optional[str] = None) -> str:
    """
    Creates a clickable terminal link for a URL that works in most modern terminals.
    
    Args:
        url: The URL
        text: Text to display for the link (defaults to url)
        
    Returns:
        Formatted string with terminal hyperlink
    """
    if text is None:
        text = url
    
    # Format as a terminal hyperlink
    # This works in many modern terminals like iTerm2, GNOME Terminal, etc.
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"