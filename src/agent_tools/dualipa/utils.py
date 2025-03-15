"""
Utility functions for the DuaLipa module.

This module provides common utility functions used across the DuaLipa pipeline.
"""

from typing import Any, Dict, List, Optional, Union
import os
import sys
from pathlib import Path
import re

def format_string(text: str, **kwargs: Any) -> str:
    """
    Format a string with optional formatting parameters.
    
    This function provides string formatting with error handling.
    
    Args:
        text: The string to format
        **kwargs: Additional formatting parameters
        
    Returns:
        The formatted string
    """
    if not kwargs:
        return text.strip()
        
    try:
        # Apply any provided formatting
        formatted_text = text.format(**kwargs)
        return formatted_text.strip()
    except Exception as e:
        # Fallback if formatting fails
        return f"{text.strip()} (Error: {str(e)})"

def path_to_relative(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convert a path to a relative path from a base directory.
    
    Args:
        path: The path to convert
        base_path: Base directory to make path relative to (defaults to current working directory)
        
    Returns:
        Relative path as a string
    """
    if base_path is None:
        base_path = os.getcwd()
        
    path_obj = Path(path)
    base_path_obj = Path(base_path)
    
    try:
        relative_path = path_obj.relative_to(base_path_obj)
        return str(relative_path)
    except ValueError:
        # If the path is not relative to the base, return the original path
        return str(path)

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path 