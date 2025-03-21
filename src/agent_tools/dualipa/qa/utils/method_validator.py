"""Method validator for QA module functions.

This utility provides validation for method signatures, docstrings, and behavior 
conformance against specifications.

Official documentation:
- inspect: https://docs.python.org/3/library/inspect.html
- typing: https://docs.python.org/3/library/typing.html
- docstring_parser: https://pypi.org/project/docstring-parser/
"""

import inspect
import typing
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional
import asyncio
import textwrap
import logging

logger = logging.getLogger(__name__)


def verify_async_function(func: Callable) -> Dict[str, Any]:
    """Verify that a function is correctly implemented as async.
    
    Args:
        func: The function to validate
        
    Returns:
        Dict with validation results
        
    Raises:
        ValueError: If the function is not async
    """
    if not asyncio.iscoroutinefunction(func):
        raise ValueError(f"Function {func.__name__} must be async (use 'async def')")
        
    result = {
        "is_async": True,
        "name": func.__name__,
        "module": func.__module__,
        "signature": str(inspect.signature(func)),
    }
    
    # Verify docstring
    doc = inspect.getdoc(func)
    if not doc:
        result["docstring_exists"] = False
        logger.warning(f"Function {func.__name__} is missing a docstring")
    else:
        result["docstring_exists"] = True
        if "Args:" not in doc or "Returns:" not in doc:
            logger.warning(f"Function {func.__name__} docstring is missing Args or Returns section")
            result["docstring_complete"] = False
        else:
            result["docstring_complete"] = True
            
    return result


def validate_dedent_usage(source_code: str) -> Dict[str, Any]:
    """Validate that textwrap.dedent is used for all multiline strings.
    
    Args:
        source_code: The source code to check
        
    Returns:
        Dict with validation results
    """
    result = {
        "multiline_strings": 0,
        "dedent_calls": 0,
        "properly_dedented": True,
    }
    
    # Simple check for triple quotes not preceded by dedent
    triple_quotes = source_code.count('"""') + source_code.count("'''")
    if triple_quotes > 0:
        result["multiline_strings"] = triple_quotes // 2  # Each string has opening and closing quotes
        
    dedent_calls = source_code.count("textwrap.dedent")
    result["dedent_calls"] = dedent_calls
    
    # Check for multiline strings without dedent
    if result["multiline_strings"] > dedent_calls:
        result["properly_dedented"] = False
        logger.warning(
            f"Found {result['multiline_strings']} multiline strings but only "
            f"{dedent_calls} calls to textwrap.dedent"
        )
    
    return result


def verify_methods(module) -> Dict[str, Any]:
    """Verify all methods in a module for compliance with standards.
    
    Args:
        module: The module to verify
        
    Returns:
        Dict with validation results for each function
    """
    results = {}
    
    # Get all functions in the module
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            # Verify async functions
            try:
                results[name] = verify_async_function(obj)
            except ValueError as e:
                # Not an async function, skip
                pass
                
            # Verify source code for dedent usage
            try:
                source = inspect.getsource(obj)
                results[f"{name}_source"] = validate_dedent_usage(source)
            except Exception as e:
                logger.error(f"Could not get source for {name}: {e}")
    
    return results