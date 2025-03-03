"""Function validation using existing Python tools.

This module provides a simple interface to validate Python functions using:
1. mypy for static type checking
2. pylint for code quality (in deep analysis mode)
3. pydantic for runtime validation
4. method_validator for preventing method hallucination
"""

import ast
import hashlib
from functools import lru_cache
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, TypedDict
from dataclasses import dataclass
from loguru import logger
from pylint.lint import Run
from pylint.reporters import JSONReporter
from astroid import MANAGER
import os
import json
from io import StringIO
import time

from .analyzer import MethodAnalyzer, validate_method
from .constants import STDLIB_PACKAGES, COMMON_UTIL_PACKAGES
from .cache import init_db, store_result, get_result

class ValidationResult(TypedDict):
    type_issues: List[str]
    method_issues: List[str]
    quality_issues: List[str]
    style_issues: List[str]
    suggestions: List[str]

# Cache for in-memory results with TTL
_analysis_cache: Dict[str, Tuple[ValidationResult, float]] = {}
_CACHE_TTL = 3600  # 1 hour

def _get_cache_key(func_text: str) -> str:
    """Generate a cache key for a function."""
    return hashlib.md5(func_text.encode()).hexdigest()

def _clear_pylint_cache() -> None:
    """Clear pylint's internal cache between runs."""
    MANAGER.clear_cache()

def _extract_method_calls(func_text: str) -> Set[Tuple[str, str]]:
    """Extract all method calls from function text as (package, method) pairs.
    
    This function extracts:
    1. Package methods (e.g., requests.get)
    2. Built-in functions (e.g., print, len)
    3. Ignores object methods (e.g., str.upper, list.append)
    """
    method_calls = set()
    try:
        tree = ast.parse(func_text)
        
        # First pass: collect imports and function name
        imports = {}
        function_name = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        imports[name.asname or name.name] = node.module
            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
        
        # Second pass: collect method calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Handle package.method calls
                    if isinstance(node.func.value, ast.Name):
                        value_id = node.func.value.id
                        # Check if this is an imported module
                        if value_id in imports:
                            method_calls.add((imports[value_id], node.func.attr))
                        # Skip if the value looks like a variable
                        # (e.g., name.upper where name is a parameter)
                        elif value_id[0].isupper() or value_id in {
                            'os', 'sys', 'json', 'requests', 'pathlib',
                            'subprocess', 'datetime', 're', 'math', 'random',
                            'collections', 'itertools', 'functools'
                        }:
                            method_calls.add((value_id, node.func.attr))
                elif isinstance(node.func, ast.Name):
                    # Handle builtin method calls
                    func_id = node.func.id
                    # Skip if this is the function being validated
                    if func_id == function_name:
                        continue
                    if func_id in imports:
                        # This is an imported function
                        method_calls.add((imports[func_id], func_id))
                    else:
                        # Assume builtin
                        method_calls.add(("builtins", func_id))
    except Exception as e:
        logger.error(f"Failed to extract method calls: {e}")
    return method_calls

def _get_cached_result(cache_key: str) -> Optional[ValidationResult]:
    """Get cached result if valid."""
    if cache_key in _analysis_cache:
        result, timestamp = _analysis_cache[cache_key]
        if time.time() - timestamp < _CACHE_TTL:
            return result
        del _analysis_cache[cache_key]
    return None

def _cache_result(cache_key: str, result: ValidationResult) -> None:
    """Cache a validation result with timestamp."""
    _analysis_cache[cache_key] = (result, time.time())

class FunctionValidator:
    """Class for validating Python functions."""
    
    _db_initialized = False  # Class-level flag for DB initialization
    
    def __init__(self, deep_analysis: bool = False):
        """Initialize the validator.
        
        Args:
            deep_analysis: Whether to run full analysis with pylint
        """
        self.deep_analysis = deep_analysis
        self._ensure_db()
        
    @classmethod
    def _ensure_db(cls) -> None:
        """Ensure database is initialized only once per process."""
        if not cls._db_initialized:
            init_db()
            cls._db_initialized = True

    def validate_function(self, function_text: Optional[str] = None, file_path: Optional[str] = None) -> ValidationResult:
        """Validate a function and return any issues found."""
        if not function_text and not file_path:
            raise ValueError("Either function_text or file_path must be provided")

        # Get function text if file path provided
        if file_path:
            with open(file_path, 'r') as f:
                function_text = f.read()
        
        if not function_text:
            raise ValueError("No function text available")

        # Generate cache key
        cache_key = _get_cache_key(function_text)
        
        # Try memory cache first
        if cache_key in _analysis_cache:
            result, timestamp = _analysis_cache[cache_key]
            if time.time() - timestamp < _CACHE_TTL:
                return result
            del _analysis_cache[cache_key]
        
        # Try persistent cache
        cached = get_result("function_validator", cache_key)
        if cached:
            result, _ = cached
            # Update memory cache
            _analysis_cache[cache_key] = (result, time.time())
            return result
        
        # Run validation
        result = self._validate_function_internal(function_text, file_path)
        
        # Update both caches
        _analysis_cache[cache_key] = (result, time.time())
        store_result("function_validator", cache_key, cache_key, result)
        
        return result

    def _run_pylint(self, function_text: Optional[str], file_path: Optional[str]) -> List[Dict[str, Any]]:
        """Run pylint on the function and return any issues found."""
        if not self.deep_analysis:
            return []
            
        try:
            # Create a temporary file with the function content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                if file_path:
                    with open(file_path, 'r') as f:
                        temp_file.write(f.read())
                elif function_text:
                    temp_file.write(function_text)
                else:
                    return []
                temp_file.flush()
                temp_path = temp_file.name

            # Run pylint with JSON reporter
            output = StringIO()
            reporter = JSONReporter(output)
            Run([temp_path, '--output-format=json'], reporter=reporter, exit=False)
            
            # Parse JSON output
            output_str = output.getvalue()
            if not output_str:
                return []
                
            messages = json.loads(output_str)
            
            # Format messages into user-friendly categories
            formatted_messages = []
            for msg in messages:
                severity = msg['type']
                symbol = msg['symbol']
                message = {
                    'type': severity,
                    'message': msg['message'],
                    'line': msg['line'],
                    'symbol': symbol
                }
                
                # Add category-specific information
                if severity == 'error' or severity == 'fatal':
                    message['category'] = 'Critical Error'
                elif symbol in ['broad-exception-caught', 'no-member', 'not-callable', 'undefined-variable']:
                    message['category'] = 'Warning'
                elif symbol in ['undefined-variable', 'no-member', 'not-callable', 'no-name-in-module']:
                    message['category'] = 'Method/Attribute Issue'
                elif symbol in ['trailing-whitespace', 'missing-final-newline', 'wrong-import-order', 'missing-module-docstring']:
                    message['category'] = 'Style Issue'
                else:
                    message['category'] = 'Quality Issue'
                    
                # Skip certain messages in non-deep analysis mode
                if not self.deep_analysis and message['category'] == 'Style Issue':
                    continue
                    
                formatted_messages.append(message)
                
            return formatted_messages
            
        except Exception as e:
            logger.warning(f"Error running pylint: {e}")
            return []
            
        finally:
            # Clean up temporary file
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def _validate_function_internal(self, function_text: Optional[str], file_path: Optional[str]) -> ValidationResult:
        """Internal validation logic."""
        issues: ValidationResult = {
            'type_issues': [],
            'method_issues': [],
            'quality_issues': [],
            'style_issues': [],
            'suggestions': []
        }

        # Run quick validation first
        if not self.deep_analysis:
            return issues

        # Only run pylint for deep analysis
        pylint_messages = self._run_pylint(function_text, file_path)
        
        has_style_issues = False
        has_quality_issues = False
        has_type_issues = False
        has_method_issues = False
        
        for msg in pylint_messages:
            category = msg['category']
            formatted_msg = f"Line {msg['line']}: {msg['message']} ({msg['symbol']})"
            
            if category == 'Critical Error':
                issues['type_issues'].append(formatted_msg)
                has_type_issues = True
            elif category == 'Warning':
                issues['quality_issues'].append(formatted_msg)
                has_quality_issues = True
            elif category == 'Method/Attribute Issue':
                issues['method_issues'].append(formatted_msg)
                has_method_issues = True
            elif category == 'Style Issue':
                issues['style_issues'].append(formatted_msg)
                has_style_issues = True
            else:
                issues['quality_issues'].append(formatted_msg)
                has_quality_issues = True

        # Add suggestions based on found issues
        if has_quality_issues:
            issues['suggestions'].extend([
                "Consider using more specific exception handling",
                "Review variable and method naming for clarity",
                "Add docstrings and type hints where missing"
            ])
        
        if has_style_issues:
            issues['suggestions'].extend([
                "Fix trailing whitespace and missing newlines",
                "Follow PEP 8 style guidelines for imports and formatting"
            ])
            
        if has_type_issues:
            issues['suggestions'].extend([
                "Add type hints to improve code clarity",
                "Fix type-related errors to prevent runtime issues"
            ])
            
        if has_method_issues:
            issues['suggestions'].extend([
                "Check method and attribute names for correctness",
                "Ensure all required methods are properly imported"
            ])

        # Return early if no issues found
        if not any([has_type_issues, has_method_issues, has_quality_issues, has_style_issues]):
            return issues

        # Add specific suggestions based on message types
        for msg in pylint_messages:
            symbol = msg['symbol']
            if symbol == 'broad-exception-caught':
                issues['suggestions'].append(
                    "Replace 'except Exception:' with specific exception types"
                )
            elif symbol == 'wrong-import-order':
                issues['suggestions'].append(
                    "Place standard library imports before third-party imports"
                )
            elif symbol == 'missing-module-docstring':
                issues['suggestions'].append(
                    "Add a module-level docstring to describe the module's purpose"
                )
            elif symbol == 'trailing-whitespace':
                issues['suggestions'].append(
                    "Remove trailing whitespace from all lines"
                )
            elif symbol == 'missing-final-newline':
                issues['suggestions'].append(
                    "Add a newline at the end of the file"
                )

        return issues

def improve_function(func_text: str) -> str:
    """Suggest improvements for a Python function.
    
    This is a thin wrapper around FunctionValidator that applies
    suggested improvements to the function text.
    
    Args:
        func_text: The function text to improve
        
    Returns:
        Improved function text with type hints and best practices applied
    """
    # Quick validation first
    validator = FunctionValidator(deep_analysis=False)
    quick_result = validator.validate_function(function_text=func_text)
    
    # If quick validation finds critical issues, fix them
    if quick_result['type_issues'] or quick_result['method_issues']:
        # Get validation results with deep analysis
        validator = FunctionValidator(deep_analysis=True)
        result = validator.validate_function(function_text=func_text)
        
        # Try to parse function and apply improvements
        try:
            tree = ast.parse(func_text)
            if isinstance(tree.body[0], ast.FunctionDef):
                func_def = tree.body[0]
                
                # Add type hints if missing
                if any('type hints' in s for s in result['suggestions']):
                    # Extract function signature
                    first_line = func_text.split("\n")[0]
                    if "def " in first_line and ":" in first_line:
                        # Add Any type hints to parameters
                        params = first_line[first_line.find("(")+1:first_line.find(")")].split(",")
                        improved_params = []
                        for param in params:
                            param = param.strip()
                            if param and ":" not in param:
                                improved_params.append(f"{param}: Any")
                            else:
                                improved_params.append(param)
                                
                        # Create improved signature
                        improved_sig = f"def {func_def.name}({', '.join(improved_params)}) -> Any:"
                        func_text = improved_sig + func_text[first_line.find(":"):]
                
                # Add docstring if missing
                if any('docstring' in s for s in result['suggestions']):
                    if not ast.get_docstring(func_def):
                        indent = " " * 4
                        docstring = f'{indent}"""\n{indent}Description of the function.\n{indent}"""\n'
                        func_body = func_text[func_text.find(":") + 1:]
                        func_text = func_text[:func_text.find(":") + 1] + "\n" + docstring + func_body
                
                # Fix broad exception handling
                if any('broad-exception-caught' in s for s in result['suggestions']):
                    func_text = func_text.replace("except Exception", "except (ValueError, TypeError)")
                    
        except Exception as e:
            logger.debug(f"Could not parse function for improvements: {e}")
    
    # Cache the improved version
    cache_key = _get_cache_key(func_text)
    _cache_result(cache_key, quick_result)
    
    return func_text

__all__ = ['FunctionValidator', 'improve_function'] 