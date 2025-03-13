#!/usr/bin/env python3
"""
Rule Enforcer Module

This module provides decorators and utilities to enforce database rule checking
before any code generation or execution.
"""

import asyncio
import functools
import inspect
from typing import Callable, Dict, Any, List, Optional, Union
import logging
from loguru import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleCheckError(Exception):
    """Exception raised when mandatory rule checking is not performed."""
    pass

class RuleCheckRegistry:
    """Registry to track rule checks performed during execution."""
    
    def __init__(self):
        self.checks_performed = {}
        self.validation_tokens = {}
        
    def register_check(self, function_name: str, query: str, rules_found: List[Dict[str, Any]]) -> str:
        """
        Register that a rule check was performed.
        
        Args:
            function_name: Name of the function being checked
            query: The query used to search for rules
            rules_found: The rules that were found
            
        Returns:
            str: A validation token for this check
        """
        token = f"rule_check_{len(self.checks_performed)}"
        self.checks_performed[function_name] = {
            "query": query,
            "rules_found": rules_found,
            "token": token
        }
        self.validation_tokens[token] = function_name
        return token
    
    def verify_check(self, function_name: str, token: Optional[str] = None) -> bool:
        """
        Verify that a rule check was performed for a function.
        
        Args:
            function_name: Name of the function to check
            token: Optional validation token
            
        Returns:
            bool: True if check was performed, False otherwise
        """
        if token:
            # Verify by token
            return token in self.validation_tokens and self.validation_tokens[token] == function_name
        else:
            # Verify by function name
            return function_name in self.checks_performed
    
    def get_rules_for_function(self, function_name: str) -> List[Dict[str, Any]]:
        """
        Get the rules that were found for a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            List: The rules that were found
        """
        if function_name in self.checks_performed:
            return self.checks_performed[function_name]["rules_found"]
        return []
    
    def clear(self):
        """Clear the registry."""
        self.checks_performed = {}
        self.validation_tokens = {}

# Global registry instance
registry = RuleCheckRegistry()

def require_rule_check(db_func: Callable, query_param: str = "query"):
    """
    Decorator to enforce database rule checking before function execution.
    
    Args:
        db_func: The database function to use for rule checking
        query_param: The parameter name that contains the query
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Extract parameters
            params = bound_args.arguments
            
            # Get the query parameter
            query = params.get(query_param)
            if not query:
                raise ValueError(f"Missing required parameter: {query_param}")
            
            # Get the database parameter (first parameter of db_func)
            db_sig = inspect.signature(db_func)
            db_param_name = list(db_sig.parameters.keys())[0]
            db = params.get(db_param_name)
            if not db:
                raise ValueError(f"Missing required parameter: {db_param_name}")
            
            # Perform rule check
            logger.info(f"Performing mandatory rule check for {func.__name__} with query: {query}")
            rules = await db_func(db, query)
            
            # Register the check
            token = registry.register_check(func.__name__, query, rules)
            
            # Add token to kwargs
            kwargs["rule_check_token"] = token
            
            # Execute the function
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Extract parameters
            params = bound_args.arguments
            
            # Get the query parameter
            query = params.get(query_param)
            if not query:
                raise ValueError(f"Missing required parameter: {query_param}")
            
            # Get the database parameter (first parameter of db_func)
            db_sig = inspect.signature(db_func)
            db_param_name = list(db_sig.parameters.keys())[0]
            db = params.get(db_param_name)
            if not db:
                raise ValueError(f"Missing required parameter: {db_param_name}")
            
            # Perform rule check (synchronously)
            logger.info(f"Performing mandatory rule check for {func.__name__} with query: {query}")
            rules = asyncio.run(db_func(db, query))
            
            # Register the check
            token = registry.register_check(func.__name__, query, rules)
            
            # Add token to kwargs
            kwargs["rule_check_token"] = token
            
            # Execute the function
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def verify_rule_check(func_name: str, token: Optional[str] = None) -> bool:
    """
    Verify that a rule check was performed for a function.
    
    Args:
        func_name: Name of the function to check
        token: Optional validation token
        
    Returns:
        bool: True if check was performed, False otherwise
    """
    return registry.verify_check(func_name, token)

def get_rules_for_function(func_name: str) -> List[Dict[str, Any]]:
    """
    Get the rules that were found for a function.
    
    Args:
        func_name: Name of the function
        
    Returns:
        List: The rules that were found
    """
    return registry.get_rules_for_function(func_name)

def clear_registry():
    """Clear the rule check registry."""
    registry.clear() 