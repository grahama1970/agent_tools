"""Tests for the function improver module."""

import pytest
from typing import NoReturn
from agent_tools.method_validator.function_improver import (
    validate_function,
    improve_function,
    ValidationResult
)

def test_validate_function_basic() -> None:
    """Test basic function validation."""
    # Test function with missing type hints and docstring
    func_text = """def hello(name):
    print(f"Hello {name}")
    return name.upper()
"""
    
    result = validate_function(func_text)
    
    assert isinstance(result, ValidationResult)
    assert not result.is_valid  # Function should be invalid due to missing type hints
    assert len(result.type_errors) > 0  # Should have type errors
    assert "type hints" in result.suggestions[0]  # Should suggest adding type hints
    assert not result.invalid_methods  # No invalid method calls

def test_validate_function_with_type_hints() -> None:
    """Test validation of function with type hints."""
    func_text = '''def hello(name: str) -> str:
    """Add a greeting."""
    print(f"Hello {name}")
    return name.upper()
'''
    
    result = validate_function(func_text)
    assert result.is_valid  # Function with type hints should be valid
    assert not result.type_errors  # No type errors
    assert not result.invalid_methods  # No invalid method calls

def test_validate_function_with_quality_issues() -> None:
    """Test validation of function with quality issues."""
    # Function with unused import and variable
    func_text = """def process_data():
    import sys
    x = 5
    return True
"""
    
    result = validate_function(func_text)
    assert result.quality_issues
    assert any('unused' in issue.lower() for issue in result.quality_issues)

def test_improve_function() -> None:
    """Test function improvement."""
    func_text = """def hello(name):
    print(f"Hello {name}")
    return name.upper()
"""
    
    improved = improve_function(func_text)
    
    # Check that type hints were added
    assert "name: Any" in improved
    assert "-> Any:" in improved
    assert "print(f\"Hello {name}\")" in improved  # Original code preserved 

def test_improve_problematic_function():
    """Test that improve_function handles problematic functions gracefully."""
    # Function with multiple issues
    problematic_func = '''
def process_data(data,filter):
    """process some data"""
    import json,sys,os;from datetime import datetime
    result=[];filtered=[]
    logger.log_info("Processing data...")
    data.process_items()
    try:
        return result
    except:
        logger.error_log("An error occurred")
        return None
'''
    
    # Should not raise an exception
    result = improve_function(problematic_func)
    
    # Should still return a string
    assert isinstance(result, str)
    
    # Should contain validation results
    assert "process_data" in result
    assert any("type hints" in s for s in validate_function(result).suggestions) 