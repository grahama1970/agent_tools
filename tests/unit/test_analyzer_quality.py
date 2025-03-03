"""Tests for code quality and type checking functionality in analyzer.py."""

import pytest
from typing import Any, Dict, List
from agent_tools.method_validator.analyzer import MethodInfo

def test_function_no_type_hints() -> None:
    """Test analyzing a function without type hints."""
    def sample_function(x, y):  # Intentionally no type hints
        return x + y
    
    info = MethodInfo(sample_function, "sample_function")
    type_info = info.type_info
    
    print("\nType errors:", type_info["errors"])  # Debug output
    print("Type suggestions:", type_info["suggestions"])  # Debug output
    
    assert len(type_info["errors"]) > 0
    assert any("Missing type annotation" in error for error in type_info["errors"])
    assert "Add type hints" in type_info["suggestions"][0]

def test_function_with_type_hints() -> None:
    """Test analyzing a function with proper type hints."""
    def sample_function(x: int, y: int) -> int:
        return x + y
    
    info = MethodInfo(sample_function, "sample_function")
    type_info = info.type_info
    
    assert len(type_info["errors"]) == 0
    assert len(type_info["suggestions"]) == 0

def test_code_quality_issues() -> None:
    """Test detecting code quality issues."""
    def poorly_formatted_function() -> int:
        x=1;y=2 # Multiple statements, no spaces
        if x==y:z=3 # Poor formatting
        return z
    
    info = MethodInfo(poorly_formatted_function, "poorly_formatted_function")
    quality_info = info.quality_info
    
    print("\nQuality issues:", quality_info["issues"])  # Debug output
    print("Quality suggestions:", quality_info["suggestions"])  # Debug output
    
    assert len(quality_info["issues"]) > 0
    assert any("=" in issue or ";" in issue or "whitespace" in issue.lower() 
              for issue in quality_info["issues"])
    assert any("Fix code formatting" in suggestion 
              for suggestion in quality_info["suggestions"])

def test_well_formatted_code() -> None:
    """Test analyzing well-formatted code."""
    def well_formatted_function(x: int, y: int) -> int:
        """Add two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Sum of x and y
        """
        result = x + y
        return result
    
    info = MethodInfo(well_formatted_function, "well_formatted_function")
    quality_info = info.quality_info
    
    # Check for actual issues in the output
    print("Quality issues:", quality_info["issues"])  # Debug output
    assert all(not issue.strip().startswith("error:") for issue in quality_info["issues"])
    assert not any("=" in issue or ";" in issue or "whitespace" in issue.lower() 
                  for issue in quality_info["issues"])

def test_method_info_complete_analysis() -> None:
    """Test that MethodInfo performs complete analysis including quality and types."""
    def test_function(x: int, y: str) -> Dict[str, Any]:
        result = {"x": x, "y": y}
        return result
    
    info = MethodInfo(test_function, "test_function")
    
    # Check that all analysis fields are present
    assert hasattr(info, "quality_info")
    assert hasattr(info, "type_info")
    
    # Verify dictionary structure
    assert isinstance(info.to_dict(), dict)
    assert "quality_info" in info.to_dict()
    assert "type_info" in info.to_dict()
    
    # Check dictionary content types
    assert isinstance(info.to_dict()["quality_info"], dict)
    assert isinstance(info.to_dict()["type_info"], dict)
    assert isinstance(info.to_dict()["quality_info"]["issues"], list)
    assert isinstance(info.to_dict()["type_info"]["errors"], list) 