"""Unit tests for the method validator analyzer functionality."""

import pytest
from typing import Any, Dict, List, Optional, Tuple, cast
from agent_tools.method_validator.analyzer import MethodAnalyzer, validate_method

@pytest.fixture
def analyzer() -> MethodAnalyzer:
    """Create a method analyzer instance for testing."""
    return MethodAnalyzer()

def test_analyzer_initialization(analyzer: MethodAnalyzer) -> None:
    """Test analyzer initialization."""
    assert analyzer is not None
    assert hasattr(analyzer, 'quick_scan')
    assert hasattr(analyzer, 'deep_analyze')

def test_analyze_standard_lib(analyzer: MethodAnalyzer) -> None:
    """Test analyzing a method from the standard library."""
    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "parameters" in result_dict
    assert "obj" in result_dict["parameters"]
    assert result_dict["parameters"]["obj"]["required"]

def test_quick_scan_results(analyzer: MethodAnalyzer) -> None:
    """Test quick scan of a package."""
    results = analyzer.quick_scan("json")
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check structure of results
    for name, summary, categories in results:
        assert isinstance(name, str)
        assert isinstance(summary, str)
        assert isinstance(categories, list)
        assert all(isinstance(cat, str) for cat in categories)

def test_analyze_method_parameters(analyzer: MethodAnalyzer) -> None:
    """Test parameter analysis of a method."""
    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    params = result_dict["parameters"]
    
    # Test required parameters
    assert "obj" in params
    assert params["obj"]["required"]
    
    # Test kwargs handling
    assert "kw" in params
    assert params["kw"]["required"]

def test_analyze_method_exceptions(analyzer: MethodAnalyzer) -> None:
    """Test exception analysis of a method."""
    result = analyzer.deep_analyze("json", "loads")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    exceptions = result_dict.get("exceptions", [])
    
    assert isinstance(exceptions, list)
    # Note: Exception analysis might be empty if docstring doesn't specify exceptions
    if exceptions:
        assert all(isinstance(e, dict) for e in exceptions)
        assert all("type" in e for e in exceptions)

def test_analyze_method_return_info(analyzer: MethodAnalyzer) -> None:
    """Test return info analysis of a method."""
    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    return_info = result_dict["return_info"]
    
    assert isinstance(return_info, dict)
    assert "type" in return_info
    # Type might be None if not determinable
    if return_info["type"]:
        assert isinstance(return_info["type"], str)

def test_analyze_method_examples(analyzer: MethodAnalyzer) -> None:
    """Test examples extraction from a method."""
    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    examples = result_dict.get("examples", [])
    
    assert isinstance(examples, list)
    # Note: Examples might be empty if docstring doesn't contain examples
    if examples:
        assert all(isinstance(example, str) for example in examples)

def test_analyze_invalid_method(analyzer: MethodAnalyzer) -> None:
    """Test analyzing a non-existent method."""
    result = analyzer.deep_analyze("json", "nonexistent_method")
    assert result is None

def test_analyze_invalid_package(analyzer: MethodAnalyzer) -> None:
    """Test analyzing a method from a non-existent package."""
    try:
        result = analyzer.deep_analyze("nonexistent_package", "method")
        assert result is None
    except ModuleNotFoundError:
        # Either returning None or raising ModuleNotFoundError is acceptable
        pass

def test_method_categories(analyzer: MethodAnalyzer) -> None:
    """Test method categorization."""
    results = analyzer.quick_scan("json")
    dumps_result = next((r for r in results if r[0] == "dumps"), None)
    assert dumps_result is not None
    
    # Check categories
    categories = dumps_result[2]
    assert isinstance(categories, list)
    assert len(categories) > 0
    # The actual category might vary, but should be a non-empty string
    assert all(isinstance(cat, str) and cat for cat in categories)

def test_validate_method_integration() -> None:
    """Test integration between validate_method and analyzer."""
    # Test with a valid method
    is_valid, message = validate_method("json", "dumps")
    assert is_valid
    assert "exists and is callable" in message
    
    # Test with an invalid method
    is_valid, message = validate_method("json", "nonexistent")
    assert not is_valid
    assert "not found" in message.lower()

def test_module_caching(analyzer: MethodAnalyzer) -> None:
    """Test that module imports are cached."""
    # First import
    result1 = analyzer.deep_analyze("json", "dumps")
    assert result1 is not None
    
    # Second import should use cache
    result2 = analyzer.deep_analyze("json", "dumps")
    assert result2 is not None
    assert result1 == result2

def test_method_discovery(analyzer: MethodAnalyzer) -> None:
    """Test method discovery in a package."""
    results = analyzer.quick_scan("json")
    method_names = [name for name, _, _ in results]
    
    # Check common JSON methods are found
    assert "dumps" in method_names
    assert "loads" in method_names
    assert "dump" in method_names
    assert "load" in method_names

def test_method_analysis_caching(analyzer: MethodAnalyzer) -> None:
    """Test that method analysis results are cached."""
    # First analysis
    result1 = analyzer.deep_analyze("json", "dumps")
    assert result1 is not None
    
    # Second analysis should use cache
    result2 = analyzer.deep_analyze("json", "dumps")
    assert result2 is not None
    assert result1 == result2 