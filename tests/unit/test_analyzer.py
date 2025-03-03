"""Unit tests for the method validator analyzer functionality."""

import pytest
from unittest.mock import patch, MagicMock, create_autospec
from typing import Any, Dict, List, Optional, Tuple, cast
from agent_tools.method_validator.analyzer import MethodAnalyzer, validate_method
import inspect

@pytest.fixture
def analyzer() -> MethodAnalyzer:
    """Create a method analyzer instance for testing."""
    return MethodAnalyzer()

def test_analyzer_initialization(analyzer: MethodAnalyzer) -> None:
    """Test analyzer initialization."""
    assert analyzer is not None
    assert hasattr(analyzer, 'quick_scan')
    assert hasattr(analyzer, 'deep_analyze')

def mock_function(*args: Any, **kwargs: Any) -> Any:
    """Mock function for testing."""
    return None

def create_mock_function(name: str, annotations: Dict[str, Any], doc: str) -> Any:
    """Create a mock function with proper signature."""
    def mock_fn(*args: Any, **kwargs: Any) -> Any:
        return None
    mock_fn.__name__ = name
    mock_fn.__doc__ = doc
    mock_fn.__module__ = 'json'
    mock_fn.__annotations__ = annotations
    # Create a signature from annotations
    params = []
    for param_name, param_type in annotations.items():
        if param_name != 'return':
            params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param_type,
                    default=inspect.Parameter.empty
                )
            )
    mock_fn.__signature__ = inspect.Signature(
        params,
        return_annotation=annotations.get('return', inspect.Signature.empty)
    )
    return mock_fn

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_standard_lib(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test analyzing a method from the standard library."""
    # Mock the json module
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={'obj': Any, 'return': str},
        doc="""Serialize obj to a JSON formatted str."""
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json
    
    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "parameters" in result_dict
    assert "obj" in result_dict["parameters"]
    assert result_dict["parameters"]["obj"]["required"]

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_quick_scan_results(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test quick scan of a package."""
    # Mock the json module with some methods
    mock_json = MagicMock()
    mock_json.__all__ = ['dumps', 'loads', 'dump', 'load']
    
    # Set up dumps method
    mock_dumps = create_autospec(mock_function)
    mock_dumps.__doc__ = """Serialize obj to JSON."""
    mock_dumps.__name__ = 'dumps'
    mock_dumps.__module__ = 'json'
    mock_dumps.__annotations__ = {
        'obj': Any,
        'skipkeys': bool,
        'ensure_ascii': bool,
        'check_circular': bool,
        'allow_nan': bool,
        'cls': Optional[Any],
        'indent': Optional[int],
        'separators': Optional[Tuple[str, str]],
        'default': Optional[Any],
        'sort_keys': bool,
        'return': str,
    }
    mock_json.dumps = mock_dumps
    
    # Set up loads method
    mock_loads = create_autospec(mock_function)
    mock_loads.__doc__ = """Parse JSON string."""
    mock_loads.__name__ = 'loads'
    mock_loads.__module__ = 'json'
    mock_loads.__annotations__ = {
        's': str,
        'return': Any
    }
    mock_json.loads = mock_loads
    
    mock_import.return_value = mock_json
    
    results = analyzer.quick_scan("json")
    assert results is not None
    assert len(results) >= 2  # At least dumps and loads
    assert any(name == "dumps" for name, _, _ in results)
    assert any(name == "loads" for name, _, _ in results)

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_method_parameters(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test parameter analysis of a method."""
    # Mock the json module with dumps method
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={
            'obj': Any,
            'skipkeys': bool,
            'ensure_ascii': bool,
            'check_circular': bool,
            'allow_nan': bool,
            'cls': Optional[Any],
            'indent': Optional[int],
            'separators': Optional[Tuple[str, str]],
            'default': Optional[Any],
            'sort_keys': bool,
            'return': str,
        },
        doc="""Serialize obj to JSON."""
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json

    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "parameters" in result_dict
    assert len(result_dict["parameters"]) > 0
    assert "obj" in result_dict["parameters"]
    assert result_dict["parameters"]["obj"]["required"]

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_method_exceptions(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test exception analysis of a method."""
    # Mock the json module with loads method
    mock_json = MagicMock()
    mock_loads = create_mock_function(
        name='loads',
        annotations={'s': str, 'return': Any},
        doc="""Parse JSON string.
    
        Raises:
            ValueError: If the string is not valid JSON
            TypeError: If the input is not a string
        """
    )
    mock_json.loads = mock_loads
    mock_import.return_value = mock_json

    result = analyzer.deep_analyze("json", "loads")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "exceptions" in result_dict
    assert len(result_dict["exceptions"]) >= 2
    assert any(e["type"] == "ValueError" for e in result_dict["exceptions"])
    assert any(e["type"] == "TypeError" for e in result_dict["exceptions"])

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_method_return_info(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test return info analysis of a method."""
    # Mock the json module with dumps method
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={'obj': Any, 'return': str},
        doc="""Serialize obj to JSON.
    
        Returns:
            str: The JSON string representation
        """
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json

    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "return_info" in result_dict
    assert result_dict["return_info"]["type"] == "str"
    assert result_dict["return_info"]["description"] is not None

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_method_examples(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test examples extraction from a method."""
    # Mock the json module with dumps method
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={'obj': Any, 'return': str},
        doc="""Serialize obj to JSON.
    
        Examples:
            >>> dumps({"key": "value"})
            '{"key": "value"}'
        """
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json

    result = analyzer.deep_analyze("json", "dumps")
    assert result is not None
    result_dict = cast(Dict[str, Any], result)
    assert isinstance(result_dict, dict)
    assert "examples" in result_dict
    assert len(result_dict["examples"]) > 0
    assert '{"key": "value"}' in result_dict["examples"][0]

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_invalid_method(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test analyzing a non-existent method."""
    mock_json = MagicMock()
    mock_json.nonexistent_method = None
    mock_import.return_value = mock_json
    
    result = analyzer.deep_analyze("json", "nonexistent_method")
    assert result is None

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_analyze_invalid_package(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test analyzing a method from a non-existent package."""
    try:
        mock_import.side_effect = ModuleNotFoundError
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

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_module_caching(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test that module imports are cached."""
    # Mock the json module
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={'obj': Any, 'return': str},
        doc="""Serialize obj to JSON."""
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json

    # First import
    result1 = analyzer.deep_analyze("json", "dumps")
    assert result1 is not None

    # Second import should use cache
    result2 = analyzer.deep_analyze("json", "dumps")
    assert result2 is not None

    # Import should only be called once
    assert mock_import.call_count == 1

def test_method_discovery(analyzer: MethodAnalyzer) -> None:
    """Test method discovery in a package."""
    results = analyzer.quick_scan("json")
    method_names = [name for name, _, _ in results]
    
    # Check common JSON methods are found
    assert "dumps" in method_names
    assert "loads" in method_names
    assert "dump" in method_names
    assert "load" in method_names

@patch('agent_tools.method_validator.analyzer.importlib.import_module')
def test_method_analysis_caching(mock_import: MagicMock, analyzer: MethodAnalyzer) -> None:
    """Test that method analysis results are cached."""
    # Mock the json module
    mock_json = MagicMock()
    mock_dumps = create_mock_function(
        name='dumps',
        annotations={'obj': Any, 'return': str},
        doc="""Serialize obj to JSON."""
    )
    mock_json.dumps = mock_dumps
    mock_import.return_value = mock_json

    # First analysis
    result1 = analyzer.deep_analyze("json", "dumps")
    assert result1 is not None

    # Second analysis should use cache
    result2 = analyzer.deep_analyze("json", "dumps")
    assert result2 is not None

    # Import should only be called once
    assert mock_import.call_count == 1 