"""Tests for the CLI module."""

from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from agent_tools.method_validator.cli import cli
from agent_tools.method_validator.function_improver import ValidationResult

runner = CliRunner()

def test_list_methods_command() -> None:
    """Test the list-methods command."""
    result = runner.invoke(cli, ["validate", "requests", "--list-all"])
    assert result.exit_code == 0
    assert "get" in result.stdout
    # numpy should not be listed as it's a common utility
    assert "numpy" not in result.stdout

def test_analyze_method_command() -> None:
    """Test the analyze-method command."""
    result = runner.invoke(cli, ["validate", "requests", "--method", "get"])
    assert result.exit_code == 0
    assert "get" in result.stdout

def test_script_analysis() -> None:
    """Test the script analysis command."""
    # Create a test script with non-standard library imports
    script_content = """import tensorflow as tf
import torch
import numpy as np

def main():
    x = tf.constant(1)
    y = torch.tensor(2)
    z = np.array([3])
    return x + y + z
"""
    with runner.isolated_filesystem():
        with open("test_script.py", "w") as f:
            f.write(script_content)
        result = runner.invoke(cli, ["validate", "--script", "test_script.py"])
        assert result.exit_code == 0
        assert "tensorflow" in result.stdout
        assert "torch" in result.stdout
        assert "numpy" not in result.stdout  # numpy is a common utility

def test_check_function(tmp_path: str) -> None:
    """Test the check-function command."""
    # Create a test function file
    func_file = tmp_path / "test_func.py"
    func_file.write_text("""def hello(name):
    print(f"Hello {name}")
    return name.upper()
""")
    
    # Mock validation results
    invalid_result = ValidationResult(
        is_valid=False,
        type_errors=["Missing type annotation for 'name'"],
        quality_issues=[],
        suggestions=["Add type hints"],
        invalid_methods=[],
        alternative_methods={}
    )
    
    valid_result = ValidationResult(
        is_valid=True,
        type_errors=[],
        quality_issues=[],
        suggestions=[],
        invalid_methods=[],
        alternative_methods={}
    )
    
    with patch('agent_tools.method_validator.function_improver.validate_function') as mock_validate, \
         patch('agent_tools.method_validator.function_improver.improve_function') as mock_improve:
        
        # First test - invalid function
        mock_validate.return_value = invalid_result
        mock_improve.return_value = """def hello(name: str) -> str:
    print(f"Hello {name}")
    return name.upper()
"""
        result = runner.invoke(cli, ['check-function', '--file', str(func_file)])
        assert result.exit_code == 1
        assert "Type Issues" in result.output
        assert "Missing type annotation" in result.output
        
        # Second test - valid function
        mock_validate.return_value = valid_result
        result = runner.invoke(cli, ['check-function'], input="""def greet(name: str) -> str:
    \"\"\"Greet someone.\"\"\"
    return f"Hello {name}"
""")
        assert result.exit_code == 0
        assert "No issues found" in result.output
        
        # Third test - with apply flag
        mock_validate.return_value = invalid_result
        result = runner.invoke(cli, ['check-function', '--file', str(func_file), '--apply'])
        assert result.exit_code == 1
        assert "Suggested Fix" in result.output