import json
from click.testing import CliRunner
from agent_tools.method_validator.cli import main

# Create a test runner
runner = CliRunner()

def test_list_methods_command() -> None:
    """Test the list-methods command."""
    result = runner.invoke(main, ["requests", "--list-all"])
    assert result.exit_code == 0
    print(f"Full output: {result.stdout}")

    # Parse the entire output as JSON
    try:
        json_output = json.loads(result.stdout)
        assert isinstance(json_output, list)
        assert len(json_output) > 0
        # Verify that get method is in the list (a common requests method)
        assert any(method[0] == "get" for method in json_output)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"JSON content attempted to parse: {result.stdout}")
        assert False, f"Failed to parse JSON output: {e}"

def test_analyze_method_command() -> None:
    """Test the analyze-method command."""
    result = runner.invoke(main, ["requests", "--method", "get"])
    assert result.exit_code == 0
    print(f"Full output: {result.stdout}")
    
    # Parse the entire output as JSON
    try:
        json_output = json.loads(result.stdout)
        assert isinstance(json_output, dict)
        # Verify that the method info contains parameters and return info
        assert "parameters" in json_output
        assert "return_info" in json_output
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        assert False, f"Failed to parse JSON output: {e}"

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
        result = runner.invoke(main, ["--script", "test_script.py"])
        assert result.exit_code == 0
        # Verify that tensorflow and torch are found
        assert "tensorflow" in result.stdout
        assert "torch" in result.stdout
        # numpy should not be listed as it's a common utility
        assert "numpy" not in result.stdout