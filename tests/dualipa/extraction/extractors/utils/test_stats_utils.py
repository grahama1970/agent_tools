#!/usr/bin/env python3
"""Tests for stats dictionary consistency across extraction methods."""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import json
import inspect
import importlib

# Configure paths properly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Testing agent_tools availability...")

# Check if the module is available
try:
    agent_tools_spec = importlib.util.find_spec("agent_tools")
    if agent_tools_spec:
        print(f"agent_tools found at: {agent_tools_spec.origin}")
        print(f"agent_tools submodule locations: {agent_tools_spec.submodule_search_locations}")
    else:
        print("agent_tools module not found")
except ImportError:
    print("Error checking for agent_tools module")

# Try more direct imports
try:
    # Import extraction modules directly
    from agent_tools.dualipa.extraction.extractors.code.code_extractor import (
        extract_code_blocks,
        extract_python_blocks,
        extract_js_ts_blocks,
        extract_generic_blocks,
        validate_block,
        verify_block
    )
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
    print("Successfully imported extraction functions")
except ImportError as e:
    print(f"Error importing from agent_tools: {e}")
    import traceback
    print("Traceback:")
    print(traceback.format_exc())
    pytest.fail("Required extraction modules not available")

# Import the function from multiple modules
try:
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats as stats_utils_init
    from agent_tools.dualipa.extraction.extractors.code.code_extractor import init_stats as code_extractor_init
    from agent_tools.dualipa.extraction.extractors.github.repo_utils import init_stats as repo_utils_init
    from agent_tools.dualipa.extraction.extractors.code.hierarchy import init_stats as code_hierarchy_init
except ImportError as e:
    pytest.fail(f"Failed to import init_stats from required modules: {e}")

@pytest.fixture
def python_sample():
    """Sample Python code fixture."""
    return """
def hello_world():
    """'"""Simple function that returns a greeting."""'"""
    return "Hello, world!"

class SampleClass:
    """'"""A sample class with methods."""'"""
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        return f"Hello, {self.name}!"
"""

@pytest.fixture
def javascript_sample():
    """Sample JavaScript code fixture."""
    return """
/**
 * Simple function that returns a greeting.
 */
function helloWorld() {
    return "Hello, world!";
}

/**
 * A sample class for greeting.
 */
class Greeter {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return `Hello, ${this.name}!`;
    }
}

// Export the function and class
module.exports = {
    helloWorld,
    Greeter
};
"""

@pytest.fixture
def markdown_sample():
    """Sample Markdown code fixture."""
    return """
# Sample Markdown

This is a sample markdown file with code blocks.

## Python Example

```python
def hello_world():
    return "Hello, world!"
```

## JavaScript Example

```javascript
function helloWorld() {
    return "Hello, world!";
}
```
"""

def verify_stats_fields(stats):
    """Verify that the stats dictionary has all the required fields with correct types."""
    # Required fields and their expected types
    required_fields = {
        # Source and output information
        "source": str,
        "repo_url": str,
        "output_path": str,
        
        # Timing information
        "start_time": str,
        "end_time": (str, type(None)),  # Can be None before extraction completes
        "duration_seconds": (int, float),
        
        # File and block counts
        "total_files": int,
        "documentation_files": int,
        "code_files": int,
        "code_blocks": int,
        "doc_blocks": int,
        "skipped_files": int,
        "error_files": int,
        
        # Categorization
        "languages": dict,
        "file_types": dict,
        
        # Error tracking
        "errors": list,
        
        # Block storage
        "file_blocks": dict
    }
    
    # Verify all fields exist with correct types
    for field, expected_type in required_fields.items():
        assert field in stats, f"Missing required field '{field}' in stats dictionary"
        
        # Handle fields that can be multiple types
        if isinstance(expected_type, tuple):
            assert isinstance(stats[field], expected_type), f"Field '{field}' has incorrect type: {type(stats[field])} (expected one of {expected_type})"
        else:
            assert isinstance(stats[field], expected_type), f"Field '{field}' has incorrect type: {type(stats[field])} (expected {expected_type})"
    
    return True

def test_stats_consistency_across_extraction_methods(python_sample, javascript_sample, markdown_sample):
    """Test that all extraction methods update the stats dictionary consistently."""
    if not HAS_DEPENDENCIES:
        pytest.fail("Required dependencies not available for testing stats consistency")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Test Python extraction
        python_file = output_dir / "sample.py"
        with open(python_file, "w") as f:
            f.write(python_sample)
        
        python_stats = initialize_stats_dict(source=python_file, output_dir=output_dir)
        _extract_python_blocks(python_file, python_sample, output_dir, python_stats)
        
        # Test JavaScript extraction
        js_file = output_dir / "sample.js"
        with open(js_file, "w") as f:
            f.write(javascript_sample)
        
        js_stats = initialize_stats_dict(source=js_file, output_dir=output_dir)
        _extract_js_ts_blocks(js_file, javascript_sample, output_dir, js_stats, language="javascript")
        
        # Test Markdown extraction
        md_file = output_dir / "sample.md"
        with open(md_file, "w") as f:
            f.write(markdown_sample)
        
        md_stats = initialize_stats_dict(source=md_file, output_dir=output_dir)
        _extract_markdown_blocks(md_file, markdown_sample, output_dir, md_stats)
        
        # Verify each stats dictionary has all required fields
        print("Verifying Python extraction stats consistency...")
        assert verify_stats_fields(python_stats), "Python extraction stats has inconsistent structure"
        
        print("Verifying JavaScript extraction stats consistency...")
        assert verify_stats_fields(js_stats), "JavaScript extraction stats has inconsistent structure"
        
        print("Verifying Markdown extraction stats consistency...")
        assert verify_stats_fields(md_stats), "Markdown extraction stats has inconsistent structure"
        
        # Verify that fields are updated correctly in each extraction
        assert python_stats["code_blocks"] > 0, "Python extraction should update code_blocks count"
        assert python_stats["file_blocks"], "Python extraction should update file_blocks dictionary"
        
        assert js_stats["code_blocks"] > 0, "JavaScript extraction should update code_blocks count"
        assert js_stats["file_blocks"], "JavaScript extraction should update file_blocks dictionary"
        
        assert md_stats["doc_blocks"] > 0, "Markdown extraction should update doc_blocks count"
        assert md_stats["file_blocks"], "Markdown extraction should update file_blocks dictionary"
        
        # Verify languages are tracked correctly
        assert "python" in python_stats["languages"], "Python extraction should update languages dictionary"
        assert "javascript" in js_stats["languages"], "JavaScript extraction should update languages dictionary"
        assert "markdown" in md_stats["languages"], "Markdown extraction should update languages dictionary"
        
        # Verify file_types are tracked correctly
        assert ".py" in python_stats["file_types"], "Python extraction should update file_types dictionary"
        assert ".js" in js_stats["file_types"], "JavaScript extraction should update file_types dictionary"
        assert ".md" in md_stats["file_types"], "Markdown extraction should update file_types dictionary"
        
        print("All extraction methods update the stats dictionary consistently")

def test_initialize_stats_dict_consistency():
    """Test that initialize_stats_dict is used consistently across all modules."""
    if not HAS_DEPENDENCIES:
        pytest.fail("Required dependencies not available for testing stats consistency")
    
    # Import the function from multiple modules
    try:
        from agent_tools.dualipa.extraction.extractors.code.code_extractor import initialize_stats_dict as code_extractor_init
        from agent_tools.dualipa.extraction.extractors.github.repo_utils import initialize_stats_dict as extract_repo_init
        
        # Try importing from other modules that may use it
        try:
            from agent_tools.dualipa.extraction.extractors.code.hierarchy import initialize_stats_dict as code_hierarchy_init
            modules_with_init = [
                ("code_extractor", code_extractor_init),
                ("extract_repo", extract_repo_init),
                ("code_hierarchy", code_hierarchy_init)
            ]
        except ImportError:
            modules_with_init = [
                ("code_extractor", code_extractor_init),
                ("extract_repo", extract_repo_init)
            ]
    except ImportError as e:
        pytest.fail(f"Failed to import initialize_stats_dict from required modules: {e}")
    
    # Verify all modules use the same function
    for i, (module_name, init_func) in enumerate(modules_with_init):
        # Skip first module (it's our reference)
        if i == 0:
            continue
            
        # Check that the function is the same object
        assert init_func is modules_with_init[0][1], f"{module_name} is not using the same initialize_stats_dict function as code_extractor"
    
    # Verify that the function is called with proper arguments
    # Create stats dictionary with each function
    for module_name, init_func in modules_with_init:
        stats = init_func(source="test_source", output_dir=Path("/tmp"))
        assert verify_stats_fields(stats), f"{module_name}.initialize_stats_dict does not produce a valid stats dictionary"
    
    print("All modules use the same initialize_stats_dict function consistently")

if __name__ == "__main__":
    # Run the tests directly if executed as a script
    pytest.main(["-xvs", __file__]) 