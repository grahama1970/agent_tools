"""
Smoke tests for DuaLipa compliance with best practices.

These tests verify basic functionality and compliance with project standards:
1. All modules have proper imports
2. All modules have demo functions
3. Core functionality works properly

Official Documentation References:
- pytest: https://docs.pytest.org/en/stable/
- loguru: https://loguru.readthedocs.io/en/stable/
"""

import os
import sys
import importlib
import inspect
import pytest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import asyncio

# Get the parent directory to import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))

# Import DuaLipa modules to test
try:
    from agent_tools.dualipa import __version__
    from agent_tools.dualipa import code_extractor
    from agent_tools.dualipa import format_dataset
    from agent_tools.dualipa import github_utils
    from agent_tools.dualipa import language_detection
    from agent_tools.dualipa import llm_generator
    from agent_tools.dualipa import markdown_parser
    from agent_tools.dualipa import qa_validator
    from agent_tools.dualipa import pipeline
except ImportError as e:
    pytest.skip(f"Failed to import DuaLipa modules: {e}", allow_module_level=True)

# Core modules that should be present
CORE_MODULES = [
    "code_extractor",
    "format_dataset",
    "github_utils",
    "language_detection",
    "llm_generator", 
    "markdown_parser",
    "qa_validator",
    "pipeline"
]

# Required function patterns for all modules
REQUIRED_PATTERNS = [
    "demo_",            # Demo functions
]

# Optional function patterns (not all modules will have these)
OPTIONAL_PATTERNS = [
    "extract_",         # Extraction functions  
    "generate_",        # Generation functions
    "validate_",        # Validation functions
    "parse_",           # Parsing functions
    "detect_",          # Detection functions
]


def test_version():
    """Test that package has a version defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_module_imports():
    """Test that all core modules can be imported."""
    for module_name in CORE_MODULES:
        module_path = f"agent_tools.dualipa.{module_name}"
        try:
            module = importlib.import_module(module_path)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {module_path}: {e}")


def test_modules_have_demo_functions():
    """Test that all modules have demo functions."""
    module_demo_map = {}
    
    for module_name in CORE_MODULES:
        module = globals()[module_name]
        demo_functions = []
        
        for name, obj in inspect.getmembers(module):
            if (name.startswith("demo_") and inspect.isfunction(obj) and 
                obj.__module__ == f"agent_tools.dualipa.{module_name}"):
                demo_functions.append(name)
        
        module_demo_map[module_name] = demo_functions
        assert len(demo_functions) > 0, f"Module {module_name} should have at least one demo function"
    
    # Print summary of demo functions for reference
    print("\nDemo functions found:")
    for module, functions in module_demo_map.items():
        print(f"  {module}: {', '.join(functions)}")


def test_modules_have_main_block():
    """Test that all modules have an if __name__ == "__main__" block."""
    for module_name in CORE_MODULES:
        # Check if module source code contains if __name__ == "__main__"
        module = globals()[module_name]
        module_path = inspect.getfile(module)
        
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        assert 'if __name__ == "__main__"' in source, f"Module {module_name} should have a main block"


def test_modules_have_docstrings():
    """Test that all modules and their public functions have docstrings."""
    for module_name in CORE_MODULES:
        module = globals()[module_name]
        
        # Check module docstring
        assert module.__doc__ is not None, f"Module {module_name} should have a docstring"
        
        # Check public function docstrings
        for name, obj in inspect.getmembers(module):
            if (inspect.isfunction(obj) and 
                not name.startswith("_") and 
                obj.__module__ == f"agent_tools.dualipa.{module_name}"):
                assert obj.__doc__ is not None, f"Function {module_name}.{name} should have a docstring"


def test_modules_have_required_functions():
    """Test that all modules have at least one function matching each required pattern."""
    for module_name in CORE_MODULES:
        module = globals()[module_name]
        
        for pattern in REQUIRED_PATTERNS:
            pattern_found = False
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) and 
                    name.startswith(pattern) and 
                    obj.__module__ == f"agent_tools.dualipa.{module_name}"):
                    pattern_found = True
                    break
            
            assert pattern_found, f"Module {module_name} should have at least one function matching pattern '{pattern}'"


def test_code_extractor_basic_functionality():
    """Test basic functionality of code_extractor module."""
    # Verify extract_repository function exists and has expected parameters
    assert hasattr(code_extractor, "extract_repository")
    
    # Check signature of extract_repository
    sig = inspect.signature(code_extractor.extract_repository)
    params = sig.parameters
    
    assert "source" in params
    assert "output_path" in params
    assert "max_files" in params


def test_language_detection_basic_functionality():
    """Test basic functionality of language_detection module."""
    # Check that detect_language function exists
    assert hasattr(language_detection, "detect_language")
    
    # Test detection of a few common languages
    test_cases = [
        ("test.py", "python"),
        ("test.js", "javascript"),
        ("test.html", "html"),
        ("test.md", "markdown"),
    ]
    
    for filename, expected_language in test_cases:
        detected = language_detection.detect_language(filename)
        assert detected == expected_language, f"Expected {filename} to be detected as {expected_language}, got {detected}"


def test_github_utils_basic_functionality():
    """Test basic functionality of github_utils module."""
    # Check for key functions
    assert hasattr(github_utils, "is_github_url")
    
    # Test URL detection
    assert github_utils.is_github_url("https://github.com/username/repo")
    assert not github_utils.is_github_url("https://example.com")


def test_markdown_parser_basic_functionality():
    """Test basic functionality of markdown_parser module."""
    # Check that extract_code_blocks function exists
    assert hasattr(markdown_parser, "extract_code_blocks")
    
    # Test extraction of code blocks
    test_md = """# Test
    
```python
def test_function():
    return "test"
```

```javascript
function testFunction() {
    return "test";
}
```
"""
    
    code_blocks = markdown_parser.extract_code_blocks(test_md)
    assert len(code_blocks) == 2
    assert "python" in code_blocks
    assert "javascript" in code_blocks


def test_qa_validator_basic_functionality():
    """Test basic functionality of qa_validator module."""
    # Check that validate_and_enhance_qa_pairs function exists
    assert hasattr(qa_validator, "validate_and_enhance_qa_pairs")
    
    # Test basic validation
    test_pairs = [
        {"question": "What is the purpose of function X?", "answer": "Function X does Y."},
        {"question": "How does function X work?", "answer": "It works by doing Y step by step."}
    ]
    
    # Create a sync wrapper to call the async function
    def sync_validate(pairs, content="", function_name=None):
        """Sync wrapper for async validate_and_enhance_qa_pairs"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            qa_validator.validate_and_enhance_qa_pairs(
                pairs, content, function_name
            )
        )
    
    validated = sync_validate(test_pairs)
    assert len(validated) <= len(test_pairs)  # Should have same or fewer pairs after validation


def test_llm_generator_config():
    """Test basic configuration of llm_generator module."""
    # Check that key functions exist
    assert hasattr(llm_generator, "check_litellm_available")
    assert hasattr(llm_generator, "generate_code_related_qa_pairs")
    assert hasattr(llm_generator, "generate_markdown_related_qa_pairs")
    assert hasattr(llm_generator, "generate_reverse_qa_pairs")
    
    # Check litellm availability (just verify the function runs, not the result)
    try:
        llm_generator.check_litellm_available()
    except Exception as e:
        pytest.fail(f"check_litellm_available should not raise exceptions: {e}")


def test_format_dataset_basic_functionality():
    """Test basic functionality of format_dataset module."""
    # Check that format_for_lora function exists
    assert hasattr(format_dataset, "format_for_lora")
    
    # Check that generate_basic_code_qa_pairs function exists
    assert hasattr(format_dataset, "generate_basic_code_qa_pairs")


def test_pipeline_basic_functionality():
    """Test basic functionality of pipeline module."""
    # Check that Pipeline class exists
    assert hasattr(pipeline, "Pipeline")
    
    # Check that run_pipeline function exists
    assert hasattr(pipeline, "run_pipeline")
    
    # Check that demo_pipeline function exists
    assert hasattr(pipeline, "demo_pipeline")
    
    # Check Pipeline class has expected methods
    pipeline_class = pipeline.Pipeline
    assert hasattr(pipeline_class, "run")
    assert inspect.iscoroutinefunction(pipeline_class.run), "Pipeline.run should be an async function"
    
    # Check run_pipeline function parameters
    run_pipeline_params = inspect.signature(pipeline.run_pipeline).parameters
    assert "repo_path" in run_pipeline_params
    assert "output_dir" in run_pipeline_params
    assert "extract_kwargs" in run_pipeline_params
    assert "format_kwargs" in run_pipeline_params
    assert "train_kwargs" in run_pipeline_params


if __name__ == "__main__":
    # Allow running this file directly (not through pytest)
    print("Running smoke tests for DuaLipa...")
    
    # Run tests
    test_version()
    test_module_imports()
    test_modules_have_demo_functions()
    test_modules_have_main_block()
    test_modules_have_docstrings()
    test_modules_have_required_functions()
    test_code_extractor_basic_functionality()
    test_language_detection_basic_functionality()
    test_github_utils_basic_functionality()
    test_markdown_parser_basic_functionality()
    test_qa_validator_basic_functionality()
    test_llm_generator_config()
    test_format_dataset_basic_functionality()
    test_pipeline_basic_functionality()
    
    print("All smoke tests passed!")