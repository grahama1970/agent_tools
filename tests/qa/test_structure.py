"""Test project structure existence.

This module tests that the project structure exists as expected.
Test first, implementation second following TDD approach.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

import os
import pathlib
import pytest


def test_structure_exists():
    """Test that the project structure exists as expected.
    
    This test verifies that all required directories and files exist.
    It is part of Task 1.1 in task.md.
    """
    # Base path for the project
    base_path = pathlib.Path("/home/grahama/workspace/experiments/agent_tools")
    qa_path = base_path / "src" / "agent_tools" / "dualipa" / "qa"
    
    # Check main directories
    assert (qa_path).exists(), "QA module directory doesn't exist"
    assert (qa_path / "models").exists(), "Models directory doesn't exist"
    assert (qa_path / "utils").exists(), "Utils directory doesn't exist"
    assert (qa_path / "llm").exists(), "LLM directory doesn't exist"
    assert (qa_path / "docs").exists(), "Docs directory doesn't exist"
    
    # Check key files
    assert (qa_path / "__init__.py").exists(), "QA module __init__.py doesn't exist"
    assert (qa_path / "processor.py").exists(), "processor.py doesn't exist"
    assert (qa_path / "models" / "qa_models.py").exists(), "qa_models.py doesn't exist"
    assert (qa_path / "models" / "__init__.py").exists(), "models/__init__.py doesn't exist"
    assert (qa_path / "utils" / "__init__.py").exists(), "utils/__init__.py doesn't exist"
    assert (qa_path / "utils" / "method_validator.py").exists(), "method_validator.py doesn't exist"
    assert (qa_path / "utils" / "security.py").exists(), "security.py doesn't exist"
    assert (qa_path / "utils" / "validation.py").exists(), "validation.py doesn't exist"
    assert (qa_path / "utils" / "deduplication.py").exists(), "deduplication.py doesn't exist"
    assert (qa_path / "llm" / "__init__.py").exists(), "llm/__init__.py doesn't exist"
    assert (qa_path / "llm" / "retry_llm_call.py").exists(), "retry_llm_call.py doesn't exist"
    
    # Test structure has been created
    test_path = base_path / "tests" / "qa"
    assert (test_path).exists(), "Test directory doesn't exist"
    assert (test_path / "conftest.py").exists(), "conftest.py doesn't exist"
    assert (test_path / "test_processor.py").exists(), "test_processor.py doesn't exist"
    assert (test_path / "test_models").exists(), "test_models directory doesn't exist"
    assert (test_path / "test_utils").exists(), "test_utils directory doesn't exist"
    assert (test_path / "test_llm").exists(), "test_llm directory doesn't exist"