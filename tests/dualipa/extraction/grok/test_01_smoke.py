"""
Smoke tests to ensure basic functionality and imports.
These tests verify that the environment and dependencies are correctly set up.
"""

import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports_available():
    """Verify that all required modules can be imported."""
    try:
        from agent_tools.dualipa import code_extractor, github_utils
        from agent_tools.dualipa.code_extractor import extract_repository
        assert True
    except ImportError as e:
        pytest.fail(f"Required modules not available: {e}")

def test_simple_assertion():
    """Basic sanity check."""
    assert True, "Smoke test failed"