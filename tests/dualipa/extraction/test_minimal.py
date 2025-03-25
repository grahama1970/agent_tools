#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal test for extraction module."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_minimal_imports():
    """Test that imports work."""
    # Import the module
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import initialize_stats_dict
    from agent_tools.dualipa.extraction.extractors.code.code_extractor import extract_code_blocks
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write("def test_func():\n    pass\n")
        f.flush()
        
        # Initialize stats
        stats = initialize_stats_dict(source=f.name, output_dir=tempfile.gettempdir())
        
        # Check that stats has expected fields
        assert "source" in stats
        assert "output_dir" in stats
        assert "file_blocks" in stats
        assert isinstance(stats["file_blocks"], dict)
        
        # Print success message
        print("Minimal import test passed!")

if __name__ == "__main__":
    test_minimal_imports()