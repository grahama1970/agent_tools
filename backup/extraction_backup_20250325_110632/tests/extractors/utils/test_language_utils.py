"""
Test language detection and utilities.

This module tests language detection, language info retrieval,
and language-specific patterns.
"""

import os
import sys
import pytest
from pathlib import Path

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
try:
    from agent_tools.dualipa.extraction.extractors.utils.language_utils import (
        detect_language,
        get_language_info,
        is_supported_language,
        get_comment_pattern,
        get_block_comment_patterns
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Error importing language utils: {e}")
    raise ImportError(f"Required language utils not available: {e}. Fix the dependencies to run these tests.")

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required language utils not available")

def test_detect_language_from_extension():
    """Test language detection from file extensions."""
    # Python files
    assert detect_language(Path("test.py")) == "python", "Failed to detect Python"
    assert detect_language(Path("script.py")) == "python", "Failed to detect Python"
    
    # JavaScript files
    assert detect_language(Path("app.js")) == "javascript", "Failed to detect JavaScript"
    assert detect_language(Path("component.jsx")) == "javascript", "Failed to detect JSX"
    
    # TypeScript files
    assert detect_language(Path("service.ts")) == "typescript", "Failed to detect TypeScript"
    assert detect_language(Path("component.tsx")) == "typescript", "Failed to detect TSX"
    
    # Java files
    assert detect_language(Path("Main.java")) == "java", "Failed to detect Java"
    
    # C/C++ files
    assert detect_language(Path("program.c")) == "c", "Failed to detect C"
    assert detect_language(Path("class.cpp")) == "cpp", "Failed to detect C++"
    assert detect_language(Path("header.h")) == "c", "Failed to detect C header"
    assert detect_language(Path("class.hpp")) == "cpp", "Failed to detect C++ header"

def test_detect_language_edge_cases():
    """Test language detection edge cases."""
    # Missing extension
    assert detect_language(Path("test")) is None, "Should not detect language without extension"
    
    # Unknown extension
    assert detect_language(Path("file.xyz")) is None, "Should not detect unknown extension"
    
    # Mixed case extensions
    assert detect_language(Path("test.PY")) == "python", "Should handle uppercase extension"
    assert detect_language(Path("test.Py")) == "python", "Should handle mixed case extension"
    
    # Multiple dots
    assert detect_language(Path("test.min.js")) == "javascript", "Should handle multiple dots"
    assert detect_language(Path("test.spec.ts")) == "typescript", "Should handle multiple dots"

def test_detect_language_with_path():
    """Test language detection with full file paths."""
    # Absolute paths
    assert detect_language(Path("/path/to/test.py")) == "python", "Failed to detect Python from absolute path"
    assert detect_language(Path("/var/www/html/app.js")) == "javascript", "Failed to detect JavaScript from absolute path"
    
    # Relative paths
    assert detect_language(Path("./src/main.ts")) == "typescript", "Failed to detect TypeScript from relative path"
    assert detect_language(Path("../lib/utils.java")) == "java", "Failed to detect Java from relative path"
    
    # Nested paths
    assert detect_language(Path("deep/nested/path/script.py")) == "python", "Failed to detect Python from nested path"

def test_detect_language_with_content():
    """Test language detection with file content."""
    # Create temporary files with content
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        # Python content in .txt file
        f.write(b"""#!/usr/bin/env python3
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
        f.flush()
        # Even though content is Python, extension is not
        assert detect_language(Path(f.name)) is None, "Should not detect Python from content alone"

def test_detect_language_with_shebang():
    """Test language detection with shebang lines."""
    # Create temporary files with shebangs
    with tempfile.NamedTemporaryFile() as f:
        # Python shebang
        f.write(b"#!/usr/bin/env python3\n")
        f.flush()
        # No extension, so no language detection
        assert detect_language(Path(f.name)) is None, "Should not detect Python from shebang alone"

def test_detect_language_with_empty_files():
    """Test language detection with empty files."""
    # Create temporary empty files
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        assert detect_language(Path(f.name)) == "python", "Should detect Python from empty file"
    
    with tempfile.NamedTemporaryFile(suffix=".js") as f:
        assert detect_language(Path(f.name)) == "javascript", "Should detect JavaScript from empty file"

if __name__ == "__main__":
    pytest.main([__file__]) 