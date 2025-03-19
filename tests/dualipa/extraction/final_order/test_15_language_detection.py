"""
TEST EXPECTATIONS

test_detect_language:
Input: File path and content
Expected Output: Language identifier string or None

CRITICAL RULES:
1. Language Detection Rules:
   - Must detect language from file extension
   - Must handle unknown extensions
   - Must handle case-insensitive extensions
   - Must support all tree-sitter languages

2. Language Support Rules:
   - Must support Python
   - Must support JavaScript/TypeScript
   - Must support Java
   - Must support C++
   - Must support Go
   - Must support Rust
   - Must support Ruby
   - Must support Bash

3. Edge Case Rules:
   - Must handle missing extensions
   - Must handle empty files
   - Must handle binary files
   - Must handle text files with wrong extensions
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.code_hierarchy import _get_language_for_file
    IMPORTS_AVAILABLE = True
except ImportError as import_error:
    import traceback
    print(f"IMPORT ERROR: {import_error}")
    print("Traceback:")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required code extractor modules not available")

def test_detect_language_from_extension():
    """Test language detection from file extensions."""
    # Python files
    assert _get_language_for_file(Path("test.py")) == "python", "Failed to detect Python"
    assert _get_language_for_file(Path("script.py")) == "python", "Failed to detect Python"
    
    # JavaScript files
    assert _get_language_for_file(Path("app.js")) == "javascript", "Failed to detect JavaScript"
    assert _get_language_for_file(Path("component.jsx")) == "javascript", "Failed to detect JSX"
    
    # TypeScript files
    assert _get_language_for_file(Path("service.ts")) == "typescript", "Failed to detect TypeScript"
    assert _get_language_for_file(Path("component.tsx")) == "typescript", "Failed to detect TSX"
    
    # Java files
    assert _get_language_for_file(Path("Main.java")) == "java", "Failed to detect Java"
    
    # C/C++ files
    assert _get_language_for_file(Path("program.c")) == "c", "Failed to detect C"
    assert _get_language_for_file(Path("class.cpp")) == "cpp", "Failed to detect C++"
    assert _get_language_for_file(Path("header.h")) == "c", "Failed to detect C header"
    assert _get_language_for_file(Path("class.hpp")) == "cpp", "Failed to detect C++ header"

def test_detect_language_edge_cases():
    """Test language detection edge cases."""
    # Missing extension
    assert _get_language_for_file(Path("test")) is None, "Should not detect language without extension"
    
    # Unknown extension
    assert _get_language_for_file(Path("file.xyz")) is None, "Should not detect unknown extension"
    
    # Mixed case extensions
    assert _get_language_for_file(Path("test.PY")) == "python", "Should handle uppercase extension"
    assert _get_language_for_file(Path("test.Py")) == "python", "Should handle mixed case extension"
    
    # Multiple dots
    assert _get_language_for_file(Path("test.min.js")) == "javascript", "Should handle multiple dots"
    assert _get_language_for_file(Path("test.spec.ts")) == "typescript", "Should handle multiple dots"

def test_detect_language_with_path():
    """Test language detection with full file paths."""
    # Absolute paths
    assert _get_language_for_file(Path("/path/to/test.py")) == "python", "Failed to detect Python from absolute path"
    assert _get_language_for_file(Path("/var/www/html/app.js")) == "javascript", "Failed to detect JavaScript from absolute path"
    
    # Relative paths
    assert _get_language_for_file(Path("./src/main.ts")) == "typescript", "Failed to detect TypeScript from relative path"
    assert _get_language_for_file(Path("../lib/utils.java")) == "java", "Failed to detect Java from relative path"
    
    # Nested paths
    assert _get_language_for_file(Path("deep/nested/path/script.py")) == "python", "Failed to detect Python from nested path"

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
        assert _get_language_for_file(Path(f.name)) is None, "Should not detect Python from content alone"

def test_detect_language_with_shebang():
    """Test language detection with shebang lines."""
    # Create temporary files with shebangs
    with tempfile.NamedTemporaryFile() as f:
        # Python shebang
        f.write(b"#!/usr/bin/env python3\n")
        f.flush()
        # No extension, so no language detection
        assert _get_language_for_file(Path(f.name)) is None, "Should not detect Python from shebang alone"

def test_detect_language_with_empty_files():
    """Test language detection with empty files."""
    # Create temporary empty files
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        assert _get_language_for_file(Path(f.name)) == "python", "Should detect Python from empty file"
    
    with tempfile.NamedTemporaryFile(suffix=".js") as f:
        assert _get_language_for_file(Path(f.name)) == "javascript", "Should detect JavaScript from empty file"

if __name__ == "__main__":
    pytest.main([__file__]) 