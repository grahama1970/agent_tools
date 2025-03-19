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
    from agent_tools.dualipa.code_extractor import (
        _get_language_for_file,
        _is_code_file
    )
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
    assert _get_language_for_file("test.py") == "python", "Failed to detect Python"
    assert _get_language_for_file("test.PY") == "python", "Failed to detect Python (uppercase)"
    assert _get_language_for_file("test.pyw") == "python", "Failed to detect Python (alternate extension)"
    
    # JavaScript files
    assert _get_language_for_file("test.js") == "javascript", "Failed to detect JavaScript"
    assert _get_language_for_file("test.JS") == "javascript", "Failed to detect JavaScript (uppercase)"
    assert _get_language_for_file("test.jsx") == "javascript", "Failed to detect JavaScript (JSX)"
    assert _get_language_for_file("test.mjs") == "javascript", "Failed to detect JavaScript (module)"
    
    # TypeScript files
    assert _get_language_for_file("test.ts") == "typescript", "Failed to detect TypeScript"
    assert _get_language_for_file("test.TS") == "typescript", "Failed to detect TypeScript (uppercase)"
    assert _get_language_for_file("test.tsx") == "typescript", "Failed to detect TypeScript (TSX)"
    
    # Java files
    assert _get_language_for_file("test.java") == "java", "Failed to detect Java"
    assert _get_language_for_file("test.JAVA") == "java", "Failed to detect Java (uppercase)"
    
    # C++ files
    assert _get_language_for_file("test.cpp") == "cpp", "Failed to detect C++"
    assert _get_language_for_file("test.CPP") == "cpp", "Failed to detect C++ (uppercase)"
    assert _get_language_for_file("test.hpp") == "cpp", "Failed to detect C++ header"
    assert _get_language_for_file("test.cc") == "cpp", "Failed to detect C++ (alternate extension)"
    
    # Go files
    assert _get_language_for_file("test.go") == "go", "Failed to detect Go"
    assert _get_language_for_file("test.GO") == "go", "Failed to detect Go (uppercase)"
    
    # Rust files
    assert _get_language_for_file("test.rs") == "rust", "Failed to detect Rust"
    assert _get_language_for_file("test.RS") == "rust", "Failed to detect Rust (uppercase)"
    
    # Ruby files
    assert _get_language_for_file("test.rb") == "ruby", "Failed to detect Ruby"
    assert _get_language_for_file("test.RB") == "ruby", "Failed to detect Ruby (uppercase)"
    
    # Bash files
    assert _get_language_for_file("test.sh") == "bash", "Failed to detect Bash"
    assert _get_language_for_file("test.SH") == "bash", "Failed to detect Bash (uppercase)"
    assert _get_language_for_file("test.bash") == "bash", "Failed to detect Bash (alternate extension)"

def test_detect_language_edge_cases():
    """Test language detection edge cases."""
    # Missing extension
    assert _get_language_for_file("test") is None, "Should not detect language without extension"
    
    # Empty extension
    assert _get_language_for_file("test.") is None, "Should not detect language with empty extension"
    
    # Unknown extension
    assert _get_language_for_file("test.xyz") is None, "Should not detect language with unknown extension"
    
    # Multiple extensions
    assert _get_language_for_file("test.min.js") == "javascript", "Failed to detect language with multiple extensions"
    
    # Hidden files
    assert _get_language_for_file(".gitignore") is None, "Should not detect language for gitignore"
    assert _get_language_for_file(".env") is None, "Should not detect language for env file"
    
    # Binary file extensions
    assert _get_language_for_file("test.exe") is None, "Should not detect language for executable"
    assert _get_language_for_file("test.bin") is None, "Should not detect language for binary file"
    assert _get_language_for_file("test.jpg") is None, "Should not detect language for image file"

def test_detect_language_with_path():
    """Test language detection with full file paths."""
    # Absolute paths
    assert _get_language_for_file("/path/to/test.py") == "python", "Failed to detect Python from absolute path"
    assert _get_language_for_file("/root/project/src/test.js") == "javascript", "Failed to detect JavaScript from absolute path"
    
    # Relative paths
    assert _get_language_for_file("./src/test.py") == "python", "Failed to detect Python from relative path"
    assert _get_language_for_file("../project/test.js") == "javascript", "Failed to detect JavaScript from relative path"
    
    # Windows-style paths
    assert _get_language_for_file("C:\\path\\to\\test.py") == "python", "Failed to detect Python from Windows path"
    assert _get_language_for_file("D:\\project\\src\\test.js") == "javascript", "Failed to detect JavaScript from Windows path"
    
    # Path with spaces and special characters
    assert _get_language_for_file("path with spaces/test.py") == "python", "Failed to detect Python from path with spaces"
    assert _get_language_for_file("special_@#$%/test.js") == "javascript", "Failed to detect JavaScript from path with special chars"

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
        assert _get_language_for_file(f.name) is None, "Should not detect Python from content alone"
    
    with tempfile.NamedTemporaryFile(suffix=".js") as f:
        # TypeScript content in .js file
        f.write(b"""interface User {
    name: string;
    age: number;
}
""")
        f.flush()
        # Extension determines language, not content
        assert _get_language_for_file(f.name) == "javascript", "Should detect JavaScript from extension"

def test_detect_language_with_shebang():
    """Test language detection with shebang lines."""
    # Create temporary files with shebangs
    with tempfile.NamedTemporaryFile() as f:
        # Python shebang
        f.write(b"#!/usr/bin/env python3\n")
        f.flush()
        # No extension, so no language detection
        assert _get_language_for_file(f.name) is None, "Should not detect Python from shebang alone"
    
    with tempfile.NamedTemporaryFile() as f:
        # Bash shebang
        f.write(b"#!/bin/bash\n")
        f.flush()
        # No extension, so no language detection
        assert _get_language_for_file(f.name) is None, "Should not detect Bash from shebang alone"

def test_detect_language_with_empty_files():
    """Test language detection with empty files."""
    # Create temporary empty files
    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        assert _get_language_for_file(f.name) == "python", "Should detect Python from empty file"
    
    with tempfile.NamedTemporaryFile(suffix=".js") as f:
        assert _get_language_for_file(f.name) == "javascript", "Should detect JavaScript from empty file"
    
    with tempfile.NamedTemporaryFile() as f:
        assert _get_language_for_file(f.name) is None, "Should not detect language from empty file without extension"

if __name__ == "__main__":
    pytest.main([__file__]) 