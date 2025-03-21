"""
Tests for code hierarchy analysis.
Focus on core functionality without over-complication.
"""

import pytest
from pathlib import Path
from agent_tools.dualipa.extraction.extractors.code.hierarchy import analyze_code_hierarchy

def test_cpp_class_hierarchy(tmp_path):
    """Test hierarchy analysis of a C++ class with methods."""
    cpp_file = tmp_path / "test.cpp"
    cpp_file.write_text("""
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    int subtract(int a, int b) {
        return a - b;
    }
};
""")
    
    hierarchy, stats = analyze_code_hierarchy(str(cpp_file))
    
    # Check class was found
    assert "Calculator" in hierarchy["classes"]
    calc_class = hierarchy["classes"]["Calculator"]
    
    # Check class structure
    assert calc_class["line_start"] > 0
    assert calc_class["line_end"] > calc_class["line_start"]
    
    # Check functions were found
    assert len(hierarchy["functions"]) == 2  # add and subtract
    assert "add" in hierarchy["functions"]
    assert "subtract" in hierarchy["functions"]
    
    # Check statistics
    assert stats["classes"] == 1
    assert stats["functions"] == 2

def test_rust_function_hierarchy(tmp_path):
    """Test hierarchy analysis of Rust functions."""
    rust_file = tmp_path / "test.rs"
    rust_file.write_text("""
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
""")
    
    hierarchy, stats = analyze_code_hierarchy(str(rust_file))
    
    # Check functions were found
    assert len(hierarchy["functions"]) == 2
    assert "add" in hierarchy["functions"]
    assert "greet" in hierarchy["functions"]
    
    # Check function structure
    add_func = hierarchy["functions"]["add"]
    assert add_func["line_start"] > 0
    assert add_func["line_end"] > add_func["line_start"]
    
    # Check statistics
    assert stats["functions"] == 2
    assert stats.get("classes", 0) == 0  # No classes in this example 