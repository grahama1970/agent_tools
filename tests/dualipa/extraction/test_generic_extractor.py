"""
Tests for generic code extraction.
Focus on core functionality without over-complication.
"""

import pytest
from pathlib import Path
from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks

def test_c_function_extraction(tmp_path):
    """Test extraction of a basic C function."""
    c_file = tmp_path / "test.c"
    c_file.write_text("""
int add(int a, int b) {
    return a + b;
}
""")
    
    blocks, stats = extract_generic_blocks(str(c_file))
    assert len(blocks) == 1
    assert blocks[0]["type"] == "function"
    assert blocks[0]["name"] == "add"
    assert "return" in blocks[0]["content"]
    assert stats["functions"] == 1

def test_cpp_class_extraction(tmp_path):
    """Test extraction of a basic C++ class with method."""
    cpp_file = tmp_path / "test.cpp"
    cpp_file.write_text("""
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};
""")
    
    blocks, stats = extract_generic_blocks(str(cpp_file))
    assert len(blocks) == 2  # Class + function
    # Find class block
    class_block = next(b for b in blocks if b["type"] == "class")
    assert class_block["name"] == "Calculator"
    assert "add" in class_block["content"]
    # Find function block
    func_block = next(b for b in blocks if b["type"] == "function")
    assert func_block["name"] == "add"
    assert stats["classes"] == 1
    assert stats["functions"] == 1

def test_rust_function_extraction(tmp_path):
    """Test extraction of a basic Rust function."""
    rust_file = tmp_path / "test.rs"
    rust_file.write_text("""
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}
""")
    
    blocks, stats = extract_generic_blocks(str(rust_file))
    assert len(blocks) == 1
    assert blocks[0]["type"] == "function"
    assert blocks[0]["name"] == "greet"
    assert "Hello" in blocks[0]["content"]
    assert stats["functions"] == 1 