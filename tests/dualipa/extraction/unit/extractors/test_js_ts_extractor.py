"""
Tests for JavaScript/TypeScript code extraction.
Focus on core functionality without over-complication.
"""

import pytest
from pathlib import Path
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks

def test_basic_function_extraction(tmp_path):
    """Test extraction of a basic JavaScript function."""
    js_file = tmp_path / "test.js"
    js_file.write_text("""
function greet(name) {
    return `Hello ${name}!`;
}
""")
    
    blocks, stats = extract_js_ts_blocks(str(js_file))
    assert len(blocks) == 1
    assert blocks[0]["type"] == "function"
    assert blocks[0]["name"] == "greet"
    assert "Hello" in blocks[0]["content"]
    assert stats["functions"] == 1

def test_basic_class_extraction(tmp_path):
    """Test extraction of a basic TypeScript class."""
    ts_file = tmp_path / "test.ts"
    ts_file.write_text("""
class Person {
    name: string;
    
    constructor(name: string) {
        this.name = name;
    }
    
    greet() {
        return `Hello ${this.name}!`;
    }
}
""")
    
    blocks, stats = extract_js_ts_blocks(str(ts_file))
    assert len(blocks) == 2  # Class + method
    assert blocks[0]["type"] == "class"
    assert blocks[0]["name"] == "Person"
    assert blocks[1]["type"] == "method"
    assert blocks[1]["name"] == "greet"
    assert stats["classes"] == 1
    assert stats["methods"] == 1

def test_react_component_extraction(tmp_path):
    """Test extraction of a basic React component."""
    tsx_file = tmp_path / "test.tsx"
    tsx_file.write_text("""
import React from 'react';

const Greeting = ({ name }: { name: string }) => {
    return <div>Hello {name}!</div>;
};

export default Greeting;
""")
    
    blocks, stats = extract_js_ts_blocks(str(tsx_file))
    assert len(blocks) == 1
    assert blocks[0]["type"] == "react_component"
    assert "Greeting" in blocks[0]["content"]
    assert stats["react_components"] == 1
    assert stats["imports"] == 1
    assert stats["exports"] == 1 