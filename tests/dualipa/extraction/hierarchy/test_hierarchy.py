"""Test suite for hierarchy module functionality.

This module tests the code hierarchy analysis functionality, including:
- Python hierarchy analysis
- JavaScript/TypeScript hierarchy analysis
- Generic language hierarchy analysis
- Backward compatibility

Test files:
- tests/dualipa/extraction/fixtures/python_sample.py
- tests/dualipa/extraction/fixtures/typescript_sample.ts
- tests/dualipa/extraction/fixtures/cpp_sample.cpp
"""

import sys
import pytest
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_tools.dualipa.extraction.extractors.hierarchy import (
    analyze_code_hierarchy,
    build_code_hierarchy
)
from src.agent_tools.dualipa.extraction.extractors.hierarchy.python.parser import analyze_python_hierarchy
from src.agent_tools.dualipa.extraction.extractors.hierarchy.js_ts.parser import analyze_js_ts_hierarchy
from src.agent_tools.dualipa.extraction.extractors.hierarchy.generic.parser import analyze_generic_hierarchy


# Create sample files for testing
@pytest.fixture
def python_sample_file():
    """Create a temporary Python file for testing."""
    content = textwrap.dedent('''
    from abc import ABC, abstractmethod
    
    class Animal(ABC):
        def __init__(self, name: str):
            self.name = name
            
        @abstractmethod
        def make_sound(self) -> str:
            pass
            
    class Dog(Animal):
        def make_sound(self) -> str:
            return "Woof!"
            
    class Cat(Animal):
        def make_sound(self) -> str:
            return "Meow!"
    ''').strip()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def typescript_sample_file():
    """Create a temporary TypeScript file for testing."""
    content = textwrap.dedent('''
    import { Component } from 'react';
    
    interface Props {
        name: string;
        age?: number;
    }
    
    class Person implements Props {
        name: string;
        age?: number;
        
        constructor(name: string, age?: number) {
            this.name = name;
            this.age = age;
        }
        
        static create(name: string): Person {
            return new Person(name);
        }
        
        async getDetails(): Promise<string> {
            return `${this.name}, ${this.age || 'unknown age'}`;
        }
    }
    
    export default Person;
    ''').strip()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)


@pytest.fixture
def cpp_sample_file():
    """Create a temporary C++ file for testing."""
    content = textwrap.dedent('''
    #include <iostream>
    #include <string>
    
    class Shape {
    public:
        virtual double area() const = 0;
        virtual void display() const {
            std::cout << "Shape with area: " << area() << std::endl;
        }
    };
    
    class Circle : public Shape {
    private:
        double radius;
        
    public:
        Circle(double r) : radius(r) {}
        
        double area() const override {
            return 3.14159 * radius * radius;
        }
    };
    
    int main() {
        Circle c(5.0);
        c.display();
        return 0;
    }
    ''').strip()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)


@pytest.mark.skip(reason="TODO: Implement proper AST visitor pattern for Python hierarchy analysis")
def test_python_hierarchy_analysis(python_sample_file):
    """Test Python hierarchy analysis."""
    with open(python_sample_file, 'r') as f:
        content = f.read()
    
    # Initialize stats
    stats = {"classes": 0, "functions": 0, "imports": 0, "errors": []}
    
    # Analyze hierarchy
    hierarchy, stats = analyze_python_hierarchy(content, python_sample_file, stats)
    
    # Check results
    assert hierarchy["language"] == "python"
    assert hierarchy["file_path"] == python_sample_file
    assert len(hierarchy["classes"]) == 3
    assert "Animal" in hierarchy["classes"]
    assert "Dog" in hierarchy["classes"]
    assert "Cat" in hierarchy["classes"]
    
    # Check inheritance
    assert "ABC" in hierarchy["classes"]["Animal"]["bases"]
    
    # Check methods
    animal_methods = {m["name"] for m in hierarchy["classes"]["Animal"]["methods"]}
    assert "__init__" in animal_methods
    assert "make_sound" in animal_methods
    
    # Check stats
    assert stats["classes"] == 3
    assert stats["imports"] == 1
    assert not stats["errors"]


@pytest.mark.skip(reason="TODO: Implement proper tree-sitter visitor for JS/TS hierarchy analysis")
def test_js_ts_hierarchy_analysis(typescript_sample_file):
    """Test JavaScript/TypeScript hierarchy analysis."""
    with open(typescript_sample_file, 'r') as f:
        content = f.read()
    
    # Initialize stats
    stats = {"classes": 0, "interfaces": 0, "imports": 0, "exports": 0, "errors": []}
    
    # Analyze hierarchy
    hierarchy, stats = analyze_js_ts_hierarchy(content, typescript_sample_file, "typescript", stats)
    
    # Check results
    assert hierarchy["language"] == "typescript"
    assert hierarchy["file_path"] == typescript_sample_file
    assert len(hierarchy["classes"]) == 1
    assert "Person" in hierarchy["classes"]
    assert "Props" in hierarchy.get("interfaces", {})
    
    # Check implementation
    assert "Props" in hierarchy["classes"]["Person"]["implements"]
    
    # Check methods
    person_methods = {m["name"] for m in hierarchy["classes"]["Person"]["methods"]}
    assert "constructor" in person_methods
    assert "create" in person_methods
    assert "getDetails" in person_methods
    
    # Check method properties
    for method in hierarchy["classes"]["Person"]["methods"]:
        if method["name"] == "create":
            assert method["is_static"]
        if method["name"] == "getDetails":
            assert method["is_async"]
    
    # Check imports/exports
    assert len(hierarchy["imports"]) == 1
    assert len(hierarchy["exports"]) == 1
    
    # Check stats
    assert stats["classes"] == 1
    assert stats["interfaces"] == 1
    assert stats["imports"] == 1
    assert stats["exports"] == 1
    assert not stats["errors"]


def test_generic_hierarchy_analysis(cpp_sample_file):
    """Test generic hierarchy analysis for C++."""
    with open(cpp_sample_file, 'r') as f:
        content = f.read()
    
    # Initialize stats
    stats = {"classes": 0, "functions": 0, "errors": []}
    
    # Analyze hierarchy
    hierarchy, stats = analyze_generic_hierarchy(content, cpp_sample_file, "cpp", stats)
    
    # Check results
    assert hierarchy["language"] == "cpp"
    assert hierarchy["file_path"] == cpp_sample_file
    assert len(hierarchy["classes"]) == 2
    assert "Shape" in hierarchy["classes"]
    assert "Circle" in hierarchy["classes"]
    assert "main" in hierarchy["functions"]
    
    # Check stats
    assert stats["classes"] == 2
    assert stats["functions"] == 1
    assert not stats["errors"]


@pytest.mark.skip(reason="Depends on other hierarchy analysis tests that need proper visitor pattern implementations")
def test_analyze_code_hierarchy_integration(python_sample_file, typescript_sample_file, cpp_sample_file):
    """Test the integrated analyze_code_hierarchy function."""
    # Test with Python file
    python_hierarchy, python_stats = analyze_code_hierarchy(python_sample_file)
    assert python_hierarchy["language"] == "python"
    assert len(python_hierarchy["classes"]) == 3
    
    # Test with TypeScript file
    ts_hierarchy, ts_stats = analyze_code_hierarchy(typescript_sample_file)
    assert ts_hierarchy["language"] == "typescript"
    assert len(ts_hierarchy["classes"]) == 1
    
    # Test with C++ file
    cpp_hierarchy, cpp_stats = analyze_code_hierarchy(cpp_sample_file)
    assert cpp_hierarchy["language"] == "cpp"
    assert len(cpp_hierarchy["classes"]) == 2


@pytest.mark.skip(reason="Fixing build_code_hierarchy integration")
def test_build_code_hierarchy():
    """Test building a hierarchy from code blocks."""
    # Create sample blocks
    blocks = [
        {
            "uuid": "123",
            "file_path": "test.py",
            "type": "code",
            "language": "python",
            "line_start": 1,
            "depth": 0
        },
        {
            "uuid": "456",
            "file_path": "test.py",
            "type": "code",
            "language": "python",
            "line_start": 10,
            "depth": 1
        },
        {
            "uuid": "789",
            "file_path": "test2.js",
            "type": "code",
            "language": "javascript",
            "line_start": 1,
            "depth": 0
        }
    ]
    
    # Build hierarchy
    stats = build_code_hierarchy(blocks)
    
    # Check results
    assert stats["files_processed"] == 2
    assert stats["blocks_processed"] == 3
    
    # Check parent-child relationships
    assert blocks[0]["child_uuids"] == ["456"]
    assert blocks[1]["parent_uuid"] == "123"


@pytest.mark.skip(reason="Fixing backward compatibility in progress")
def test_backward_compatibility():
    """Test backward compatibility with old module structure."""
    # Import the old-style module
    from src.agent_tools.dualipa.extraction.extractors.code.hierarchy import (
        analyze_code_hierarchy as old_analyze_code_hierarchy,
        build_code_hierarchy as old_build_code_hierarchy
    )
    
    # Create a simple block list
    blocks = [{"uuid": "123", "file_path": "test.py", "type": "code", "language": "python"}]
    
    # Test both functions
    new_stats = build_code_hierarchy(blocks)
    old_stats = old_build_code_hierarchy(blocks)
    
    # They should be equivalent
    assert type(new_stats) == type(old_stats)
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def test(): pass")
        temp_file = f.name
    
    try:
        # Test analyze_code_hierarchy
        new_hierarchy, new_stats = analyze_code_hierarchy(temp_file)
        old_hierarchy, old_stats = old_analyze_code_hierarchy(temp_file)
        
        # They should be equivalent
        assert new_hierarchy["language"] == old_hierarchy["language"]
        assert new_hierarchy["file_path"] == old_hierarchy["file_path"]
    finally:
        # Cleanup
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])