"""
Simple test to verify extractors are working.
"""

import os
from pathlib import Path
from loguru import logger

from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks

def test_python_extractor():
    """Test Python code extraction."""
    # Create a test Python file
    test_content = '''
def hello_world():
    """Say hello."""
    print("Hello, World!")

class TestClass:
    def test_method(self):
        return "test"
'''
    
    # Create temporary output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test file
    test_file = output_dir / "test.py"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Run extraction
    blocks, stats = extract_python_blocks(str(test_file))
    
    # Print results
    print("\nPython Extractor Test Results:")
    print(f"Blocks extracted: {len(blocks)}")
    print(f"Stats: {stats}")
    
    return len(blocks) > 0

def test_js_extractor():
    """Test JavaScript code extraction."""
    # Create a test JavaScript file
    test_content = '''
function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    sayHello() {
        console.log(`Hi, I'm ${this.name}`);
    }
}
'''
    
    # Create temporary output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test file
    test_file = output_dir / "test.js"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Run extraction
    blocks, stats = extract_js_ts_blocks(str(test_file))
    
    # Print results
    print("\nJavaScript Extractor Test Results:")
    print(f"Blocks extracted: {len(blocks)}")
    print(f"Stats: {stats}")
    
    return len(blocks) > 0

def test_generic_extractor():
    """Test generic code extraction."""
    # Create a test file with generic code
    test_content = '''
// First block
void main() {
    printf("Hello, World!");
}

// Second block
int add(int a, int b) {
    return a + b;
}
'''
    
    # Create temporary output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test file
    test_file = output_dir / "test.c"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Run extraction
    blocks, stats = extract_generic_blocks(str(test_file))
    
    # Print results
    print("\nGeneric Extractor Test Results:")
    print(f"Blocks extracted: {len(blocks)}")
    print(f"Stats: {stats}")
    
    return len(blocks) > 0

def main():
    """Run all extractor tests."""
    print("Running extractor tests...")
    
    results = {
        "python": test_python_extractor(),
        "javascript": test_js_extractor(),
        "generic": test_generic_extractor()
    }
    
    print("\nTest Summary:")
    for extractor, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{extractor}: {status}")

if __name__ == "__main__":
    main() 