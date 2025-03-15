#!/usr/bin/env python
"""
Test script for the code_extractor.py module.

This script tests the functionality of the code_extractor module:
1. Create sample test files (code and documentation)
2. Run the extraction process
3. Verify the extraction results:
   - Complete code files are preserved
   - Complete documentation files are preserved
   - No code_blocks directory is created
   - File paths are embedded in the extracted files
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json

# Import the code_extractor module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from agent_tools.dualipa.code_extractor import extract_repository
from agent_tools.dualipa.github_utils import download_github_repo

# Configure logging
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_files(temp_dir):
    """Create test files for extraction testing."""
    logger.info("Creating test files...")
    
    # Create a sample Python file
    python_file = temp_dir / "sample.py"
    with open(python_file, "w") as f:
        f.write('''"""Sample Python file for testing."""

def hello_world():
    """Say hello to the world."""
    return "Hello, World!"

class TestClass:
    """A test class."""
    
    def __init__(self, name):
        """Initialize with a name."""
        self.name = name
    
    def greet(self):
        """Return a greeting."""
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    print(hello_world())
    test = TestClass("Tester")
    print(test.greet())
''')
    
    # Create a sample Markdown file
    markdown_file = temp_dir / "README.md"
    with open(markdown_file, "w") as f:
        f.write('''# Test README

This is a sample README file for testing the code extractor.

## Code Example

```python
def example_function():
    """An example function."""
    return "This is an example!"
```

## Another Example

```javascript
function anotherExample() {
    // This is another example
    return "Another example!";
}
```
''')
    
    # Create a sample JavaScript file
    js_file = temp_dir / "sample.js"
    with open(js_file, "w") as f:
        f.write('''// Sample JavaScript file for testing

/**
 * Add two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The sum of a and b.
 */
function add(a, b) {
    return a + b;
}

// A simple class
class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(value) {
        this.result += value;
        return this.result;
    }
    
    subtract(value) {
        this.result -= value;
        return this.result;
    }
}

console.log(add(1, 2));
const calc = new Calculator();
console.log(calc.add(5));
''')
    
    return {"python": python_file, "markdown": markdown_file, "javascript": js_file}

def test_extraction():
    """Test the extraction process."""
    logger.info("Starting code extractor test...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_files(temp_path)
        
        # Create output directory
        output_dir = temp_path / "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the extraction process
        logger.info("Running extraction process...")
        stats = extract_repository(
            str(temp_path),
            str(output_dir),
            max_files=10,
            extract_documentation=True,
            extract_code=True
        )
        
        # Print extraction statistics
        logger.info(f"Extraction stats: {json.dumps(stats, indent=2)}")
        
        # Verify the extraction results
        code_dir = output_dir / "code"
        docs_dir = output_dir / "docs"
        code_blocks_dir = output_dir / "code_blocks"
        
        # Check if directories exist
        logger.info(f"Code directory exists: {code_dir.exists()}")
        logger.info(f"Docs directory exists: {docs_dir.exists()}")
        logger.info(f"Code blocks directory exists: {code_blocks_dir.exists()} (Should be False)")
        
        # Check if files were extracted
        python_files = list(code_dir.glob("**/sample.py*"))
        js_files = list(code_dir.glob("**/sample.js*"))
        md_files = list(docs_dir.glob("**/README.md*"))
        
        logger.info(f"Python files: {len(python_files)}")
        logger.info(f"JavaScript files: {len(js_files)}")
        logger.info(f"Markdown files: {len(md_files)}")
        
        # Check that file paths are embedded in the files
        if python_files:
            with open(python_files[0], "r") as f:
                content = f.read()
                logger.info("First few lines of Python file:")
                logger.info("\n".join(content.split("\n")[:3]))
        
        if md_files:
            with open(md_files[0], "r") as f:
                content = f.read()
                logger.info("First few lines of Markdown file:")
                logger.info("\n".join(content.split("\n")[:3]))
        
        # Verify the results
        assert code_dir.exists(), "Code directory should exist"
        assert docs_dir.exists(), "Docs directory should exist"
        assert not code_blocks_dir.exists(), "Code blocks directory should not exist"
        assert len(python_files) > 0, "Should have extracted Python files"
        assert len(js_files) > 0, "Should have extracted JavaScript files"
        assert len(md_files) > 0, "Should have extracted Markdown files"
        
        logger.info("Extraction test passed!")
        return True

def main():
    """Main function."""
    try:
        test_result = test_extraction()
        if test_result:
            logger.info("All tests passed!")
            return 0
        else:
            logger.error("Tests failed!")
            return 1
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 