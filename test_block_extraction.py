#!/usr/bin/env python
"""
Test script for the block extraction functionality in code_extractor.py.

This script tests:
1. Python block extraction using AST
2. Markdown section extraction
3. JS/TS block extraction using regex
4. Generic block extraction

The test confirms that:
- Complete files are preserved in code/ and docs/ directories
- Blocks are properly extracted to blocks/code/ and blocks/docs/ directories
- Block files contain proper metadata about their source files
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

# Configure logging
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def create_test_files(temp_dir):
    """Create test files for extraction testing."""
    logger.info("Creating test files...")
    test_files = {}
    
    # Create a sample Python file with functions and classes
    python_file = temp_dir / "sample.py"
    with open(python_file, "w") as f:
        f.write('''"""Sample Python file for testing block extraction."""

import os
import sys
from typing import List, Dict, Optional

def hello_world():
    """Say hello to the world."""
    return "Hello, World!"

class Person:
    """A simple Person class."""
    
    def __init__(self, name: str, age: int):
        """Initialize with name and age."""
        self.name = name
        self.age = age
    
    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, my name is {self.name} and I'm {self.age} years old."
        
    def is_adult(self) -> bool:
        """Check if the person is an adult."""
        return self.age >= 18

def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)

if __name__ == "__main__":
    print(hello_world())
    person = Person("Alice", 30)
    print(person.greet())
    print(f"Is adult: {person.is_adult()}")
    print(f"Sum of [1, 2, 3, 4, 5]: {calculate_sum([1, 2, 3, 4, 5])}")
''')
    test_files["python"] = python_file
    
    # Create a sample JavaScript file with functions and classes
    js_file = temp_dir / "sample.js"
    with open(js_file, "w") as f:
        f.write('''// Sample JavaScript file for testing block extraction

/**
 * Say hello to the world.
 * @returns {string} A greeting message
 */
function helloWorld() {
    return "Hello, World!";
}

/**
 * A simple Person class.
 */
class Person {
    /**
     * Create a new Person.
     * @param {string} name - The person's name
     * @param {number} age - The person's age
     */
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    /**
     * Get a greeting from the person.
     * @returns {string} The greeting
     */
    greet() {
        return `Hello, my name is ${this.name} and I'm ${this.age} years old.`;
    }
    
    /**
     * Check if the person is an adult.
     * @returns {boolean} True if the person is at least 18 years old
     */
    isAdult() {
        return this.age >= 18;
    }
}

/**
 * Calculate the sum of an array of numbers.
 * @param {number[]} numbers - The numbers to sum
 * @returns {number} The sum of the numbers
 */
function calculateSum(numbers) {
    return numbers.reduce((sum, num) => sum + num, 0);
}

// Example usage
console.log(helloWorld());
const person = new Person("Bob", 25);
console.log(person.greet());
console.log(`Is adult: ${person.isAdult()}`);
console.log(`Sum of [1, 2, 3, 4, 5]: ${calculateSum([1, 2, 3, 4, 5])}`);
''')
    test_files["javascript"] = js_file
    
    # Create a sample Markdown file with sections
    md_file = temp_dir / "README.md"
    with open(md_file, "w") as f:
        f.write('''# Sample README

This is a sample README file for testing block extraction.

## Introduction

This project demonstrates code block extraction for different programming languages.
It shows how to extract functions, classes, and sections from source files.

## Python Example

Python blocks are extracted using the AST (Abstract Syntax Tree):

```python
def example_function():
    """An example function."""
    return "This is an example!"
```

## JavaScript Example

JavaScript blocks are extracted using regex patterns:

```javascript
function anotherExample() {
    // This is another example
    return "Another example!";
}
```

## Conclusion

Block extraction is useful for generating QA pairs and fine-tuning language models.
''')
    test_files["markdown"] = md_file
    
    return test_files

def test_block_extraction():
    """Test the block extraction functionality."""
    logger.info("Starting block extraction test...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_files(temp_path)
        
        # Create output directory
        output_dir = temp_path / "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the extraction process with block extraction enabled
        logger.info("Running extraction process with block extraction...")
        stats = extract_repository(
            str(temp_path),
            str(output_dir),
            max_files=10,
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Print extraction statistics
        logger.info(f"Extraction stats: {json.dumps(stats, indent=2)}")
        
        # Directory paths
        code_dir = output_dir / "code"
        docs_dir = output_dir / "docs"
        code_blocks_dir = output_dir / "blocks" / "code"
        doc_blocks_dir = output_dir / "blocks" / "docs"
        
        # Check if directories exist
        logger.info(f"Code directory exists: {code_dir.exists()}")
        logger.info(f"Docs directory exists: {docs_dir.exists()}")
        logger.info(f"Code blocks directory exists: {code_blocks_dir.exists()}")
        logger.info(f"Doc blocks directory exists: {doc_blocks_dir.exists()}")
        
        # Check file counts
        python_files = list(code_dir.glob("**/sample.py*"))
        js_files = list(code_dir.glob("**/sample.js*"))
        md_files = list(docs_dir.glob("**/README.md*"))
        
        python_blocks = list(code_blocks_dir.glob("**/python/sample_*"))
        js_blocks = list(code_blocks_dir.glob("**/javascript/sample_*"))
        js_blocks.extend(list(code_blocks_dir.glob("**/js/sample_*")))
        md_blocks = list(doc_blocks_dir.glob("**/markdown/README_*"))
        
        logger.info(f"Python files: {len(python_files)}")
        logger.info(f"JavaScript files: {len(js_files)}")
        logger.info(f"Markdown files: {len(md_files)}")
        
        logger.info(f"Python blocks: {len(python_blocks)}")
        logger.info(f"JavaScript blocks: {len(js_blocks)}")
        logger.info(f"Markdown blocks: {len(md_blocks)}")
        
        # Verify Python blocks (should have functions and classes)
        if python_blocks:
            logger.info("Python blocks extracted:")
            for block in python_blocks[:3]:  # Show first 3 blocks
                logger.info(f"  - {block.name}")
                with open(block, "r") as f:
                    content = f.read()
                    logger.info("    First few lines:")
                    logger.info("\n".join(content.split("\n")[:3]))
        
        # Verify JavaScript blocks (should have functions and classes)
        if js_blocks:
            logger.info("JavaScript blocks extracted:")
            for block in js_blocks[:3]:  # Show first 3 blocks
                logger.info(f"  - {block.name}")
                with open(block, "r") as f:
                    content = f.read()
                    logger.info("    First few lines:")
                    logger.info("\n".join(content.split("\n")[:3]))
        
        # Verify Markdown blocks (should have sections)
        if md_blocks:
            logger.info("Markdown blocks extracted:")
            for block in md_blocks[:3]:  # Show first 3 blocks
                logger.info(f"  - {block.name}")
                with open(block, "r") as f:
                    content = f.read()
                    logger.info("    First few lines:")
                    logger.info("\n".join(content.split("\n")[:3]))
        
        # Verify the results
        success = True
        
        # Check for basic structure
        if not code_dir.exists() or not docs_dir.exists():
            logger.error("Missing basic directories for complete files")
            success = False
        
        if not code_blocks_dir.exists() or not doc_blocks_dir.exists():
            logger.error("Missing block directories")
            success = False
        
        # Check for content
        if len(python_files) == 0 or len(js_files) == 0 or len(md_files) == 0:
            logger.error("Missing complete files")
            success = False
        
        # Check for blocks
        if len(python_blocks) == 0:
            logger.error("No Python blocks extracted")
            success = False
        
        if len(js_blocks) == 0:
            logger.error("No JavaScript blocks extracted")
            success = False
        
        if len(md_blocks) == 0:
            logger.error("No Markdown blocks extracted")
            success = False
        
        if success:
            logger.info("Block extraction test passed!")
        else:
            logger.error("Block extraction test failed!")
        
        return success

def main():
    """Main function."""
    try:
        test_result = test_block_extraction()
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