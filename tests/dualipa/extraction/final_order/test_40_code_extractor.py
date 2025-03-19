"""
TEST EXPECTATIONS

1. test_extract_python_blocks:
   Input: Python file with functions and classes
   Expected Output:
   {
       "code_blocks": > 0,
       "total_files": 1,
       "file_blocks": {
           "example.py": [
               {
                   "block_type": "function",
                   "name": "simple_function",
                   "content": "def simple_function()..."
               },
               {
                   "block_type": "function",
                   "name": "function_with_args",
                   "content": "def function_with_args(arg1: str, arg2: int = 0)..."
               },
               {
                   "block_type": "class",
                   "name": "SimpleClass",
                   "content": "class SimpleClass..."
               }
           ]
       }
   }

2. test_extract_js_ts_blocks:
   Input: JavaScript file with functions and classes
   Expected Output:
   {
       "file_blocks": {
           "example.js": [
               {
                   "block_type": "function",
                   "name": "greet",
                   "content": "function greet(name)..."
               },
               {
                   "block_type": "class",
                   "name": "Person",
                   "content": "class Person..."
               }
           ]
       }
   }

3. test_extract_markdown_blocks:
   Input: Markdown file with sections and code blocks
   Expected Output:
   {
       "doc_blocks": > 0,
       "total_files": 1,
       "file_blocks": {
           "example.md": [
               {
                   "block_type": "section",
                   "title": "Example_Documentation",
                   "content": "# Example Documentation..."
               },
               {
                   "block_type": "code_block",
                   "language": "python",
                   "content": "from my_package import MyClass..."
               }
           ]
       }
   }

4. test_extract_repository_integration:
   Input: Repository with Python, JavaScript, Markdown, and text files
   Expected Output:
   {
       "total_files": 4,
       "languages": ["python", "javascript", "markdown", "text"],
       "file_blocks": {
           "main.py": [...],
           "script.js": [...],
           "README.md": [...],
           "notes.txt": [...]
       }
   }

CRITICAL RULES:
1. Block Extraction Rules:
   - Each block must have a block_type
   - Each block must have a name (if applicable)
   - Each block must have content
   - Each block must preserve original formatting

2. Language Detection Rules:
   - .py files → python
   - .js files → javascript
   - .ts files → typescript
   - .md files → markdown
   - Unknown extensions → text

3. Stats Tracking Rules:
   - Track total files processed
   - Track blocks per file
   - Track languages encountered
   - Track errors during extraction

4. Output File Rules:
   - blocks.json must be created
   - extraction_stats.json must be created
   - All output files must be valid JSON
   - All paths must be relative to output directory

Input:
- source_path (Path): Path to source file or directory
- output_path (Path): Path to output directory
- max_files (int, optional): Maximum number of files to process

Output Structure:
{
    "total_files": int,
    "code_blocks": int,
    "doc_blocks": int,
    "languages": Set[str],
    "file_blocks": Dict[str, List[Dict]],
    "errors": List[Dict]
}

This test verifies the code extraction functionality across different file types
and ensures proper block extraction, stats tracking, and output file generation.
"""

import pytest
import os
import tempfile
import shutil
import json
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import directly from the module
from agent_tools.dualipa.code_extractor import (
    extract_repository, 
    _extract_python_blocks, 
    _extract_js_ts_blocks, 
    _extract_markdown_blocks, 
    _extract_generic_blocks,
    initialize_stats_dict
)
from agent_tools.dualipa.language_detection import detect_language

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    # Clean up after the test
    shutil.rmtree(dir_path)


@pytest.fixture
def python_file_fixture(temp_dir):
    """Create a Python file with known functions and classes for testing."""
    py_file = Path(temp_dir) / "example.py"
    content = """
import os
import sys
from typing import List, Dict, Any

def simple_function():
    \"\"\"A simple function that returns a string.\"\"\"
    return "This is a simple function"

def function_with_args(arg1: str, arg2: int = 0) -> str:
    \"\"\"
    A function with arguments and type hints.
    
    Args:
        arg1: A string argument
        arg2: An integer argument with default value
        
    Returns:
        A formatted string
    \"\"\"
    return f"{arg1}: {arg2}"

class SimpleClass:
    \"\"\"A simple class with methods.\"\"\"
    
    def __init__(self, name: str):
        \"\"\"Initialize with a name.\"\"\"
        self.name = name
    
    def get_name(self) -> str:
        \"\"\"Return the name.\"\"\"
        return self.name
    
    def set_name(self, name: str) -> None:
        \"\"\"Set the name.\"\"\"
        self.name = name

# A global variable
GLOBAL_CONSTANT = "This is a global constant"

if __name__ == "__main__":
    simple_function()
"""
    
    py_file.write_text(content)
    return py_file


@pytest.fixture
def javascript_file_fixture(temp_dir):
    """Create a JavaScript file with known functions and classes for testing."""
    js_file = Path(temp_dir) / "example.js"
    content = """
/**
 * A simple JavaScript function
 * @param {string} name - The name to greet
 * @returns {string} A greeting message
 */
function greet(name) {
    return `Hello, ${name}!`;
}

/**
 * A Person class
 */
class Person {
    /**
     * Create a new Person
     * @param {string} name - The person's name
     * @param {number} age - The person's age
     */
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    /**
     * Get a greeting for this person
     * @returns {string} A personalized greeting
     */
    getGreeting() {
        return greet(this.name);
    }
    
    /**
     * Check if the person is an adult
     * @returns {boolean} True if the person is an adult
     */
    isAdult() {
        return this.age >= 18;
    }
}

// A constant
const MAX_AGE = 120;

// Export the functions and classes
module.exports = {
    greet,
    Person,
    MAX_AGE
};
"""
    
    js_file.write_text(content)
    return js_file


@pytest.fixture
def markdown_file_fixture(temp_dir):
    """Create a Markdown file with sections and code blocks for testing."""
    md_file = Path(temp_dir) / "example.md"
    content = """# Example Documentation

This is an example markdown file with sections and code blocks.

## Installation

To install the package, run:
```
pip install my-package
```

## Usage

Here's how to use the package:

```
from my_package import MyClass

# Create an instance
obj = MyClass()

# Call a method
result = obj.process_data()
```

## API Reference

### MyClass

```
class MyClass:
    def __init__(self, config=None):
        \"\"\"
        Initialize the class.
        
        Args:
            config (dict, optional): Configuration dictionary
        \"\"\"
        self.config = config or {}
    
    def process_data(self, data=None):
        \"\"\"
        Process the data.
        
        Args:
            data (Any, optional): The data to process
            
        Returns:
            The processed data
        \"\"\"
        return data
```

## Configuration

Configuration is done using a JSON file:

```
{
    "api_key": "your-api-key",
    "max_retries": 3,
    "timeout": 30
}
```
"""
    
    md_file.write_text(content)
    return md_file


def test_extract_python_blocks(python_file_fixture, temp_dir):
    """Test extracting Python blocks."""
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=python_file_fixture, output_dir=output_dir)
    
    content = python_file_fixture.read_text()
    
    _extract_python_blocks(python_file_fixture, content, output_dir, stats)
    
    assert stats["code_blocks"] > 0, "No code blocks were extracted"
    assert stats["total_files"] == 1, "Total file count should be 1"
    assert str(python_file_fixture) in stats["file_blocks"], "File not in file_blocks dictionary"
    
    blocks = stats["file_blocks"][str(python_file_fixture)]
    
    assert len(blocks) > 0, "No blocks were extracted"
    
    function_names = [b.get("name", "") for b in blocks if b.get("block_type") == "function"]
    assert "simple_function" in function_names, "simple_function not found"
    assert "function_with_args" in function_names, "function_with_args not found"
    
    class_blocks = [b for b in blocks if b.get("block_type") == "class"]
    assert len(class_blocks) > 0, "No class blocks found"
    assert class_blocks[0].get("name") == "SimpleClass", "SimpleClass not found"
    
    for block in blocks:
        output_file = block.get("output_file")
        if output_file:
            assert Path(output_file).exists(), f"Output file does not exist: {output_file}"


def test_extract_js_ts_blocks(javascript_file_fixture, temp_dir):
    """Test extracting JavaScript blocks."""
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source="js file", output_dir=output_dir)
    
    content = javascript_file_fixture.read_text()
    
    _extract_js_ts_blocks(javascript_file_fixture, content, output_dir, stats, "javascript")
    
    assert str(javascript_file_fixture) in stats["file_blocks"], "JavaScript file not in file_blocks dictionary"
    blocks = stats["file_blocks"][str(javascript_file_fixture)]
    assert len(blocks) > 0, "No JavaScript blocks were extracted"
    
    function_names = [b.get("name", "") for b in blocks if b.get("block_type") == "function"]
    assert "greet" in function_names, "greet function not found in JavaScript"
    
    class_names = [b.get("name", "") for b in blocks if b.get("block_type") == "class"]
    assert "Person" in class_names, "Person class not found in JavaScript"


def test_extract_markdown_blocks(markdown_file_fixture, temp_dir):
    """Test extracting Markdown blocks."""
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=markdown_file_fixture, output_dir=output_dir)
    
    content = markdown_file_fixture.read_text()
    
    _extract_markdown_blocks(markdown_file_fixture, content, output_dir, stats)
    
    assert stats["doc_blocks"] > 0, "No doc blocks were extracted"
    assert stats["total_files"] == 1, "Total file count should be 1"
    assert str(markdown_file_fixture) in stats["file_blocks"], "File not in file_blocks dictionary"
    
    blocks = stats["file_blocks"][str(markdown_file_fixture)]
    
    assert len(blocks) > 0, "No blocks were extracted"
    
    titles = [b.get("title", "") for b in blocks if b.get("block_type") == "section"]
    assert any("Example_Documentation" in t for t in titles), "Main title not found"
    assert any("Installation" in t for t in titles), "Installation section not found"
    
    code_blocks = [b for b in blocks if b.get("block_type") == "code_block"]
    code_languages = [b.get("language", "") for b in code_blocks]
    
    assert "bash" in code_languages, "Bash code block not found"
    assert "python" in code_languages, "Python code block not found"
    assert "json" in code_languages, "JSON code block not found"


def test_extract_generic_blocks(temp_dir):
    """Test extracting generic blocks from various file types."""
    text_file = Path(temp_dir) / "example.txt"
    text_file.write_text("This is a simple text file.\n\nIt has multiple paragraphs.")
    
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source="generic files", output_dir=output_dir)
    
    content = text_file.read_text()
    
    _extract_generic_blocks(text_file, content, output_dir, stats, "text")
    
    assert str(text_file) in stats["file_blocks"], "Text file not in file_blocks dictionary"
    blocks = stats["file_blocks"][str(text_file)]
    assert len(blocks) > 0, "No blocks were extracted from text file"
    
    text_block = blocks[0]
    assert "This is a simple text file" in text_block.get("content", ""), "Expected text content not found"


def test_get_language_for_file_ext():
    """Test the language detection from file extensions."""
    assert detect_language("test.py") == "python", "Python language not detected"
    assert detect_language("test.js") == "javascript", "JavaScript language not detected"
    assert detect_language("test.md") == "markdown", "Markdown language not detected"
    assert detect_language("test.unknown") == "unknown", "Unknown extension should default to unknown"
    assert detect_language("noextension") == "unknown", "No extension should default to unknown"


def test_extract_repository_integration(temp_dir):
    """Test the integration of the extract_repository function with all file types."""
    repo_dir = Path(temp_dir) / "repo"
    repo_dir.mkdir(exist_ok=True)
    
    (repo_dir / "main.py").write_text("""
def hello():
    \"\"\"Say hello.\"\"\"
    return "Hello, world!"

class Greeter:
    def greet(self, name):
        \"\"\"Greet someone.\"\"\"
        return f"Hello, {name}!"
""")
    
    (repo_dir / "script.js").write_text("""
function sayHello() {
    return "Hello, world!";
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
""")
    
    (repo_dir / "README.md").write_text("""# Mock Repository

This is a mock repository for testing.

## Usage

```
from main import hello

result = hello()
print(result)
```
""")
    
    (repo_dir / "notes.txt").write_text("These are some notes about the project.")
    
    output_dir = Path(temp_dir) / "output"
    
    stats = extract_repository(
        source=repo_dir,
        output_path=output_dir,
        max_files=10
    )
    
    assert (output_dir / "blocks.json").exists(), "blocks.json not created"
    assert (output_dir / "extraction_stats.json").exists(), "extraction_stats.json not created"
    
    with open(output_dir / "blocks.json", "r") as f:
        blocks = json.load(f)
    
    assert len(blocks) > 0, "No blocks were extracted"
    
    file_paths = [block.get("file", "") for block in blocks]
    assert any("main.py" in p for p in file_paths), "Python file blocks not found"
    assert any("script.js" in p for p in file_paths), "JavaScript file blocks not found"
    assert any("README.md" in p for p in file_paths), "Markdown file blocks not found"
    assert any("notes.txt" in p for p in file_paths), "Text file blocks not found"
    
    assert stats["total_files"] == 4, "Expected 4 total files"
    assert "python" in stats["languages"], "Python language not detected"
    assert "javascript" in stats["languages"], "JavaScript language not detected"
    assert "markdown" in stats["languages"], "Markdown language not detected"
    assert "text" in stats["languages"], "Text language not detected"
    
    assert len(stats["file_blocks"]) == 4, "Expected 4 entries in file_blocks"


if __name__ == "__main__":
    pytest.main()

