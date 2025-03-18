"""
Tests for code extractor functionality.

Official Documentation References:
- ast: https://docs.python.org/3/library/ast.html
- tree_sitter: https://tree-sitter.github.io/tree-sitter/
- json: https://docs.python.org/3/library/json.html
- tempfile: https://docs.python.org/3/library/tempfile.html
- os.path: https://docs.python.org/3/library/os.path.html
"""

import pytest
import os
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Import directly from the module
try:
    from agent_tools.dualipa.code_extractor import (
        extract_repository, 
        _extract_python_blocks, 
        _extract_with_tree_sitter,
        _extract_js_ts_blocks, 
        _extract_markdown_blocks, 
        _extract_generic_blocks,
        _get_language_for_file_ext,
        initialize_stats_dict
    )
    IMPORTS_AVAILABLE = True
except ImportError as import_error:
    import traceback
    print(f"IMPORT ERROR: {import_error}")
    print("Traceback:")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False
    
# Only skip tests if imports failed
if not IMPORTS_AVAILABLE:
    pytest.fail(f"Required code extractor modules not available: {import_error}. Fix the dependencies to run these tests.")


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
    py_file = os.path.join(temp_dir, "example.py")
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
    
    with open(py_file, "w") as f:
        f.write(content)
    
    return py_file


@pytest.fixture
def javascript_file_fixture(temp_dir):
    """Create a JavaScript file with known functions and classes for testing."""
    js_file = os.path.join(temp_dir, "example.js")
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
    
    with open(js_file, "w") as f:
        f.write(content)
    
    return js_file


@pytest.fixture
def typescript_file_fixture(temp_dir):
    """Create a TypeScript file with known types, functions and classes for testing."""
    ts_file = os.path.join(temp_dir, "example.ts")
    content = """
/**
 * Type definition for a user
 */
interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
}

/**
 * Type for API response
 */
type ApiResponse<T> = {
    data: T;
    status: number;
    message: string;
}

/**
 * A function to get users
 * @returns {Promise<ApiResponse<User[]>>} The API response with users
 */
async function getUsers(): Promise<ApiResponse<User[]>> {
    // Simulated API call
    const response = await fetch('/api/users');
    return await response.json();
}

/**
 * A service class for user operations
 */
class UserService {
    private baseUrl: string;
    
    /**
     * Create a new UserService
     * @param {string} baseUrl - The base URL for API calls
     */
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    /**
     * Get a user by ID
     * @param {number} id - The user ID
     * @returns {Promise<User>} The user
     */
    async getUserById(id: number): Promise<User> {
        const response = await fetch(`${this.baseUrl}/users/${id}`);
        return await response.json();
    }
    
    /**
     * Create a new user
     * @param {Omit<User, 'id'>} userData - The user data without ID
     * @returns {Promise<User>} The created user
     */
    async createUser(userData: Omit<User, 'id'>): Promise<User> {
        const response = await fetch(`${this.baseUrl}/users`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        });
        return await response.json();
    }
}

// Default API URL
export const API_URL = 'https://api.example.com';

// Export types and classes
export { User, ApiResponse, UserService, getUsers };
"""
    
    with open(ts_file, "w") as f:
        f.write(content)
    
    return ts_file


@pytest.fixture
def markdown_file_fixture(temp_dir):
    """Create a Markdown file with sections and code blocks for testing."""
    md_file = os.path.join(temp_dir, "example.md")
    content = """# Example Documentation

This is an example markdown file with sections and code blocks.

## Installation

To install the package, run:

```bash
pip install my-package
```

## Usage

Here's how to use the package:

```python
from my_package import MyClass

# Create an instance
obj = MyClass()

# Call a method
result = obj.process_data()
```

## API Reference

### MyClass

```python
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

```json
{
    "api_key": "your-api-key",
    "max_retries": 3,
    "timeout": 30
}
```
"""
    
    with open(md_file, "w") as f:
        f.write(content)
    
    return md_file


def test_extract_python_blocks(python_file_fixture, temp_dir):
    """
    Test extracting Python blocks with specific expected results.
    This is a blind test with known expected output.
    """
    # Setup output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats dictionary with all required keys
    stats = initialize_stats_dict(source=python_file_fixture, output_dir=Path(output_dir))
    
    # Read the content of the Python file
    with open(python_file_fixture, "r") as f:
        content = f.read()
    
    # Extract Python blocks - convert file_path and output_dir to Path objects
    _extract_python_blocks(Path(python_file_fixture), content, Path(output_dir), stats)
    
    # Verify that blocks were extracted and stats were updated
    assert stats["code_blocks"] > 0, "No code blocks were extracted"
    assert stats["code_files"] == 1, "Code file count should be 1"
    assert str(python_file_fixture) in stats["file_blocks"], "File not in file_blocks dictionary"
    
    # Get the blocks for the Python file
    blocks = stats["file_blocks"][str(python_file_fixture)]
    
    # Verify the exact number of blocks (2 functions, 1 class with 3 methods)
    assert len(blocks) == 6, f"Expected 6 blocks, got {len(blocks)}"
    
    # Check for specific functions
    function_names = [b.get("name", "") for b in blocks if b.get("block_type") == "function"]
    assert "simple_function" in function_names, "simple_function not found"
    assert "function_with_args" in function_names, "function_with_args not found"
    
    # Check for the class and its methods
    class_blocks = [b for b in blocks if b.get("block_type") == "class"]
    assert len(class_blocks) == 1, "Expected 1 class block"
    assert class_blocks[0].get("name") == "SimpleClass", "SimpleClass not found"
    
    method_names = [b.get("name", "") for b in blocks 
                   if b.get("block_type") == "function" and "SimpleClass" in b.get("content", "")]
    assert "__init__" in method_names, "__init__ method not found"
    assert "get_name" in method_names, "get_name method not found"
    assert "set_name" in method_names, "set_name method not found"
    
    # Check content of a specific block (simple_function)
    simple_function_block = next((b for b in blocks if b.get("name") == "simple_function"), None)
    assert simple_function_block is not None, "simple_function block not found"
    assert "A simple function that returns a string" in simple_function_block.get("content", ""), "Expected docstring not found"
    assert "return \"This is a simple function\"" in simple_function_block.get("content", ""), "Expected return statement not found"
    
    # Verify the output files were created
    for block in blocks:
        output_file = block.get("output_file")
        assert output_file is not None, "Block missing output_file"
        assert os.path.exists(output_file), f"Output file does not exist: {output_file}"
        
        # Check that the output file contains the expected content
        with open(output_file, "r") as f:
            content = f.read()
            assert len(content) > 0, "Output file is empty"
            assert block.get("content", "") == content, "Output file content does not match block content"


def test_extract_js_ts_blocks(javascript_file_fixture, typescript_file_fixture, temp_dir):
    """
    Test extracting JavaScript and TypeScript blocks with specific expected results.
    """
    # Setup output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats dictionary with all required keys
    stats = initialize_stats_dict(source="js and ts files", output_dir=Path(output_dir))
    
    # Read file contents
    with open(javascript_file_fixture, "r") as f:
        js_content = f.read()
        
    with open(typescript_file_fixture, "r") as f:
        ts_content = f.read()
    
    # Extract JavaScript blocks with proper parameters
    _extract_js_ts_blocks(Path(javascript_file_fixture), js_content, Path(output_dir), stats, "javascript")
    
    # Verify JavaScript extraction
    assert str(javascript_file_fixture) in stats["file_blocks"], "JavaScript file not in file_blocks dictionary"
    js_blocks = stats["file_blocks"][str(javascript_file_fixture)]
    assert len(js_blocks) > 0, "No JavaScript blocks were extracted"
    
    # Check for specific JavaScript elements
    js_function_names = [b.get("name", "") for b in js_blocks if b.get("block_type") == "function"]
    assert "greet" in js_function_names, "greet function not found in JavaScript"
    
    js_class_names = [b.get("name", "") for b in js_blocks if b.get("block_type") == "class"]
    assert "Person" in js_class_names, "Person class not found in JavaScript"
    
    # Extract TypeScript blocks with proper parameters
    _extract_js_ts_blocks(Path(typescript_file_fixture), ts_content, Path(output_dir), stats, "typescript")
    
    # Verify TypeScript extraction
    assert str(typescript_file_fixture) in stats["file_blocks"], "TypeScript file not in file_blocks dictionary"
    ts_blocks = stats["file_blocks"][str(typescript_file_fixture)]
    assert len(ts_blocks) > 0, "No TypeScript blocks were extracted"
    
    # Check for specific TypeScript elements
    ts_interface_names = [b.get("name", "") for b in ts_blocks if b.get("block_type") == "interface"]
    assert "User" in ts_interface_names, "User interface not found in TypeScript"
    
    ts_function_names = [b.get("name", "") for b in ts_blocks if b.get("block_type") == "function"]
    assert "getUsers" in ts_function_names, "getUsers function not found in TypeScript"
    
    ts_class_names = [b.get("name", "") for b in ts_blocks if b.get("block_type") == "class"]
    assert "UserService" in ts_class_names, "UserService class not found in TypeScript"
    
    # Verify the content of a specific TypeScript block
    user_service_block = next((b for b in ts_blocks if b.get("name") == "UserService"), None)
    assert user_service_block is not None, "UserService block not found"
    assert "class UserService" in user_service_block.get("content", ""), "Expected class definition not found"
    assert "getUserById" in user_service_block.get("content", ""), "Expected method not found"


def test_extract_markdown_blocks(markdown_file_fixture, temp_dir):
    """
    Test extracting Markdown blocks with specific expected results.
    """
    # Setup output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats dictionary with all required keys
    stats = initialize_stats_dict(source=markdown_file_fixture, output_dir=Path(output_dir))
    
    # Read the content of the Markdown file
    with open(markdown_file_fixture, "r") as f:
        content = f.read()
    
    # Extract Markdown blocks with proper parameters
    _extract_markdown_blocks(Path(markdown_file_fixture), content, Path(output_dir), stats)
    
    # Verify that blocks were extracted and stats were updated
    assert stats["doc_blocks"] > 0, "No doc blocks were extracted"
    assert stats["documentation_files"] == 1, "Documentation file count should be 1"
    assert str(markdown_file_fixture) in stats["file_blocks"], "File not in file_blocks dictionary"
    
    # Get the blocks for the Markdown file
    blocks = stats["file_blocks"][str(markdown_file_fixture)]
    
    # Verify the number of blocks (1 main title, 4 sections, and 4 code blocks)
    assert len(blocks) >= 5, f"Expected at least 5 blocks, got {len(blocks)}"
    
    # Check for specific section titles
    titles = [b.get("title", "") for b in blocks if b.get("block_type") == "section"]
    assert any("Example_Documentation" in t for t in titles), "Main title not found"
    assert any("Installation" in t for t in titles), "Installation section not found"
    assert any("Usage" in t for t in titles), "Usage section not found"
    assert any("API_Reference" in t for t in titles), "API Reference section not found"
    
    # Check for code blocks with different languages
    code_blocks = [b for b in blocks if b.get("block_type") == "code_block"]
    code_languages = [b.get("language", "") for b in code_blocks]
    
    assert "bash" in code_languages, "Bash code block not found"
    assert "python" in code_languages, "Python code block not found"
    assert "json" in code_languages, "JSON code block not found"
    
    # Check content of a specific code block (python)
    python_code_block = next((b for b in code_blocks if b.get("language") == "python"), None)
    assert python_code_block is not None, "Python code block not found"
    assert "from my_package import MyClass" in python_code_block.get("content", ""), "Expected import statement not found"
    
    # Verify the output files were created
    for block in blocks:
        output_file = block.get("output_file")
        if output_file:  # Some blocks might not have output files
            assert os.path.exists(output_file), f"Output file does not exist: {output_file}"
            
            # Check that the output file contains the expected content
            with open(output_file, "r") as f:
                content = f.read()
                assert len(content) > 0, "Output file is empty"


def test_extract_generic_blocks(temp_dir):
    """
    Test extracting generic blocks from various file types.
    """
    # Create a generic text file
    text_file = os.path.join(temp_dir, "example.txt")
    with open(text_file, "w") as f:
        f.write("""This is a simple text file.
        
It has multiple paragraphs.

- This is a list item
- Another list item

And some more text.""")
    
    # Create a CSV file
    csv_file = os.path.join(temp_dir, "example.csv")
    with open(csv_file, "w") as f:
        f.write("""Name,Age,City
John,30,New York
Jane,25,San Francisco
Bob,40,Chicago""")
    
    # Setup output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize stats dictionary with all required keys
    stats = initialize_stats_dict(source="generic files", output_dir=Path(output_dir))
    
    # Read the content of text file
    with open(text_file, "r") as f:
        text_content = f.read()
    
    # Extract generic blocks from text file with proper parameters
    _extract_generic_blocks(Path(text_file), text_content, Path(output_dir), stats, "text")
    
    # Verify text file extraction
    assert str(text_file) in stats["file_blocks"], "Text file not in file_blocks dictionary"
    text_blocks = stats["file_blocks"][str(text_file)]
    assert len(text_blocks) > 0, "No blocks were extracted from text file"
    
    # Verify content of text file block
    text_block = text_blocks[0]  # Should only be one block for the whole file
    assert "This is a simple text file" in text_block.get("content", ""), "Expected text content not found"
    assert "list item" in text_block.get("content", ""), "Expected list items not found"
    
    # Read the content of CSV file
    with open(csv_file, "r") as f:
        csv_content = f.read()
    
    # Extract generic blocks from CSV file with proper parameters
    _extract_generic_blocks(Path(csv_file), csv_content, Path(output_dir), stats, "csv")
    
    # Verify CSV file extraction
    assert str(csv_file) in stats["file_blocks"], "CSV file not in file_blocks dictionary"
    csv_blocks = stats["file_blocks"][str(csv_file)]
    assert len(csv_blocks) > 0, "No blocks were extracted from CSV file"
    
    # Verify content of CSV file block
    csv_block = csv_blocks[0]  # Should only be one block for the whole file
    assert "Name,Age,City" in csv_block.get("content", ""), "Expected header row not found"
    assert "John,30,New York" in csv_block.get("content", ""), "Expected data row not found"
    
    # Verify output files were created
    for file_path, blocks in stats["file_blocks"].items():
        for block in blocks:
            output_file = block.get("output_file")
            assert output_file is not None, f"Block from {file_path} missing output_file"
            assert os.path.exists(output_file), f"Output file does not exist: {output_file}"


def test_get_language_for_file_ext():
    """
    Test the language detection from file extensions.
    """
    # Test common file extensions
    assert _get_language_for_file_ext(".py") == "python", "Python language not detected"
    assert _get_language_for_file_ext(".js") == "javascript", "JavaScript language not detected"
    assert _get_language_for_file_ext(".ts") == "typescript", "TypeScript language not detected"
    assert _get_language_for_file_ext(".tsx") == "typescript", "TSX language not detected"
    assert _get_language_for_file_ext(".jsx") == "javascript", "JSX language not detected"
    assert _get_language_for_file_ext(".md") == "markdown", "Markdown language not detected"
    assert _get_language_for_file_ext(".rst") == "rst", "RST language not detected"
    assert _get_language_for_file_ext(".txt") == "text", "Text language not detected"
    assert _get_language_for_file_ext(".c") == "c", "C language not detected"
    assert _get_language_for_file_ext(".cpp") == "cpp", "C++ language not detected"
    assert _get_language_for_file_ext(".java") == "java", "Java language not detected"
    
    # Test unknown extension
    assert _get_language_for_file_ext(".unknown") == "text", "Unknown extension should default to text"
    
    # Test no extension
    assert _get_language_for_file_ext("") == "text", "No extension should default to text"


def test_extract_repository_integration(temp_dir):
    """
    Test the integration of the extract_repository function with all file types.
    """
    # Create a mock repository
    repo_dir = os.path.join(temp_dir, "repo")
    os.makedirs(repo_dir, exist_ok=True)
    
    # Create a Python file
    py_file = os.path.join(repo_dir, "main.py")
    with open(py_file, "w") as f:
        f.write("""
def hello():
    \"\"\"Say hello.\"\"\"
    return "Hello, world!"

class Greeter:
    def greet(self, name):
        \"\"\"Greet someone.\"\"\"
        return f"Hello, {name}!"
""")
    
    # Create a JavaScript file
    js_file = os.path.join(repo_dir, "script.js")
    with open(js_file, "w") as f:
        f.write("""
function sayHello() {
    return "Hello, world!";
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
""")
    
    # Create a Markdown file
    md_file = os.path.join(repo_dir, "README.md")
    with open(md_file, "w") as f:
        f.write("""# Mock Repository

This is a mock repository for testing.

## Usage

```python
from main import hello

result = hello()
print(result)
```
""")
    
    # Create a text file
    txt_file = os.path.join(repo_dir, "notes.txt")
    with open(txt_file, "w") as f:
        f.write("These are some notes about the project.")
    
    # Create an output directory
    output_dir = os.path.join(temp_dir, "output")
    
    # Run the extraction
    stats = extract_repository(
        source=repo_dir,
        output_path=output_dir,
        max_files=10
    )
    
    # Verify that files were processed
    assert os.path.exists(os.path.join(output_dir, "blocks.json")), "blocks.json not created"
    assert os.path.exists(os.path.join(output_dir, "extraction_stats.json")), "extraction_stats.json not created"
    
    # Load the blocks
    with open(os.path.join(output_dir, "blocks.json"), "r") as f:
        blocks = json.load(f)
    
    # Verify blocks were extracted from each file type
    assert len(blocks) > 0, "No blocks were extracted"
    
    file_paths = [block.get("file", "") for block in blocks]
    assert any(p.endswith("main.py") for p in file_paths), "Python file blocks not found"
    assert any(p.endswith("script.js") for p in file_paths), "JavaScript file blocks not found"
    assert any(p.endswith("README.md") for p in file_paths), "Markdown file blocks not found"
    assert any(p.endswith("notes.txt") for p in file_paths), "Text file blocks not found"
    
    # Verify stats
    assert stats["total_files"] == 4, "Expected 4 total files"
    assert stats["code_files"] == 2, "Expected 2 code files (py and js)"
    assert stats["documentation_files"] == 2, "Expected 2 documentation files (md and txt)"
    
    # Verify languages detected
    assert "python" in stats["languages"], "Python language not detected"
    assert "javascript" in stats["languages"], "JavaScript language not detected"
    assert "markdown" in stats["languages"], "Markdown language not detected"
    assert "text" in stats["languages"], "Text language not detected"
    
    # Verify file_blocks structure
    assert len(stats["file_blocks"]) == 4, "Expected 4 entries in file_blocks"
    
    # Check that all block files exist
    for block in blocks:
        output_file = block.get("output_file")
        if output_file:
            assert os.path.exists(output_file), f"Output file does not exist: {output_file}" 

if __name__ == "__main__":
    pytest.main()