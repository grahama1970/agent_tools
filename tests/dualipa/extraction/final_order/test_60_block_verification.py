"""
TEST EXPECTATIONS

1. test_verify_python_block:
   Input: Python code blocks (valid and invalid)
   Expected Output:
   - Valid blocks pass verification
   - Invalid blocks fail verification
   - Proper error messages for invalid blocks

2. test_verify_javascript_block:
   Input: JavaScript code blocks (valid and invalid)
   Expected Output:
   - Valid blocks pass verification
   - Invalid blocks fail verification
   - Proper error messages for invalid blocks

3. test_verify_blocks_from_extraction:
   Input: Blocks extracted from Python files
   Expected Output:
   - All extracted blocks are valid Python code
   - Proper error messages for any invalid blocks
   - Stats tracking is accurate

4. test_verify_multifile_extraction:
   Input: Blocks from multiple files (Python, JavaScript)
   Expected Output:
   - All blocks are valid in their respective languages
   - Language-specific verification rules are applied
   - Stats tracking is accurate

CRITICAL RULES:
1. Verification Rules:
   - Each block must be syntactically valid
   - Each block must be complete (no partial functions/classes)
   - Each block must preserve original formatting

2. Error Handling:
   - Clear error messages for invalid blocks
   - Skip tests if dependencies are missing
   - Log verification failures

3. Stats Tracking:
   - Track number of blocks verified
   - Track verification failures
   - Track language-specific issues
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import json

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"

# Test if the repositories exist
HAS_TEST_REPOS = RUST_ANALYZER_PATH.exists() and REACT_PATH.exists()
if not HAS_TEST_REPOS:
    print(f"Warning: Test repositories not found at: {RUST_ANALYZER_PATH}, {REACT_PATH}")
    print("Some tests will be skipped")

# Import the required modules
try:
    # Try to import the module
    from agent_tools.dualipa import code_extractor
    from agent_tools.dualipa.code_extractor import (
        _extract_python_blocks,
        _extract_js_ts_blocks,
        _extract_markdown_blocks,
        initialize_stats_dict
    )
    # Import verification function correctly
    from agent_tools.dualipa.verification.verify_code_blocks import verify_code_block
    
    HAS_DEPENDENCIES = True
except ImportError as e:
    # Instead of silently skipping, fail loudly with a clear error message
    raise ImportError(f"Required verification modules not available: {e}. Fix the dependencies to run these tests.")

# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required verification modules not available")

def create_test_block(language, content=None, valid=True):
    """Create a test code block for verification."""
    if content is None:
        if language == "python":
            content = "def test_function():\n    return 'test'" if valid else "def test_function(:\n    return 'test'"
        elif language == "javascript":
            content = "function test() {\n    return 'test';\n}" if valid else "function test() {\n    return 'test';\n"
        elif language == "rust":
            content = "fn test() {\n    println!(\"test\");\n}" if valid else "fn test() {\n    println!(\"test\");\n"
        else:
            content = "// Sample content"
    
    return {
        "language": language,
        "content": content,
        "path": f"test.{language}",
        "start_line": 1,
        "end_line": content.count('\n') + 1,
        "type": "function",
        "name": "test_function"
    }

@pytest.fixture(scope="session")
def requests_repo():
    """Fixture to provide the requests repository path."""
    requests_path = project_root / "test_repos" / "requests"
    if not requests_path.exists():
        pytest.skip("Requests repository not available")
    return requests_path

@pytest.fixture(scope="session")
def react_repo():
    """Fixture to provide the React repository path."""
    react_path = project_root / "test_repos" / "react"
    if not react_path.exists():
        pytest.skip("React repository not available")
    return react_path

def test_verify_python_block():
    """Test verification of Python code blocks."""
    # Create a valid Python block
    valid_block = {
        "language": "python",
        "content": """
def hello_world():
    print("Hello, world!")
    return "Hello, world!"
""",
        "file": "sample.py"
    }
    
    # Create an invalid Python block with syntax error
    invalid_block = {
        "language": "python",
        "content": """
def hello_world()
    print("Hello, world!")
    return "Hello, world!"
""",  # Missing colon
        "file": "sample.py"
    }
    
    # Test verification
    assert verify_code_block(valid_block), "Valid Python block should pass verification"
    assert not verify_code_block(invalid_block), "Invalid Python block should fail verification"
    
    print("Python block verification tests passed!")

def test_verify_javascript_block():
    """Test verification of JavaScript code blocks."""
    # Create a valid JavaScript block
    valid_block = {
        "language": "javascript",
        "content": """
function helloWorld() {
    console.log("Hello, world!");
    return "Hello, world!";
}
""",
        "file": "sample.js"
    }
    
    # Create an invalid JavaScript block with syntax error
    invalid_block = {
        "language": "javascript",
        "content": """
function helloWorld() {
    console.log("Hello, world!")
    return "Hello, world!"
""",  # Missing closing brace
        "file": "sample.js"
    }
    
    # Test verification
    assert verify_code_block(valid_block), "Valid JavaScript block should pass verification"
    assert not verify_code_block(invalid_block), "Invalid JavaScript block should fail verification"
    
    print("JavaScript block verification tests passed!")

def test_verify_blocks_from_extraction():
    """Test verifying blocks generated by the extraction process."""
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w') as f:
        f.write("""
def hello_world():
    print("Hello, world!")
    
class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""")
        f.flush()
        
        # Extract code blocks
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create test file for extraction
            with open(f.name, 'r') as file:
                content = file.read()
            
            # Set up stats dictionary for extraction with proper keys
            stats = initialize_stats_dict(source=f.name, output_dir=output_dir)
            
            # Extract blocks
            _extract_python_blocks(Path(f.name), content, output_dir, stats)
            
            assert stats["code_blocks"] > 0, "Should extract at least one code block"
            
            # Verify blocks
            blocks = []
            blocks_dir = output_dir / "blocks" / "code" / "python"
            if blocks_dir.exists():
                print(f"Found blocks directory at {blocks_dir}")
                block_files = list(blocks_dir.glob("*.py"))
                print(f"Found {len(block_files)} block files")
                
                for block_file in block_files:
                    with open(block_file, 'r') as bf:
                        block_content = bf.read()
                    
                    block = {
                        "language": "python",
                        "content": block_content,
                        "file": str(block_file)
                    }
                    
                    blocks.append(block)
                    print(f"Added block from {block_file.name}")
            
            # Verify each block
            verified_count = 0
            for block in blocks:
                result = verify_code_block(block)
                assert result, f"Block should be valid Python: {block['file']}"
                verified_count += 1
            
            print(f"Successfully verified {verified_count} blocks")
            assert verified_count > 0, "Should verify at least one block"

def test_verify_multifile_extraction(requests_repo, react_repo):
    """Test verifying blocks from multiple files of different languages."""
    # Get paths to specific test files we know exist
    python_file = requests_repo / "setup.py"
    js_file = react_repo / "fixtures" / "devtools" / "scheduling-profiler" / "run.js"
    
    test_files = []
    if python_file.exists():
        test_files.append(("python", python_file))
    if js_file.exists():
        test_files.append(("javascript", js_file))
    
    if not test_files:
        pytest.skip("No test files found in repositories")
    
    print(f"Testing with {len(test_files)} files")
    for lang, file in test_files:
        print(f"- {lang}: {file}")

    # Extract and verify blocks
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        total_verified = 0
        
        # Process each test file
        for lang, file_path in test_files:
            print(f"Processing {lang} file: {file_path}")
            
            # Read file content
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            
            # Set up stats dictionary for extraction with proper keys
            stats = initialize_stats_dict(source=str(file_path), output_dir=output_dir)
            
            # Extract blocks
            if lang == "python":
                _extract_python_blocks(file_path, content, output_dir, stats)
            elif lang == "javascript":
                _extract_js_ts_blocks(file_path, content, output_dir, stats)
            
            # Verify blocks
            blocks_dir = output_dir / "blocks" / "code" / lang
            if blocks_dir.exists():
                block_files = list(blocks_dir.glob(f"*.{lang}"))
                print(f"Found {len(block_files)} block files for {lang}")
                
                for block_file in block_files:
                    with open(block_file, 'r') as bf:
                        block_content = bf.read()
                    
                    block = {
                        "language": lang,
                        "content": block_content,
                        "file": str(block_file)
                    }
                    
                    result = verify_code_block(block)
                    assert result, f"Block should be valid {lang}: {block_file.name}"
                    total_verified += 1
                    print(f"Verified {block_file.name}")
        
        print(f"Successfully verified {total_verified} blocks across {len(test_files)} languages")
        assert total_verified > 0, "Should verify at least one block"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 