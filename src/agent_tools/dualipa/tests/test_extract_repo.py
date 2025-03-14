"""
Test the extract_repo module.

These tests verify that code and documentation can be extracted
correctly from repositories.

Official Documentation References:
- gitpython: https://gitpython.readthedocs.io/
- tempfile: https://docs.python.org/3/library/tempfile.html
- pytest: https://docs.pytest.org/
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import dualipa modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try to import the extract_repo module
try:
    from extract_repo import extract_from_repo, _extract_basic_file
except ImportError as e:
    pytest.skip(f"Skipping extract_repo tests: {e}", allow_module_level=True)


# Fixture for a test repository with sample files
@pytest.fixture
def test_repo():
    """Create a temporary directory with test files to simulate a repository."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create Python file
        with open(os.path.join(temp_dir, "example.py"), "w") as f:
            f.write("""
def hello_world():
    \"\"\"Say hello to the world.
    
    Returns:
        str: A greeting message
    \"\"\"
    return "Hello, World!"
            """)
        
        # Create Markdown file
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write("""# Test Repository
            
This is a test repository for the DuaLipa extractor.

## Usage

```python
from example import hello_world

print(hello_world())
```
            """)
        
        # Create text file
        with open(os.path.join(temp_dir, "notes.txt"), "w") as f:
            f.write("These are some notes about the repository.")
        
        # Create file to be ignored
        os.makedirs(os.path.join(temp_dir, "__pycache__"), exist_ok=True)
        with open(os.path.join(temp_dir, "__pycache__", "ignored.py"), "w") as f:
            f.write("# This should be ignored")
        
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def test_extract_from_repo(test_repo):
    """Test that files can be extracted from a repository."""
    # Create output directory
    output_dir = tempfile.mkdtemp()
    try:
        # Extract files from the test repository
        extract_from_repo(test_repo, output_dir)
        
        # Check if the output file exists
        output_file = os.path.join(output_dir, "extracted_data.json")
        assert os.path.exists(output_file), "Output file should exist"
        
        # Load the extracted data
        with open(output_file, "r") as f:
            data = json.load(f)
        
        # Verify the structure of the data
        assert "repo_path" in data
        assert "files" in data
        assert len(data["files"]) == 3  # example.py, README.md, notes.txt
        
        # Verify that the Python file was extracted
        py_file = next((f for f in data["files"] if f["path"].endswith(".py")), None)
        assert py_file is not None
        assert "hello_world" in py_file["content"]
        
        # Verify that the Markdown file was extracted with special handling
        md_file = next((f for f in data["files"] if f["path"].endswith(".md")), None)
        assert md_file is not None
        assert "Test Repository" in md_file["content"]
        assert "sections" in md_file
        assert "code_blocks" in md_file
        
        # Verify that the text file was extracted
        txt_file = next((f for f in data["files"] if f["path"].endswith(".txt")), None)
        assert txt_file is not None
        assert "notes about the repository" in txt_file["content"]
        
        # Verify that ignored files were not extracted
        ignored_files = [f for f in data["files"] if "__pycache__" in f["path"]]
        assert len(ignored_files) == 0
    finally:
        shutil.rmtree(output_dir)


def test_extract_with_custom_extensions(test_repo):
    """Test extraction with custom file extensions."""
    output_dir = tempfile.mkdtemp()
    try:
        # Extract only Python files
        extract_from_repo(test_repo, output_dir, file_extensions=[".py"])
        
        # Load the extracted data
        output_file = os.path.join(output_dir, "extracted_data.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        
        # Verify that only Python files were extracted
        assert len(data["files"]) == 1
        assert data["files"][0]["path"].endswith(".py")
    finally:
        shutil.rmtree(output_dir)


def test_extract_with_custom_ignore_patterns(test_repo):
    """Test extraction with custom ignore patterns."""
    output_dir = tempfile.mkdtemp()
    try:
        # Create a file that would normally be included but we'll ignore
        with open(os.path.join(test_repo, "ignore_me.py"), "w") as f:
            f.write("# This should be ignored")
        
        # Extract with custom ignore patterns
        extract_from_repo(
            test_repo, 
            output_dir, 
            ignore_patterns=["__pycache__", "ignore_me"]
        )
        
        # Load the extracted data
        output_file = os.path.join(output_dir, "extracted_data.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        
        # Verify that the ignored file was not extracted
        ignored_files = [f for f in data["files"] if "ignore_me.py" in f["path"]]
        assert len(ignored_files) == 0
    finally:
        shutil.rmtree(output_dir)


def test_extract_from_github_url():
    """Test extraction from a GitHub URL."""
    # Mock the GitHub cloning function
    with patch("extract_repo.clone_github_repo") as mock_clone:
        # Set up a temporary directory structure
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a test file in the temporary directory
            with open(os.path.join(temp_dir, "test.py"), "w") as f:
                f.write("print('Hello from GitHub')")
            
            # Mock the clone function to return our temp directory
            mock_clone.return_value = temp_dir
            
            # Set up output directory
            output_dir = tempfile.mkdtemp()
            try:
                # Extract from the "GitHub URL"
                extract_from_repo("https://github.com/username/repo", output_dir)
                
                # Verify that clone_github_repo was called
                mock_clone.assert_called_once()
                
                # Load the extracted data
                output_file = os.path.join(output_dir, "extracted_data.json")
                with open(output_file, "r") as f:
                    data = json.load(f)
                
                # Verify that the test file was extracted
                assert len(data["files"]) == 1
                assert data["files"][0]["path"] == "test.py"
                assert "Hello from GitHub" in data["files"][0]["content"]
            finally:
                shutil.rmtree(output_dir)
        finally:
            shutil.rmtree(temp_dir)


def test_extract_basic_file():
    """Test the basic file extraction helper function."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write("Test content")
    
    try:
        # Create a data structure to update
        extracted_data = {"files": []}
        
        # Extract the file
        _extract_basic_file(temp_path, "test_file.txt", extracted_data)
        
        # Verify the extraction
        assert len(extracted_data["files"]) == 1
        assert extracted_data["files"][0]["path"] == "test_file.txt"
        assert extracted_data["files"][0]["content"] == "Test content"
    finally:
        os.unlink(temp_path)


def test_extract_nonexistent_repo():
    """Test that extraction fails gracefully for nonexistent repositories."""
    with pytest.raises(FileNotFoundError):
        extract_from_repo("/path/does/not/exist", tempfile.mkdtemp()) 