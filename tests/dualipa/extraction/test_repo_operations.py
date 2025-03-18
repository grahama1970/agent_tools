"""
Tests for basic GitHub repository operations.

Official Documentation References:
- requests: https://requests.readthedocs.io/en/latest/
- git: https://gitpython.readthedocs.io/en/stable/
- tempfile: https://docs.python.org/3/library/tempfile.html
- os: https://docs.python.org/3/library/os.html
- asyncio: https://docs.python.org/3/library/asyncio.html
- aiohttp: https://docs.aiohttp.org/en/stable/
"""

import pytest
import os
import tempfile
import shutil
import asyncio
import aiohttp
import requests
import git
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import directly from the package
try:
    from agent_tools.dualipa.extract_repo import extract_from_repo
    from agent_tools.dualipa.code_extractor import (
        extract_repository,
        _extract_python_blocks,
        _extract_js_ts_blocks,
        _extract_markdown_blocks,
        _extract_generic_blocks
    )
    from agent_tools.dualipa.github_utils import download_github_repo, clone_github_repo
    HAS_DEPENDENCIES = True
except ImportError as e:
    import traceback
    traceback.print_exc()
    HAS_DEPENDENCIES = False
    pytest.fail(f"Required repository operation modules not available: {e}. Fix the dependencies to run these tests.")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    # Clean up after the test
    shutil.rmtree(dir_path)


@pytest.fixture
def fixture_repo(temp_dir):
    """Create a fixture repository with test files."""
    # Create a Python file
    py_file = os.path.join(temp_dir, "test_file.py")
    with open(py_file, "w") as f:
        f.write("""
def test_function():
    \"\"\"This is a test function.\"\"\"
    return "Hello, world!"

class TestClass:
    \"\"\"This is a test class.\"\"\"
    
    def method(self):
        \"\"\"This is a test method.\"\"\"
        return "Hello from TestClass!"
""")
    
    # Create a TypeScript file
    ts_file = os.path.join(temp_dir, "test_file.ts")
    with open(ts_file, "w") as f:
        f.write("""
function greet(name: string): string {
    return `Hello, ${name}!`;
}

class Person {
    name: string;
    
    constructor(name: string) {
        this.name = name;
    }
    
    sayHello(): void {
        console.log(greet(this.name));
    }
}

interface Shape {
    area(): number;
}
""")
    
    # Create a Markdown file
    md_file = os.path.join(temp_dir, "test_file.md")
    with open(md_file, "w") as f:
        f.write("""# Test Markdown File

This is a test markdown file.

## Section 1

Content for section 1.

```python
def example_code():
    return "This is some example code"
```

## Section 2

Content for section 2.
""")
    
    return {
        "root": temp_dir,
        "py_file": py_file,
        "ts_file": ts_file,
        "md_file": md_file
    }


def test_clone_small_repo():
    """Test cloning a small GitHub repository."""
    # Create a temporary directory for the cloned repo
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone a very small test repo
        repo_url = "https://github.com/git-fixtures/basic.git"
        repo = git.Repo.clone_from(repo_url, temp_dir)
        
        # Verify that it was cloned correctly
        assert isinstance(repo, git.Repo)
        assert os.path.exists(os.path.join(temp_dir, ".git"))
        assert repo.git.rev_parse("HEAD")  # Check that we can get the HEAD commit hash
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_repo_file_extraction():
    """Test extracting files from a repository with specific extensions."""
    # Create a temporary directory for the cloned repo
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone a test repo with various file types
        repo_url = "https://github.com/python-git/python.git"  # Python documentation repo (small)
        git.Repo.clone_from(repo_url, temp_dir, depth=1)  # Use depth=1 for faster cloning
        
        # Extract Python and Markdown files
        extensions = {".py", ".md", ".rst"}
        extracted_files = []
        
        for root, _, files in os.walk(temp_dir):
            # Skip .git directory
            if os.path.basename(root) == ".git":
                continue
                
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    file_path = os.path.join(root, file)
                    extracted_files.append(file_path)
        
        # Verify extraction
        assert len(extracted_files) > 0
        
        # Check that we have at least one file of each expected extension
        exts_found = {os.path.splitext(f)[1].lower() for f in extracted_files}
        assert ".py" in exts_found
        
        # Read content of one Python file to verify it's valid
        py_files = [f for f in extracted_files if f.endswith(".py")]
        if py_files:
            with open(py_files[0], "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Verify it contains typical Python content
                assert "def " in content or "class " in content or "import " in content
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_api_repo_contents():
    """Test fetching repository contents using the GitHub API."""
    # Use the GitHub API to get the contents of a repository
    repo_owner = "huggingface"
    repo_name = "transformers"
    path = "README.md"
    
    # Make a request to the GitHub API
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    response = requests.get(url)
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "README.md"
    assert data["type"] == "file"


@pytest.mark.asyncio
async def test_async_repo_download():
    """Test asynchronously downloading files from a repository."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Define a list of files to download from the GitHub repo
        repo_owner = "arangodb"
        repo_name = "arangodb"
        files_to_download = [
            "README.md",
            "utils/gantt.py",  
            "js/apps/system/_admin/aardvark/APP/react/src/global.d.ts"
        ]
        
        async def download_file(session, owner, repo, path, save_as):
            """Download a file from a GitHub repository."""
            # Construct the raw URL
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/devel/{path}"
            
            # Send the request
            async with session.get(url) as response:
                if response.status == 404:
                    return False, f"File not found: {path}"
                
                if response.status != 200:
                    return False, f"Error downloading {path}: {response.status}"
                
                # Read the content
                content = await response.text()
                
                # Save to disk
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
                with open(save_as, "w", encoding="utf-8") as f:
                    f.write(content)
                
                return True, save_as
        
        # Download the files asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file_path in files_to_download:
                save_path = os.path.join(temp_dir, file_path)
                task = download_file(session, repo_owner, repo_name, file_path, save_path)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # Verify the results
        successful_downloads = [r for r in results if r[0]]
        assert len(successful_downloads) > 0
        
        # Check that at least one file was downloaded and has content
        for success, path in successful_downloads:
            if success and os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    assert len(content) > 0
                    break
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_repo_structure_parsing():
    """Test parsing repository structure."""
    # Create a temporary directory to act as a repository
    temp_dir = tempfile.mkdtemp()
    try:
        # Create directories and files to simulate a repository structure
        os.makedirs(os.path.join(temp_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "docs"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "tests"), exist_ok=True)
        
        # Create Python files
        with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
            f.write("def main(): print('Hello, world!')")
        
        with open(os.path.join(temp_dir, "src", "utils.py"), "w") as f:
            f.write("def helper(): return 'Helper function'")
        
        # Create a Markdown doc
        with open(os.path.join(temp_dir, "docs", "readme.md"), "w") as f:
            f.write("# Test Repository\n\nThis is a test repository.")
        
        # Create test file
        with open(os.path.join(temp_dir, "tests", "test_main.py"), "w") as f:
            f.write("def test_main(): assert True")
        
        # Parse the repository structure
        structure = {"dirs": [], "files": []}
        
        for root, dirs, files in os.walk(temp_dir):
            rel_path = os.path.relpath(root, temp_dir)
            if rel_path != ".":
                structure["dirs"].append(rel_path)
            
            for file in files:
                file_path = os.path.join(rel_path, file)
                if file_path.startswith("./"):
                    file_path = file_path[2:]
                structure["files"].append(file_path)
        
        # Verify structure
        assert "src" in structure["dirs"]
        assert "docs" in structure["dirs"]
        assert "tests" in structure["dirs"]
        
        # Check for files
        assert any(f.endswith("main.py") for f in structure["files"])
        assert any(f.endswith("utils.py") for f in structure["files"])
        assert any(f.endswith("readme.md") for f in structure["files"])
        assert any(f.endswith("test_main.py") for f in structure["files"])
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_extract_blocks_from_fixture(fixture_repo, temp_dir):
    """
    Test extraction of code blocks from test fixture files with expected content.
    This is a blind test with known expected output.
    """
    # Directory to store extracted blocks
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all files from the fixture repo
    stats = extract_repository(
        source=fixture_repo["root"],
        output_path=output_dir,
        max_files=10
    )
    
    # Verify blocks.json was created
    blocks_json_path = os.path.join(output_dir, "blocks.json")
    assert os.path.exists(blocks_json_path), "blocks.json not created"
    
    # Load the blocks
    with open(blocks_json_path, "r") as f:
        blocks = json.load(f)
    
    # Check that blocks were extracted
    assert len(blocks) > 0, "No blocks were extracted"
    
    # Verify we have blocks from all file types
    languages = set(block.get("language", "") for block in blocks)
    assert "python" in languages, "No Python blocks extracted"
    assert "typescript" in languages or "javascript" in languages, "No TypeScript/JavaScript blocks extracted"
    assert "markdown" in languages, "No Markdown blocks extracted"
    
    # Check for specific expected blocks from Python file
    py_file_path = str(Path(fixture_repo["py_file"]))
    py_blocks = [b for b in blocks if b.get("file") == py_file_path]
    
    # Verify Python blocks
    function_names = [b.get("name", "") for b in py_blocks if b.get("block_type") == "function"]
    class_names = [b.get("name", "") for b in py_blocks if b.get("block_type") == "class"]
    
    assert "test_function" in function_names, "Expected Python function 'test_function' not found"
    assert "TestClass" in class_names, "Expected Python class 'TestClass' not found"
    assert "method" in function_names, "Expected Python method 'method' not found"
    
    # Check for specific expected blocks from TypeScript file
    ts_file_path = str(Path(fixture_repo["ts_file"]))
    ts_blocks = [b for b in blocks if b.get("file") == ts_file_path]
    
    # Verify TypeScript blocks
    ts_names = [b.get("name", "") for b in ts_blocks]
    assert "greet" in ts_names, "Expected TypeScript function 'greet' not found"
    assert "Person" in ts_names, "Expected TypeScript class 'Person' not found"
    assert "Shape" in ts_names, "Expected TypeScript interface 'Shape' not found"
    
    # Check for specific expected blocks from Markdown file
    md_file_path = str(Path(fixture_repo["md_file"]))
    md_blocks = [b for b in blocks if b.get("file") == md_file_path]
    
    # Verify Markdown blocks
    md_titles = [b.get("title", "") for b in md_blocks]
    assert any("Test_Markdown_File" in title for title in md_titles), "Expected Markdown title not found"
    assert any("Section_1" in title for title in md_titles), "Expected Markdown section not found"
    assert any("Section_2" in title for title in md_titles), "Expected Markdown section not found"
    
    # Verify content of a specific block (test_function)
    test_function_block = next((b for b in py_blocks if b.get("name") == "test_function"), None)
    assert test_function_block is not None, "test_function block not found"
    assert "This is a test function" in test_function_block.get("content", ""), "Expected docstring not found in test_function"
    
    # Verify structure of blocks
    required_fields = ["type", "language", "content", "file", "output_file"]
    for block in blocks:
        for field in required_fields:
            assert field in block, f"Block missing required field: {field}"
        
        # Check field types
        assert isinstance(block["type"], str)
        assert isinstance(block["language"], str)
        assert isinstance(block["content"], str)
        assert isinstance(block["file"], str)
        
        # Verify block has actual content
        assert len(block["content"]) > 0, "Block has empty content"
        
        # Verify output file exists
        output_file = block.get("output_file")
        if output_file:
            assert os.path.exists(output_file), f"Output file doesn't exist: {output_file}"


def test_extraction_stats_structure(fixture_repo, temp_dir):
    """
    Test that the extraction stats have the expected structure and content.
    This verifies the metadata collected during extraction.
    """
    # Directory to store extraction results
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all files from the fixture repo
    stats = extract_repository(
        source=fixture_repo["root"],
        output_path=output_dir,
        max_files=10
    )
    
    # Verify stats.json was created
    stats_json_path = os.path.join(output_dir, "extraction_stats.json")
    assert os.path.exists(stats_json_path), "extraction_stats.json not created"
    
    # Load the stats
    with open(stats_json_path, "r") as f:
        stats_data = json.load(f)
    
    # Verify the stats have the required fields
    required_fields = [
        "source", "output_path", "start_time", "end_time", "duration_seconds",
        "total_files", "code_files", "documentation_files", "code_blocks",
        "doc_blocks", "languages", "file_types", "errors", "file_blocks"
    ]
    
    for field in required_fields:
        assert field in stats_data, f"Stats missing required field: {field}"
    
    # Verify counts match expectations
    assert stats_data["total_files"] == 3, "Expected 3 total files"
    assert stats_data["code_files"] == 2, "Expected 2 code files"
    assert stats_data["documentation_files"] == 1, "Expected 1 documentation file"
    
    # Verify languages detected
    assert "python" in stats_data["languages"], "Python language not detected"
    assert "typescript" in stats_data["languages"] or "javascript" in stats_data["languages"], "TypeScript/JavaScript language not detected"
    
    # Verify file types detected
    assert ".py" in stats_data["file_types"], ".py file type not detected"
    assert ".ts" in stats_data["file_types"], ".ts file type not detected"
    assert ".md" in stats_data["file_types"], ".md file type not detected"
    
    # Verify file_blocks structure
    assert isinstance(stats_data["file_blocks"], dict), "file_blocks should be a dictionary"
    
    # Check that each file in the repo has entries in file_blocks
    py_file_path = str(Path(fixture_repo["py_file"]))
    ts_file_path = str(Path(fixture_repo["ts_file"]))
    md_file_path = str(Path(fixture_repo["md_file"]))
    
    assert py_file_path in stats_data["file_blocks"], "Python file not in file_blocks"
    assert ts_file_path in stats_data["file_blocks"], "TypeScript file not in file_blocks"
    assert md_file_path in stats_data["file_blocks"], "Markdown file not in file_blocks"
    
    # Verify blocks for each file
    py_blocks = stats_data["file_blocks"][py_file_path]
    ts_blocks = stats_data["file_blocks"][ts_file_path]
    md_blocks = stats_data["file_blocks"][md_file_path]
    
    assert len(py_blocks) > 0, "No Python blocks extracted"
    assert len(ts_blocks) > 0, "No TypeScript blocks extracted"
    assert len(md_blocks) > 0, "No Markdown blocks extracted" 