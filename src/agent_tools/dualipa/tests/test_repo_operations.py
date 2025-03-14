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
from pathlib import Path


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
    assert "content" in data  # The content is base64 encoded
    assert "download_url" in data  # There should be a raw download URL


@pytest.mark.asyncio
async def test_async_repo_download():
    """Test asynchronously downloading repository content using aiohttp."""
    # Create a temporary directory for downloaded files
    temp_dir = tempfile.mkdtemp()
    try:
        # Define the files to download from a repository
        files_to_download = [
            {
                "owner": "huggingface",
                "repo": "transformers",
                "path": "README.md",
                "save_as": os.path.join(temp_dir, "transformers_readme.md")
            },
            {
                "owner": "pytorch",
                "repo": "pytorch",
                "path": "README.md",
                "save_as": os.path.join(temp_dir, "pytorch_readme.md")
            }
        ]
        
        # Function to download a file
        async def download_file(session, owner, repo, path, save_as):
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    with open(save_as, "w", encoding="utf-8") as f:
                        f.write(content)
                    return True
                return False
        
        # Download files asynchronously
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file_info in files_to_download:
                task = download_file(
                    session, 
                    file_info["owner"], 
                    file_info["repo"], 
                    file_info["path"], 
                    file_info["save_as"]
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        # Verify the downloads
        assert any(results)  # At least one download should succeed
        
        # Check the content of downloaded files
        downloaded_files = os.listdir(temp_dir)
        assert len(downloaded_files) > 0
        
        # Verify the content of one of the files
        for file_path in [os.path.join(temp_dir, file) for file in downloaded_files]:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # It should contain typical README content
                    assert len(content) > 0
                    assert any(keyword in content.lower() for keyword in 
                              ["installation", "getting started", "documentation", "license"])
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_repo_structure_parsing():
    """Test parsing repository structure to identify relevant files."""
    # Create a temporary directory with a mock repository structure
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a typical repository structure
        # - Root level
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write("# Test Repository\nThis is a test repository for parsing.")
            
        with open(os.path.join(temp_dir, "setup.py"), "w") as f:
            f.write("from setuptools import setup\nsetup(name='test', version='0.1')")
            
        # - Source directory
        src_dir = os.path.join(temp_dir, "src")
        os.makedirs(src_dir)
        
        with open(os.path.join(src_dir, "main.py"), "w") as f:
            f.write("def main():\n    print('Hello world')\n\nif __name__ == '__main__':\n    main()")
            
        # - Tests directory
        tests_dir = os.path.join(temp_dir, "tests")
        os.makedirs(tests_dir)
        
        with open(os.path.join(tests_dir, "test_main.py"), "w") as f:
            f.write("import unittest\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        pass")
            
        # - Docs directory
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir)
        
        with open(os.path.join(docs_dir, "index.rst"), "w") as f:
            f.write("Welcome to the documentation!")
        
        # Parse the repository to find Python, Markdown, and RST files
        python_files = []
        doc_files = []
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext == ".py":
                    python_files.append(file_path)
                elif file_ext in [".md", ".rst"]:
                    doc_files.append(file_path)
        
        # Verify the parsing
        assert len(python_files) == 3  # setup.py, main.py, test_main.py
        assert len(doc_files) == 2  # README.md, index.rst
        
        # Verify file paths
        python_file_names = [os.path.basename(f) for f in python_files]
        assert "setup.py" in python_file_names
        assert "main.py" in python_file_names
        assert "test_main.py" in python_file_names
        
        doc_file_names = [os.path.basename(f) for f in doc_files]
        assert "README.md" in doc_file_names
        assert "index.rst" in doc_file_names
    finally:
        # Clean up
        shutil.rmtree(temp_dir) 