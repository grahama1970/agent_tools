"""
Tests for GitHub repository operations.
These tests ensure repositories can be downloaded/cloned correctly.
Depends on: Smoke tests passing.
"""

import os
import pytest
import shutil
import tempfile
from pathlib import Path
from agent_tools.dualipa.github_utils import clone_github_repo

@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)

def test_clone_small_repo(temp_dir):
    """Test cloning a small GitHub repository."""
    repo_url = "https://github.com/git-fixtures/basic.git"
    repo_path = clone_github_repo(repo_url, temp_dir)
    assert Path(repo_path).exists(), "Repository not cloned"
    assert (Path(repo_path) / ".git").exists(), "Not a valid Git repository"

def test_repo_structure(temp_dir):
    """Test parsing a simple repository structure."""
    os.makedirs(os.path.join(temp_dir, "src"), exist_ok=True)
    with open(os.path.join(temp_dir, "src", "main.py"), "w") as f:
        f.write("def main(): pass")
    files = [f for f in os.walk(temp_dir)][0][2]
    assert "main.py" in files, "File not found in structure"