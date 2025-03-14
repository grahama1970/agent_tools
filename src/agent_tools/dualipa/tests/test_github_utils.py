"""
Test the github_utils module.

These tests verify that the GitHub utilities work correctly for
repository operations like URL parsing and cloning.

Official Documentation References:
- git: https://gitpython.readthedocs.io/en/stable/
- pytest: https://docs.pytest.org/en/latest/
- pytest-mock: https://pytest-mock.readthedocs.io/en/latest/
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the parent directory to the path to import dualipa modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Try to import the github_utils module
try:
    from github_utils import (
        parse_github_url,
        clone_github_repo,
        fetch_repo_contents_async,
        GIT_AVAILABLE,
        REQUESTS_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Skipping github_utils tests: {e}", allow_module_level=True)


def test_parse_github_url():
    """Test that GitHub URLs are parsed correctly."""
    # Test HTTPS URL format
    url = "https://github.com/username/repo"
    result = parse_github_url(url)
    assert result["owner"] == "username"
    assert result["repo"] == "repo"
    assert result["protocol"] == "https"
    
    # Test HTTPS URL with .git extension
    url = "https://github.com/username/repo.git"
    result = parse_github_url(url)
    assert result["owner"] == "username"
    assert result["repo"] == "repo"
    assert result["protocol"] == "https"
    
    # Test SSH URL format
    url = "git@github.com:username/repo.git"
    result = parse_github_url(url)
    assert result["owner"] == "username"
    assert result["repo"] == "repo"
    assert result["protocol"] == "ssh"
    
    # Test URL with branch
    url = "https://github.com/username/repo/tree/main"
    result = parse_github_url(url)
    assert result["owner"] == "username"
    assert result["repo"] == "repo"
    assert result["branch"] == "main"
    assert result["protocol"] == "https"
    
    # Test URL with subdirectory
    url = "https://github.com/username/repo/tree/main/folder"
    result = parse_github_url(url)
    assert result["owner"] == "username"
    assert result["repo"] == "repo"
    assert result["branch"] == "main"
    assert result["subdir"] == "folder"
    assert result["protocol"] == "https"
    
    # Test invalid URL
    with pytest.raises(ValueError):
        parse_github_url("invalid_url")


@pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not installed")
def test_clone_github_repo():
    """Test that GitHub repositories can be cloned."""
    # We'll use a mock for git.Repo.clone_from to avoid actual cloning
    with patch('github_utils.git.Repo.clone_from') as mock_clone:
        # Set up a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Mock the clone operation
            mock_clone.return_value = MagicMock()
            
            # Clone the repository
            repo_path = clone_github_repo("https://github.com/username/repo", temp_dir)
            
            # Verify that clone_from was called with the correct arguments
            mock_clone.assert_called_once()
            call_args = mock_clone.call_args[0]
            assert call_args[0] == "https://github.com/username/repo"
            assert call_args[1] == temp_dir
            
            # Verify that the returned path is correct
            assert repo_path == temp_dir
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="Requests not installed")
def test_fetch_repo_contents_async():
    """Test that repository contents can be fetched asynchronously."""
    # Mock the requests.get response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "name": "file1.py",
            "path": "file1.py",
            "type": "file",
            "download_url": "https://raw.githubusercontent.com/username/repo/main/file1.py"
        },
        {
            "name": "folder",
            "path": "folder",
            "type": "dir",
            "url": "https://api.github.com/repos/username/repo/contents/folder"
        }
    ]
    
    with patch('github_utils.requests.get', return_value=mock_response):
        import asyncio
        
        # Run the async function
        result = asyncio.run(fetch_repo_contents_async("username", "repo"))
        
        # Verify the result
        assert len(result) == 2
        assert result[0]["name"] == "file1.py"
        assert result[0]["type"] == "file"
        assert result[1]["name"] == "folder"
        assert result[1]["type"] == "dir"


def test_git_available_flag():
    """Test that the GIT_AVAILABLE flag is set correctly."""
    try:
        import git
        assert GIT_AVAILABLE is True
    except ImportError:
        assert GIT_AVAILABLE is False


def test_requests_available_flag():
    """Test that the REQUESTS_AVAILABLE flag is set correctly."""
    try:
        import requests
        assert REQUESTS_AVAILABLE is True
    except ImportError:
        assert REQUESTS_AVAILABLE is False 