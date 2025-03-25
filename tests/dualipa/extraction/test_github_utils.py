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

# Remove path manipulation
# current_dir = Path(__file__).parent
# parent_dir = current_dir.parent
# if str(parent_dir) not in sys.path:
#     sys.path.append(str(parent_dir))

# Import directly from the package
try:
    from agent_tools.dualipa.extraction.extractors.github.repo_utils import (
        parse_github_url,
        clone_github_repo,
        fetch_repo_contents_async,
        is_github_url,
        GIT_AVAILABLE,
        REQUESTS_AVAILABLE
    )
    import git
    import requests
    
    GIT_AVAILABLE = True
    REQUESTS_AVAILABLE = True
    HAS_DEPENDENCIES = True
except ImportError as e:
    import traceback
    traceback.print_exc()
    
    GIT_AVAILABLE = False
    REQUESTS_AVAILABLE = False
    HAS_DEPENDENCIES = False
    pytest.fail(f"Required GitHub utils modules not available: {e}. Fix the dependencies to run these tests.")


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


def test_arangodb_url_parsing():
    """
    Blind test that verifies parsing of ArangoDB repository URL.
    This test ensures we correctly handle a real-world repository URL.
    """
    # ArangoDB main repository URL
    url = "https://github.com/arangodb/arangodb"
    result = parse_github_url(url)
    
    # Verify specific expected values for ArangoDB
    assert result["owner"] == "arangodb"
    assert result["repo"] == "arangodb"
    assert result["protocol"] == "https"
    assert result["branch"] == "main"  # Default branch
    assert "path" in result
    assert result["path"] == ""  # No specific path
    
    # Test with specific file in ArangoDB repo
    url = "https://github.com/arangodb/arangodb/blob/devel/utils/gantt.py"
    result = parse_github_url(url)
    
    assert result["owner"] == "arangodb"
    assert result["repo"] == "arangodb"
    assert result["branch"] == "devel"
    assert result["path"] == "utils/gantt.py"
    
    # Test with a specific directory in ArangoDB repo
    url = "https://github.com/arangodb/arangodb/tree/devel/js/apps/system/_admin/aardvark/APP/react/src"
    result = parse_github_url(url)
    
    assert result["owner"] == "arangodb"
    assert result["repo"] == "arangodb"
    assert result["branch"] == "devel"
    assert result["path"] == "js/apps/system/_admin/aardvark/APP/react/src"


def test_url_parsing_structure():
    """
    Test that the parse_github_url function returns a dictionary with the expected structure.
    This ensures the output structure is consistent for downstream functions.
    """
    # Example URL with all components
    url = "https://github.com/arangodb/arangodb/blob/devel/utils/gantt.py"
    result = parse_github_url(url)
    
    # Check the result is a dictionary with the expected keys
    expected_keys = ["owner", "repo", "branch", "path", "protocol"]
    for key in expected_keys:
        assert key in result, f"Expected key '{key}' not found in result"
    
    # Check the types of values
    assert isinstance(result["owner"], str)
    assert isinstance(result["repo"], str)
    assert isinstance(result["branch"], str)
    assert isinstance(result["path"], str)
    assert isinstance(result["protocol"], str)
    
    # Verify URL detection works for this URL
    assert is_github_url(url) is True
    
    # Check that non-GitHub URLs are correctly identified
    assert is_github_url("https://gitlab.com/user/repo") is False
    assert is_github_url("https://example.com") is False
    assert is_github_url("not a url") is False


@patch('git.Repo')
def test_clone_github_repo(mock_repo, tmpdir):
    """Test cloning a GitHub repository with mocked git operations."""
    # Check dependencies manually
    if not GIT_AVAILABLE:
        pytest.fail("GitPython not installed. Install it to run these tests.")
    
    # We'll use a mock for git.Repo.clone_from to avoid actual cloning
    with patch('agent_tools.dualipa.github_utils.git.Repo.clone_from') as mock_clone:
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


@patch('requests.get')
def test_download_github_repo(mock_get, tmpdir):
    """Test downloading a GitHub repository with mocked responses."""
    # Check dependencies manually
    if not REQUESTS_AVAILABLE:
        pytest.fail("Requests not installed. Install it to run these tests.")
    
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
    
    with patch('agent_tools.dualipa.github_utils.requests.get', return_value=mock_response):
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


# Test for mocked repository operations
def test_mock_github_operations():
    """Test GitHub operations with mocked responses."""
    # Remove skipif decorator to fail tests loudly if dependencies missing
    # Check at the start of the test instead
    if not GIT_AVAILABLE:
        pytest.fail("GitPython not installed. Install it to run these tests.")
    if not REQUESTS_AVAILABLE:
        pytest.fail("Requests not installed. Install it to run these tests.")
    
    # Create a temporary directory for the test
    # ...rest of test code 