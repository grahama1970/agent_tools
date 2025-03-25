"""
Test the github_utils module using local test repositories.

Official Documentation References:
- GitPython: https://gitpython.readthedocs.io/en/stable/
- GitHub API: https://docs.github.com/en/rest
- Requests: https://requests.readthedocs.io/en/latest/
- aiohttp: https://docs.aiohttp.org/en/stable/
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

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

# Test repository paths
TEST_REPOS_DIR = Path("/home/grahama/workspace/experiments/agent_tools/test_repos")
PYTHON_SAMPLE = TEST_REPOS_DIR / "python-sample"
TS_SAMPLE = TEST_REPOS_DIR / "typescript-sample"
REQUESTS_REPO = TEST_REPOS_DIR / "requests"

def verify_repo_structure(repo_path: Path, repo_type: str = None) -> bool:
    """Verify that a repository has a valid structure.
    
    Args:
        repo_path: Path to the repository
        repo_type: Optional type of repository (python, typescript, etc.)
        
    Returns:
        bool: True if repository structure is valid
    """
    if not repo_path.exists() or not repo_path.is_dir():
        return False
        
    # Check for either .git directory or files
    has_content = (repo_path / ".git").exists() or any(f.is_file() for f in repo_path.iterdir())
    if not has_content:
        return False
    
    # Language-specific checks
    if repo_type == "python":
        # Python repos should have either setup.py, pyproject.toml, or requirements.txt
        has_python_files = any(
            (repo_path / f).exists() 
            for f in ["setup.py", "pyproject.toml", "requirements.txt"]
        )
        has_py_files = any(f.suffix == ".py" for f in repo_path.rglob("*.py"))
        return has_python_files and has_py_files
        
    elif repo_type == "typescript":
        # TypeScript repos should have package.json and tsconfig.json
        has_ts_config = (repo_path / "tsconfig.json").exists()
        has_package_json = (repo_path / "package.json").exists()
        has_ts_files = any(f.suffix == ".ts" for f in repo_path.rglob("*.ts"))
        return has_package_json and (has_ts_config or has_ts_files)
        
    # Default check just verifies directory exists and has files
    return True

def test_parse_github_url():
    """Test that GitHub URLs are parsed correctly using real repository examples."""
    # Test with real repository URLs
    urls_and_expected = [
        (
            "https://github.com/psf/requests",
            {"owner": "psf", "repo": "requests", "protocol": "https"}
        ),
        (
            "git@github.com:psf/requests.git",
            {"owner": "psf", "repo": "requests", "protocol": "ssh"}
        ),
        (
            "https://github.com/psf/requests/tree/main/requests",
            {
                "owner": "psf",
                "repo": "requests",
                "protocol": "https",
                "branch": "main",
                "subdir": "requests"
            }
        )
    ]
    
    for url, expected in urls_and_expected:
        result = parse_github_url(url)
        for key, value in expected.items():
            assert result[key] == value, f"Failed parsing {url} for key {key}"

def test_local_repository_operations(tmp_path: Path):
    """Test repository operations using local test repositories."""
    # Test with python-sample repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    # Copy repository to temporary location
    temp_repo = tmp_path / "python-sample"
    shutil.copytree(PYTHON_SAMPLE, temp_repo)
    
    assert verify_repo_structure(temp_repo), "Failed to copy Python sample repository"
    
    # Test with typescript repository
    if not TS_SAMPLE.exists():
        pytest.skip(f"TypeScript sample repository not found at {TS_SAMPLE}")
    
    temp_ts_repo = tmp_path / "typescript-sample"
    shutil.copytree(TS_SAMPLE, temp_ts_repo)
    
    assert verify_repo_structure(temp_ts_repo), "Failed to copy TypeScript sample repository"

@pytest.mark.skipif(not GIT_AVAILABLE, reason="GitPython not installed")
def test_repository_cloning(tmp_path: Path):
    """Test repository cloning using local repositories."""
    # Test with Python repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    # Test cloning Python repository
    python_clone_dir = tmp_path / "cloned-python"
    # Use git.Repo.clone_from directly for local repos
    git.Repo.clone_from(str(PYTHON_SAMPLE), str(python_clone_dir))
    assert verify_repo_structure(python_clone_dir, "python"), \
        "Failed to verify Python repository structure"
    
    # Test with TypeScript repository
    if not TS_SAMPLE.exists():
        pytest.skip(f"TypeScript sample repository not found at {TS_SAMPLE}")
    
    # Test cloning TypeScript repository
    ts_clone_dir = tmp_path / "cloned-typescript"
    git.Repo.clone_from(str(TS_SAMPLE), str(ts_clone_dir))
    assert verify_repo_structure(ts_clone_dir, "typescript"), \
        "Failed to verify TypeScript repository structure"
    
    # Test with Requests repository (real Git repository)
    if not REQUESTS_REPO.exists():
        pytest.skip(f"Requests repository not found at {REQUESTS_REPO}")
    
    # Test cloning with specific branch
    requests_clone_dir = tmp_path / "cloned-requests"
    git.Repo.clone_from(str(REQUESTS_REPO), str(requests_clone_dir))
    assert verify_repo_structure(requests_clone_dir, "python"), \
        "Failed to verify Requests repository structure"
    assert (requests_clone_dir / ".git").exists(), \
        "Cloned repository missing .git directory"
    assert (requests_clone_dir / "setup.py").exists(), \
        "Cloned repository missing setup.py"

@pytest.mark.asyncio
async def test_fetch_contents_from_local():
    """Test fetching contents from local repositories."""
    # Test with python-sample repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    # Get expected contents
    expected_contents = []
    for item in PYTHON_SAMPLE.rglob("*"):
        if item.is_file():
            expected_contents.append({
                "name": item.name,
                "path": str(item.relative_to(PYTHON_SAMPLE)),
                "type": "file"
            })
    
    # Test the actual async function
    contents = await fetch_repo_contents_async(
        owner="local",
        repo=str(PYTHON_SAMPLE),
        path=""  # Add path parameter
    )
    
    assert len(contents) > 0, "No files found in Python sample repository"
    assert len(contents) == len(expected_contents), "Mismatch in number of files"
    
    # Verify specific files exist
    setup_py = next((f for f in contents if f["name"] == "setup.py"), None)
    assert setup_py is not None, "setup.py not found"
    assert setup_py["type"] == "file"
    
    # Test with typescript repository
    if not TS_SAMPLE.exists():
        pytest.skip(f"TypeScript sample repository not found at {TS_SAMPLE}")
    
    ts_contents = await fetch_repo_contents_async(
        owner="local",
        repo=str(TS_SAMPLE),
        path=""  # Add path parameter
    )
    
    assert len(ts_contents) > 0, "No files found in TypeScript sample repository"
    package_json = next((f for f in ts_contents if f["name"] == "package.json"), None)
    assert package_json is not None, "package.json not found"
    assert package_json["type"] == "file"

def test_url_detection():
    """Test URL detection with both valid and invalid URLs."""
    # Valid GitHub URLs
    valid_urls = [
        "https://github.com/psf/requests",
        "https://github.com/psf/requests.git",
        "git@github.com:psf/requests.git",
        "https://github.com/psf/requests/tree/main"
    ]
    
    for url in valid_urls:
        assert is_github_url(url), f"Failed to recognize valid GitHub URL: {url}"
    
    # Invalid URLs
    invalid_urls = [
        "https://gitlab.com/user/repo",
        "https://bitbucket.org/user/repo",
        "not a url",
        "git@gitlab.com:user/repo.git"
    ]
    
    for url in invalid_urls:
        assert not is_github_url(url), f"Incorrectly recognized invalid URL: {url}"

@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="Requests not installed")
def test_network_error_handling():
    """Test handling of network errors when accessing repositories."""
    # Test invalid repository URL
    with pytest.raises(ValueError, match="Not a GitHub URL"):
        clone_github_repo("https://not-github.com/user/repo")
    
    # Test non-existent repository
    with pytest.raises(git.GitCommandError, match="Repository not found"):
        clone_github_repo("https://github.com/user/repo")
    
    # Test invalid local path
    with pytest.raises(ValueError, match="Not a GitHub URL"):
        clone_github_repo(str(TEST_REPOS_DIR / "nonexistent"))
    
    # Test private repository (should fail with Repository not found)
    with pytest.raises(git.GitCommandError, match="Repository not found"):
        clone_github_repo("https://github.com/private/repo")

def test_url_parsing_errors():
    """Test error handling in URL parsing."""
    invalid_urls = [
        ("https://not-github.com/user/repo", "Not a GitHub URL"),
        ("github.com/user/repo", "Not a GitHub URL"),
        ("https://github.com/", "Invalid GitHub repository path"),
        ("https://github.com/user", "Invalid GitHub repository path"),
        ("git@not-github.com:user/repo.git", "Invalid GitHub SSH URL")
    ]
    
    for url, expected_error in invalid_urls:
        with pytest.raises(ValueError) as excinfo:
            parse_github_url(url)
        assert str(excinfo.value) == expected_error, f"Expected error '{expected_error}' but got '{str(excinfo.value)}' for URL '{url}'"

def test_local_path_handling():
    """Test handling of local repository paths."""
    # Test with Python repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    # Test direct cloning with git.Repo
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.clone_from(str(PYTHON_SAMPLE), temp_dir)
        assert repo.git_dir is not None, "Failed to clone local repository"
        assert Path(temp_dir).exists(), "Clone directory not created"
        assert (Path(temp_dir) / ".git").exists(), "Git directory not created"

def test_cleanup(tmp_path: Path):
    """Test cleanup of temporary directories."""
    # Test with Python repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    # Test cleanup after successful clone
    clone_dir = tmp_path / "cleanup-test"
    repo = git.Repo.clone_from(str(PYTHON_SAMPLE), str(clone_dir))
    assert clone_dir.exists(), "Clone directory not created"
    
    # Clean up
    shutil.rmtree(clone_dir)
    assert not clone_dir.exists(), "Clone directory not cleaned up"
    
    # Test cleanup after failed clone
    bad_dir = tmp_path / "bad-clone"
    try:
        git.Repo.clone_from("https://github.com/invalid/repo", str(bad_dir))
    except:
        pass
    assert not bad_dir.exists() or not any(bad_dir.iterdir()), "Failed clone directory not cleaned up"

def test_repository_structure():
    """Test repository structure verification."""
    # Test with Python repository
    if not PYTHON_SAMPLE.exists():
        pytest.skip(f"Python sample repository not found at {PYTHON_SAMPLE}")
    
    assert verify_repo_structure(PYTHON_SAMPLE, "python"), \
        "Failed to verify Python repository structure"
    
    # Test with TypeScript repository
    if not TS_SAMPLE.exists():
        pytest.skip(f"TypeScript sample repository not found at {TS_SAMPLE}")
    
    assert verify_repo_structure(TS_SAMPLE, "typescript"), \
        "Failed to verify TypeScript repository structure"
    
    # Test with invalid path
    assert not verify_repo_structure(Path("/nonexistent")), \
        "Should fail for non-existent directory"
    
    # Test with empty directory
    with tempfile.TemporaryDirectory() as temp_dir:
        assert not verify_repo_structure(Path(temp_dir)), \
            "Should fail for empty directory" 