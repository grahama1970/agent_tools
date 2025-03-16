"""
GitHub utilities for DuaLipa.

Provides functionality to interact with GitHub repositories,
including cloning, downloading, and accessing repository content.

Official Documentation References:
- pygithub: https://pygithub.readthedocs.io/en/latest/
- gitpython: https://gitpython.readthedocs.io/en/stable/tutorial.html
- loguru: https://loguru.readthedocs.io/en/stable/
"""

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple, Any
from urllib.parse import urlparse
import json
from loguru import logger

try:
    import git
    GIT_AVAILABLE = True
    logger.info("GitPython is available for repository operations")
except ImportError:
    GIT_AVAILABLE = False
    logger.warning("GitPython not available, full Git functionality will be limited")

try:
    from github import Github, Auth, Repository, ContentFile
    GITHUB_API_AVAILABLE = True
    logger.info("PyGithub is available for GitHub API access")
except ImportError:
    GITHUB_API_AVAILABLE = False
    logger.warning("PyGithub not available, GitHub API access will be limited")

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Check if any GitHub library is available
if not (GIT_AVAILABLE or GITHUB_API_AVAILABLE):
    logger.error("No GitHub libraries available. Install gitpython or PyGithub.")


def parse_github_url(url: str) -> Dict[str, str]:
    """Parse a GitHub URL into components.
    
    Args:
        url: The GitHub URL to parse
        
    Returns:
        Dictionary with extracted components (owner, repo, path, branch)
    
    Examples:
        >>> parse_github_url("https://github.com/username/repo")
        {'owner': 'username', 'repo': 'repo', 'path': '', 'branch': 'main'}
        
        >>> parse_github_url("https://github.com/username/repo/tree/main/src")
        {'owner': 'username', 'repo': 'repo', 'path': 'src', 'branch': 'main'}
    """
    result = {
        "owner": "",
        "repo": "",
        "path": "",
        "branch": "main"  # Default to main branch
    }
    
    if not url:
        return result
    
    # Handle various URL formats
    # Remove any trailing .git
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]
    
    # Handle SSH URLs like git@github.com:username/repo.git
    if url.startswith('git@github.com:'):
        # Convert SSH format to HTTPS format for parsing
        username_repo = url[15:]  # Remove git@github.com:
        if username_repo.endswith('.git'):
            username_repo = username_repo[:-4]  # Remove .git
        
        parts = username_repo.split('/')
        if parts:
            result["owner"] = parts[0]
            if len(parts) > 1:
                result["repo"] = parts[1]
                # Handle any additional path components
                if len(parts) > 2:
                    result["path"] = '/'.join(parts[2:])
        return result
    
    # Parse URL for HTTP/HTTPS format
    parsed_url = urlparse(url)
    
    # Skip if not a GitHub URL
    if not parsed_url.netloc.endswith("github.com"):
        return result
    
    # Extract owner and repo from path
    path_parts = [p for p in parsed_url.path.strip('/').split("/") if p]
    if len(path_parts) >= 2:
        result["owner"] = path_parts[0]
        result["repo"] = path_parts[1]
    else:
        # Not enough parts to extract owner and repo
        return result
    
    # Extract branch and path if available
    if len(path_parts) >= 4 and path_parts[2] in ["tree", "blob"]:
        result["branch"] = path_parts[3] or "main"  # Default to main if empty
        if len(path_parts) >= 5:
            result["path"] = "/".join(path_parts[4:])
    
    return result


def is_github_url(url: str) -> bool:
    """Check if a URL is a GitHub repository URL.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is a GitHub repository URL, False otherwise
        
    Examples:
        >>> is_github_url("https://github.com/username/repo")
        True
        >>> is_github_url("https://github.com/username/repo.git")
        True
        >>> is_github_url("git@github.com:username/repo.git")
        True
        >>> is_github_url("https://gitlab.com/username/repo")
        False
    """
    if not url or not isinstance(url, str):
        return False
    
    # Handle SSH format (git@github.com:username/repo.git)
    if url.startswith('git@github.com:'):
        parts = url[15:].split('/')
        # Check if there's a username/repo part after the colon
        return len(parts) > 0 and '/' not in parts[0] and ':' not in parts[0]
    
    # Handle HTTP/HTTPS formats
    parsed_url = urlparse(url)
    
    # Check if the domain is github.com
    if not parsed_url.netloc.endswith("github.com"):
        return False
    
    # Check if there are at least owner/repo in the path
    path_parts = [p for p in parsed_url.path.strip('/').split("/") if p]
    return len(path_parts) >= 2


def get_clone_url(owner: str, repo: str) -> str:
    """Generate a clone URL for a GitHub repository.
    
    Args:
        owner: GitHub repository owner/username
        repo: GitHub repository name
        
    Returns:
        HTTPS clone URL for the repository
        
    Examples:
        >>> get_clone_url("username", "repo")
        'https://github.com/username/repo.git'
    """
    return f"https://github.com/{owner}/{repo}.git"


def clone_github_repo(url: str, temp_dir: str = None) -> str:
    """Clone a GitHub repository to a local directory.
    
    Args:
        url: GitHub repository URL
        temp_dir: Directory to clone the repository to (created if not exists)
        
    Returns:
        Path to the cloned repository
        
    Raises:
        ValueError: If GitPython is not available or URL is invalid
        git.GitCommandError: If the clone operation fails
    """
    if not GIT_AVAILABLE:
        raise ValueError("GitPython is required to clone repositories")
    
    # Parse the GitHub URL
    try:
        repo_info = parse_github_url(url)
        
        # Validate that we have necessary information
        if not repo_info["owner"] or not repo_info["repo"]:
            raise ValueError(f"Invalid GitHub URL format: {url}\nCould not extract owner/repo information.")
        
    except Exception as e:
        logger.error(f"Failed to parse GitHub URL '{url}': {e}")
        raise ValueError(f"Invalid GitHub URL: {url}. Error: {str(e)}")
    
    # Create a temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="dualipa_repo_")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Cloning repository {repo_info['owner']}/{repo_info['repo']} to {temp_dir}")
    
    # Clone the repository
    try:
        # Construct full clone URL
        clone_url = get_clone_url(repo_info['owner'], repo_info['repo'])
        
        # Clone the repository to the temporary directory
        repo = git.Repo.clone_from(clone_url, temp_dir)
        
        # Checkout the specified branch if provided and different from default
        branch = repo_info.get('branch')
        if branch and branch.strip() and branch not in ["main", "master"]:
            try:
                logger.info(f"Checking out branch: {branch}")
                repo.git.checkout(branch)
            except git.GitCommandError as branch_error:
                logger.warning(f"Failed to checkout branch '{branch}': {branch_error}")
                logger.info("Continuing with default branch")
        
        logger.info(f"Repository cloned successfully to {temp_dir}")
        
        return temp_dir
    except git.GitCommandError as e:
        logger.error(f"Git command error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during repository cloning: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(f"Failed to clone repository: {str(e)}")


async def fetch_repo_contents_async(owner: str, repo: str, path: str = '', ref: str = 'main') -> Dict[str, Any]:
    """Fetch repository contents asynchronously using GitHub API.
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: Path within the repository
        ref: Branch or tag reference
        
    Returns:
        Repository contents from GitHub API
        
    Raises:
        ValueError: If aiohttp is not available
        RuntimeError: If the API request fails
    """
    if not GITHUB_API_AVAILABLE:
        raise ValueError("PyGithub is required for API access")
    
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    
    logger.info(f"Fetching repository contents from {url}")
    
    g = Github()
    repo = g.get_repo(f"{owner}/{repo}")
    
    try:
        contents = repo.get_contents(path, ref=ref)
        if isinstance(contents, list):
            return {"error": "Path is a directory, not a file"}
        
        content = contents.decoded_content.decode('utf-8')
        return {"content": content}
    except Exception as e:
        error_msg = f"Error getting file from GitHub: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_file_contents_from_github(
    repo_url: str, 
    file_path: str, 
    branch: str = "main",
    token: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get file contents directly from GitHub API.
    
    Args:
        repo_url: URL of the GitHub repository
        file_path: Path to the file within the repository
        branch: Branch name to fetch from
        token: GitHub API token (optional)
        
    Returns:
        Tuple containing (content, error_message)
        If successful, error_message will be None
        If failed, content will be None and error_message will contain the error
    """
    if not GITHUB_API_AVAILABLE:
        return None, "PyGithub not available"
    
    try:
        # Extract owner and repo name from URL
        parse_result = urlparse(repo_url)
        path_parts = parse_result.path.strip('/').split('/')
        
        if len(path_parts) < 2:
            return None, f"Invalid GitHub URL format: {repo_url}"
        
        owner, repo_name = path_parts[0], path_parts[1]
        
        # Create GitHub instance with or without token
        if token:
            auth = Auth.Token(token)
            g = Github(auth=auth)
        else:
            g = Github()
        
        # Get repository
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        # Get file contents
        content_file = repo.get_contents(file_path, ref=branch)
        
        # Decode content
        if isinstance(content_file, list):
            return None, "Path is a directory, not a file"
        
        decoded_content = content_file.decoded_content.decode('utf-8')
        return decoded_content, None
        
    except Exception as e:
        error_msg = f"Error getting file from GitHub: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def download_github_repo(url: str, output_dir: Optional[str] = None) -> str:
    """Download a GitHub repository to a local directory.
    
    This is a simplified wrapper for clone_github_repo that handles edge cases like
    empty branch names safely.
    
    Args:
        url: GitHub repository URL
        output_dir: Directory to clone into (creates a temp dir if None)
        
    Returns:
        Path to the cloned repository
        
    Raises:
        ValueError: If the URL is invalid or cloning fails
        
    Examples:
        >>> download_github_repo("https://github.com/username/repo")
        '/path/to/cloned/repo'
    """
    if not GIT_AVAILABLE:
        raise ValueError("GitPython is required to clone repositories")
    
    # Validate URL format
    if not url or not isinstance(url, str):
        raise ValueError(f"Invalid URL: {url}")
    
    # Try to clean/normalize the URL if needed
    url = url.strip()
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dualipa_repo_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use the improved clone_github_repo function
        return clone_github_repo(url, output_dir)
    except ValueError as e:
        logger.error(f"Invalid GitHub URL: {e}")
        raise
    except git.GitCommandError as e:
        logger.error(f"Git command error: {e}")
        raise ValueError(f"Failed to clone repository: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ValueError(f"Failed to download repository: {str(e)}")


if __name__ == "__main__":
    # Simple test to verify the URL handling works correctly
    test_urls = [
        "https://github.com/username/repo",
        "https://github.com/username/repo.git",
        "https://github.com/username/repo/tree/main/src",
        "git@github.com:username/repo.git",
        "https://invalid-url",
        "not-a-url"
    ]
    
    print("URL Parsing Results:")
    for url in test_urls:
        is_github = is_github_url(url)
        print(f"{url:40} -> {'✅ GitHub URL' if is_github else '❌ Not a GitHub URL'}")
        
        if is_github:
            info = parse_github_url(url)
            print(f"  Owner: {info['owner']}")
            print(f"  Repo: {info['repo']}")
            print(f"  Branch: {info['branch']}")
            print(f"  Path: {info['path']}")
            print() 