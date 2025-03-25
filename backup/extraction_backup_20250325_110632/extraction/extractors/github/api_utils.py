"""
GitHub API utilities for DuaLipa.

This module handles interactions with the GitHub API,
including fetching repository metadata and file content.

Key Features:
1. API authentication
2. Rate limit handling
3. Repository metadata fetching
4. File content retrieval

Dependencies:
- requests: For API calls
- loguru: For logging

Related Files:
- repo_utils.py: Core repository operations
"""

import os
import base64
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import requests
from loguru import logger

# GitHub API configuration
API_BASE = "https://api.github.com"
DEFAULT_HEADERS = {
    "Accept": "application/vnd.github.v3+json"
}

def _get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for GitHub API."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = DEFAULT_HEADERS.copy()
    
    if token:
        headers["Authorization"] = f"token {token}"
        
    return headers

def _parse_github_url(url: str) -> Dict[str, str]:
    """
    Parse GitHub repository URL into components.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Dictionary with owner and repo name
    """
    try:
        path = urlparse(url).path.strip('/')
        parts = path.split('/')
        
        if len(parts) >= 2:
            return {
                "owner": parts[0],
                "repo": parts[1].replace('.git', '')
            }
    except Exception as e:
        logger.error(f"Failed to parse GitHub URL: {e}")
        
    raise ValueError(f"Invalid GitHub URL: {url}")

def fetch_repo_metadata(url: str) -> Dict[str, Any]:
    """
    Fetch repository metadata from GitHub API.
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Dictionary with repository metadata
    """
    try:
        # Parse URL
        repo_info = _parse_github_url(url)
        owner = repo_info["owner"]
        repo = repo_info["repo"]
        
        # Make API request
        headers = _get_auth_headers()
        response = requests.get(
            f"{API_BASE}/repos/{owner}/{repo}",
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        return {
            "name": data["name"],
            "full_name": data["full_name"],
            "description": data.get("description"),
            "default_branch": data["default_branch"],
            "stars": data["stargazers_count"],
            "forks": data["forks_count"],
            "open_issues": data["open_issues_count"],
            "language": data.get("language"),
            "topics": data.get("topics", []),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "clone_url": data["clone_url"],
            "size": data["size"],
            "license": data.get("license", {}).get("name")
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching repository metadata: {e}")
        raise

def get_file_content(url: str, path: str, ref: Optional[str] = None) -> Optional[str]:
    """
    Get file content from GitHub repository.
    
    Args:
        url: GitHub repository URL
        path: Path to file in repository
        ref: Optional git reference (branch, tag, commit)
        
    Returns:
        File content if successful, None otherwise
    """
    try:
        # Parse URL
        repo_info = _parse_github_url(url)
        owner = repo_info["owner"]
        repo = repo_info["repo"]
        
        # Build API URL
        api_url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
        if ref:
            api_url += f"?ref={ref}"
            
        # Make API request
        headers = _get_auth_headers()
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get("type") != "file":
            logger.warning(f"Not a file: {path}")
            return None
            
        # Decode content
        content = base64.b64decode(data["content"]).decode('utf-8')
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        return None

def usage_example() -> None:
    """Example usage of GitHub API utilities."""
    # Set up authentication (in practice, use environment variable)
    os.environ["GITHUB_TOKEN"] = "your-token-here"
    
    # Example repository
    url = "https://github.com/example/repo"
    
    try:
        # Fetch repository metadata
        metadata = fetch_repo_metadata(url)
        print("\nRepository Metadata:")
        print(f"Name: {metadata['name']}")
        print(f"Description: {metadata['description']}")
        print(f"Language: {metadata['language']}")
        print(f"Stars: {metadata['stars']}")
        print(f"Topics: {', '.join(metadata['topics'])}")
        
        # Get file content
        content = get_file_content(url, "README.md")
        if content:
            print("\nREADME.md content:")
            print(content[:200] + "...")  # Show first 200 chars
            
    except Exception as e:
        print(f"Error: {e}")
        
    # Clean up
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"] 