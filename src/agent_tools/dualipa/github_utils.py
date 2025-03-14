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
        url: GitHub repository URL
        
    Returns:
        Dictionary with owner, repo, ref (branch/tag), and path components
        
    Raises:
        ValueError: If the URL is not a valid GitHub repository URL
    """
    # Pattern for GitHub URLs
    patterns = [
        # Standard GitHub URL
        r'https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)(?:/tree/(?P<ref>[^/]+))?(?:/(?P<path>.*))?',
        # Git URL
        r'git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+)(?:.git)?(?:/tree/(?P<ref>[^/]+))?(?:/(?P<path>.*))?'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            result = match.groupdict()
            return {
                'owner': result.get('owner', ''),
                'repo': result.get('repo', '').replace('.git', ''),
                'ref': result.get('ref', 'main'),  # Default to 'main' if not specified
                'path': result.get('path', '')
            }
            
    raise ValueError(f"Invalid GitHub URL: {url}")


def clone_github_repo(url: str, temp_dir: Optional[str] = None) -> str:
    """Clone a GitHub repository to a temporary directory.
    
    Args:
        url: GitHub repository URL
        temp_dir: Optional temporary directory to use, creates one if not provided
        
    Returns:
        Path to the cloned repository
        
    Raises:
        ValueError: If GitPython is not available
        git.GitCommandError: If the clone operation fails
    """
    if not GIT_AVAILABLE:
        raise ValueError("GitPython is required to clone repositories")
    
    # Parse the GitHub URL
    try:
        repo_info = parse_github_url(url)
    except ValueError as e:
        logger.error(f"Failed to parse GitHub URL: {e}")
        raise
    
    # Create a temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="dualipa_repo_")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"Cloning repository {repo_info['owner']}/{repo_info['repo']} to {temp_dir}")
    
    # Clone the repository
    try:
        # Construct full clone URL
        clone_url = f"https://github.com/{repo_info['owner']}/{repo_info['repo']}.git"
        
        # Clone the repository to the temporary directory
        repo = git.Repo.clone_from(clone_url, temp_dir)
        
        # Checkout the specified reference if not the default
        if repo_info['ref'] != 'main':
            repo.git.checkout(repo_info['ref'])
        
        logger.info(f"Repository cloned successfully to {temp_dir}")
        
        return temp_dir
    except git.GitCommandError as e:
        logger.error(f"Git command error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


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


def debug_github_utils() -> None:
    """Simple debug function to test GitHub utilities functionality.
    
    This function tests:
    1. URL parsing
    2. Repository cloning (if GitPython is available)
    3. API access (if PyGithub is available)
    """
    test_url = "https://github.com/huggingface/transformers/tree/main/examples/pytorch"
    
    # Test URL parsing
    print("Testing GitHub URL parsing...")
    try:
        repo_info = parse_github_url(test_url)
        print(f"Parsed URL components: {repo_info}")
    except Exception as e:
        print(f"URL parsing failed: {e}")
    
    # Test repo cloning if GitPython is available
    if GIT_AVAILABLE:
        print("\nTesting repository cloning...")
        try:
            # Use a very small test repo for quick cloning
            test_clone_url = "https://github.com/octocat/Hello-World"
            temp_dir = tempfile.mkdtemp(prefix="dualipa_test_")
            clone_path = clone_github_repo(test_clone_url, temp_dir)
            print(f"Repository cloned to: {clone_path}")
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Repository cloning failed: {e}")
    
    # Test API access if PyGithub is available
    if GITHUB_API_AVAILABLE:
        print("\nTesting GitHub API access...")
        try:
            api_url = "https://api.github.com/repos/octocat/Hello-World/contents"
            response = fetch_repo_contents_async("octocat", "Hello-World")
            if response.get("error"):
                print(f"API access failed: {response['error']}")
            else:
                print(f"API access successful, found {len(response['content'].splitlines())} lines")
        except Exception as e:
            print(f"API access failed: {e}")
    
    print("\nDebug tests completed")


def demo_github_utils() -> None:
    """Demonstrate the GitHub utilities with examples.
    
    This function shows how to use the main components of the GitHub utilities:
    1. Checking if a URL is a GitHub repository
    2. Parsing GitHub URLs
    3. Cloning repositories
    4. Accessing file contents
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("GitHub Utils Demo")
        logger.info("=================")
        
        # Example GitHub URLs
        github_urls = [
            "https://github.com/huggingface/transformers",
            "https://github.com/huggingface/transformers/blob/main/README.md",
            "https://github.com/some-user/private-repo",
            "https://gitlab.com/some-user/some-repo",  # Not GitHub
            "https://example.com/not-a-repo"           # Not a repo URL
        ]
        
        # 1. Check which URLs are GitHub repositories
        logger.info("\n1. Identifying GitHub repository URLs:")
        for url in github_urls:
            is_github = is_github_url(url)
            logger.info(f"  {url} -> {'GitHub URL' if is_github else 'Not a GitHub URL'}")
        
        # 2. Parse GitHub URLs
        logger.info("\n2. Parsing GitHub URLs:")
        valid_github_url = "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
        owner, repo, path, branch = parse_github_url(valid_github_url)
        
        logger.info(f"  URL: {valid_github_url}")
        logger.info(f"  Owner: {owner}")
        logger.info(f"  Repo: {repo}")
        logger.info(f"  Path: {path}")
        logger.info(f"  Branch: {branch}")
        
        # 3. Create a clone URL
        logger.info("\n3. Creating clone URL:")
        clone_url = get_clone_url("huggingface", "transformers")
        logger.info(f"  Clone URL: {clone_url}")
        
        # 4. Access repository info (if PyGithub is available)
        if GITHUB_API_AVAILABLE:
            logger.info("\n4. Accessing repository information:")
            repo_url = "https://github.com/huggingface/transformers"
            
            logger.info(f"  Checking for README.md in {repo_url}")
            content, error = get_file_contents_from_github(repo_url, "README.md")
            
            if content:
                preview = content[:150] + "..." if len(content) > 150 else content
                logger.info(f"  README.md preview:\n{preview}")
            else:
                logger.warning(f"  Could not access README.md: {error}")
        else:
            logger.warning("\n4. PyGithub not available, skipping repository access example")
        
        logger.info("\nGitHub Utils Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in GitHub utils demo: {e}")


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_github_utils()
    
    # Example of processing a GitHub repository URL
    if len(sys.argv) > 1:
        try:
            repo_url = sys.argv[1]
            logger.info(f"Processing GitHub URL: {repo_url}")
            
            if not is_github_url(repo_url):
                logger.error(f"Not a valid GitHub URL: {repo_url}")
                sys.exit(1)
                
            owner, repo, path, branch = parse_github_url(repo_url)
            logger.info(f"Owner: {owner}, Repo: {repo}, Path: {path}, Branch: {branch}")
            
            # Clone to temporary directory if requested
            if "--clone" in sys.argv:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Cloning repository to {temp_dir}")
                    
                    if GIT_AVAILABLE:
                        clone_url = get_clone_url(owner, repo)
                        result = clone_repository(clone_url, temp_dir)
                        
                        if result:
                            logger.info(f"Successfully cloned repository to {temp_dir}")
                            
                            # Count files
                            file_count = sum(1 for _ in Path(temp_dir).rglob('*') if _.is_file())
                            logger.info(f"Repository contains {file_count} files")
                        else:
                            logger.error("Failed to clone repository")
                    else:
                        logger.error("GitPython not available for cloning")
            
            # Get file contents if path is provided and --content is specified
            if path and "--content" in sys.argv:
                if GITHUB_API_AVAILABLE:
                    content, error = get_file_contents_from_github(f"https://github.com/{owner}/{repo}", path, branch)
                    
                    if content:
                        logger.info(f"Content of {path}:")
                        print(content[:500] + "..." if len(content) > 500 else content)
                        
                        # Save to file if requested
                        if "--save" in sys.argv:
                            output_file = Path(path).name
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Saved content to {output_file}")
                    else:
                        logger.error(f"Could not access file: {error}")
                else:
                    logger.error("PyGithub not available for accessing file content")
                    
        except Exception as e:
            logger.error(f"Error processing URL: {e}") 