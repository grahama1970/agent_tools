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
        {'owner': 'username', 'repo': 'repo', 'path': '', 'branch': ''}
        
        >>> parse_github_url("https://github.com/username/repo/tree/main/src")
        {'owner': 'username', 'repo': 'repo', 'path': 'src', 'branch': 'main'}
    """
    result = {
        "owner": "",
        "repo": "",
        "path": "",
        "branch": "main"
    }
    
    if not url:
        return result
    
    # Parse URL
    parsed_url = urlparse(url)
    
    # Skip if not a GitHub URL
    if not parsed_url.netloc.endswith("github.com"):
        return result
    
    # Extract owner and repo from path
    path_parts = [p for p in parsed_url.path.split("/") if p]
    if len(path_parts) >= 2:
        result["owner"] = path_parts[0]
        result["repo"] = path_parts[1]
        
        # Remove .git suffix if present
        if result["repo"].endswith(".git"):
            result["repo"] = result["repo"][:-4]
    
    # Extract branch and path if available
    if len(path_parts) >= 4 and path_parts[2] in ["tree", "blob"]:
        result["branch"] = path_parts[3]
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
    
    # Handle SSH format directly
    if url.startswith('git@github.com:'):
        # Check if there's at least a username/repo part after the colon
        parts = url[15:].strip('/').split('/')
        return len(parts) > 0 and '.' in parts[0]  # Expecting at least one part with a .git extension
    
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
        clone_url = get_clone_url(repo_info['owner'], repo_info['repo'])
        
        # Clone the repository to the temporary directory
        repo = git.Repo.clone_from(clone_url, temp_dir)
        
        # Checkout the specified reference if provided and not empty
        if repo_info['branch'] and repo_info['branch'] != 'main':
            logger.info(f"Checking out branch: {repo_info['branch']}")
            repo.git.checkout(repo_info['branch'])
        
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
        clone_url = get_clone_url(owner, repo)
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


def download_github_repo(url: str, output_dir: Optional[str] = None) -> str:
    """Download a GitHub repository to a local directory.
    
    This is a simplified wrapper for clone_github_repo that handles edge cases like
    empty branch names safely.
    
    Args:
        url: GitHub repository URL
        output_dir: Directory to clone into (creates a temp dir if None)
        
    Returns:
        Path to the cloned repository
        
    Examples:
        >>> download_github_repo("https://github.com/username/repo")
        '/path/to/cloned/repo'
    """
    if not GIT_AVAILABLE:
        raise ValueError("GitPython is required to clone repositories")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="dualipa_repo_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Parse the GitHub URL
    repo_info = parse_github_url(url)
    logger.info(f"Downloading repository {repo_info['owner']}/{repo_info['repo']} to {output_dir}")
    
    # Construct clone URL
    clone_url = get_clone_url(repo_info['owner'], repo_info['repo'])
    
    try:
        # Clone the repository
        repo = git.Repo.clone_from(clone_url, output_dir)
        
        # Only checkout branch if it's explicitly specified and not empty
        if repo_info['branch'] and repo_info['branch'] != 'main':
            logger.info(f"Checking out branch: {repo_info['branch']}")
            repo.git.checkout(repo_info['branch'])
        
        logger.info(f"Repository downloaded successfully to {output_dir}")
        return output_dir
    except git.GitCommandError as e:
        logger.error(f"Git command error: {e}")
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def extract_repository(
    source: str, 
    output_path: str,
    max_files: int = 1000,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    extract_documentation: bool = True,
    extract_code: bool = True
) -> Dict[str, Any]:
    """
    Extract code and documentation from a repository.
    
    Args:
        source: Repository URL or local path
        output_path: Path to save the extracted data
        max_files: Maximum number of files to extract
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        extract_documentation: Whether to extract documentation files
        extract_code: Whether to extract code files
        
    Returns:
        Dictionary with statistics about the extraction process
    """
    stats = {
        "total_files": 0,
        "code_files": 0,
        "documentation_files": 0,
        "code_blocks": 0,
        "languages": {},
        "file_types": {},
        "errors": []
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle GitHub repository URL
    if is_github_url(source):
        logger.info(f"Cloning GitHub repository: {source}")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use download_github_repo which has better error handling
                try:
                    repo_dir = download_github_repo(source, temp_dir)
                    
                    if not repo_dir or not os.path.exists(repo_dir):
                        error_msg = f"Failed to clone repository: {source}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
                        return stats
                    
                    # If a specific path in the repo was provided, use it
                    repo_info = parse_github_url(source)
                    repo_path = os.path.join(repo_dir, repo_info.get('path', '')) if repo_info.get('path') else repo_dir
                    
                    # Process the repository
                    return _process_repository(
                        repo_path, 
                        output_dir, 
                        stats, 
                        max_files,
                        include_patterns,
                        exclude_patterns,
                        extract_documentation,
                        extract_code,
                        extract_blocks=True  # Enable block extraction by default
                    )
                except ValueError as e:
                    error_msg = f"Error with GitHub repository: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    return stats
                except Exception as e:
                    error_msg = f"Unexpected error processing GitHub repository: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    return stats
        except Exception as e:
            error_msg = f"Error creating temporary directory: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
    
    # Handle local directory
    elif os.path.isdir(source):
        logger.info(f"Processing local directory: {source}")
        return _process_repository(
            source, 
            output_dir, 
            stats, 
            max_files,
            include_patterns,
            exclude_patterns,
            extract_documentation,
            extract_code,
            extract_blocks=True  # Enable block extraction by default
        )
    
    # Handle local file
    elif os.path.isfile(source):
        logger.info(f"Processing single file: {source}")
        try:
            file_path = Path(source)
            language = detect_language(file_path)
            
            # Process the file based on its type
            if _is_code_file(file_path.name) and extract_code:
                _process_code_file(file_path, output_dir, stats, language, extract_blocks=True)
            elif _is_documentation_file(file_path.name) and extract_documentation:
                _process_documentation_file(file_path, output_dir, stats, extract_blocks=True)
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                
            return stats
        except Exception as e:
            error_msg = f"Error processing file {source}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
    
    # Invalid source
    else:
        error_msg = f"Invalid source: {source}. Must be a GitHub URL, local directory, or file."
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return stats


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
                
            repo_info = parse_github_url(repo_url)
            logger.info(f"Owner: {repo_info['owner']}, Repo: {repo_info['repo']}, Path: {repo_info['path']}, Branch: {repo_info['branch']}")
            
            # Clone to temporary directory if requested
            if "--clone" in sys.argv:
                with tempfile.TemporaryDirectory() as temp_dir:
                    logger.info(f"Cloning repository to {temp_dir}")
                    
                    if GIT_AVAILABLE:
                        # Use the download_github_repo function
                        try:
                            repo_path = download_github_repo(repo_url, temp_dir)
                            logger.info(f"Successfully cloned repository to {repo_path}")
                            
                            # Count files
                            file_count = sum(1 for _ in Path(repo_path).rglob('*') if _.is_file())
                            logger.info(f"Repository contains {file_count} files")
                        except Exception as e:
                            logger.error(f"Failed to clone repository: {e}")
                    else:
                        logger.error("GitPython not available for cloning")
            
            # Get file contents if path is provided and --content is specified
            if repo_info['path'] and "--content" in sys.argv:
                if GITHUB_API_AVAILABLE:
                    content, error = get_file_contents_from_github(f"https://github.com/{repo_info['owner']}/{repo_info['repo']}", repo_info['path'], repo_info['branch'])
                    
                    if content:
                        logger.info(f"Content of {repo_info['path']}:")
                        print(content[:500] + "..." if len(content) > 500 else content)
                        
                        # Save to file if requested
                        if "--save" in sys.argv:
                            output_file = Path(repo_info['path']).name
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            logger.info(f"Saved content to {output_file}")
                    else:
                        logger.error(f"Could not access file: {error}")
                else:
                    logger.error("PyGithub not available for accessing file content")
                    
        except Exception as e:
            logger.error(f"Error processing URL: {e}") 