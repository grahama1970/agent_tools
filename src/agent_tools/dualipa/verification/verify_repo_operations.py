#!/usr/bin/env python3
"""
Verify GitHub repository operations.

This script tests the GitHub repository operations in the DuaLipa library,
including cloning repositories, downloading files, and accessing the API.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the required modules
try:
    from agent_tools.dualipa.github_utils import (
        is_github_url,
        parse_github_url,
        clone_github_repo,
        get_file_contents_from_github,
        download_github_repo
    )
    print("Successfully imported GitHub utils")
except ImportError as e:
    print(f"Error importing GitHub utils: {e}")
    sys.exit(1)

def print_header(text, underline='='):
    """Print a header with underline."""
    print(f"\n{text}")
    print(underline * len(text))

def verify_github_url_parsing():
    """Verify GitHub URL parsing."""
    print_header("Testing GitHub URL parsing", "-")
    
    test_urls = [
        "https://github.com/username/repo",
        "https://github.com/username/repo.git",
        "https://github.com/username/repo/tree/main",
        "https://github.com/username/repo/tree/branch/path/to/file",
        "git@github.com:username/repo.git",
        "https://not-github.com/username/repo",
        "not-a-url"
    ]
    
    for url in test_urls:
        is_github = is_github_url(url)
        print(f"\nURL: {url}")
        print(f"Is GitHub URL: {'✅' if is_github else '❌'}")
        
        if is_github:
            info = parse_github_url(url)
            print(f"  Owner: {info.get('owner', '')}")
            print(f"  Repo: {info.get('repo', '')}")
            print(f"  Branch: {info.get('branch', '')}")
            print(f"  Path: {info.get('path', '')}")
    
    return True

def verify_repo_cloning():
    """Verify repository cloning."""
    print_header("Testing repository cloning", "-")
    
    # Try cloning a small public repository
    repo_url = "https://github.com/psf/requests"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Cloning {repo_url} to {temp_dir}...")
            clone_path = clone_github_repo(repo_url, temp_dir)
            
            # Check if the repository was cloned successfully
            if os.path.exists(os.path.join(clone_path, "README.md")):
                print(f"✅ Repository cloned successfully")
                return True
            else:
                print(f"❌ Repository clone seems incomplete")
                return False
        except Exception as e:
            print(f"❌ Error cloning repository: {str(e)}")
            return False

def verify_file_download():
    """Verify file download from GitHub."""
    print_header("Testing file download from GitHub", "-")
    
    # Try downloading a specific file from a public repository
    repo_url = "https://github.com/psf/requests"
    file_path = "README.md"
    
    try:
        print(f"Downloading {file_path} from {repo_url}...")
        content, error = get_file_contents_from_github(repo_url, file_path)
        
        if content and not error:
            print(f"✅ File downloaded successfully ({len(content)} bytes)")
            print(f"First 100 characters: {content[:100]}...")
            return True
        else:
            print(f"❌ Error downloading file: {error}")
            return False
    except Exception as e:
        print(f"❌ Error downloading file: {str(e)}")
        return False

def verify_repo_download():
    """Verify repository download."""
    print_header("Testing repository download", "-")
    
    # Try downloading a small public repository
    repo_url = "https://github.com/psf/requests"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Downloading {repo_url} to {temp_dir}...")
            download_path = download_github_repo(repo_url, temp_dir)
            
            # Check if the repository was downloaded successfully
            if os.path.exists(os.path.join(download_path, "README.md")):
                print(f"✅ Repository downloaded successfully")
                return True
            else:
                print(f"❌ Repository download seems incomplete")
                return False
        except Exception as e:
            print(f"❌ Error downloading repository: {str(e)}")
            return False

def main():
    """Run all verification tests."""
    print_header("GitHub Repository Operations Verification")
    
    # Run all verification tests
    url_parsing_success = verify_github_url_parsing()
    repo_cloning_success = verify_repo_cloning()
    file_download_success = verify_file_download()
    repo_download_success = verify_repo_download()
    
    # Calculate overall success
    all_success = (
        url_parsing_success and
        repo_cloning_success and
        file_download_success and
        repo_download_success
    )
    
    # Print summary
    print_header("Verification Summary")
    print(f"URL Parsing: {'✅' if url_parsing_success else '❌'}")
    print(f"Repository Cloning: {'✅' if repo_cloning_success else '❌'}")
    print(f"File Download: {'✅' if file_download_success else '❌'}")
    print(f"Repository Download: {'✅' if repo_download_success else '❌'}")
    print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 