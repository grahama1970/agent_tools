#!/usr/bin/env python3
"""
This script fixes the GitHub URL handling in the github_utils module,
with a focus on properly supporting SSH URL formats.
"""

import os
import sys
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the original functions
try:
    from agent_tools.dualipa.github_utils import is_github_url, parse_github_url
    print("Successfully imported GitHub utils")
except ImportError as e:
    print(f"Error importing GitHub utils: {e}")
    sys.exit(1)

def test_original_functions():
    print("\n===== Testing Original GitHub URL Handling =====\n")
    
    # Test URLs
    test_urls = [
        "https://github.com/username/repo",
        "https://github.com/username/repo.git",
        "https://github.com/username/repo/tree/main",
        "https://github.com/username/repo/tree/main/src",
        "git@github.com:username/repo.git",
        "https://invalid-url",
        "not-a-url"
    ]
    
    print("Original URL Parsing Results:")
    print(f"{'URL':<40} | {'Is GitHub URL':<12}")
    print("-" * 55)
    
    for url in test_urls:
        is_github = is_github_url(url)
        print(f"{url:<40} | {'✅' if is_github else '❌':<12}")

def improved_is_github_url(url: str) -> bool:
    """Improved version of is_github_url that handles SSH format correctly."""
    if not url or not isinstance(url, str):
        return False
    
    # Handle SSH format (git@github.com:username/repo.git)
    if url.startswith('git@github.com:'):
        parts = url[15:].split('/')
        # Check if there's a username/repo part after the colon
        return len(parts) > 0 and '/' not in parts[0] and ':' not in parts[0]
    
    # Handle HTTP/HTTPS formats as before
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    
    # Check if the domain is github.com
    if not parsed_url.netloc.endswith("github.com"):
        return False
    
    # Check if there are at least owner/repo in the path
    path_parts = [p for p in parsed_url.path.strip('/').split("/") if p]
    return len(path_parts) >= 2

def test_improved_function():
    print("\n===== Testing Improved GitHub URL Handling =====\n")
    
    # Test URLs
    test_urls = [
        "https://github.com/username/repo",
        "https://github.com/username/repo.git",
        "https://github.com/username/repo/tree/main",
        "https://github.com/username/repo/tree/main/src",
        "git@github.com:username/repo.git",
        "https://invalid-url",
        "not-a-url"
    ]
    
    print("Improved URL Parsing Results:")
    print(f"{'URL':<40} | {'Is GitHub URL':<12}")
    print("-" * 55)
    
    for url in test_urls:
        is_github = improved_is_github_url(url)
        print(f"{url:<40} | {'✅' if is_github else '❌':<12}")

def main():
    print("\n===== GitHub URL Handling Fix =====\n")
    
    # Test the original function
    test_original_functions()
    
    # Test the improved function
    test_improved_function()
    
    print("\n===== Function Implementation That Fixes the Issue =====\n")
    print("def is_github_url(url: str) -> bool:")
    print('    """Check if a URL is a GitHub repository URL."""')
    print("    if not url or not isinstance(url, str):")
    print("        return False")
    print("    ")
    print("    # Handle SSH format (git@github.com:username/repo.git)")
    print("    if url.startswith('git@github.com:'):")
    print("        parts = url[15:].split('/')")
    print("        # Check if there's a username/repo part after the colon")
    print("        return len(parts) > 0 and '/' not in parts[0] and ':' not in parts[0]")
    print("    ")
    print("    # Handle HTTP/HTTPS formats")
    print("    from urllib.parse import urlparse")
    print("    parsed_url = urlparse(url)")
    print("    ")
    print("    # Check if the domain is github.com")
    print("    if not parsed_url.netloc.endswith('github.com'):")
    print("        return False")
    print("    ")
    print("    # Check if there are at least owner/repo in the path")
    print("    path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]")
    print("    return len(path_parts) >= 2")
    
    print("\nYou can copy this implementation to fix the SSH URL handling issue.")
    print("Replace the is_github_url function in github_utils.py with this implementation.")

if __name__ == "__main__":
    main() 