#!/usr/bin/env python3
"""
Simple test script to verify the fixed GitHub URL handling
"""

import os
import sys
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    # Import from the fixed implementation
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import src.agent_tools.dualipa.github_utils_fixed as github_utils
    is_github_url = github_utils.is_github_url
    parse_github_url = github_utils.parse_github_url
    print("Successfully imported fixed GitHub utils")
except ImportError as e:
    print(f"Error importing GitHub utils: {e}")
    sys.exit(1)

def main():
    print("\n===== Testing Fixed GitHub URL Handling =====\n")
    
    # Test URLs including SSH format
    test_urls = [
        "https://github.com/username/repo",
        "https://github.com/username/repo.git",
        "https://github.com/username/repo/tree/main",
        "https://github.com/username/repo/tree/main/src",
        "git@github.com:username/repo.git",
        "git@github.com:username/repo",
        "https://invalid-url",
        "not-a-url"
    ]
    
    print("URL Parsing Results:")
    print(f"{'URL':<45} | {'Is GitHub URL':<12} | {'Owner':<12} | {'Repo':<12} | {'Branch':<12} | {'Path':<12}")
    print("-" * 110)
    
    for url in test_urls:
        is_github = is_github_url(url)
        owner = repo = branch = path = "N/A"
        
        if is_github:
            try:
                info = parse_github_url(url)
                owner = info.get("owner", "")
                repo = info.get("repo", "")
                branch = info.get("branch", "")
                path = info.get("path", "")
            except Exception as e:
                owner = repo = branch = path = f"Error: {str(e)}"
        
        print(f"{url:<45} | {'✅' if is_github else '❌':<12} | {owner:<12} | {repo:<12} | {branch:<12} | {path:<12}")
    
    print("\n===== Test completed =====\n")
    
    print("To fix the GitHub URL handling issue, replace the is_github_url function in github_utils.py with this implementation:")
    print("\ndef is_github_url(url: str) -> bool:")
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
    print("    parsed_url = urlparse(url)")
    print("    ")
    print("    # Check if the domain is github.com")
    print("    if not parsed_url.netloc.endswith('github.com'):")
    print("        return False")
    print("    ")
    print("    # Check if there are at least owner/repo in the path")
    print("    path_parts = [p for p in parsed_url.path.strip('/').split('/') if p]")
    print("    return len(path_parts) >= 2")

if __name__ == "__main__":
    main() 