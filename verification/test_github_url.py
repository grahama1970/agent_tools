#!/usr/bin/env python3
"""
Simple test script to verify GitHub URL handling
"""

import os
import sys
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from agent_tools.dualipa.github_utils import is_github_url, parse_github_url
    print("Successfully imported GitHub utils")
except ImportError as e:
    print(f"Error importing GitHub utils: {e}")
    sys.exit(1)

def main():
    print("\n===== Testing GitHub URL Handling =====\n")
    
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
    
    print("URL Parsing Results:")
    print(f"{'URL':<40} | {'Is GitHub URL':<12} | {'Owner':<12} | {'Repo':<12} | {'Branch':<12} | {'Path':<12}")
    print("-" * 100)
    
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
        
        print(f"{url:<40} | {'✅' if is_github else '❌':<12} | {owner:<12} | {repo:<12} | {branch:<12} | {path:<12}")
    
    print("\n===== Test completed =====\n")

if __name__ == "__main__":
    main() 