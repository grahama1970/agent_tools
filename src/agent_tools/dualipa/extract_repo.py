"""
Extract code and documentation from a repository.

Official Documentation References:
- os: https://docs.python.org/3/library/os.html
- json: https://docs.python.org/3/library/json.html
- tempfile: https://docs.python.org/3/library/tempfile.html
- git: https://gitpython.readthedocs.io/en/stable/
- aiohttp: https://docs.aiohttp.org/en/stable/
"""

import os
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union
import argparse
from pathlib import Path
from loguru import logger
import re

# Import local modules
try:
    from .github_utils import parse_github_url, clone_github_repo, GIT_AVAILABLE
    from .markdown_parser import get_markdown_files, process_markdown_file
    GITHUB_UTILS_AVAILABLE = True
    MARKDOWN_PARSER_AVAILABLE = True
except ImportError:
    GITHUB_UTILS_AVAILABLE = False
    MARKDOWN_PARSER_AVAILABLE = False
    logger.warning("Some modules not available. Functionality will be limited.")


def extract_from_repo(repo_path: str, output_dir: str, file_extensions: List[str] = None, 
                      ignore_patterns: List[str] = None) -> None:
    """Extract code and documentation from a repository.
    
    This function:
    1. Analyzes a repository (local or remote GitHub URL)
    2. Extracts relevant files based on extensions and ignore patterns
    3. Processes markdown files specially to extract structured content
    4. Saves the extracted data to a JSON file
    
    Args:
        repo_path: Path to the repository or GitHub URL
        output_dir: Directory where extracted data will be saved
        file_extensions: List of file extensions to extract (default: ['.py', '.md', '.txt'])
        ignore_patterns: List of regex patterns to ignore (default: ['__pycache__', '\.git', '\.venv'])
        
    Raises:
        FileNotFoundError: If the repository path doesn't exist
        ValueError: If the GitHub URL is invalid
    """
    # Set default extensions and ignore patterns
    if file_extensions is None:
        file_extensions = ['.py', '.md', '.txt']
    
    if ignore_patterns is None:
        ignore_patterns = ['__pycache__', '\.git', '\.venv', 'node_modules']
    
    # Check if repo_path is a GitHub URL
    is_github_url = repo_path.startswith(('https://github.com', 'http://github.com', 'git@github.com'))
    temp_dir = None
    
    try:
        # Handle GitHub URL
        if is_github_url and GITHUB_UTILS_AVAILABLE:
            logger.info(f"Detected GitHub URL: {repo_path}")
            temp_dir = tempfile.mkdtemp(prefix="dualipa_repo_")
            repo_path = clone_github_repo(repo_path, temp_dir)
            logger.info(f"Repository cloned to temporary directory: {repo_path}")
        
        # Validate repository path
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Compile ignore patterns
        ignore_regexes = [re.compile(pattern) for pattern in ignore_patterns]
        
        # Initialize data structure
        extracted_data = {
            "repo_path": repo_path,
            "files": []
        }
        
        # Extract files
        for root, dirs, files in os.walk(repo_path):
            # Check if current directory should be ignored
            if any(regex.search(root) for regex in ignore_regexes):
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Skip files with extensions not in the list
                if file_ext not in file_extensions:
                    continue
                
                # Skip files matching ignore patterns
                if any(regex.search(file) for regex in ignore_regexes):
                    continue
                
                # Process the file
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Special handling for markdown files if markdown parser is available
                if file_ext == '.md' and MARKDOWN_PARSER_AVAILABLE:
                    try:
                        md_data = process_markdown_file(file_path)
                        extracted_data["files"].append({
                            "path": relative_path,
                            "content": md_data["content"],
                            "sections": md_data["sections"],
                            "code_blocks": md_data["code_blocks"]
                        })
                        logger.info(f"Processed markdown file: {relative_path}")
                    except Exception as e:
                        logger.error(f"Error processing markdown file {relative_path}: {e}")
                        # Fall back to basic extraction
                        _extract_basic_file(file_path, relative_path, extracted_data)
                else:
                    # Basic extraction for other files
                    _extract_basic_file(file_path, relative_path, extracted_data)
        
        # Save extracted data
        output_file = os.path.join(output_dir, "extracted_data.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4)
        
        logger.info(f"Extracted {len(extracted_data['files'])} files. Data saved to {output_file}")
        print(f"Extracted {len(extracted_data['files'])} files. Data saved to {output_file}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        raise
    finally:
        # Clean up temp directory if created
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_basic_file(file_path: str, relative_path: str, extracted_data: Dict[str, Any]) -> None:
    """Basic file extraction helper function.
    
    Args:
        file_path: Path to the file
        relative_path: Path relative to the repository root
        extracted_data: Data structure to update
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        extracted_data["files"].append({
            "path": relative_path,
            "content": content
        })
        logger.info(f"Extracted file: {relative_path}")
    except UnicodeDecodeError:
        logger.warning(f"Skipping binary file: {relative_path}")
    except Exception as e:
        logger.error(f"Error reading file {relative_path}: {e}")


def debug_extract_repo() -> None:
    """Simple debug function to test repository extraction functionality.
    
    This function:
    1. Creates a temporary test repository
    2. Adds sample files
    3. Extracts from the repository
    4. Cleans up
    """
    import subprocess
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dualipa_test_repo_")
    output_dir = tempfile.mkdtemp(prefix="dualipa_test_output_")
    
    try:
        print(f"Created test repository in {temp_dir}")
        
        # Create a Python file
        py_file = os.path.join(temp_dir, "example.py")
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write("""
def hello_world():
    \"\"\"Say hello to the world.
    
    Returns:
        str: A greeting message
    \"\"\"
    return "Hello, World!"
            """)
        
        # Create a Markdown file
        md_file = os.path.join(temp_dir, "README.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("""# Test Repository
            
This is a test repository for the DuaLipa extractor.

## Usage

```python
from example import hello_world

print(hello_world())
```
            """)
        
        # Create a directory to ignore
        os.makedirs(os.path.join(temp_dir, "__pycache__"), exist_ok=True)
        
        # Test extraction
        print("Testing local repository extraction...")
        extract_from_repo(temp_dir, output_dir)
        
        # Check if the output file exists
        output_file = os.path.join(output_dir, "extracted_data.json")
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Extraction successful! Extracted {len(data['files'])} files.")
            
            # List extracted files
            for i, file in enumerate(data['files']):
                print(f"{i+1}. {file['path']}")
                if 'sections' in file:
                    print(f"   - Contains {len(file['sections'])} markdown sections")
                if 'code_blocks' in file:
                    print(f"   - Contains {len(file['code_blocks'])} code blocks")
        else:
            print(f"Extraction failed! Output file {output_file} not found.")
        
        # Test GitHub URL extraction if available
        if GITHUB_UTILS_AVAILABLE and GIT_AVAILABLE:
            print("\nTesting GitHub URL extraction...")
            try:
                # Use a very small test repo
                github_output_dir = os.path.join(output_dir, "github")
                os.makedirs(github_output_dir, exist_ok=True)
                extract_from_repo("https://github.com/octocat/Hello-World", github_output_dir)
                
                # Check if the output file exists
                github_output_file = os.path.join(github_output_dir, "extracted_data.json")
                if os.path.exists(github_output_file):
                    with open(github_output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"GitHub extraction successful! Extracted {len(data['files'])} files.")
                else:
                    print(f"GitHub extraction failed! Output file {github_output_file} not found.")
            except Exception as e:
                print(f"GitHub extraction failed: {e}")
        
    except Exception as e:
        print(f"Debug test failed: {e}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        print("Debug test completed and temporary files cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract code from a repository")
    parser.add_argument("repo_path", help="Path to the repository or GitHub URL")
    parser.add_argument("output_dir", help="Output directory for extracted data")
    parser.add_argument("--extensions", nargs="+", default=['.py', '.md', '.txt'],
                        help="File extensions to extract")
    parser.add_argument("--ignore", nargs="+", default=['__pycache__', '\.git', '\.venv', 'node_modules'],
                        help="Regex patterns to ignore")
    
    args = parser.parse_args()
    
    extract_from_repo(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        file_extensions=args.extensions,
        ignore_patterns=args.ignore
    )
