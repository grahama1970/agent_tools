"""
Code extraction module for DuaLipa.

This module provides functions to extract code from repositories,
parse files, and generate datasets for LoRA fine-tuning.

Official Documentation References:
- gitpython: https://gitpython.readthedocs.io/en/stable/tutorial.html
- loguru: https://loguru.readthedocs.io/en/stable/
- tqdm: https://tqdm.github.io/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

import os
import json
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple, Any
import re
import time
from loguru import logger
from tqdm import tqdm

# Import local modules
try:
    from .github_utils import clone_repository, is_github_url, parse_github_url, get_clone_url
    from .language_detection import detect_language
    from .markdown_parser import extract_code_blocks
except ImportError:
    # Handle case where this module is run standalone
    logger.warning("Running in standalone mode, attempting relative imports")
    from github_utils import clone_repository, is_github_url, parse_github_url, get_clone_url
    from language_detection import detect_language
    from markdown_parser import extract_code_blocks

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Common extensions for code and documentation files
CODE_FILE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', 
    '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala'
}

DOCUMENTATION_FILE_EXTENSIONS = {
    '.md', '.rst', '.txt', '.ipynb', '.tex'
}

# Files to ignore
IGNORED_FILES = {
    'LICENSE', '.gitignore', '.gitattributes', '.gitmodules',
    'setup.py', 'requirements.txt', 'package.json', 'package-lock.json',
    'yarn.lock', 'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'poetry.lock',
    'Cargo.toml', 'Cargo.lock', 'Gemfile', 'Gemfile.lock'
}

# Directories to ignore
IGNORED_DIRECTORIES = {
    '.git', '.github', '.vscode', '.idea', '__pycache__', 
    'node_modules', 'venv', 'env', '.env', 'build', 'dist', 
    'target', 'out', 'bin', 'obj', 'tmp', 'temp', 'tests'
}

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
                owner, repo, path, branch = parse_github_url(source)
                clone_url = get_clone_url(owner, repo)
                
                if not clone_repository(clone_url, temp_dir):
                    error_msg = f"Failed to clone repository: {source}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    return stats
                
                # If a specific path in the repo was provided, use it
                repo_dir = os.path.join(temp_dir, path) if path else temp_dir
                return _process_repository(
                    repo_dir, 
                    output_dir, 
                    stats, 
                    max_files,
                    include_patterns,
                    exclude_patterns,
                    extract_documentation,
                    extract_code
                )
        except Exception as e:
            error_msg = f"Error processing GitHub repository: {str(e)}"
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
            extract_code
        )
    
    # Handle local file
    elif os.path.isfile(source):
        logger.info(f"Processing single file: {source}")
        try:
            file_path = Path(source)
            language = detect_language(file_path)
            
            # Process the file based on its type
            if _is_code_file(file_path.name) and extract_code:
                _process_code_file(file_path, output_dir, stats, language)
            elif _is_documentation_file(file_path.name) and extract_documentation:
                _process_documentation_file(file_path, output_dir, stats)
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


def demo_code_extractor() -> None:
    """Demonstrate the code extractor functionality with examples.
    
    This function shows how to use the main components of the code extractor:
    1. Extracting code from a Python file
    2. Extracting code blocks from a Markdown file
    3. Extracting a repository (using a local example)
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("Code Extractor Demo")
        logger.info("===================")
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Create a sample Python file
            logger.info("\n1. Extracting code from a Python file:")
            python_file = temp_path / "sample.py"
            with open(python_file, "w") as f:
                f.write('''"""Sample Python file for demonstration."""

def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number recursively.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def calculate_factorial(n: int) -> int:
    """Calculate the factorial of n recursively.
    
    Args:
        n: The number to calculate factorial for
        
    Returns:
        The factorial of n
    """
    if n <= 1:
        return 1
    return n * calculate_factorial(n-1)

if __name__ == "__main__":
    print(f"Fibonacci of 10: {calculate_fibonacci(10)}")
    print(f"Factorial of 5: {calculate_factorial(5)}")
''')
            
            # 2. Create a sample Markdown file
            logger.info("\n2. Extracting code blocks from a Markdown file:")
            markdown_file = temp_path / "sample.md"
            with open(markdown_file, "w") as f:
                f.write('''# Sample Markdown File

This is a sample Markdown file with code blocks for demonstration.

## Python Example

```python
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

print(greet("World"))
```

## JavaScript Example

```javascript
function calculateSum(numbers) {
    return numbers.reduce((sum, num) => sum + num, 0);
}

console.log(calculateSum([1, 2, 3, 4, 5]));
```

## Shell Command Example

```bash
# List files in the current directory
ls -la
echo "Current directory: $(pwd)"
```
''')
            
            # 3. Create a sample repository structure
            logger.info("\n3. Extracting from a sample repository structure:")
            repo_path = temp_path / "sample_repo"
            repo_path.mkdir()
            
            # Create a README
            with open(repo_path / "README.md", "w") as f:
                f.write("# Sample Repository\n\nThis is a sample repository for testing the code extractor.")
            
            # Create a src directory with Python files
            src_path = repo_path / "src"
            src_path.mkdir()
            
            # Create a Python module
            with open(src_path / "main.py", "w") as f:
                f.write('''"""Main module for the sample repository."""

from utils import format_string

def main():
    """Main function."""
    message = format_string("Hello, World!")
    print(message)

if __name__ == "__main__":
    main()
''')
            
            with open(src_path / "utils.py", "w") as f:
                f.write('''"""Utility functions for the sample repository."""

def format_string(text: str) -> str:
    """Format a string by adding brackets around it.
    
    Args:
        text: The input string
        
    Returns:
        The formatted string
    """
    return f"[{text}]"
''')
            
            # Create a tests directory
            tests_path = repo_path / "tests"
            tests_path.mkdir()
            
            with open(tests_path / "test_utils.py", "w") as f:
                f.write('''"""Tests for the utils module."""

import unittest
from src.utils import format_string

class TestUtils(unittest.TestCase):
    """Test class for utils module."""
    
    def test_format_string(self):
        """Test the format_string function."""
        result = format_string("test")
        self.assertEqual(result, "[test]")

if __name__ == "__main__":
    unittest.main()
''')
            
            # Execute the extraction for each example
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # 1. Extract from Python file
            logger.info("\nExtracting from Python file...")
            python_stats = extract_repository(
                str(python_file),
                str(output_dir / "python_output"),
                extract_documentation=True,
                extract_code=True
            )
            logger.info(f"Python file extraction stats: {json.dumps(python_stats, indent=2)}")
            
            # 2. Extract from Markdown file
            logger.info("\nExtracting from Markdown file...")
            markdown_stats = extract_repository(
                str(markdown_file),
                str(output_dir / "markdown_output"),
                extract_documentation=True,
                extract_code=True
            )
            logger.info(f"Markdown file extraction stats: {json.dumps(markdown_stats, indent=2)}")
            
            # 3. Extract from repository
            logger.info("\nExtracting from repository...")
            repo_stats = extract_repository(
                str(repo_path),
                str(output_dir / "repo_output"),
                max_files=10,
                include_patterns=["*.py", "*.md"],
                exclude_patterns=["*test*"],
                extract_documentation=True,
                extract_code=True
            )
            logger.info(f"Repository extraction stats: {json.dumps(repo_stats, indent=2)}")
            
            # Display file counts
            python_output_files = list(Path(output_dir / "python_output").glob("**/*"))
            markdown_output_files = list(Path(output_dir / "markdown_output").glob("**/*"))
            repo_output_files = list(Path(output_dir / "repo_output").glob("**/*"))
            
            logger.info(f"\nPython output files: {len(python_output_files)}")
            logger.info(f"Markdown output files: {len(markdown_output_files)}")
            logger.info(f"Repository output files: {len(repo_output_files)}")
            
        logger.info("\nCode Extractor Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in code extractor demo: {e}")


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_code_extractor()
    
    # Process repository if URL or path is provided
    if len(sys.argv) > 1:
        try:
            source = sys.argv[1]
            
            # Determine output path (2nd argument or default)
            output_path = sys.argv[2] if len(sys.argv) > 2 else "extracted_data"
            
            # Extract repository with command line arguments
            max_files = 1000
            include_patterns = None
            exclude_patterns = None
            extract_documentation = True
            extract_code = True
            
            # Parse command line flags
            if "--max-files" in sys.argv and sys.argv.index("--max-files") + 1 < len(sys.argv):
                max_files = int(sys.argv[sys.argv.index("--max-files") + 1])
            
            if "--include" in sys.argv and sys.argv.index("--include") + 1 < len(sys.argv):
                include_patterns = sys.argv[sys.argv.index("--include") + 1].split(",")
            
            if "--exclude" in sys.argv and sys.argv.index("--exclude") + 1 < len(sys.argv):
                exclude_patterns = sys.argv[sys.argv.index("--exclude") + 1].split(",")
            
            if "--no-docs" in sys.argv:
                extract_documentation = False
            
            if "--no-code" in sys.argv:
                extract_code = False
            
            logger.info(f"Processing source: {source}")
            logger.info(f"Output path: {output_path}")
            logger.info(f"Max files: {max_files}")
            logger.info(f"Include patterns: {include_patterns}")
            logger.info(f"Exclude patterns: {exclude_patterns}")
            logger.info(f"Extract documentation: {extract_documentation}")
            logger.info(f"Extract code: {extract_code}")
            
            # Execute extraction
            start_time = time.time()
            stats = extract_repository(
                source,
                output_path,
                max_files=max_files,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                extract_documentation=extract_documentation,
                extract_code=extract_code
            )
            end_time = time.time()
            
            # Print stats
            logger.info(f"\nExtraction completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Total files processed: {stats['total_files']}")
            logger.info(f"Code files extracted: {stats['code_files']}")
            logger.info(f"Documentation files extracted: {stats['documentation_files']}")
            logger.info(f"Code blocks extracted: {stats['code_blocks']}")
            
            if stats["errors"]:
                logger.warning(f"Encountered {len(stats['errors'])} errors during extraction")
                for error in stats["errors"][:5]:  # Show at most 5 errors
                    logger.warning(f"- {error}")
                
                if len(stats["errors"]) > 5:
                    logger.warning(f"... and {len(stats['errors']) - 5} more errors")
            
            # Write stats to file
            stats_file = os.path.join(output_path, "extraction_stats.json")
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Extraction statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error processing repository: {e}") 