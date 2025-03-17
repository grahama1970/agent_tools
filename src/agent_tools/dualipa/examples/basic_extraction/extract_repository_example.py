#!/usr/bin/env python3
"""Example demonstrating how to extract code from a repository."""

import tempfile
import os
from pathlib import Path
import shutil

from agent_tools.dualipa.code_extractor import extract_repository

def demonstrate_repository_extraction():
    """Show how to extract code from a repository."""
    
    # Create a temporary directory to simulate a repository
    with tempfile.TemporaryDirectory() as repo_dir:
        # Convert to Path
        repo_path = Path(repo_dir)
        
        # Create a simple file structure
        (repo_path / "src").mkdir(exist_ok=True)
        (repo_path / "docs").mkdir(exist_ok=True)
        
        # Create some sample files
        with open(repo_path / "src" / "main.py", "w") as f:
            f.write("""
def main():
    \"\"\"Main function.\"\"\"
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
            
        with open(repo_path / "docs" / "README.md", "w") as f:
            f.write("""
# Example Project

## Overview
This is an example project.

## Usage
```python
from example import main
main()
```
""")
        
        # Create output directory
        with tempfile.TemporaryDirectory() as output_dir:
            # Convert to Path object
            output_path = Path(output_dir)
            
            # Extract repository with both files and blocks
            print(f"Extracting repository at {repo_path} to {output_path}")
            result = extract_repository(
                repo_path=repo_path,
                output_dir=output_path,
                extract_blocks=True  # Extract both files and blocks
            )
            
            print(f"Extraction completed. Processed {result.get('file_count', 0)} files.")
            print(f"Extracted {result.get('block_count', 0)} blocks.")
            
            # Show the output directory structure
            print("\nOutput directory structure:")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                for file in files:
                    print(f"{indent}    {file}")

if __name__ == "__main__":
    demonstrate_repository_extraction() 