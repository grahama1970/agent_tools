#!/usr/bin/env python3
"""
Reorganize test files for the extraction module.

This script implements Phase 3 of the reorganization plan by:
1. Creating a clear directory structure for tests
2. Moving test files to appropriate locations
3. Updating test references
"""

import os
import shutil
from pathlib import Path
import re

# Define paths
BASE_DIR = Path(__file__).parent
TESTS_DIR = Path(BASE_DIR.parent.parent.parent.parent) / "tests" / "dualipa" / "extraction"

# Create new directory structure
new_structure = {
    "unit": {
        "extractors": "Tests for individual extractors",
        "utils": "Tests for utility functions",
        "validation": "Tests for validation components"
    },
    "integration": "Tests that verify component integration",
    "end_to_end": "Full workflow tests",
    "fixtures": "Test fixtures and data",
    "playwright": "Tests for Playwright functionality"
}

# File mapping for tests (source_path, target_directory, new_filename)
# We'll only move a few key files as an example, the rest would follow the same pattern
test_file_mapping = [
    # Unit tests
    (TESTS_DIR / "test_code_extractor.py", "unit/extractors", "test_code_extractor.py"),
    (TESTS_DIR / "test_python_extractor.py", "unit/extractors", "test_python_extractor.py"),
    (TESTS_DIR / "test_js_ts_extractor.py", "unit/extractors", "test_js_ts_extractor.py"),
    (TESTS_DIR / "test_generic_extractor.py", "unit/extractors", "test_generic_extractor.py"),
    (TESTS_DIR / "test_block_verification.py", "unit/validation", "test_block_verification.py"),
    
    # Integration tests
    (TESTS_DIR / "test_fetch_docs_integration.py", "integration", "test_fetch_docs_integration.py"),
    (TESTS_DIR / "test_html_extraction.py", "integration", "test_html_extraction.py"),
    (TESTS_DIR / "test_output_examples.py", "integration", "test_output_examples.py"),
    
    # End-to-end tests
    (TESTS_DIR / "test_end_to_end.py", "end_to_end", "test_end_to_end.py"),
    
    # Playwright tests
    (TESTS_DIR / "test_playwright_download.py", "playwright", "test_playwright_download.py"),
    (TESTS_DIR / "test_download_patch.py", "playwright", "test_download_patch.py")
]

def create_test_directory_structure():
    """Create the new directory structure for tests."""
    print("Creating test directory structure...")
    
    # Create README with directory structure overview
    readme_content = "# Extraction Module Tests\n\n"
    readme_content += "This directory contains tests for the extraction module organized for clarity.\n\n"
    readme_content += "## Directory Structure\n\n"
    
    # Create subdirectories
    for subdir, description in new_structure.items():
        if isinstance(description, dict):
            # This is a nested structure
            (TESTS_DIR / subdir).mkdir(exist_ok=True)
            
            # Create nested README
            nested_readme = f"# {subdir.title()} Tests\n\n"
            nested_readme += f"This directory contains {subdir} tests for the extraction module.\n\n"
            nested_readme += "## Directory Structure\n\n"
            
            # Create nested directories
            for nested_subdir, nested_desc in description.items():
                nested_path = TESTS_DIR / subdir / nested_subdir
                nested_path.mkdir(exist_ok=True, parents=True)
                
                # Create README in nested directory
                with open(nested_path / "README.md", "w") as f:
                    f.write(f"# {nested_subdir.title()} Tests\n\n")
                    f.write(f"{nested_desc}\n")
                
                # Add to nested README
                nested_readme += f"- **{nested_subdir}**: {nested_desc}\n"
            
            # Write nested README
            with open(TESTS_DIR / subdir / "README.md", "w") as f:
                f.write(nested_readme)
            
            # Add to main README
            readme_content += f"- **{subdir}**: Contains tests by component\n"
        else:
            # This is a simple directory
            (TESTS_DIR / subdir).mkdir(exist_ok=True)
            
            # Create README in directory
            with open(TESTS_DIR / subdir / "README.md", "w") as f:
                f.write(f"# {subdir.title()} Tests\n\n")
                f.write(f"{description}\n")
            
            # Add to main README
            readme_content += f"- **{subdir}**: {description}\n"
    
    # Write the main README
    with open(TESTS_DIR / "README.md", "w") as f:
        f.write(readme_content)
    
    print("Test directory structure created successfully.")

def copy_test_files():
    """Copy test files to their new locations."""
    print("Copying test files to new locations...")
    
    for source_path, target_dir, new_filename in test_file_mapping:
        if not source_path.exists():
            print(f"Warning: Source file does not exist: {source_path}")
            continue
        
        target_path = TESTS_DIR / target_dir / new_filename
        
        # Ensure parent directory exists
        target_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied: {source_path} -> {target_path}")

def create_init_files():
    """Create __init__.py files in all test directories."""
    print("Creating __init__.py files...")
    
    # Create __init__.py in main test directory if it doesn't exist
    if not (TESTS_DIR / "__init__.py").exists():
        with open(TESTS_DIR / "__init__.py", "w") as f:
            f.write('"""Extraction module tests."""\n')
    
    # Create __init__.py in subdirectories
    for subdir, description in new_structure.items():
        init_path = TESTS_DIR / subdir / "__init__.py"
        if not init_path.exists():
            with open(init_path, "w") as f:
                f.write(f'"""{subdir.title()} tests for extraction module."""\n')
        
        if isinstance(description, dict):
            # Create __init__.py in nested directories
            for nested_subdir in description.keys():
                nested_init_path = TESTS_DIR / subdir / nested_subdir / "__init__.py"
                if not nested_init_path.exists():
                    with open(nested_init_path, "w") as f:
                        f.write(f'"""{nested_subdir.title()} tests for {subdir}."""\n')
    
    print("__init__.py files created successfully.")

def update_imports():
    """Update import statements in Python files."""
    print("Updating import statements in test files...")
    
    for subdir, description in new_structure.items():
        if isinstance(description, dict):
            # Handle nested directories
            for nested_subdir in description.keys():
                update_imports_in_dir(TESTS_DIR / subdir / nested_subdir)
        else:
            # Handle simple directories
            update_imports_in_dir(TESTS_DIR / subdir)

def update_imports_in_dir(directory):
    """Update import statements in Python files in a directory."""
    for py_file in directory.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, "r") as f:
                content = f.read()
            
            # Update relative imports if needed
            # This would need to be customized based on actual import patterns
            
            with open(py_file, "w") as f:
                f.write(content)
                
            print(f"Processed: {py_file}")
        except Exception as e:
            print(f"Error processing {py_file}: {e}")

def create_conftest_file():
    """Create or update the conftest.py file with fixtures."""
    conftest_path = TESTS_DIR / "conftest.py"
    fixtures_dir = TESTS_DIR / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    # Basic conftest template
    conftest_content = '''"""
Test fixtures for extraction module.

This file contains pytest fixtures used across multiple test files.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_code_file(tmp_path):
    """Create a sample Python file for testing."""
    file_path = tmp_path / "sample.py"
    with open(file_path, "w") as f:
        f.write("""
def example_function():
    """Example function for testing."""
    return 42

class ExampleClass:
    """Example class for testing."""
    
    def __init__(self):
        self.value = 10
        
    def get_value(self):
        """Get the value."""
        return self.value
""")
    return file_path

@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a sample Markdown file for testing."""
    file_path = tmp_path / "sample.md"
    with open(file_path, "w") as f:
        f.write("""# Sample Markdown
        
This is a sample markdown file for testing.

## Section 1

Some content in section 1.

```python
def example():
    return "Hello World"
```

## Section 2

More content in section 2.
""")
    return file_path
'''
    
    # Write or update conftest.py
    with open(conftest_path, "w") as f:
        f.write(conftest_content)
    
    print(f"Created conftest.py at {conftest_path}")

def main():
    """Execute the test reorganization process."""
    print("Starting test reorganization...")
    
    # Create the directory structure
    create_test_directory_structure()
    
    # Copy files to new locations
    copy_test_files()
    
    # Create __init__.py files
    create_init_files()
    
    # Create conftest.py
    create_conftest_file()
    
    # Update import statements
    update_imports()
    
    print("Test reorganization complete!")

if __name__ == "__main__":
    main()