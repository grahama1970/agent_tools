#!/usr/bin/env python3
"""
Master reorganization script for extraction module.

This script implements the full reorganization plan by:
1. Running each phase of reorganization in sequence
2. Verifying that each phase completed successfully
3. Creating a summary report of changes
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path
import shutil
import time

# Define paths
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
EXAMPLES_DIR = BASE_DIR / "examples"
INTEGRATION_DIR = BASE_DIR / "integration"
PROJECT_ROOT = BASE_DIR.parent.parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "tests" / "dualipa" / "extraction"

# Define reorganization scripts
scripts = [
    {
        "name": "reorganize_docs.py",
        "description": "Consolidate documentation files",
        "path": BASE_DIR / "reorganize_docs.py"
    },
    {
        "name": "reorganize_examples.py",
        "description": "Reorganize example files",
        "path": BASE_DIR / "reorganize_examples.py"
    },
    {
        "name": "reorganize_tests.py",
        "description": "Reorganize test files",
        "path": BASE_DIR / "reorganize_tests.py"
    },
    {
        "name": "reorganize_integration.py",
        "description": "Improve integration points",
        "path": BASE_DIR / "reorganize_integration.py"
    }
]

def run_script(script_path):
    """Run a Python script and return success status."""
    print(f"Running {script_path}...")
    try:
        # Load the script as a module and execute its main function
        spec = importlib.util.spec_from_file_location("script", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call the main function if it exists
        if hasattr(module, "main"):
            module.main()
        
        return True
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False

def verify_directories():
    """Verify that all expected directories were created."""
    expected_dirs = {
        "docs": DOCS_DIR.exists(),
        "examples": EXAMPLES_DIR.exists(),
        "integration": INTEGRATION_DIR.exists()
    }
    
    # Check subdirectories in docs
    if DOCS_DIR.exists():
        expected_dirs.update({
            f"docs/{subdir}": (DOCS_DIR / subdir).exists()
            for subdir in ["api", "concepts", "examples", "integration", "validation", "guides", "testing"]
        })
    
    # Check subdirectories in examples
    if EXAMPLES_DIR.exists():
        expected_dirs.update({
            f"examples/{subdir}": (EXAMPLES_DIR / subdir).exists()
            for subdir in ["basic", "end_to_end", "validation", "integration"]
        })
        
        # Check nested subdirectories in end_to_end
        if (EXAMPLES_DIR / "end_to_end").exists():
            expected_dirs.update({
                f"examples/end_to_end/{subdir}": (EXAMPLES_DIR / "end_to_end" / subdir).exists()
                for subdir in ["code", "markdown", "html", "playwright"]
            })
    
    # Print verification results
    print("\nDirectory Verification:")
    for dir_name, exists in expected_dirs.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}")
    
    # Return overall success
    return all(exists for exists in expected_dirs.values())

def run_tests():
    """Run basic tests to verify functionality."""
    print("\nRunning basic tests...")
    
    # Run pytest to verify basic functionality
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(TESTS_DIR / "test_simple.py"), "-v"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check result
        if result.returncode == 0:
            print("Basic tests passed successfully!")
            return True
        else:
            print("Basic tests failed. Output:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def create_backup():
    """Create a backup of the current directory structure."""
    print("\nCreating backup...")
    
    # Create backup timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_ROOT / "backup" / f"extraction_backup_{timestamp}"
    
    try:
        # Create directories
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup extraction directory
        shutil.copytree(BASE_DIR, backup_dir / "extraction", dirs_exist_ok=True)
        
        # Backup tests directory
        shutil.copytree(TESTS_DIR, backup_dir / "tests", dirs_exist_ok=True)
        
        print(f"Backup created at: {backup_dir}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def create_summary():
    """Create a summary of the reorganization."""
    summary_path = BASE_DIR / "REORGANIZATION_SUMMARY.md"
    
    summary_content = """# Reorganization Summary

This document summarizes the reorganization of the extraction module directory structure.

## Directory Structure

The extraction module now has the following directory structure:

```
extraction/
├── docs/              # Documentation
│   ├── api/           # API reference documentation
│   ├── concepts/      # Concept explanations
│   ├── examples/      # Example usage scenarios
│   ├── integration/   # Integration points with other modules
│   ├── validation/    # Validation framework documentation
│   ├── guides/        # How-to guides
│   └── testing/       # Testing approach and strategies
│
├── examples/          # Examples
│   ├── basic/         # Basic extraction examples
│   ├── end_to_end/    # Complete workflow examples
│   │   ├── code/      # Code extraction examples
│   │   ├── markdown/  # Markdown extraction examples
│   │   ├── html/      # HTML extraction examples
│   │   └── playwright/ # Playwright integration examples
│   ├── validation/    # Validation examples
│   └── integration/   # Integration examples with other systems
│
├── integration/       # Integration modules
│   ├── fetch_docs_adapter.py     # Fetch docs integration
│   ├── fetch_docs_config.py      # Fetch docs configuration
│   ├── qa_adapter.py             # QA system integration
│   ├── qa_config.py              # QA system configuration
│   ├── validation_adapter.py     # Validation integration
│   └── validation_config.py      # Validation configuration
```

## Test Structure

The test directory now has the following structure:

```
tests/dualipa/extraction/
├── unit/              # Unit tests for individual components
│   ├── extractors/    # Tests for extractors
│   ├── utils/         # Tests for utilities
│   └── validation/    # Tests for validation components
├── integration/       # Tests that verify component integration
├── end_to_end/        # Full workflow tests
├── fixtures/          # Test fixtures and data
└── playwright/        # Tests for Playwright functionality
```

## Changes Made

1. **Documentation Consolidation**:
   - Gathered all documentation into a clear structure
   - Organized documentation by topic and purpose
   - Created index files for easy navigation

2. **Examples Reorganization**:
   - Organized examples by purpose and type
   - Simplified example code and documentation
   - Created clear structure for end-to-end examples

3. **Test Structure Improvement**:
   - Organized tests by type and purpose
   - Created fixtures for common test data
   - Reduced duplication across test files

4. **Integration Improvement**:
   - Created clear adapter modules for external systems
   - Defined consistent integration interfaces
   - Documented integration points and usage

## Benefits

- **Improved Discoverability**: Clear directory structure makes files easier to find
- **Reduced Duplication**: Consolidated duplicate files and functionality
- **Better Documentation**: Comprehensive and well-organized documentation
- **Clearer Integration**: Well-defined interfaces for external systems
- **Simplified Testing**: Consistent testing approach with shared fixtures

## Next Steps

1. **Complete Testing**: Ensure all tests pass with the new structure
2. **Update Documentation**: Update any references to old file locations
3. **Communicate Changes**: Inform team members of new structure
4. **CI Integration**: Update CI scripts to reflect new directory structure
"""
    
    with open(summary_path, "w") as f:
        f.write(summary_content)
    
    print(f"Summary created at: {summary_path}")

def main():
    """Execute the full reorganization process."""
    print("Starting extraction module reorganization...")
    
    # Create backup first
    if not create_backup():
        print("Backup failed. Aborting reorganization.")
        return
    
    # Run each script in sequence
    success = True
    for script in scripts:
        print(f"\n===== Running {script['name']}: {script['description']} =====")
        if not run_script(script["path"]):
            success = False
            print(f"Error in {script['name']}. Continuing with next script...")
    
    # Verify results
    print("\n===== Verifying reorganization =====")
    dir_success = verify_directories()
    
    # Run basic tests
    test_success = run_tests()
    
    # Create summary
    create_summary()
    
    # Final report
    print("\n===== Reorganization Complete =====")
    if success and dir_success and test_success:
        print("Reorganization completed successfully!")
    else:
        print("Reorganization completed with some issues.")
        
        if not success:
            print("- Some scripts encountered errors.")
        
        if not dir_success:
            print("- Some expected directories were not created.")
        
        if not test_success:
            print("- Basic tests failed after reorganization.")
    
    print("\nSee REORGANIZATION_SUMMARY.md for details on the new structure.")

if __name__ == "__main__":
    main()