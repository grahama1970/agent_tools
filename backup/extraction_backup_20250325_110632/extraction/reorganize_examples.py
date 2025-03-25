#!/usr/bin/env python3
"""
Reorganize example files for the extraction module.

This script implements Phase 2 of the reorganization plan by:
1. Creating a clear directory structure for examples
2. Moving example files to appropriate locations
3. Updating references within examples
"""

import os
import shutil
from pathlib import Path
import re
import json

# Define paths
BASE_DIR = Path(__file__).parent
EXAMPLES_DIR = BASE_DIR / "examples"
END_TO_END_DIR = EXAMPLES_DIR / "end_to_end"

# Create new directory structure
new_structure = {
    "basic": "Basic extraction examples",
    "end_to_end": {
        "code": "Code extraction examples",
        "markdown": "Markdown extraction examples",
        "html": "HTML extraction examples",
        "playwright": "Playwright integration examples"
    },
    "validation": "Validation examples",
    "integration": "Integration examples with other systems"
}

# File mapping for examples (source_path, target_directory, new_filename)
file_mapping = [
    # Move HTML extraction examples
    (END_TO_END_DIR / "test_arangodb_extraction_transparent.py", "end_to_end/html", "arangodb_extraction.py"),
    (END_TO_END_DIR / "test_readthedocs_extraction_transparent.py", "end_to_end/html", "readthedocs_extraction.py"),
    (END_TO_END_DIR / "arangodb_validator.py", "end_to_end/html", "arangodb_validator.py"),
    
    # Move code extraction examples
    (END_TO_END_DIR / "test_code_extraction.py", "end_to_end/code", "code_extraction.py"),
    (END_TO_END_DIR / "test_js_extraction.py", "end_to_end/code", "js_extraction.py"),
    (END_TO_END_DIR / "length_function_test.py", "end_to_end/code", "length_function_test.py"),
    
    # Move validation examples
    (END_TO_END_DIR / "validate_extraction_format.py", "validation", "validate_extraction_format.py"),
    (END_TO_END_DIR / "validate_hierarchy.py", "validation", "validate_hierarchy.py"),
    (END_TO_END_DIR / "validate_qa_compatibility.py", "validation", "validate_qa_compatibility.py"),
    (END_TO_END_DIR / "validation.py", "validation", "validation.py"),
    (END_TO_END_DIR / "example_extract.py", "validation", "example_extract.py"),
    (END_TO_END_DIR / "quick_extract.py", "validation", "quick_extract.py"),
    
    # Move expected format files
    (END_TO_END_DIR / "expected_format_template.json", "validation", "expected_format_template.json"),
    
    # Move integration examples
    (END_TO_END_DIR / "test_fetch_docs_integration.py", "integration", "fetch_docs_integration.py"),
    (END_TO_END_DIR / "test_extraction_qa_integration.py", "integration", "qa_integration.py"),
    (END_TO_END_DIR / "test_playwright_fetch.py", "end_to_end/playwright", "playwright_fetch.py"),
    (END_TO_END_DIR / "download_site_patch.py", "end_to_end/playwright", "download_site_patch.py"),
    
    # Move Markdown extraction examples
    (END_TO_END_DIR / "test_deepseek_extraction.py", "end_to_end/markdown", "deepseek_extraction.py")
]

def create_directory_structure():
    """Create the new directory structure for examples."""
    print("Creating examples directory structure...")
    
    # Ensure the examples directory exists
    EXAMPLES_DIR.mkdir(exist_ok=True)
    
    # Create README with directory structure overview
    readme_content = "# Extraction Module Examples\n\n"
    readme_content += "This directory contains examples for the extraction module.\n\n"
    readme_content += "## Directory Structure\n\n"
    
    # Create subdirectories
    for subdir, description in new_structure.items():
        if isinstance(description, dict):
            # This is a nested structure
            (EXAMPLES_DIR / subdir).mkdir(exist_ok=True)
            
            # Create nested README
            nested_readme = f"# {subdir.title()} Examples\n\n"
            nested_readme += f"This directory contains {subdir} examples for the extraction module.\n\n"
            nested_readme += "## Directory Structure\n\n"
            
            # Create nested directories
            for nested_subdir, nested_desc in description.items():
                nested_path = EXAMPLES_DIR / subdir / nested_subdir
                nested_path.mkdir(exist_ok=True, parents=True)
                
                # Create README in nested directory
                with open(nested_path / "README.md", "w") as f:
                    f.write(f"# {nested_subdir.title()} Examples\n\n")
                    f.write(f"{nested_desc}\n")
                
                # Add to nested README
                nested_readme += f"- **{nested_subdir}**: {nested_desc}\n"
            
            # Write nested README
            with open(EXAMPLES_DIR / subdir / "README.md", "w") as f:
                f.write(nested_readme)
            
            # Add to main README
            readme_content += f"- **{subdir}**: Contains examples organized by type\n"
        else:
            # This is a simple directory
            (EXAMPLES_DIR / subdir).mkdir(exist_ok=True)
            
            # Create README in directory
            with open(EXAMPLES_DIR / subdir / "README.md", "w") as f:
                f.write(f"# {subdir.title()} Examples\n\n")
                f.write(f"{description}\n")
            
            # Add to main README
            readme_content += f"- **{subdir}**: {description}\n"
    
    # Write the main README
    with open(EXAMPLES_DIR / "README.md", "w") as f:
        f.write(readme_content)
    
    print("Examples directory structure created successfully.")

def copy_files():
    """Copy files to their new locations."""
    print("Copying example files to new locations...")
    
    for source_path, target_dir, new_filename in file_mapping:
        if not source_path.exists():
            print(f"Warning: Source file does not exist: {source_path}")
            continue
        
        target_path = EXAMPLES_DIR / target_dir / new_filename
        
        # Ensure parent directory exists
        target_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied: {source_path} -> {target_path}")

def copy_json_examples():
    """Copy JSON example files to their appropriate directories."""
    print("Copying JSON example files...")
    
    # Create expected_formats directory
    expected_formats_dir = EXAMPLES_DIR / "validation" / "expected_formats"
    expected_formats_dir.mkdir(exist_ok=True, parents=True)
    
    # Copy expected format files from end_to_end directory
    for json_file in END_TO_END_DIR.glob("*expected_format.json"):
        target_path = expected_formats_dir / json_file.name
        shutil.copy2(json_file, target_path)
        print(f"Copied JSON example: {json_file} -> {target_path}")
    
    # Copy expected format files from expected_formats directory
    source_expected_formats = END_TO_END_DIR / "expected_formats"
    if source_expected_formats.exists():
        for json_file in source_expected_formats.glob("*.json"):
            target_path = expected_formats_dir / json_file.name
            shutil.copy2(json_file, target_path)
            print(f"Copied JSON example: {json_file} -> {target_path}")

def update_imports():
    """Update import statements in Python files."""
    print("Updating import statements in Python files...")
    
    for subdir, description in new_structure.items():
        if isinstance(description, dict):
            # Handle nested directories
            for nested_subdir in description.keys():
                update_imports_in_dir(EXAMPLES_DIR / subdir / nested_subdir)
        else:
            # Handle simple directories
            update_imports_in_dir(EXAMPLES_DIR / subdir)

def update_imports_in_dir(directory):
    """Update import statements in Python files in a directory."""
    for py_file in directory.glob("*.py"):
        try:
            with open(py_file, "r") as f:
                content = f.read()
            
            # Update relative imports
            # Example: from ..end_to_end import -> from ...end_to_end import
            content = re.sub(r'from \.\.examples\.end_to_end', r'from ...end_to_end', content)
            content = re.sub(r'from \.\.end_to_end', r'from ...end_to_end', content)
            
            # Update direct imports from validation files
            for src_file, target_dir, new_filename in file_mapping:
                if src_file.suffix == '.py' and target_dir.startswith('validation'):
                    old_import = f"from {src_file.stem}"
                    new_import = f"from ...validation.{new_filename[:-3]}"
                    content = content.replace(old_import, new_import)
            
            with open(py_file, "w") as f:
                f.write(content)
                
            print(f"Updated imports in: {py_file}")
        except Exception as e:
            print(f"Error updating imports in {py_file}: {e}")

def create_index_files():
    """Create index files for example categories."""
    for subdir, description in new_structure.items():
        if isinstance(description, dict):
            # Handle nested directories
            for nested_subdir, nested_desc in description.items():
                dir_path = EXAMPLES_DIR / subdir / nested_subdir
                create_index_for_dir(dir_path, f"{nested_subdir.title()} Examples", nested_desc)
        else:
            # Handle simple directories
            dir_path = EXAMPLES_DIR / subdir
            create_index_for_dir(dir_path, f"{subdir.title()} Examples", description)

def create_index_for_dir(dir_path, title, description):
    """Create an index file for a specific directory."""
    py_files = list(dir_path.glob("*.py"))
    json_files = list(dir_path.glob("*.json"))
    
    # Skip if too few files exist
    if len(py_files) + len(json_files) <= 1:
        return
    
    # Create INDEX.md
    with open(dir_path / "INDEX.md", "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"{description}\n\n")
        
        if py_files:
            f.write("## Python Examples\n\n")
            for file_path in sorted(py_files):
                try:
                    with open(file_path, "r") as py_file:
                        # Extract docstring if available
                        content = py_file.read()
                        docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
                        description = docstring_match.group(1).strip() if docstring_match else f"{file_path.stem} example"
                        # Just get the first line
                        short_desc = description.split('\n')[0]
                    
                    f.write(f"- **[{file_path.name}]({file_path.name})**: {short_desc}\n")
                except:
                    f.write(f"- **[{file_path.name}]({file_path.name})**\n")
        
        if json_files:
            f.write("\n## JSON Examples\n\n")
            for file_path in sorted(json_files):
                try:
                    with open(file_path, "r") as json_file:
                        # Try to extract some info from the JSON
                        data = json.load(json_file)
                        if isinstance(data, dict) and "description" in data:
                            description = data["description"]
                        else:
                            description = f"{file_path.stem} example format"
                    
                    f.write(f"- **[{file_path.name}]({file_path.name})**: {description}\n")
                except:
                    f.write(f"- **[{file_path.name}]({file_path.name})**\n")
    
    print(f"Created index for: {dir_path}")

def main():
    """Execute the reorganization process."""
    print("Starting examples reorganization...")
    
    # Create the directory structure
    create_directory_structure()
    
    # Copy files to new locations
    copy_files()
    
    # Copy JSON examples
    copy_json_examples()
    
    # Update import statements
    update_imports()
    
    # Create index files
    create_index_files()
    
    print("Examples reorganization complete!")

if __name__ == "__main__":
    main()