#!/usr/bin/env python3
"""
Reorganize documentation files for the extraction module.

This script implements Phase 1 of the reorganization plan by:
1. Creating a clear directory structure for docs
2. Moving documentation files to appropriate locations
3. Updating references within docs
"""

import os
import shutil
from pathlib import Path
import re
import sys

# Define paths
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
END_TO_END_DIR = BASE_DIR / "examples" / "end_to_end"
FETCH_DOCS_DIR = Path(BASE_DIR.parent.parent.parent) / "fetch_docs"

# Create new directory structure
new_structure = {
    "api": "API reference documentation",
    "concepts": "Concept explanations and architecture docs",
    "examples": "Example usage scenarios",
    "integration": "Integration points with other modules",
    "validation": "Validation framework documentation",
    "guides": "How-to guides and tutorials",
    "testing": "Testing approach and strategies"
}

# File mapping from current location to new location
# Format: (source_path, target_directory, new_filename)
file_mapping = [
    # Move files from docs directory
    (DOCS_DIR / "extraction_format.md", "concepts", "extraction_format.md"),
    (DOCS_DIR / "fetch_docs_integration.md", "integration", "fetch_docs_integration.md"),
    (DOCS_DIR / "function_reference.md", "api", "function_reference.md"),
    (DOCS_DIR / "module_relationships.md", "concepts", "module_relationships.md"),
    (DOCS_DIR / "task.md", "guides", "task.md"),
    (DOCS_DIR / "tdd_strategy.md", "testing", "tdd_strategy.md"),
    (DOCS_DIR / "tree_sitter_support.md", "concepts", "tree_sitter_support.md"),
    (DOCS_DIR / "extraction_qa_integration.md", "integration", "qa_integration.md"),
    (DOCS_DIR / "cascading_parser_strategy.md", "concepts", "cascading_parser_strategy.md"),
    
    # Move files from end_to_end directory
    (END_TO_END_DIR / "FRICTIONLESS_VALIDATION.md", "validation", "frictionless_validation.md"),
    (END_TO_END_DIR / "VALIDATION_FRAMEWORK.md", "validation", "validation_framework.md"),
    (END_TO_END_DIR / "VALIDATION_IMPLEMENTATION_SUMMARY.md", "validation", "implementation_summary.md"),
    (END_TO_END_DIR / "PARENT_CHILD_REQUIREMENTS.md", "concepts", "parent_child_requirements.md"),
    (END_TO_END_DIR / "PLAYWRIGHT_SUPPORT.md", "integration", "playwright_support.md"),
    (END_TO_END_DIR / "QA_INTEGRATION.md", "integration", "qa_integration_guide.md"),
    (END_TO_END_DIR / "TDD_STRATEGY.md", "testing", "tdd_strategy_end_to_end.md"),
    (END_TO_END_DIR / "CODE_EXTRACTION.md", "guides", "code_extraction_guide.md"),
    (END_TO_END_DIR / "README.md", "examples", "end_to_end_examples.md"),
    
    # Move relevant files from fetch_docs
    (FETCH_DOCS_DIR / "docs" / "integration_guide.md", "integration", "fetch_docs_usage.md")
]

def create_directory_structure():
    """Create the new directory structure for docs."""
    print("Creating directory structure...")
    
    # Ensure the docs directory exists
    DOCS_DIR.mkdir(exist_ok=True)
    
    # Create README with directory structure overview
    readme_content = "# Extraction Module Documentation\n\n"
    readme_content += "This directory contains documentation for the extraction module.\n\n"
    readme_content += "## Directory Structure\n\n"
    
    # Create subdirectories and add to README
    for subdir, description in new_structure.items():
        (DOCS_DIR / subdir).mkdir(exist_ok=True)
        
        # Create a README in each subdirectory
        with open(DOCS_DIR / subdir / "README.md", "w") as f:
            f.write(f"# {subdir.title()} Documentation\n\n")
            f.write(f"{description}\n")
        
        # Add to main README
        readme_content += f"- **{subdir}**: {description}\n"
    
    # Write the main README
    with open(DOCS_DIR / "README.md", "w") as f:
        f.write(readme_content)
    
    print("Directory structure created successfully.")

def copy_files():
    """Copy files to their new locations."""
    print("Copying files to new locations...")
    
    for source_path, target_dir, new_filename in file_mapping:
        if not source_path.exists():
            print(f"Warning: Source file does not exist: {source_path}")
            continue
        
        target_path = DOCS_DIR / target_dir / new_filename
        
        # Copy the file
        shutil.copy2(source_path, target_path)
        print(f"Copied: {source_path} -> {target_path}")

def update_references():
    """Update references in documentation files."""
    print("Updating references in documentation files...")
    
    # Map of old paths to new paths for reference updating
    path_mapping = {}
    for source_path, target_dir, new_filename in file_mapping:
        if source_path.exists():
            # Get relative paths from DOCS_DIR
            old_rel_path = os.path.relpath(source_path, DOCS_DIR)
            new_rel_path = f"{target_dir}/{new_filename}"
            path_mapping[old_rel_path] = new_rel_path
    
    # Update references in all markdown files
    for subdir, _, _ in new_structure.items():
        dir_path = DOCS_DIR / subdir
        for file_path in dir_path.glob("*.md"):
            update_file_references(file_path, path_mapping)

def update_file_references(file_path, path_mapping):
    """Update references in a specific file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Update markdown links
        for old_path, new_path in path_mapping.items():
            # Match markdown links: [text](path)
            old_pattern = r'\[([^\]]+)\]\(' + re.escape(str(old_path)) + r'\)'
            new_replacement = r'[\1](' + new_path + r')'
            content = re.sub(old_pattern, new_replacement, content)
        
        with open(file_path, "w") as f:
            f.write(content)
            
        print(f"Updated references in: {file_path}")
    except Exception as e:
        print(f"Error updating references in {file_path}: {e}")

def copy_examples():
    """Copy examples to the examples directory."""
    examples_dir = DOCS_DIR / "examples" / "validation"
    examples_dir.mkdir(exist_ok=True, parents=True)
    
    example_files = [
        END_TO_END_DIR / "example_extract.py",
        END_TO_END_DIR / "quick_extract.py"
    ]
    
    for example_file in example_files:
        if example_file.exists():
            target_path = examples_dir / example_file.name
            shutil.copy2(example_file, target_path)
            print(f"Copied example: {example_file} -> {target_path}")

def create_index_files():
    """Create index files for each documentation category."""
    for subdir, description in new_structure.items():
        dir_path = DOCS_DIR / subdir
        files = list(dir_path.glob("*.md"))
        
        # Skip if only README exists
        if len(files) <= 1:
            continue
        
        # Create INDEX.md
        with open(dir_path / "INDEX.md", "w") as f:
            f.write(f"# {subdir.title()} Documentation Index\n\n")
            f.write(f"{description}\n\n")
            f.write("## Available Documentation\n\n")
            
            for file_path in sorted(files):
                if file_path.name == "README.md" or file_path.name == "INDEX.md":
                    continue
                
                # Extract the title from the file
                try:
                    with open(file_path, "r") as doc_file:
                        first_line = doc_file.readline().strip()
                        title = first_line.lstrip("#").strip() if first_line.startswith("#") else file_path.stem.replace("_", " ").title()
                except:
                    title = file_path.stem.replace("_", " ").title()
                
                f.write(f"- [{title}]({file_path.name})\n")
        
        print(f"Created index for: {subdir}")

def main():
    """Execute the reorganization process."""
    print("Starting documentation reorganization...")
    
    # Create the directory structure
    create_directory_structure()
    
    # Copy files to new locations
    copy_files()
    
    # Copy examples
    copy_examples()
    
    # Update references in files
    update_references()
    
    # Create index files
    create_index_files()
    
    print("Documentation reorganization complete!")

if __name__ == "__main__":
    main()