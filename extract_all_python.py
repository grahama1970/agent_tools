#!/usr/bin/env python3
"""
Extract All Python Files

This script extracts all Python files from a repository and combines them into a QA-compatible JSON.
It's simple and focused: find all Python files, extract their content, and format properly.
"""

import os
import sys
import json
import uuid
import datetime
from pathlib import Path

# Constants
REPO_PATH = "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb"
OUTPUT_PATH = "/home/grahama/workspace/experiments/agent_tools/arangodb_qa_complete_python.json"

def find_all_python_files(repo_path):
    """Find all Python files in the repository."""
    python_files = []
    
    # Use os.walk for reliability
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                if "3rdParty" not in rel_path:  # Skip third-party code
                    python_files.append(full_path)
    
    print(f"Found {len(python_files)} Python files in repository")
    return python_files

def extract_file_content(file_path):
    """Extract content from a file and create a section."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create a basic block for the file
        section_uuid = str(uuid.uuid4())
        section = {
            "uuid": section_uuid,
            "id": section_uuid,
            "name": os.path.basename(file_path),
            "type": "file",
            "language": "python",
            "content": content,
            "file_path": file_path,
            "child_uuids": [],
            "parent_uuid": None,
            "extraction_focus": ["code"],
            "summary_instructions": "Explain the purpose of this Python file",
            "breadcrumb": [os.path.basename(file_path)]
        }
        
        return section
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None

def create_qa_compatible_json(sections):
    """Create a QA-compatible JSON structure."""
    # Create section relationships
    relationships = {
        "parent_child": {},
        "imports": {},
        "inheritance": {}
    }
    
    for section in sections:
        section_uuid = section.get("uuid")
        if section_uuid:
            relationships["parent_child"][section_uuid] = {
                "parent": section.get("parent_uuid"),
                "children": section.get("child_uuids", [])
            }
    
    # Create metadata
    metadata = {
        "model_used": "dualipa-extraction",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "statistics": {
            "total_blocks": len(sections),
            "total_files": len(sections),
            "block_types": {"file": len(sections)},
            "languages": {"python": len(sections)}
        }
    }
    
    # Create final output
    output = {
        "sections": sections,
        "section_relationships": relationships,
        "extraction_metadata": metadata
    }
    
    return output

def main():
    # Step 1: Find all Python files
    python_files = find_all_python_files(REPO_PATH)
    
    # Step 2: Extract content from each file
    sections = []
    for file_path in python_files:
        print(f"Extracting {os.path.basename(file_path)}")
        section = extract_file_content(file_path)
        if section:
            sections.append(section)
    
    # Step 3: Create QA-compatible JSON
    output = create_qa_compatible_json(sections)
    
    # Step 4: Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved QA-compatible JSON with {len(sections)} Python files to {OUTPUT_PATH}")
    
    # Step 5: Verify specific file is included
    target_file = "generate_ii_sa_dataset.py"
    found = any(target_file in section.get("file_path", "") for section in sections)
    if found:
        print(f"✓ Verified that {target_file} is included in the extraction")
    else:
        print(f"❌ WARNING: {target_file} was not found in the extraction")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())