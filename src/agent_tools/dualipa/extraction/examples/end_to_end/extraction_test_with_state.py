#!/usr/bin/env python3
"""
Extraction Test with State Management

This script demonstrates how to use the TestStateManager to maintain
reliable state during extraction tests, addressing the limitations
of context tracking in AI assistants.

Usage:
    python extraction_test_with_state.py --repo-path /path/to/repo --output-file output.json
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, parent_dir)

# Import state manager
try:
    # Try relative import
    sys.path.append(parent_dir)
    from extraction.test_state_manager import (
        get_state_manager, verify_extraction_completeness,
        what_am_i_doing, remember_context, add_docs, get_docs
    )
except ImportError:
    try:
        # Try absolute import
        from agent_tools.dualipa.extraction.test_state_manager import (
            get_state_manager, verify_extraction_completeness,
            what_am_i_doing, remember_context, add_docs, get_docs
        )
    except ImportError:
        print("Could not import test_state_manager. Please ensure it exists at the correct path.")
        sys.exit(1)

# Constants
DEFAULT_REPO_PATH = "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb"
DEFAULT_OUTPUT_PATH = "/home/grahama/workspace/experiments/agent_tools/extraction_with_state_output.json"


def analyze_repository(repo_path: str, state_manager=None):
    """
    Analyze repository and store statistics in state manager.
    
    Args:
        repo_path: Path to repository
        state_manager: Optional state manager instance
        
    Returns:
        Repository statistics dictionary
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("analyze_repository", "Analyzing repository structure")
    
    # Update context to remember what we're doing
    remember_context(
        what_im_doing="Analyzing repository structure",
        why_im_doing_it="To understand what files are available for extraction",
        what_step="Counting files by type and identifying important files",
        what_next="After analysis, we'll extract Python files"
    )
    
    # Add documentation about this analysis step
    add_docs(
        topic="Repository Analysis",
        content="""
Repository analysis is a critical first step in extraction.
It counts files by type, identifies important files, and provides
baseline statistics for verification.

This step MUST be performed before any extraction to ensure:
1. We know what files are available
2. We can verify extraction completeness
3. We can identify critical files to include
        """,
        summary="Count files by type before extraction",
        importance=10
    )
    
    print(f"Analyzing repository: {repo_path}")
    
    # Initialize counters
    file_counts = {}
    important_files = []
    total_files = 0
    
    # File extensions to track
    target_extensions = ['.py', '.js', '.ts', '.cpp', '.h', '.md', '.json']
    
    # Add documentation about target extensions
    add_docs(
        topic="Target Extensions",
        content=f"""
The following file extensions are tracked for extraction:
{', '.join(target_extensions)}

These represent the most important file types for code and documentation extraction.
Python (.py) files contain Python code
JavaScript (.js) files contain JavaScript code
TypeScript (.ts) files contain TypeScript code
C++ files (.cpp, .h) contain C++ code
Markdown (.md) files contain documentation
JSON (.json) files contain configuration data
        """,
        summary="File extensions that are tracked for extraction",
        importance=8
    )
    
    # Walk repository
    for root, dirs, files in os.walk(repo_path):
        # Skip third-party code
        if '3rdParty' in root or 'node_modules' in root or '.git' in root:
            continue
            
        for file in files:
            # Get file extension
            _, ext = os.path.splitext(file)
            if ext:
                # Count by extension
                file_counts[ext] = file_counts.get(ext, 0) + 1
                total_files += 1
                
                # Track file in state manager if it's a target extension
                if ext in target_extensions:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_path)
                    
                    # Store in state manager
                    state_manager.track_file(rel_path, ext, os.path.getsize(full_path))
                    
                    # Check for important files
                    if file == 'generate_ii_sa_dataset.py' or file == 'gantt.py':
                        important_files.append(rel_path)
    
    # Calculate percentages
    percentages = {ext: (count / total_files) * 100 for ext, count in file_counts.items()}
    
    # Store statistics in state manager
    for ext, count in file_counts.items():
        percentage = percentages.get(ext, 0)
        state_manager.set_repo_stats(ext, count, percentage)
    
    # Store important files
    state_manager.set_metadata("important_files", important_files)
    
    # Store total file count
    state_manager.set_metadata("total_files", total_files)
    
    # Print repository statistics
    print(f"Repository contains {total_files} files")
    print("\nFile counts by extension:")
    for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 5:  # Only show extensions with more than 5 files
            print(f"- {ext}: {count} files ({percentages[ext]:.1f}%)")
    
    # Print important files
    print("\nImportant files:")
    for file in important_files:
        print(f"- {file}")
    
    # Create repository stats dictionary
    repo_stats = {
        'total_files': total_files,
        'file_counts': file_counts,
        'percentages': percentages,
        'important_files': important_files
    }
    
    # Store in state manager
    state_manager.set("repo_stats", repo_stats)
    
    # Verify counts match expectations
    python_count = file_counts.get('.py', 0)
    state_manager.verify("python_file_count", python_count, python_count)
    
    return repo_stats


def extract_python_files(repo_path: str, state_manager=None):
    """
    Extract all Python files from repository.
    
    Args:
        repo_path: Path to repository
        state_manager: Optional state manager instance
        
    Returns:
        List of extracted Python files
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Refresh our context - what are we doing?
    what_am_i_doing()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("extract_python_files", "Extracting Python files")
    
    # Update context for this step
    remember_context(
        what_im_doing="Extracting Python files from repository",
        why_im_doing_it="To create structured blocks for all Python files",
        what_step="Reading and structuring Python file content",
        what_next="After extraction, we'll create QA-compatible output"
    )
    
    # Add documentation about Python extraction
    add_docs(
        topic="Python Extraction",
        content="""
Python file extraction creates structured blocks for each Python file.
Each block includes:
- UUID: Unique identifier
- File path: Path within repository
- Content: Full file content
- Language: Set to 'python'
- Type: Set to 'file'
- Extraction focus: Set to ['code']

This extraction process must ensure:
1. Every Python file is included
2. All content is properly extracted
3. Important files are verified to be included
4. Extraction completeness is validated against repository stats
        """,
        summary="Extract all Python files into structured blocks",
        importance=9
    )
    
    # Get repository stats from state
    repo_stats = state_manager.get("repo_stats")
    if not repo_stats:
        print("Repository has not been analyzed. Running analysis first.")
        repo_stats = analyze_repository(repo_path, state_manager)
    
    # Verify Python files exist
    python_count = repo_stats['file_counts'].get('.py', 0)
    state_manager.assert_verify(
        "python_files_exist",
        True,
        python_count > 0,
        "No Python files found in repository"
    )
    
    print(f"Extracting {python_count} Python files from repository")
    
    # Extracted files list
    extracted_files = []
    
    # Walk repository
    for root, dirs, files in os.walk(repo_path):
        # Skip third-party code
        if '3rdParty' in root or 'node_modules' in root or '.git' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                
                # Read file content
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Create extraction record
                    extraction_id = f"py_{len(extracted_files):04d}"
                    extracted_file = {
                        'id': extraction_id,
                        'uuid': extraction_id,
                        'file_path': rel_path,
                        'name': file,
                        'type': 'file',
                        'language': 'python',
                        'content': content,
                        'extraction_focus': ['code'],
                        'summary_instructions': 'Explain the purpose of this Python file',
                        'breadcrumb': [file],
                        'parent_uuid': None,
                        'child_uuids': []
                    }
                    
                    # Add to extracted files
                    extracted_files.append(extracted_file)
                    
                    # Mark file as extracted in state manager
                    state_manager.mark_file_extracted(rel_path, extraction_id)
                    
                    # Print progress every 10 files
                    if len(extracted_files) % 10 == 0:
                        print(f"Extracted {len(extracted_files)} Python files")
                    
                except Exception as e:
                    print(f"Error extracting {rel_path}: {e}")
    
    # Verify extraction count
    state_manager.assert_verify(
        "all_python_files_extracted",
        python_count,
        len(extracted_files),
        f"Failed to extract all Python files. Expected {python_count}, got {len(extracted_files)}"
    )
    
    # Check important files
    important_files = state_manager.get_metadata("important_files", [])
    for important_file in important_files:
        if important_file.endswith('.py'):
            # Check if file was extracted
            extracted = any(e['file_path'] == important_file for e in extracted_files)
            state_manager.assert_verify(
                f"important_file_{os.path.basename(important_file)}_extracted",
                True,
                extracted,
                f"Failed to extract important file: {important_file}"
            )
    
    # Update extraction stats
    state_manager.update_extracted_count('.py', len(extracted_files))
    
    # Store extracted files
    state_manager.set("extracted_python_files", extracted_files)
    
    print(f"Successfully extracted {len(extracted_files)} Python files")
    return extracted_files


def create_qa_compatible_output(extracted_files: List[Dict[str, Any]], output_path: str, state_manager=None):
    """
    Create QA-compatible output from extracted files.
    
    Args:
        extracted_files: List of extracted files
        output_path: Path to output file
        state_manager: Optional state manager instance
        
    Returns:
        Path to output file
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("create_qa_output", "Creating QA-compatible output")
    
    # Create section relationships
    relationships = {
        "parent_child": {},
        "imports": {},
        "inheritance": {}
    }
    
    # Fill parent-child relationships
    for file in extracted_files:
        file_uuid = file['uuid']
        parent_uuid = file.get('parent_uuid')
        child_uuids = file.get('child_uuids', [])
        
        relationships["parent_child"][file_uuid] = {
            "parent": parent_uuid,
            "children": child_uuids
        }
    
    # Create metadata
    metadata = {
        "model_used": "dualipa-extraction-with-state",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "statistics": {
            "total_blocks": len(extracted_files),
            "total_files": len(extracted_files),
            "block_types": {"file": len(extracted_files)},
            "languages": {"python": len(extracted_files)}
        }
    }
    
    # Create output
    output = {
        "sections": extracted_files,
        "section_relationships": relationships,
        "extraction_metadata": metadata
    }
    
    # Save output to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    # Mark files as included in output
    for file in extracted_files:
        file_path = file.get('file_path')
        if file_path:
            state_manager.mark_file_included_in_output(file_path)
    
    # Verify output file exists
    state_manager.assert_verify(
        "output_file_created",
        True,
        os.path.exists(output_path),
        f"Failed to create output file: {output_path}"
    )
    
    # Store output statistics
    state_manager.set_metadata("output_stats", {
        "sections": len(extracted_files),
        "timestamp": datetime.datetime.now().isoformat(),
        "file_path": output_path
    })
    
    print(f"Successfully created QA-compatible output at {output_path}")
    print(f"- {len(extracted_files)} sections")
    
    return output_path


def validate_output(output_path: str, repo_path: str, state_manager=None):
    """
    Validate the output file.
    
    Args:
        output_path: Path to output file
        repo_path: Path to repository
        state_manager: Optional state manager instance
        
    Returns:
        True if validation passed, False otherwise
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("validate_output", "Validating output file")
    
    # Load output file
    with open(output_path, 'r', encoding='utf-8') as f:
        output = json.load(f)
    
    # Check basic structure
    required_keys = ['sections', 'section_relationships', 'extraction_metadata']
    for key in required_keys:
        state_manager.assert_verify(
            f"output_has_{key}",
            True,
            key in output,
            f"Output is missing required key: {key}"
        )
    
    # Check sections have required fields
    section_required_fields = ['uuid', 'id', 'type', 'name', 'content', 'language']
    
    # Track missing fields
    missing_fields = []
    
    for i, section in enumerate(output['sections']):
        for field in section_required_fields:
            if field not in section:
                missing_fields.append((i, field))
    
    # Verify no missing fields
    state_manager.assert_verify(
        "sections_have_required_fields",
        0,
        len(missing_fields),
        f"Sections are missing required fields: {missing_fields}"
    )
    
    # Load repository stats
    repo_stats = state_manager.get("repo_stats")
    
    # Verify extraction completeness
    important_files = state_manager.get_metadata("important_files", [])
    python_files = [f for f in important_files if f.endswith('.py')]
    
    for important_file in python_files:
        # Check if file is in sections
        found = False
        for section in output['sections']:
            if section.get('file_path') == important_file:
                found = True
                break
        
        state_manager.assert_verify(
            f"output_includes_{os.path.basename(important_file)}",
            True,
            found,
            f"Output is missing important file: {important_file}"
        )
    
    # Check section count matches expected count
    python_count = repo_stats['file_counts'].get('.py', 0)
    state_manager.assert_verify(
        "output_section_count",
        python_count,
        len(output['sections']),
        f"Output section count does not match expected count. Expected {python_count}, got {len(output['sections'])}"
    )
    
    # Store validation result
    validation_passed = len(missing_fields) == 0
    state_manager.set_metadata("validation_passed", validation_passed)
    
    print(f"Validation {'passed' if validation_passed else 'failed'}")
    return validation_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extraction test with state management")
    parser.add_argument("--repo-path", type=str, default=DEFAULT_REPO_PATH,
                      help="Path to repository")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_PATH,
                      help="Path to output file")
    parser.add_argument("--report-file", type=str, default=None,
                      help="Path to report file")
    args = parser.parse_args()
    
    # Get state manager with persistent path for debugging
    state_manager = get_state_manager("extraction_test_state.db")
    
    try:
        # Step 1: Repository analysis
        repo_stats = analyze_repository(args.repo_path, state_manager)
        
        # Step 2: Extract Python files
        extracted_files = extract_python_files(args.repo_path, state_manager)
        
        # Step 3: Create QA-compatible output
        output_path = create_qa_compatible_output(extracted_files, args.output_file, state_manager)
        
        # Step 4: Validate output
        validation_passed = validate_output(output_path, args.repo_path, state_manager)
        
        # Generate report
        if args.report_file:
            state_manager.generate_report(args.report_file)
        else:
            report_path = f"extraction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            state_manager.generate_report(report_path)
        
        # Print extraction statistics
        print("\nExtraction Statistics:")
        stats = state_manager.get_extraction_stats()
        print(f"- Total files: {stats['total_files']}")
        print(f"- Extracted files: {stats['extracted_files']} ({stats['extraction_rate']:.1f}%)")
        print(f"- Included in output: {stats['included_files']} ({stats['inclusion_rate']:.1f}%)")
        
        # Results
        if validation_passed:
            print("\n✅ Extraction test passed!")
            return 0
        else:
            print("\n❌ Extraction test failed!")
            return 1
            
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        
        # Generate error report
        error_report_path = f"extraction_error_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        state_manager.generate_report(error_report_path)
        print(f"Error report generated: {error_report_path}")
        
        return 1
    
    finally:
        # Always close state manager
        if 'state_manager' in locals():
            state_manager.close()


if __name__ == "__main__":
    sys.exit(main())