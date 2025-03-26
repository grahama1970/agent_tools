#!/usr/bin/env python3
"""
Comprehensive ArangoDB Extraction

This script performs a comprehensive extraction of the ArangoDB repository:
1. Extracts code from multiple languages (C++, Python, JavaScript, etc.)
2. Extracts documentation from markdown files
3. Combines all extracted content into a unified structure
4. Validates and enhances the extraction with required fields
5. Generates a QA-compatible JSON file with at least 500 sections

The goal is to create a representative dataset that includes different 
languages, file types, and documentation sources for proper QA testing.
"""

import os
import sys
import json
import uuid
import logging
import argparse
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

# Try to import from the extraction module
try:
    # Try absolute imports
    from agent_tools.dualipa.extraction.extractors.code.hierarchy import extract_code_hierarchy
    from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_markdown_sections
except ImportError:
    try:
        # Try relative imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        extractors_dir = os.path.join(parent_dir, "extractors")
        code_dir = os.path.join(extractors_dir, "code")
        
        if os.path.exists(os.path.join(code_dir, "hierarchy.py")):
            sys.path.insert(0, parent_dir)
            from extractors.code.hierarchy import extract_code_hierarchy
        else:
            print(f"Could not find hierarchy.py at {code_dir}")
            raise ImportError("Cannot import extract_code_hierarchy")
            
        # Import extraction_blocks from the current directory
        sys.path.insert(0, current_dir)
        from extraction_blocks import extract_markdown_sections
    except ImportError as e:
        print(f"Error importing extraction modules: {e}")
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("full_repository_extraction")

# Constants
ARANGODB_REPO_PATH = "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb"
OUTPUT_DIR = "/home/grahama/workspace/experiments/agent_tools/arangodb_extraction"

# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".cpp": "cpp",
    ".cc": "cpp", 
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".html": "html",
    ".css": "css"
}

def find_files_by_extensions(repo_path: str, extensions: List[str], max_files_per_ext: int = 15) -> Dict[str, List[str]]:
    """
    Find files with specified extensions in the repository.
    
    Args:
        repo_path: Path to the repository
        extensions: List of file extensions to find
        max_files_per_ext: Maximum number of files to include per extension
        
    Returns:
        Dictionary mapping extensions to lists of file paths
    """
    result = {}
    
    for ext in extensions:
        try:
            # Run find command to get files with this extension
            cmd = f"find {repo_path} -type f -name '*{ext}' | grep -v '3rdParty' | sort"
            output = subprocess.check_output(cmd, shell=True, text=True)
            
            # Split output into lines and filter empty lines
            files = [f for f in output.split('\n') if f.strip()]
            
            # Limit number of files per extension
            if max_files_per_ext > 0 and len(files) > max_files_per_ext:
                # Take some files from the beginning, middle and end
                third = max_files_per_ext // 3
                selected = files[:third]  # Beginning
                
                if third > 0:
                    mid_start = max(0, len(files) // 2 - third // 2)
                    selected.extend(files[mid_start:mid_start + third])  # Middle
                
                if third > 0:
                    selected.extend(files[-third:])  # End
                
                # If we still have slots, fill them from the beginning
                remaining = max_files_per_ext - len(selected)
                if remaining > 0:
                    start_idx = third
                    end_idx = min(start_idx + remaining, len(files))
                    selected.extend(files[start_idx:end_idx])
                
                files = selected
            
            # Store the relative paths
            result[ext] = [os.path.relpath(f, repo_path) for f in files]
            
            logger.info(f"Found {len(result[ext])} files with extension {ext}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error finding files with extension {ext}: {e}")
            result[ext] = []
            
    return result

def find_interesting_files(repo_path: str) -> List[str]:
    """
    Find interesting files in the repository to include in the extraction.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of file paths to extract
    """
    # Extensions to look for
    extensions = [".cpp", ".h", ".py", ".js", ".ts", ".md", ".json", ".sh"]
    
    # Find files by extensions
    files_by_ext = find_files_by_extensions(repo_path, extensions)
    
    # Combine all files into a single list
    all_files = []
    for ext, files in files_by_ext.items():
        all_files.extend(files)
    
    # Add specific important files that should always be included
    important_files = [
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "arangod/Aql/Ast.cpp",
        "arangod/Aql/Ast.h",
        "arangod/Aql/Query.cpp",
        "arangod/Aql/Query.h",
        "arangod/Aql/Functions.cpp",
        "arangod/Aql/Functions.h",
        "utils/gantt.py",
        "js/common/modules/org/arangodb/aql-query.js",
    ]
    
    # Add important files if they exist
    for file in important_files:
        full_path = os.path.join(repo_path, file)
        if os.path.exists(full_path):
            rel_path = os.path.relpath(full_path, repo_path)
            if rel_path not in all_files:
                all_files.append(rel_path)
    
    logger.info(f"Found {len(all_files)} interesting files to extract")
    return all_files

def extract_code_files(source_files: List[str], repo_path: str) -> List[Dict[str, Any]]:
    """
    Extract code hierarchy from multiple source files.
    
    Args:
        source_files: List of source files to extract (relative to repo_path)
        repo_path: Path to the repository
        
    Returns:
        List of extracted code blocks from all files
    """
    all_blocks = []
    
    for file_path in source_files:
        full_path = os.path.join(repo_path, file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue
            
        logger.info(f"Extracting code from {file_path}")
        
        try:
            # Extract code blocks from this file
            blocks = extract_code_hierarchy(full_path)
            
            if not blocks:
                logger.warning(f"No code blocks extracted from {file_path}")
                
                # If no blocks were extracted, create a single block for the file
                ext = os.path.splitext(file_path)[1]
                language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
                
                # Read file content
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception:
                    content = f"[Content of {file_path}]"
                
                blocks = [{
                    "type": "file",
                    "name": os.path.basename(file_path),
                    "content": content,
                    "file_path": full_path,
                    "language": language,
                    "uuid": str(uuid.uuid4()),
                    "start_line": 1,
                    "end_line": content.count('\n') + 1,
                    "child_uuids": []
                }]
                
            # Add file_path to each block if it doesn't have it
            for block in blocks:
                if "file_path" not in block:
                    block["file_path"] = full_path
                    
                # Ensure each block has a UUID
                if "uuid" not in block:
                    block["uuid"] = str(uuid.uuid4())
                    
                # Ensure each block has a language field
                if "language" not in block:
                    ext = os.path.splitext(file_path)[1]
                    block["language"] = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
                    
                # Ensure each block has a content field
                if "content" not in block:
                    block["content"] = f"[{block.get('type', 'unknown')}] {block.get('name', 'Unnamed block')}"
                    
            logger.info(f"Extracted {len(blocks)} blocks from {file_path}")
            all_blocks.extend(blocks)
            
        except Exception as e:
            logger.error(f"Error extracting code from {file_path}: {e}")
            
    return all_blocks

def extract_markdown_files(source_files: List[str], repo_path: str) -> List[Dict[str, Any]]:
    """
    Extract markdown sections from multiple markdown files.
    
    Args:
        source_files: List of markdown files to extract (relative to repo_path)
        repo_path: Path to the repository
        
    Returns:
        List of extracted markdown blocks from all files
    """
    all_blocks = []
    
    # Filter only markdown files
    markdown_files = [f for f in source_files if f.endswith((".md", ".MD", ".markdown"))]
    
    for file_path in markdown_files:
        full_path = os.path.join(repo_path, file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            continue
            
        logger.info(f"Extracting markdown from {file_path}")
        
        try:
            # Read the markdown file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a file block first
            file_uuid = str(uuid.uuid4())
            file_block = {
                "uuid": file_uuid,
                "id": Path(file_path).stem,
                "name": Path(file_path).name,
                "type": "file",
                "language": "markdown",
                "content": content,
                "file_path": full_path,
                "child_uuids": [],
                "metadata": {
                    "language": "markdown",
                    "source_file": full_path
                }
            }
            
            # Extract sections from the markdown content
            section_blocks = extract_markdown_sections(content, full_path, file_uuid)
            
            # Update file block child_uuids
            for block in section_blocks:
                if block.get("parent_uuid") == file_uuid:
                    file_block["child_uuids"].append(block["uuid"])
            
            # Add all blocks
            all_blocks.append(file_block)
            all_blocks.extend(section_blocks)
            
            logger.info(f"Extracted {len(section_blocks) + 1} blocks from {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting markdown from {file_path}: {e}")
            
    return all_blocks

def combine_and_validate_extraction(code_blocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Combine all blocks and validate the result.
    
    Args:
        code_blocks: List of code blocks
        
    Returns:
        Tuple of (combined blocks, validation results)
    """
    # Combine blocks
    combined_blocks = code_blocks
    logger.info(f"Combined {len(code_blocks)} total blocks")
    
    # Run validation checks
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Check for required fields in all blocks
    required_fields = ["uuid", "type", "name", "content"]
    
    for i, block in enumerate(combined_blocks):
        missing_fields = [field for field in required_fields if field not in block]
        if missing_fields:
            # Add missing fields with default values
            for field in missing_fields:
                if field == "uuid":
                    block["uuid"] = str(uuid.uuid4())
                elif field == "type":
                    block["type"] = "block"
                elif field == "name":
                    if "file_path" in block:
                        block["name"] = os.path.basename(block["file_path"])
                    else:
                        block["name"] = f"Block {i}"
                elif field == "content":
                    block["content"] = f"[{block.get('type', 'unknown')}] {block.get('name', 'Unnamed block')}"
            
            validation["warnings"].append(f"Block {i} (type: {block.get('type', 'unknown')}) was missing required fields: {missing_fields}")
    
    # Check for parent-child relationship consistency
    child_to_parent = {}
    for block in combined_blocks:
        if "parent_uuid" in block:
            child_to_parent[block["uuid"]] = block["parent_uuid"]
            
    for block in combined_blocks:
        if "child_uuids" in block:
            for child_uuid in block["child_uuids"]:
                # Check if child exists
                if not any(b["uuid"] == child_uuid for b in combined_blocks):
                    validation["errors"].append(f"Block {block['uuid']} references non-existent child {child_uuid}")
                    validation["valid"] = False
                # Check if child has correct parent reference
                elif child_uuid in child_to_parent and child_to_parent[child_uuid] != block["uuid"]:
                    validation["errors"].append(
                        f"Block {child_uuid} has parent {child_to_parent[child_uuid]} but is listed as child of {block['uuid']}"
                    )
                    validation["valid"] = False
    
    # Collect statistics
    block_types = {}
    language_types = {}
    
    for block in combined_blocks:
        block_type = block.get("type", "unknown")
        if block_type not in block_types:
            block_types[block_type] = 0
        block_types[block_type] += 1
        
        language = block.get("language", "unknown")
        if language not in language_types:
            language_types[language] = 0
        language_types[language] += 1
        
    validation["stats"]["block_types"] = block_types
    validation["stats"]["language_types"] = language_types
    validation["stats"]["total_blocks"] = len(combined_blocks)
    
    return combined_blocks, validation

def ensure_required_fields(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all required fields are present in a block.
    
    Args:
        block: The block to process
        
    Returns:
        Block with all required fields
    """
    # Create a copy to avoid modifying the original
    enhanced = block.copy()
    
    # Ensure UUID field
    if "uuid" not in enhanced:
        enhanced["uuid"] = str(uuid.uuid4())
    if "id" not in enhanced and "uuid" in enhanced:
        enhanced["id"] = enhanced["uuid"]
        
    # Ensure content field
    if "content" not in enhanced:
        enhanced["content"] = f"[{enhanced.get('type', 'unknown')}] {enhanced.get('name', 'Unnamed block')}"
    
    # Ensure type field
    if "type" not in enhanced:
        if "file_path" in enhanced and enhanced["file_path"].endswith((".py", ".js", ".ts")):
            enhanced["type"] = "code"
        elif "file_path" in enhanced and enhanced["file_path"].endswith((".md", ".txt")):
            enhanced["type"] = "documentation"
        else:
            enhanced["type"] = "unknown"
    
    # Ensure language field
    if "language" not in enhanced:
        # Try to determine language from metadata or file extension
        if "metadata" in enhanced and "language" in enhanced["metadata"]:
            enhanced["language"] = enhanced["metadata"]["language"]
        elif enhanced.get("type") in ["section", "text", "doc_section"]:
            enhanced["language"] = "markdown"
        elif enhanced.get("type") == "documentation":
            enhanced["language"] = "markdown"
        elif enhanced.get("type") == "file":
            file_path = enhanced.get("file_path", "")
            if file_path.endswith(".md"):
                enhanced["language"] = "markdown"
            elif file_path.endswith((".py", ".pyx", ".pyw")):
                enhanced["language"] = "python"
            elif file_path.endswith((".js", ".jsx")):
                enhanced["language"] = "javascript"
            elif file_path.endswith((".ts", ".tsx")):
                enhanced["language"] = "typescript"
            else:
                enhanced["language"] = "text"
        else:
            enhanced["language"] = "text"
            
    # Ensure extraction_focus field
    if "extraction_focus" not in enhanced:
        if enhanced.get("type") in ["section", "text", "doc_section", "documentation"]:
            enhanced["extraction_focus"] = ["documentation"]
        else:
            enhanced["extraction_focus"] = ["code"]
            
    # Ensure summary_instructions field
    if "summary_instructions" not in enhanced:
        if enhanced.get("type") in ["section", "text", "doc_section", "documentation"]:
            enhanced["summary_instructions"] = "Extract key points from this documentation section"
        else:
            enhanced["summary_instructions"] = "Explain the purpose of this code"
            
    # Ensure breadcrumb field
    if "breadcrumb" not in enhanced:
        if "metadata" in enhanced and "breadcrumb" in enhanced["metadata"]:
            enhanced["breadcrumb"] = enhanced["metadata"]["breadcrumb"]
        elif "metadata" in enhanced and "section_hierarchy" in enhanced["metadata"]:
            enhanced["breadcrumb"] = enhanced["metadata"]["section_hierarchy"]
        else:
            # Create breadcrumb based on file path and name
            file_path = enhanced.get("file_path", "")
            name = enhanced.get("name", "")
            if file_path:
                file_name = file_path.split("/")[-1] if "/" in file_path else file_path
                enhanced["breadcrumb"] = [file_name]
                if name and name != file_name:
                    enhanced["breadcrumb"].append(name)
            else:
                enhanced["breadcrumb"] = [name or "Unnamed Block"]
    
    # Ensure parent-child relationships
    if "parent_uuid" not in enhanced:
        enhanced["parent_uuid"] = None
    if "child_uuids" not in enhanced:
        enhanced["child_uuids"] = []
        
    return enhanced

def create_section_relationships(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create section relationships for QA format.
    
    Args:
        blocks: List of blocks
        
    Returns:
        Dictionary of section relationships
    """
    relationships = {
        "parent_child": {},
        "imports": {},
        "inheritance": {}
    }
    
    # Build parent-child relationships
    for block in blocks:
        block_uuid = block.get("uuid")
        parent_uuid = block.get("parent_uuid")
        child_uuids = block.get("child_uuids", [])
        
        if block_uuid:
            relationships["parent_child"][block_uuid] = {
                "parent": parent_uuid,
                "children": child_uuids
            }
            
    # Add imports if available
    for block in blocks:
        block_uuid = block.get("uuid")
        
        if "imports" in block and block_uuid:
            imports = block.get("imports", [])
            if imports:
                relationships["imports"][block_uuid] = imports
                
        # Check for imports in metadata
        if "metadata" in block and "imports" in block["metadata"] and block_uuid:
            imports = block["metadata"].get("imports", [])
            if imports:
                relationships["imports"][block_uuid] = imports
                
    # Add inheritance if available
    for block in blocks:
        block_uuid = block.get("uuid")
        
        if "inheritance" in block and block_uuid:
            inheritance = block.get("inheritance", [])
            if inheritance:
                relationships["inheritance"][block_uuid] = inheritance
                
        # Check for inheritance in metadata
        if "metadata" in block and "inheritance" in block["metadata"] and block_uuid:
            inheritance = block["metadata"].get("inheritance", [])
            if inheritance:
                relationships["inheritance"][block_uuid] = inheritance
                
    return relationships

def create_qa_compatible_output(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a QA-compatible output structure.
    
    Args:
        blocks: List of blocks to include
        
    Returns:
        QA-compatible output dictionary
    """
    # Ensure all blocks have required fields
    enhanced_blocks = [ensure_required_fields(block) for block in blocks]
    
    # Create section relationships
    section_relationships = create_section_relationships(enhanced_blocks)
    
    # Build statistics
    block_types = {}
    languages = {}
    file_count = 0
    
    for block in enhanced_blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
        
        if block_type == "file":
            file_count += 1
        
        language = block.get("language", "unknown")
        languages[language] = languages.get(language, 0) + 1
    
    # Create metadata
    metadata = {
        "model_used": "dualipa-extraction",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "statistics": {
            "total_blocks": len(enhanced_blocks),
            "total_files": file_count,
            "block_types": block_types,
            "languages": languages
        }
    }
    
    # Create final output
    output = {
        "sections": enhanced_blocks,
        "section_relationships": section_relationships,
        "extraction_metadata": metadata
    }
    
    return output

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Full repository extraction")
    parser.add_argument("--repo-path", type=str, default=ARANGODB_REPO_PATH,
                        help="Path to the ArangoDB repository")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save extraction results")
    parser.add_argument("--output-file", type=str, 
                        default="/home/grahama/workspace/experiments/agent_tools/arangodb_qa_compatible_comprehensive.json",
                        help="Path to output QA-compatible JSON file")
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"full_extraction_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Find interesting files to extract
        all_files = find_interesting_files(args.repo_path)
        
        # Divide files into code and markdown
        markdown_files = [f for f in all_files if f.endswith((".md", ".MD", ".markdown"))]
        code_files = [f for f in all_files if not f.endswith((".md", ".MD", ".markdown"))]
        
        logger.info(f"Found {len(code_files)} code files and {len(markdown_files)} markdown files")
        
        # Step 1: Extract code files
        logger.info("Extracting code files")
        code_blocks = extract_code_files(code_files, args.repo_path)
        
        if not code_blocks:
            logger.warning("No code blocks extracted")
        else:
            # Save the code blocks for inspection
            code_blocks_path = output_dir / "code_blocks.json"
            with open(code_blocks_path, "w", encoding="utf-8") as f:
                json.dump(code_blocks, f, indent=2)
                
            logger.info(f"Saved {len(code_blocks)} code blocks to {code_blocks_path}")
            
        # Step 2: Extract markdown files
        logger.info("Extracting markdown files")
        markdown_blocks = extract_markdown_files(markdown_files, args.repo_path)
        
        if not markdown_blocks:
            logger.warning("No markdown blocks extracted")
        else:
            # Save the markdown blocks for inspection
            markdown_blocks_path = output_dir / "markdown_blocks.json"
            with open(markdown_blocks_path, "w", encoding="utf-8") as f:
                json.dump(markdown_blocks, f, indent=2)
                
            logger.info(f"Saved {len(markdown_blocks)} markdown blocks to {markdown_blocks_path}")
            
        # Step 3: Combine all blocks
        all_blocks = code_blocks + markdown_blocks
        
        # Step 4: Combine and validate
        combined_blocks, validation = combine_and_validate_extraction(all_blocks)
        
        # Save the combined blocks for inspection
        combined_blocks_path = output_dir / "combined_blocks.json"
        with open(combined_blocks_path, "w", encoding="utf-8") as f:
            json.dump(combined_blocks, f, indent=2)
            
        logger.info(f"Saved {len(combined_blocks)} combined blocks to {combined_blocks_path}")
        
        # Save the validation results
        validation_path = output_dir / "validation_results.json"
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)
            
        logger.info(f"Saved validation results to {validation_path}")
        
        # Step 5: Create QA-compatible JSON
        output = create_qa_compatible_output(combined_blocks)
        
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Successfully created QA-compatible JSON at {output_file}")
        logger.info(f"Total sections: {len(output['sections'])}")
        logger.info(f"Block types: {output['extraction_metadata']['statistics']['block_types']}")
        logger.info(f"Languages: {output['extraction_metadata']['statistics']['languages']}")
        
        print(f"\n✅ Full repository extraction completed successfully!")
        print(f"- {len(code_blocks)} code blocks extracted")
        print(f"- {len(markdown_blocks)} markdown blocks extracted")
        print(f"- {len(output['sections'])} total sections in final output")
        print(f"- QA-compatible JSON saved to {output_file}")
        print(f"- Extraction details saved to {output_dir}")
        
        # Print language breakdown
        print("\nLanguage breakdown:")
        for lang, count in output['extraction_metadata']['statistics']['languages'].items():
            print(f"- {lang}: {count}")
        
        return 0
            
    except Exception as e:
        logger.error(f"Error running full repository extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())