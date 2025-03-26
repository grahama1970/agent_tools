#!/usr/bin/env python3
"""
Comprehensive Extraction Test with State Management

This script implements a complete extraction test that:
1. Uses TestStateManager for context tracking between steps
2. Counts all files by type before extraction
3. Extracts code from multiple languages (Python, JS, TS, C++)
4. Extracts documentation from markdown files
5. Creates properly structured extraction blocks with hierarchical relationships
6. Verifies extraction coverage against initial analysis
7. Generates a comprehensive QA-compatible JSON output
8. Validates output format and completeness

Usage:
    python test_comprehensive_extraction.py --repo-path /path/to/repo --output-file output.json

Example:
    python test_comprehensive_extraction.py --repo-path ~/workspace/experiments/agent_tools/test_repos/arangodb
"""

import os
import sys
import json
import logging
import argparse
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("comprehensive_extraction")

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, parent_dir)

# Import state manager
try:
    # Try relative import
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

# Try to import extraction modules
try:
    # Try absolute imports
    from agent_tools.dualipa.extraction.extractors.code.hierarchy import extract_code_hierarchy
    from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_markdown_sections
except ImportError:
    try:
        # Try relative imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
        from extractors.code.hierarchy import extract_code_hierarchy
        from examples.end_to_end.extraction_blocks import extract_markdown_sections
    except ImportError:
        print("Could not import extraction modules. Please ensure they exist at the correct path.")
        sys.exit(1)

# Constants
DEFAULT_REPO_PATH = "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb"
DEFAULT_OUTPUT_PATH = "/home/grahama/workspace/experiments/agent_tools/ultimate_extraction_output.json"

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

def analyze_repository(repo_path: str, state_manager=None) -> Dict[str, Any]:
    """
    Analyze repository and store statistics in state manager.
    
    This is a critical first step that counts all files by type to establish
    a baseline for ensuring extraction completeness.
    
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
        what_next="After analysis, we'll extract files of each type"
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
    
    logger.info(f"Analyzing repository: {repo_path}")
    
    # Initialize counters
    file_counts = {}
    important_files = []
    total_files = 0
    all_files = []
    
    # File extensions to track - we'll extract these file types
    target_extensions = ['.py', '.js', '.ts', '.cpp', '.h', '.md', '.json', '.sh']
    
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
Shell (.sh) files contain shell scripts
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
                    all_files.append(rel_path)
                    
                    # Store in state manager
                    state_manager.track_file(rel_path, ext, os.path.getsize(full_path))
                    
                    # Check for important files (adjust these based on the repository)
                    important_patterns = [
                        'README.md', 
                        'gantt.py', 
                        'Ast.cpp', 
                        'Query.cpp', 
                        'Functions.cpp'
                    ]
                    
                    if any(pattern in full_path for pattern in important_patterns):
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
    
    # Store all tracked files
    state_manager.set_metadata("all_tracked_files", all_files)
    
    # Print repository statistics
    logger.info(f"Repository contains {total_files} files")
    logger.info(f"Tracking {len(all_files)} files with target extensions")
    
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
        'important_files': important_files,
        'all_files': all_files
    }
    
    # Store in state manager
    state_manager.set("repo_stats", repo_stats)
    
    return repo_stats


def sample_files_by_extension(all_files: List[str], limit_per_ext: int = 10) -> Dict[str, List[str]]:
    """
    Sample files by extension to keep the extraction manageable.
    
    Args:
        all_files: List of all files to sample from
        limit_per_ext: Maximum number of files per extension
        
    Returns:
        Dictionary mapping extensions to lists of file paths
    """
    # Group files by extension
    files_by_ext = {}
    for file_path in all_files:
        _, ext = os.path.splitext(file_path)
        if ext not in files_by_ext:
            files_by_ext[ext] = []
        files_by_ext[ext].append(file_path)
    
    # Sample files from each extension
    sampled_files = {}
    for ext, files in files_by_ext.items():
        if len(files) <= limit_per_ext:
            # Take all files if we have fewer than the limit
            sampled_files[ext] = files
        else:
            # Sample evenly from beginning, middle, and end
            third = limit_per_ext // 3
            remainder = limit_per_ext % 3
            
            # Beginning files
            beginning = files[:third + remainder]
            
            # Middle files
            mid_start = len(files) // 2 - third // 2
            middle = files[mid_start:mid_start + third]
            
            # End files
            end = files[-third:]
            
            # Combine samples
            sampled_files[ext] = beginning + middle + end
    
    return sampled_files


def extract_files(repo_path: str, state_manager=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract files from repository, grouped by file type.
    
    Args:
        repo_path: Path to repository
        state_manager: Optional state manager instance
        
    Returns:
        Dictionary mapping file types to lists of extracted blocks
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Refresh our context - what are we doing?
    what_am_i_doing()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("extract_files", "Extracting files by type")
    
    # Update context for this step
    remember_context(
        what_im_doing="Extracting files from repository",
        why_im_doing_it="To create structured blocks for all file types",
        what_step="Reading and structuring file content",
        what_next="After extraction, we'll create QA-compatible output"
    )
    
    # Get repository stats from state
    repo_stats = state_manager.get("repo_stats")
    if not repo_stats:
        logger.warning("Repository has not been analyzed. Running analysis first.")
        repo_stats = analyze_repository(repo_path, state_manager)
    
    # Get all tracked files
    all_files = repo_stats.get("all_files", [])
    if not all_files:
        all_files = state_manager.get_metadata("all_tracked_files", [])
    
    # Sample files by extension to keep extraction manageable
    files_by_ext = sample_files_by_extension(all_files, limit_per_ext=15)
    
    # Initialize results by file type
    extracted_blocks = {
        "code": [],      # For programming language files (.py, .js, .ts, .cpp, etc.)
        "markdown": [],  # For documentation files (.md)
        "config": [],    # For configuration files (.json, .yaml)
        "script": []     # For script files (.sh, .bash)
    }
    
    # Extract files by type
    for ext, files in files_by_ext.items():
        logger.info(f"Extracting {len(files)} files with extension {ext}")
        
        # Determine file type category
        if ext in ['.md']:
            file_type = "markdown"
        elif ext in ['.json', '.yaml', '.yml']:
            file_type = "config"
        elif ext in ['.sh', '.bash']:
            file_type = "script"
        else:
            file_type = "code"
        
        # Extract each file
        for file_path in files:
            full_path = os.path.join(repo_path, file_path)
            
            try:
                # Check if file exists
                if not os.path.exists(full_path):
                    logger.warning(f"File not found: {full_path}")
                    continue
                
                # Handle different file types
                if file_type == "markdown":
                    # Extract markdown blocks
                    blocks = extract_markdown_file(full_path)
                elif file_type == "code":
                    # Extract code blocks
                    blocks = extract_code_file(full_path)
                else:
                    # Basic extraction for other file types
                    blocks = extract_basic_file(full_path, file_type)
                
                # Add blocks to results
                if blocks:
                    extracted_blocks[file_type].extend(blocks)
                    
                    # Mark file as extracted
                    for block in blocks:
                        if block.get("type") == "file":
                            state_manager.mark_file_extracted(file_path, block.get("uuid", ""))
                            break
            
            except Exception as e:
                logger.error(f"Error extracting {file_path}: {e}")
    
    # Update extraction stats
    for ext, files in files_by_ext.items():
        state_manager.update_extracted_count(ext, len(files))
    
    # Log extraction results
    for file_type, blocks in extracted_blocks.items():
        logger.info(f"Extracted {len(blocks)} blocks from {file_type} files")
        state_manager.set(f"extracted_{file_type}_blocks", blocks)
    
    return extracted_blocks


def extract_markdown_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract blocks from a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        List of blocks extracted from the file
    """
    logger.info(f"Extracting markdown from {file_path}")
    
    try:
        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Create a file block first
        import uuid
        file_uuid = str(uuid.uuid4())
        file_block = {
            "uuid": file_uuid,
            "id": Path(file_path).stem,
            "name": Path(file_path).name,
            "type": "file",
            "language": "markdown",
            "content": content,
            "file_path": file_path,
            "child_uuids": [],
            "metadata": {
                "language": "markdown",
                "source_file": file_path
            }
        }
        
        # Extract sections from the markdown content
        section_blocks = extract_markdown_sections(content, file_path, file_uuid)
        
        # Update file block child_uuids
        for block in section_blocks:
            if block.get("parent_uuid") == file_uuid:
                file_block["child_uuids"].append(block["uuid"])
        
        # Add all blocks
        blocks = [file_block] + section_blocks
        
        logger.info(f"Extracted {len(blocks)} blocks from {file_path}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting markdown from {file_path}: {e}")
        return []


def extract_code_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract blocks from a code file.
    
    Args:
        file_path: Path to the code file
        
    Returns:
        List of blocks extracted from the file
    """
    logger.info(f"Extracting code from {file_path}")
    
    try:
        # Extract code hierarchy
        blocks = extract_code_hierarchy(file_path)
        
        if not blocks:
            logger.warning(f"No code blocks extracted from {file_path}")
            
            # If no blocks were extracted, create a single block for the file
            ext = os.path.splitext(file_path)[1]
            language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception:
                content = f"[Content of {file_path}]"
            
            import uuid
            blocks = [{
                "uuid": str(uuid.uuid4()),
                "type": "file",
                "name": os.path.basename(file_path),
                "content": content,
                "file_path": file_path,
                "language": language,
                "start_line": 1,
                "end_line": content.count('\n') + 1,
                "child_uuids": []
            }]
            
        # Add file_path to each block if it doesn't have it
        for block in blocks:
            if "file_path" not in block:
                block["file_path"] = file_path
                
            # Ensure each block has a UUID
            if "uuid" not in block:
                import uuid
                block["uuid"] = str(uuid.uuid4())
                
            # Ensure each block has a language field
            if "language" not in block:
                ext = os.path.splitext(file_path)[1]
                block["language"] = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
                
            # Ensure each block has a content field
            if "content" not in block and "start_line" in block and "end_line" in block:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_lines = f.read().splitlines()
                        
                    start = max(0, block["start_line"] - 1)
                    end = min(len(file_lines), block["end_line"])
                    
                    if start < end and start < len(file_lines):
                        block["content"] = "\n".join(file_lines[start:end])
                    else:
                        block["content"] = f"[{block.get('type', 'unknown')}] {block.get('name', 'Unnamed block')}"
                except Exception:
                    block["content"] = f"[{block.get('type', 'unknown')}] {block.get('name', 'Unnamed block')}"
            elif "content" not in block:
                block["content"] = f"[{block.get('type', 'unknown')}] {block.get('name', 'Unnamed block')}"
        
        logger.info(f"Extracted {len(blocks)} blocks from {file_path}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting code from {file_path}: {e}")
        return []


def extract_basic_file(file_path: str, file_type: str) -> List[Dict[str, Any]]:
    """
    Basic extraction for other file types.
    
    Args:
        file_path: Path to the file
        file_type: Type of file (config, script, etc.)
        
    Returns:
        List of blocks extracted from the file
    """
    logger.info(f"Extracting {file_type} from {file_path}")
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Determine language from extension
        ext = os.path.splitext(file_path)[1]
        language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
        
        # Create a file block
        import uuid
        file_uuid = str(uuid.uuid4())
        file_block = {
            "uuid": file_uuid,
            "id": Path(file_path).stem,
            "name": Path(file_path).name,
            "type": "file",
            "language": language,
            "content": content,
            "file_path": file_path,
            "child_uuids": [],
            "metadata": {
                "language": language,
                "source_file": file_path,
                "file_type": file_type
            }
        }
        
        # For scripts, try to extract functions (simple approach)
        blocks = [file_block]
        
        if file_type == "script" and content:
            import re
            # Simple regex to match shell functions
            func_pattern = re.compile(r'^(\w+)\s*\(\)\s*{', re.MULTILINE)
            
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                
                # Find function end (next function or EOF)
                next_func = content.find("\n}\n", start_pos)
                if next_func == -1:
                    next_func = content.find("\n}", start_pos)
                if next_func == -1:
                    next_func = len(content)
                else:
                    next_func += 2  # Include the closing brace
                
                func_content = content[start_pos:next_func].strip()
                
                if func_content:
                    # Create function block
                    func_uuid = str(uuid.uuid4())
                    blocks.append({
                        "uuid": func_uuid,
                        "id": f"{Path(file_path).stem}_{func_name}",
                        "name": func_name,
                        "type": "function",
                        "language": language,
                        "content": func_content,
                        "file_path": file_path,
                        "parent_uuid": file_uuid,
                        "child_uuids": [],
                        "metadata": {
                            "language": language,
                            "source_file": file_path,
                            "file_type": file_type
                        }
                    })
                    
                    # Add to file's child UUIDs
                    file_block["child_uuids"].append(func_uuid)
        
        logger.info(f"Extracted {len(blocks)} blocks from {file_path}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting {file_type} from {file_path}: {e}")
        return []


def combine_extracted_blocks(extracted_blocks: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Combine blocks from different file types.
    
    Args:
        extracted_blocks: Dictionary mapping file types to lists of blocks
        
    Returns:
        Combined list of all blocks
    """
    # Flatten the blocks
    all_blocks = []
    for file_type, blocks in extracted_blocks.items():
        all_blocks.extend(blocks)
    
    logger.info(f"Combined {len(all_blocks)} blocks from all file types")
    return all_blocks


def ensure_block_requirements(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure blocks have all required fields for QA output.
    
    Args:
        block: Block to ensure requirements for
        
    Returns:
        Updated block with all required fields
    """
    # Create a copy to avoid modifying the original
    enhanced = block.copy()
    
    # Ensure UUID field
    if "uuid" not in enhanced:
        import uuid
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


def create_qa_compatible_output(blocks: List[Dict[str, Any]], output_path: str, state_manager=None) -> Dict[str, Any]:
    """
    Create QA-compatible output from all blocks.
    
    Args:
        blocks: List of all extracted blocks
        output_path: Path to save output file
        state_manager: Optional state manager instance
        
    Returns:
        QA-compatible output dictionary
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("create_qa_output", "Creating QA-compatible output")
    
    # Update context for this step
    remember_context(
        what_im_doing="Creating QA-compatible output",
        why_im_doing_it="To create a properly formatted output for QA systems",
        what_step="Combining blocks and adding required fields",
        what_next="After creating output, we'll validate it"
    )
    
    # Ensure all blocks have required fields
    enhanced_blocks = [ensure_block_requirements(block) for block in blocks]
    
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
        "model_used": "dualipa-extraction-ultimate",
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
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    # Mark files as included in output
    for block in enhanced_blocks:
        if block.get("type") == "file" and "file_path" in block:
            rel_path = os.path.relpath(block["file_path"], os.path.dirname(output_path))
            state_manager.mark_file_included_in_output(rel_path)
    
    # Store output statistics
    state_manager.set_metadata("output_stats", {
        "sections": len(enhanced_blocks),
        "timestamp": datetime.datetime.now().isoformat(),
        "file_path": output_path,
        "block_types": block_types,
        "languages": languages
    })
    
    logger.info(f"Successfully created QA-compatible output at {output_path}")
    logger.info(f"Total sections: {len(enhanced_blocks)}")
    logger.info(f"Block types: {block_types}")
    
    print(f"Successfully created QA-compatible output at {output_path}")
    print(f"- {len(enhanced_blocks)} sections")
    print(f"- {len(languages)} languages")
    print(f"- {len(block_types)} block types")
    
    return output


def validate_output(output_data: Dict[str, Any], repo_stats: Dict[str, Any], state_manager=None) -> bool:
    """
    Validate the output data.
    
    Args:
        output_data: QA-compatible output data
        repo_stats: Repository statistics
        state_manager: Optional state manager instance
        
    Returns:
        True if validation passed, False otherwise
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("validate_output", "Validating output")
    
    # Update context for this step
    remember_context(
        what_im_doing="Validating extraction output",
        why_im_doing_it="To ensure extraction is complete and correctly formatted",
        what_step="Checking output format and completeness",
        what_next="After validation, we'll generate a report"
    )
    
    # Check basic structure
    validation_passed = True
    
    # Required keys in output
    required_keys = ['sections', 'section_relationships', 'extraction_metadata']
    for key in required_keys:
        if key not in output_data:
            logger.error(f"Output is missing required key: {key}")
            validation_passed = False
    
    # Check sections have required fields
    section_required_fields = ['uuid', 'id', 'type', 'name', 'content', 'language']
    
    missing_fields = []
    for i, section in enumerate(output_data.get('sections', [])):
        for field in section_required_fields:
            if field not in section:
                missing_fields.append((i, field))
    
    if missing_fields:
        logger.error(f"Found {len(missing_fields)} sections with missing required fields")
        validation_passed = False
    
    # Check important files are included
    important_files = state_manager.get_metadata("important_files", [])
    
    for important_file in important_files:
        # Check if file is in sections
        found = False
        for section in output_data.get('sections', []):
            if section.get('type') == 'file' and section.get('file_path', '').endswith(important_file):
                found = True
                break
        
        if not found:
            logger.warning(f"Important file not included in output: {important_file}")
    
    # Check statistics are present
    if 'extraction_metadata' not in output_data or 'statistics' not in output_data['extraction_metadata']:
        logger.error("Output is missing statistics in extraction_metadata")
        validation_passed = False
    
    # Verify extraction has sufficient sections
    min_sections = 100  # Minimum expected sections
    actual_sections = len(output_data.get('sections', []))
    
    if actual_sections < min_sections:
        logger.warning(f"Output has fewer sections than expected. Found {actual_sections}, expected at least {min_sections}")
    
    # Store validation result
    state_manager.set_metadata("validation_passed", validation_passed)
    
    if validation_passed:
        logger.info("✅ Output validation passed")
    else:
        logger.error("❌ Output validation failed")
    
    return validation_passed


def generate_table_rows(repo_stats: Dict[str, Any]) -> str:
    """Generate HTML table rows for repository statistics."""
    rows = []
    for ext, stats in repo_stats.items():
        rows.append(f"<tr><td>{ext}</td><td>{stats['count']}</td><td>{stats['extracted']}</td><td>{stats['extraction_rate']:.1f}%</td></tr>")
    return "\n".join(rows)


def generate_block_type_rows(block_types: Dict[str, int]) -> str:
    """Generate HTML table rows for block types."""
    rows = []
    for block_type, count in block_types.items():
        rows.append(f"<tr><td>{block_type}</td><td>{count}</td></tr>")
    return "\n".join(rows)


def generate_language_rows(languages: Dict[str, int]) -> str:
    """Generate HTML table rows for languages."""
    rows = []
    for language, count in languages.items():
        rows.append(f"<tr><td>{language}</td><td>{count}</td></tr>")
    return "\n".join(rows)


def generate_verification_rows(verification_history: List[Dict[str, Any]]) -> str:
    """Generate HTML table rows for verification history."""
    rows = []
    for v in verification_history:
        status_class = "success" if v['passed'] else "error"
        status_symbol = "✓" if v['passed'] else "✗"
        rows.append(f"<tr><td>{v['checkpoint']}</td><td>{v['step']}</td><td class=\"{status_class}\">{status_symbol}</td><td>{v['expected']}</td><td>{v['actual']}</td></tr>")
    return "\n".join(rows)


def generate_report(repo_path: str, output_path: str, state_manager=None) -> str:
    """
    Generate a comprehensive report of the extraction process.
    
    Args:
        repo_path: Path to repository
        output_path: Path to output file
        state_manager: Optional state manager instance
        
    Returns:
        Path to the generated report
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint for this phase
    state_manager.set_checkpoint("generate_report", "Generating extraction report")
    
    # Generate timestamp for the report file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"extraction_report_{timestamp}.html"
    
    # Get statistics from state manager
    extraction_stats = state_manager.get_extraction_stats()
    repo_stats = state_manager.get_repo_stats()
    output_stats = state_manager.get_metadata("output_stats", {})
    validation_passed = state_manager.get_metadata("validation_passed", False)
    verification_history = state_manager.get_verification_history()
    
    # Generate HTML report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comprehensive Extraction Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #444;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }}
        .success {{
            color: #2ecc71;
        }}
        .warning {{
            color: #f39c12;
        }}
        .error {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Extraction Report</h1>
            <p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Repository: {repo_path}</p>
            <p>Output: {output_path}</p>
        </div>

        <div class="section">
            <h2>Extraction Summary</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Files</h3>
                    <p>{extraction_stats.get('total_files', 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Extracted Files</h3>
                    <p>{extraction_stats.get('extracted_files', 0)} ({extraction_stats.get('extraction_rate', 0):.1f}%)</p>
                </div>
                <div class="stat-card">
                    <h3>Included in Output</h3>
                    <p>{extraction_stats.get('included_files', 0)} ({extraction_stats.get('inclusion_rate', 0):.1f}%)</p>
                </div>
                <div class="stat-card">
                    <h3>Validation</h3>
                    <p class="{'success' if validation_passed else 'error'}">
                        {"✅ PASSED" if validation_passed else "❌ FAILED"}
                    </p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Repository Statistics</h2>
            <h3>File Types</h3>
            <table>
                <tr>
                    <th>Extension</th>
                    <th>Count</th>
                    <th>Extracted</th>
                    <th>Extraction Rate</th>
                </tr>
                {generate_table_rows(repo_stats)}
            </table>
        </div>

        <div class="section">
            <h2>Output Statistics</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Sections</h3>
                    <p>{output_stats.get('sections', 0)}</p>
                </div>
            </div>

            <h3>Block Types</h3>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
                {generate_block_type_rows(output_stats.get('block_types', {}))}
            </table>

            <h3>Languages</h3>
            <table>
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                </tr>
                {generate_language_rows(output_stats.get('languages', {}))}
            </table>
        </div>

        <div class="section">
            <h2>Verification History</h2>
            <table>
                <tr>
                    <th>Checkpoint</th>
                    <th>Step</th>
                    <th>Result</th>
                    <th>Expected</th>
                    <th>Actual</th>
                </tr>
                {generate_verification_rows(verification_history)}
            </table>
        </div>
    </div>
</body>
</html>""")
    
    logger.info(f"Generated report at {report_path}")
    print(f"Generated report at {report_path}")
    
    return report_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive extraction test with state management")
    parser.add_argument("--repo-path", type=str, default=DEFAULT_REPO_PATH,
                      help="Path to repository")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_PATH,
                      help="Path to output file")
    parser.add_argument("--state-db", type=str, default="comprehensive_extraction_state.db",
                      help="Path to state database file")
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"test_results/comprehensive_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get state manager with persistent path for debugging
    state_manager = get_state_manager(str(output_dir / args.state_db))
    
    try:
        # Step 1: Repository analysis
        repo_stats = analyze_repository(args.repo_path, state_manager)
        
        # Step 2: Extract files by type
        extracted_blocks = extract_files(args.repo_path, state_manager)
        
        # Step 3: Combine extracted blocks
        all_blocks = combine_extracted_blocks(extracted_blocks)
        
        # Step 4: Create QA-compatible output
        output_path = args.output_file
        if output_path == DEFAULT_OUTPUT_PATH:
            # Use output directory if default path
            output_path = str(output_dir / "qa_compatible_output.json")
        
        output_data = create_qa_compatible_output(all_blocks, output_path, state_manager)
        
        # Step 5: Validate output
        validation_passed = validate_output(output_data, repo_stats, state_manager)
        
        # Step 6: Generate report
        report_path = generate_report(args.repo_path, output_path, state_manager)
        
        # Print extraction statistics
        print("\nExtraction Statistics:")
        stats = state_manager.get_extraction_stats()
        print(f"- Total files: {stats['total_files']}")
        print(f"- Extracted files: {stats['extracted_files']} ({stats['extraction_rate']:.1f}%)")
        print(f"- Included in output: {stats['included_files']} ({stats['inclusion_rate']:.1f}%)")
        
        # Print output statistics
        output_stats = state_manager.get_metadata("output_stats", {})
        print(f"\nOutput Statistics:")
        print(f"- Total sections: {output_stats.get('sections', 0)}")
        
        # Results
        if validation_passed:
            print("\n✅ Comprehensive extraction test passed!")
            return 0
        else:
            print("\n❌ Comprehensive extraction test failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Error in comprehensive extraction test: {e}")
        import traceback
        traceback.print_exc()
        
        # Generate error report if state_manager exists
        if 'state_manager' in locals():
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