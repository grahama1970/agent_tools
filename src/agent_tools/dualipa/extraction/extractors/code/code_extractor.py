"""
Code extraction module for DuaLipa.

This module handles code block extraction from source files,
supporting multiple languages and extraction strategies.

Key Features:
1. Language-specific extraction
2. Block validation and verification
3. Statistics tracking
4. Error handling

Dependencies:
- ast: For Python parsing (https://docs.python.org/3/library/ast.html)
- tree-sitter: For JS/TS parsing (https://tree-sitter.github.io/tree-sitter/)
- loguru: For logging (https://github.com/Delgan/loguru)

Documentation Links:
- AST Module: https://docs.python.org/3/library/ast.html
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Loguru: https://loguru.readthedocs.io/

Input/Output Specifications:

extract_code_blocks(file_path: str, output_dir: Path) -> List[Dict[str, Any]]:
    Input:
        - file_path: Path to source file
        - output_dir: Output directory for extracted blocks
    Output:
        - List of dictionaries containing:
            - type: str (function, class, method)
            - name: str
            - content: str
            - line_start: int
            - line_end: int
            - metadata: Dict[str, Any]
                - language: str
                - file: str
                - decorators: List[str] (Python only)
                - returns: Optional[str]
                - args: List[str] (Python only)

extract_python_blocks(content: str, file_path: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    Input:
        - content: Python source code
        - file_path: Path to source file
        - stats: Statistics dictionary
    Output:
        - List of dictionaries (same format as extract_code_blocks)

extract_js_ts_blocks(content: str, file_path: str, language: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    Input:
        - content: JS/TS source code
        - file_path: Path to source file
        - language: 'javascript' or 'typescript'
        - stats: Statistics dictionary
    Output:
        - List of dictionaries (same format as extract_code_blocks)

extract_generic_blocks(content: str, file_path: str, language: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    Input:
        - content: Source code
        - file_path: Path to source file
        - language: Programming language
        - stats: Statistics dictionary
    Output:
        - List of dictionaries (same format as extract_code_blocks)

Related Files:
- hierarchy.py: Code hierarchy analysis
- language_utils.py: Language detection
- stats_utils.py: Statistics tracking
"""

import ast
import re
import uuid
import json
import textwrap
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger

from agent_tools.dualipa.extraction.extractors.code.hierarchy import analyze_code_hierarchy
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats, initialize_stats_dict
from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_utils import get_parser
from agent_tools.dualipa.extraction.extractors.utils.validation_utils import validate_block
from agent_tools.dualipa.extraction.extractors.utils.verification_utils import verify_block
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_helpers import (
    get_node_text, get_block_content, is_async_node, is_static_node, 
    get_ts_return_type, get_extends_class, get_implements_interfaces,
    get_extends_interfaces, get_interface_properties
)

# For backward compatibility
_extract_python_blocks = extract_python_blocks
_extract_js_ts_blocks = extract_js_ts_blocks
_extract_generic_blocks = extract_generic_blocks

def _extract_with_tree_sitter(content: str, file_path: str, language: str, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract code blocks using tree-sitter.
    
    This is a compatibility function to support older tests that expect this function.
    It delegates to extract_js_ts_blocks which has the actual implementation.
    
    Args:
        content: Source code content
        file_path: Path to the source file
        language: Programming language
        stats: Statistics dictionary
        
    Returns:
        List of extracted code blocks
    """
    # Provide backwards compatibility by delegating to the new implementation
    return extract_js_ts_blocks(content, file_path, language, stats)

# For JS/TS extraction, tree_sitter import is handled through tree_sitter_utils.py
# tree-sitter is always available through tree-sitter-language-pack

def extract_code_blocks(file_path: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract code blocks from a file.
    
    Args:
        file_path: Path to source file
        output_dir: Output directory for extracted blocks
        
    Returns:
        List of extracted code blocks
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        # Detect language
        language = detect_language(file_path)
        if language == "unknown":
            stats["errors"].append(f"Unknown language for file: {file_path}")
            return []
            
        # Get language info
        info = get_language_info(language)
        if not info:
            stats["errors"].append(f"Unsupported language: {language}")
            return []
            
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract based on language
        if language == "python":
            blocks = extract_python_blocks(content, file_path, stats)
        elif language in ("javascript", "typescript"):
            # Use the js_ts_extractor with file_path directly
            # This will automatically read the file content and detect the language
            try:
                from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
                blocks_list, js_stats = extract_js_ts_blocks(file_path, output_dir)
                # Merge stats
                stats.update(js_stats)
                blocks = blocks_list
            except Exception as e:
                logger.error(f"Error extracting JS/TS blocks: {e}")
                # Fallback to generic extraction
                blocks = extract_generic_blocks(content, file_path, language, stats)
        else:
            blocks = extract_generic_blocks(content, file_path, language, stats)
            
        # Validate and verify blocks
        valid_blocks = []
        for block in blocks:
            if validate_block(block) and verify_block(block):
                # Ensure imports/exports are defined
                if "metadata" in block and "imports" not in block["metadata"]:
                    block["metadata"]["imports"] = []
                if "metadata" in block and "exports" not in block["metadata"]:
                    block["metadata"]["exports"] = []
                valid_blocks.append(block)
            else:
                stats["errors"].append(f"Invalid block in {file_path}: {block.get('name', 'unknown')}")
        
        # Save blocks to disk
        if valid_blocks:
            # Create output directories
            blocks_dir = output_dir / "blocks" / "code" / language
            blocks_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each block to separate file
            for block in valid_blocks:
                # Generate unique filename
                block_id = block.get("uuid", block.get("id", str(uuid.uuid4())))
                safe_name = "".join(c if c.isalnum() else "_" for c in block.get("name", "unnamed"))
                filename = f"{safe_name}_{block_id}.{language}"
                
                # Save block content
                with open(blocks_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(block["content"])
                    
            # Save blocks metadata as JSON
            blocks_json_path = output_dir / "blocks.json"
            
            try:
                # Load existing data if file exists
                if blocks_json_path.exists():
                    with open(blocks_json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        existing_blocks = existing_data.get("blocks", [])
                else:
                    existing_blocks = []
                    
                # Combine existing and new blocks
                with open(blocks_json_path, 'w', encoding='utf-8') as f:
                    # Convert Path objects to strings to make them JSON serializable
                    serializable_stats = {}
                    for key, value in stats.items():
                        if isinstance(value, Path):
                            serializable_stats[key] = str(value)
                        else:
                            serializable_stats[key] = value
                            
                    # Convert metadata Path objects in blocks
                    serializable_blocks = []
                    for block in existing_blocks + valid_blocks:
                        serializable_block = {}
                        for key, value in block.items():
                            if isinstance(value, Path):
                                serializable_block[key] = str(value)
                            elif key == "metadata" and isinstance(value, dict):
                                # Handle metadata dictionary
                                serializable_metadata = {}
                                for mkey, mvalue in value.items():
                                    if isinstance(mvalue, Path):
                                        serializable_metadata[mkey] = str(mvalue)
                                    else:
                                        serializable_metadata[mkey] = mvalue
                                serializable_block[key] = serializable_metadata
                            else:
                                serializable_block[key] = value
                        serializable_blocks.append(serializable_block)
                        
                    json.dump({
                        "blocks": serializable_blocks,
                        "stats": serializable_stats
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving blocks.json: {e}")
                
        # Update stats
        update_stats(stats, valid_blocks, language)
        
        return valid_blocks
        
    except Exception as e:
        logger.error(f"Error extracting code blocks: {e}")
        return []

def extract_python_blocks(
    content: str,
    file_path: str,
    stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract code blocks from Python file using AST."""
    try:
        # Parse AST
        tree = ast.parse(content)
        
        # Track blocks
        blocks = []
        
        # Process nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                blocks.append({
                    "type": "function",
                    "name": node.name,
                    "content": ast.get_source_segment(content, node),
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "metadata": {
                        "language": "python",
                        "file": file_path,
                        "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "args": [arg.arg for arg in node.args.args]
                    }
                })
                
            elif isinstance(node, ast.ClassDef):
                blocks.append({
                    "type": "class",
                    "name": node.name,
                    "content": ast.get_source_segment(content, node),
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "metadata": {
                        "language": "python",
                        "file": file_path,
                        "decorators": [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                        "bases": [b.id for b in node.bases if isinstance(b, ast.Name)]
                    }
                })
                
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting Python blocks: {e}")
        stats["errors"].append(str(e))
        return []

def extract_js_ts_blocks(
    content: str,
    file_path: str,
    language: str,
    stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract code blocks from JavaScript/TypeScript file using tree-sitter."""
    try:
        # Parse with tree-sitter
        parser = get_parser(language)
        if not parser:
            stats["errors"].append(f"Could not get parser for {language}")
            return []
        tree = parser.parse(bytes(content, "utf8"))
        
        # Track blocks
        blocks = []
        
        def visit_node(node: Any) -> None:
            """Process a tree-sitter node."""
            try:
                if node.type == "function_declaration":
                    blocks.append({
                        "type": "function",
                        "name": get_node_text(node.child_by_field_name("name"), content),
                        "content": get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "is_async": is_async_node(node),
                            "return_type": get_ts_return_type(node, content)
                        }
                    })
                    
                elif node.type == "class_declaration":
                    blocks.append({
                        "type": "class",
                        "name": get_node_text(node.child_by_field_name("name"), content),
                        "content": get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "extends": get_extends_class(node, content),
                            "implements": get_implements_interfaces(node, content)
                        }
                    })
                    
                elif node.type == "method_definition":
                    blocks.append({
                        "type": "method",
                        "name": get_node_text(node.child_by_field_name("name"), content),
                        "content": get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "is_static": is_static_node(node),
                            "is_async": is_async_node(node),
                            "return_type": get_ts_return_type(node, content)
                        }
                    })
                    
                elif node.type == "interface_declaration" and language == "typescript":
                    blocks.append({
                        "type": "interface",
                        "name": get_node_text(node.child_by_field_name("name"), content),
                        "content": get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "extends": get_extends_interfaces(node, content),
                            "properties": get_interface_properties(node, content)
                        }
                    })
                    
                # Visit children
                for child in node.children:
                    visit_node(child)
                    
            except Exception as e:
                logger.error(f"Error visiting node: {e}")
                
        # Process the tree
        visit_node(tree.root_node)
        
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting JS/TS blocks: {e}")
        stats["errors"].append(str(e))
        return []

def extract_generic_blocks(
    content: str,
    file_path: str,
    language: str,
    stats: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Extract code blocks using pattern matching."""
    try:
        # Get language patterns
        patterns = {
            "function": {
                "c": r"(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{",
                "cpp": r"(?:virtual\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?{",
                "java": r"(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(?:\{|throws)",
                "go": r"func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*{",
                "ruby": r"(?:def)\s+(\w+)(?:\([^)]*\))?\s*(?:do|\n|$|{)",
                "php": r"(?:function|public function|private function|protected function)\s+(\w+)\s*\([^)]*\)\s*{",
                "rust": r"(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*{"
            },
            "class": {
                "c": r"(?:class|struct)\s+(\w+)(?:\s*:\s*\w+)?\s*{",
                "cpp": r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*{",
                "java": r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
                "go": r"type\s+(\w+)\s+struct\s*{",
                "ruby": r"class\s+(\w+)(?:\s*<\s*\w+)?\s*(?:do|\n|$|{)",
                "php": r"(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
                "rust": r"(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*{"
            }
        }
        
        # Track blocks
        blocks = []
        
        # Extract functions
        if language in patterns["function"]:
            pattern = patterns["function"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                blocks.append({
                    "type": "function",
                    "name": match.group(1),
                    "content": get_block_content(content, match),
                    "line_start": content.count('\n', 0, match.start()) + 1,
                    "line_end": content.count('\n', 0, match.end()) + 1,
                    "metadata": {
                        "language": language,
                        "file": file_path
                    }
                })
                
        # Extract classes
        if language in patterns["class"]:
            pattern = patterns["class"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                blocks.append({
                    "type": "class",
                    "name": match.group(1),
                    "content": get_block_content(content, match),
                    "line_start": content.count('\n', 0, match.start()) + 1,
                    "line_end": content.count('\n', 0, match.end()) + 1,
                    "metadata": {
                        "language": language,
                        "file": file_path
                    }
                })
                
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting generic blocks: {e}")
        stats["errors"].append(str(e))
        return []

# Validate and verify blocks have been moved to:
# - validation_utils.py: For validating block structure and standardizing field names
# - verification_utils.py: For verifying code syntax and semantics

# Helper functions have been moved to tree_sitter_helpers.py

def extract_repository(repo_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Extract code blocks from a repository.
    
    Args:
        repo_path: Path to the repository
        output_dir: Optional output directory for extracted blocks
        
    Returns:
        Dictionary containing extracted blocks and metadata
    """
    repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path
    
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path not found: {repo_path}")
        return {"error": f"Repository path not found: {repo_path}"}
    
    # Use temporary directory if not specified
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="extraction_"))
    else:
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Extracting from repository: {repo_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find Python files in the repository
    blocks = []
    stats = init_stats()
    
    try:
        # Walk the repository directory
        for root, dirs, files in os.walk(repo_path):
            # Exclude hidden directories and venv directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv' and d != 'node_modules' and d != '__pycache__']
            
            for file in files:
                # Skip hidden files and non-source files
                if file.startswith('.') or file.endswith(('.pyc', '.pyo', '.pyd', '.exe', '.dll', '.so')):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    # Extract blocks from each file
                    file_blocks = extract_code_blocks(file_path, output_dir)
                    blocks.extend(file_blocks)
                    
                    # Update stats
                    stats["files_processed"] += 1
                    stats["blocks_extracted"] += len(file_blocks)
                    
                    logger.debug(f"Extracted {len(file_blocks)} blocks from {relative_path}")
                    
                except Exception as e:
                    logger.warning(f"Error extracting from {relative_path}: {e}")
                    stats["errors"].append(f"Error extracting from {relative_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error walking repository: {e}")
        stats["errors"].append(f"Error walking repository: {e}")
    
    # Save stats
    stats_file = output_dir / "extraction_stats.json"
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving stats: {e}")
    
    return {
        "blocks": blocks,
        "stats": stats,
        "output_dir": str(output_dir)
    }

# Usage examples moved to usage_examples.py 