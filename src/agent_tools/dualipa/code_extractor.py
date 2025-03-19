"""Code extraction utilities for the DuaLipa pipeline.

This module provides functions for extracting code and documentation blocks from
source files. It supports multiple programming languages and documentation formats.

Key functions:
- extract_repository: Main entry point for repository processing
- _extract_python_blocks: Extract Python code blocks using AST
- _extract_js_ts_blocks: Extract JavaScript/TypeScript blocks using tree-sitter
- _extract_markdown_blocks: Extract markdown sections by headers

DuaLiPA - Dual Language Processing Agent for Code and Documentation Extraction

This module provides tools for extracting and processing code from repositories,
generating structured representations for use with language models.

The module handles:
1. Code extraction from local or GitHub repositories
2. Code block identification and extraction
3. Language detection
4. Documentation extraction
5. Structured output for analysis

Main functions:
- extract_repository: Extract content from a repository
- extract_single_file: Extract content from a single file
- format_output_as_json: Format extracted data as JSON
- format_output_as_md: Format extracted data as Markdown
- format_output_as_html: Format extracted data as HTML

Official Documentation References:
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
- tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
"""

import os
import re
import sys
import time
import json
import tempfile
import glob
import shutil
import ast
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import textwrap
import argparse
import fnmatch
from tree_sitter_languages import get_language, get_parser
from .language_detection import detect_language

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Constants
OUTPUT_DIRS = {
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs",
    "METADATA": "metadata"
}

# Common extensions for code and documentation files
CODE_FILE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', 
    '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala'
}

DOCUMENTATION_FILE_EXTENSIONS = {
    '.md', '.rst', '.txt', '.ipynb', '.tex'
}

# Files to ignore
IGNORED_FILES = {
    'LICENSE', '.gitignore', '.gitattributes', '.gitmodules',
    'setup.py', 'requirements.txt', 'package.json', 'package-lock.json',
    'yarn.lock', 'Pipfile', 'Pipfile.lock', 'pyproject.toml', 'poetry.lock',
    'Cargo.toml', 'Cargo.lock', 'Gemfile', 'Gemfile.lock'
}

# Directories to ignore
IGNORED_DIRECTORIES = {
    '.git', '.github', '.vscode', '.idea', '__pycache__', 
    'node_modules', 'venv', 'env', '.env', 'build', 'dist', 
    'target', 'out', 'bin', 'obj', 'tmp', 'temp', 'tests'
}

# Tree-sitter is always available in this project
TREE_SITTER_AVAILABLE = True

# Initialize parsers for supported languages
PARSERS = {
    'javascript': get_parser('javascript'),
    'typescript': get_parser('tsx'),  # Use tsx parser for TypeScript/TSX
    'python': get_parser('python'),
    'go': get_parser('go'),
    'rust': get_parser('rust'),
    'cpp': get_parser('cpp'),
    'java': get_parser('java'),
    'ruby': get_parser('ruby'),
    'bash': get_parser('bash')
}

TREE_SITTER_LANGUAGES = {
    'javascript': get_language('javascript'),
    'typescript': get_language('tsx'),  # Use tsx language for TypeScript/TSX
    'python': get_language('python'),
    'go': get_language('go'),
    'rust': get_language('rust'),
    'cpp': get_language('cpp'),
    'java': get_language('java'),
    'ruby': get_language('ruby'),
    'bash': get_language('bash')
}

logger.info("Successfully initialized tree-sitter parsers")

def count_tokens(text: str) -> int:
    """
    Simple token counter - splits on whitespace and punctuation.
    
    Args:
        text: Text to count tokens in
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    # Split on whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)

# JSON serializer helper function for handling Path objects
def _json_serializer(obj):
    """Custom JSON serializer to handle Path objects and other non-serializable types."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):  # Handle datetime objects
        return obj.isoformat()
    return str(obj)  # Fall back to string representation for other types

# Language detection patterns
# ... rest of the file ...

# Add this new function for stats standardization
def initialize_stats_dict(source: Union[str, Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Initialize a standardized stats dictionary with all required fields.
    
    Args:
        source: Source repository or file path
        output_dir: Output directory path
        
    Returns:
        Stats dictionary with all required fields initialized
    """
    now = datetime.now()
    return {
        # Source and output information
        "source": str(source) if source else "",
        "repo_url": str(source) if source else "",  # For backward compatibility 
        "output_path": str(output_dir) if output_dir else "",
        
        # Timing information
        "start_time": now.isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        
        # File and block counts
        "total_files": 0,
        "documentation_files": 0,
        "code_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "skipped_files": 0,
        "error_files": 0,
        
        # Categorization
        "languages": {},
        "file_types": {},
        
        # Error tracking
        "errors": [],
        
        # Block storage
        "file_blocks": {}  # Dictionary to collect blocks from each file
    }

def extract_repository(
    source: str, 
    output_path: str = None,
    max_files: int = 1000,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    extract_documentation: bool = True,
    extract_code: bool = True,
    extract_blocks: bool = True
) -> Dict[str, Any]:
    """
    Extract content from a repository source (local path or GitHub URL).
    
    Args:
        source: Path to the repository or GitHub URL
        output_path: Path to the output directory (default: temp directory)
        max_files: Maximum number of files to process (default: 1000)
        include_patterns: List of glob patterns to include files
        exclude_patterns: List of glob patterns to exclude files
        extract_documentation: Whether to extract documentation files (default: True)
        extract_code: Whether to extract code files (default: True)
        extract_blocks: Whether to extract blocks from files (default: True)
        
    Returns:
        Statistics dictionary
    """
    from agent_tools.dualipa.github_utils import (
        is_github_url, download_github_repo, verify_repo_structure, discover_files
    )
    
    # Set up output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(tempfile.gettempdir(), f"extracted_{timestamp}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics with standardized structure
    stats = initialize_stats_dict(source, output_dir)
    
    repo_dir = None
    
    try:
        # Handle GitHub repositories
        if is_github_url(source):
            logger.info(f"Cloning GitHub repository: {source}")
            try:
                # Set up temporary directory for cloning
                repo_dir = tempfile.mkdtemp(prefix="github_repo_")
                logger.debug(f"Created temporary directory: {repo_dir}")
                
                # Clone the repository
                download_github_repo(source, repo_dir)
                source = repo_dir
            except Exception as e:
                error_msg = f"Error cloning repository {source}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                return stats
        
        # Convert source to Path and verify structure
        source_path = Path(source)
        if not verify_repo_structure(source_path):
            error_msg = f"Invalid repository structure at {source}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
        
        # Use github_utils to discover files
        try:
            files = discover_files(
                source_path,
                max_files=max_files,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                ignored_dirs=IGNORED_DIRECTORIES,
                ignored_files=IGNORED_FILES
            )
        except Exception as e:
            error_msg = f"Error discovering files: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
        
        logger.info(f"Found {len(files)} files in repository")
        
        # Process each file
        for file_path in tqdm(files, desc="Processing files"):
            try:
                # Read file content
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Detect language
                language = detect_language(file_path)
                
                # Extract blocks based on language
                if language == "python":
                    _extract_python_blocks(file_path, content, output_dir, stats)
                elif language in ["javascript", "typescript", "jsx", "tsx"]:
                    _extract_js_ts_blocks(file_path, content, output_dir, stats, language)
                elif language == "markdown":
                    _extract_markdown_blocks(file_path, content, output_dir, stats)
                else:
                    _extract_generic_blocks(file_path, content, output_dir, stats, language)
                    
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                stats["error_files"] += 1
                continue
        
        # Save blocks to JSON
        blocks = []
        for file_blocks in stats["file_blocks"].values():
            blocks.extend(file_blocks)
        
        blocks_json = output_dir / "blocks.json"
        with open(blocks_json, 'w') as f:
            json.dump(blocks, f, indent=2, default=_json_serializer)
        logger.info(f"Saved {len(blocks)} blocks to blocks.json")
        
        # Save code blocks separately
        code_blocks = [b for b in blocks if b["block_type"] in {"function", "class", "method", "react_component"}]
        code_json = output_dir / "code.json"
        with open(code_json, 'w') as f:
            json.dump(code_blocks, f, indent=2, default=_json_serializer)
        logger.info(f"Saved {len(code_blocks)} entries to code.json file")
        
        # Save documentation blocks separately
        doc_blocks = [b for b in blocks if b["block_type"] in {"section", "code_block", "text"}]
        doc_json = output_dir / "documentation.json"
        with open(doc_json, 'w') as f:
            json.dump(doc_blocks, f, indent=2, default=_json_serializer)
        logger.info(f"Saved {len(doc_blocks)} entries to documentation.json file")
        
        # Save stats
        stats_json = output_dir / "extraction_stats.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2, default=_json_serializer)
        logger.info(f"Extraction completed. Statistics saved to {stats_json}")
        
        return stats
        
    finally:
        # Clean up temporary directory if created
        if repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

def _process_code_file(
    file_path: Path, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: Optional[str] = None,
    extract_blocks: bool = True
) -> None:
    """
    Process a code file and update statistics.
    
    Args:
        file_path: Path to the code file
        output_dir: Output directory to save processed files
        stats: Statistics dictionary to update
        language: Optional language identifier (auto-detected if None)
        extract_blocks: Whether to extract blocks from the code file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # If language is not provided, detect it
        if language is None:
            language = detect_language(file_path)
        
        # Update statistics
        stats["code_files"] += 1
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for code files
        code_dir = output_dir / "code_files"
        code_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory based on the language
        lang_dir = code_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        # Use appropriate comment syntax for the language
        lang_comment = {
            'python': '#',
            'javascript': '//',
            'typescript': '//',
            'java': '//',
            'c': '//',
            'cpp': '//',
            'go': '//',
            'rust': '//',
            'ruby': '#',
            'php': '//',
            'shell': '#',
            'bash': '#'
        }
        
        comment_char = lang_comment.get(language, '#')
        path_comment = f"{comment_char} Original file path: {file_path}\n\n"
        content_with_path = path_comment + content
        
        # Save the file to the output directory with a unique name
        # Add the original path as part of the filename to maintain context
        path_hash = hash(str(file_path)) % 10000
        rel_path = str(file_path).replace('/', '_').replace('\\', '_')
        output_file = lang_dir / f"{rel_path}_{path_hash:04d}{ext}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_with_path)
        
        # Extract blocks if requested
        if extract_blocks:
            try:
                blocks_extracted = 0
                
                # Use the appropriate block extraction function based on language
                if language == 'python':
                    blocks_extracted = _extract_python_blocks(file_path, content, output_dir, stats)
                elif language in ['javascript', 'typescript', 'js', 'ts']:
                    blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, language)
                else:
                    # Skip block extraction for unsupported languages
                    logger.debug(f"Skipping block extraction for unsupported language: {language}")
                
                logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
            except Exception as e:
                error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                
        logger.debug(f"Processed code file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing code file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

def _extract_python_blocks(
    file_path: Path,
    content: str,
    output_dir: Path,
    stats: Dict[str, Any]
) -> int:
    """Extract functions and classes from Python code using AST.
    Also extracts script-level code for executable Python files.
    
    Args:
        file_path: Path to the Python file (must be a Path object)
        content: Content of the file as string
        output_dir: Output directory to save extracted blocks
        stats: Statistics dictionary
        
    Returns:
        Number of blocks extracted
    """
    # Initialize file_blocks for this file if not already present
    if str(file_path) not in stats["file_blocks"]:
        stats["file_blocks"][str(file_path)] = []
        
    # Update language statistics
    stats["languages"]["python"] = stats["languages"].get("python", 0) + 1
    
    # Update file type statistics
    ext = file_path.suffix.lower()
    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    
    # Update total files count
    stats["total_files"] = stats.get("total_files", 0) + 1
    
    # Create output directory for Python blocks
    python_blocks_dir = output_dir / "blocks" / "code" / "python"
    python_blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize block counter
    blocks_extracted = 0
    try:
        # Parse the Python code using AST
        tree = ast.parse(content)
        
        # Extract all top-level blocks (functions and classes)
        for node in tree.body:
            try:
                # Extract functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    block_name = node.name
                    start_line = node.lineno
                    start_line = node.lineno
                    end_line = _find_func_end(content, start_line)
                    block_content = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    # Handle decorators
                    if node.decorator_list:
                        decorator_start = min(d.lineno for d in node.decorator_list)
                        block_content = "\n".join(content.splitlines()[decorator_start-1:end_line])
                    
                    _save_python_block(
                        block_name, block_content, file_path, python_blocks_dir, 
                        start_line, end_line, stats, "function"
                    )
                    blocks_extracted += 1
                
                # Extract classes and their methods
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    start_line = node.lineno
                    end_line = _find_class_end(content, start_line)
                    
                    # Handle class decorators
                    if node.decorator_list:
                        decorator_start = min(d.lineno for d in node.decorator_list)
                        start_line = decorator_start
                    
                    class_content = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    # Save the entire class as a block
                    _save_python_block(
                        class_name, class_content, file_path, python_blocks_dir, 
                        start_line, end_line, stats, "class"
                    )
                    blocks_extracted += 1
                    
                    # Extract methods from the class
                    for class_node in node.body:
                        if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_name = class_node.name
                            method_start_line = class_node.lineno
                            method_end_line = _find_func_end(content, method_start_line, indent_level=4)
                            
                            # Handle method decorators
                            if class_node.decorator_list:
                                decorator_start = min(d.lineno for d in class_node.decorator_list)
                                method_start_line = decorator_start
                            
                            method_content = "\n".join(content.splitlines()[method_start_line-1:method_end_line])
                            
                            _save_python_block(
                                f"{class_name}.{method_name}", method_content, file_path, 
                                python_blocks_dir, method_start_line, method_end_line, 
                                stats, "method"
                            )
                            blocks_extracted += 1
            
            except Exception as e:
                error_msg = f"Error processing Python node: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                continue
        
        # Check if we should extract script-level code
        is_special_file = any(file_path.name.lower().endswith(name) for name in [
            "setup.py", "manage.py", "app.py", "main.py", "run.py", "wsgi.py", "asgi.py"
        ])
        
        # Check for top-level executable statements
        has_top_level_executable = False
        for node in tree.body:
            if isinstance(node, (ast.Expr, ast.Assign, ast.If, ast.For, ast.While, ast.Import, ast.ImportFrom)):
                has_top_level_executable = True
                break
        
        # Extract the entire file as a script block if:
        # 1. It's a special file by name, or
        # 2. It has top-level executable statements and few or no function/class definitions
        if is_special_file or (has_top_level_executable and blocks_extracted <= 2):
            script_name = file_path.stem
            block_content = content
            
            # Create block header with metadata
            header = f"# Original file: {file_path}\n"
            header += f"# Block type: script\n"
            header += f"# Name: {script_name}\n"
            header += f"# Description: Full script file with top-level executable code\n\n"
            
            # Save the script block
            _save_python_block(
                script_name, block_content, file_path, python_blocks_dir,
                1, len(content.splitlines()), stats, "script"
            )
            blocks_extracted += 1
        
        # Update stats
        stats["code_blocks"] += blocks_extracted
        
        return blocks_extracted
    
    except SyntaxError:
        # Handle Python files that can't be parsed with AST
        logger.warning(f"SyntaxError parsing Python file: {file_path}, falling back to generic splitting")
        return _extract_generic_blocks(file_path, content, output_dir, stats, "python")
    except Exception as e:
        error_msg = f"Error extracting Python blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _find_func_end(content: str, start_line: int, indent_level: int = 0) -> int:
    """Find the end line of a function definition."""
    lines = content.splitlines()
    if start_line > len(lines):
        return start_line
        
    # Get the indentation of the function definition
    func_indent = len(lines[start_line-1]) - len(lines[start_line-1].lstrip())
    if indent_level > 0:
        func_indent = indent_level
        
    # Find the last line of the function
    end_line = start_line
    for i in range(start_line, len(lines)):
        line = lines[i]
        if line.strip() == "":
                continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= func_indent and i > start_line:
            return i
        end_line = i + 1
        
    return end_line

def _find_class_end(content: str, start_line: int) -> int:
    """Find the end line of a class definition."""
    lines = content.splitlines()
    if start_line > len(lines):
        return start_line
        
    # Get the indentation of the class definition
    class_indent = len(lines[start_line-1]) - len(lines[start_line-1].lstrip())
        
    # Find the last line of the class
    end_line = start_line
    for i in range(start_line, len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= class_indent and i > start_line:
            return i
        end_line = i + 1
        
    return end_line

def _save_python_block(
    block_name: str,
    block_content: str,
    source_file: Path,
    output_dir: Path,
    start_line: int,
    end_line: int,
    stats: Dict[str, Any],
    block_type: str
) -> None:
    """
    Save a Python code block to a file and update stats.
    
    Args:
        block_name: Name of the block (function/class/method name)
        block_content: Content of the block
        source_file: Source file path
        output_dir: Output directory
        start_line: Start line of the block
        end_line: End line of the block
        stats: Statistics dictionary
        block_type: Type of block (function, class, method, script)
    """
    # Create a sanitized filename
    safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
    block_file = output_dir / f"{safe_name}.py"
    
    # Ensure we don't overwrite existing blocks with the same name
    if block_file.exists():
        count = 1
        while block_file.exists():
            block_file = output_dir / f"{safe_name}_{count}.py"
            count += 1
    
    # Write block to file
    with open(block_file, "w", encoding="utf-8") as f:
        f.write(block_content)
            
            # Create block metadata
    block_info = {
        "type": block_type,  # Keep type for backward compatibility
        "block_type": block_type,  # Add block_type to match test expectations
        "name": block_name,
        "source_file": str(source_file),
        "source_start_line": start_line,
        "source_end_line": end_line,
        "content": block_content,
        "language": "python",
        "output_file": str(block_file),
        "extracted_at": datetime.now().isoformat()
    }
    
    # Add to stats
    stats["file_blocks"][str(source_file)].append(block_info)

def _extract_js_ts_blocks(
    file_path: Path,
    content: str,
    output_dir: Path,
    stats: Dict[str, Any],
    language: Optional[str] = None
) -> int:
    """Extract JavaScript/TypeScript code blocks using tree-sitter.
    
    Args:
        file_path: Path to the source file
        content: File content
        output_dir: Output directory for extracted blocks
        stats: Statistics dictionary
        language: Optional language override
        
    Returns:
        Number of blocks extracted
    """
    if not content.strip():
        return 0
        
    # Determine language from file extension if not provided
    ext = file_path.suffix.lower()
    if not language:
        if ext in {'.js', '.jsx'}:
            language = 'javascript'
        elif ext in {'.ts', '.tsx'}:
            language = 'typescript'
        else:
            language = 'javascript'  # Default to JavaScript
            
    # Create output directory
    blocks_dir = output_dir / "blocks" / "code" / language
    blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Update stats
    stats["languages"][language] = stats["languages"].get(language, 0) + 1
    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    stats["total_files"] = stats.get("total_files", 0) + 1
    
    # Initialize file_blocks for this file if not already present
    if str(file_path) not in stats["file_blocks"]:
        stats["file_blocks"][str(file_path)] = []
    
    # Parse with tree-sitter
    try:
        parser = PARSERS[language]
        content_bytes = content.encode('utf8')
        tree = parser.parse(content_bytes)
        root_node = tree.root_node
        
        # Extract blocks
        blocks = []
        for child in root_node.children:
            try:
                # Get block type and name
                block_type = child.type
                block_name = None
                
                # Extract name based on node type
                if block_type == 'function_declaration':
                    for sub_node in child.children:
                        if sub_node.type == 'identifier':
                            block_name = content[sub_node.start_byte:sub_node.end_byte]
                            block_type = 'function'  # Normalize to expected block_type
                            break
                elif block_type == 'class_declaration':
                    for sub_node in child.children:
                        if sub_node.type == 'identifier':
                            block_name = content[sub_node.start_byte:sub_node.end_byte]
                            block_type = 'class'  # Normalize to expected block_type
                            break
                elif block_type == 'variable_declaration':
                    for sub_node in child.children:
                        if sub_node.type == 'variable_declarator':
                            for var_node in sub_node.children:
                                if var_node.type == 'identifier':
                                    block_name = content[var_node.start_byte:var_node.end_byte]
                                elif var_node.type == 'function':
                                    # Handle function expressions
                                    block_type = 'function'
                                    # Look for the identifier in the parent declarator
                                    for sibling in sub_node.children:
                                        if sibling.type == 'identifier':
                                            block_name = content[sibling.start_byte:sibling.end_byte]
                                            break
                                    break
                            break
                
                # Only process blocks with valid names
                if block_name:
                    block_content = content[child.start_byte:child.end_byte]
                    
                    # Create block metadata
                    block_info = {
                        "type": block_type,  # Keep type for backward compatibility
                        "block_type": block_type,  # Add block_type to match test expectations
                        "name": block_name,
                        "source_file": str(file_path),
                        "content": block_content,
                        "language": language,
                        "output_file": str(blocks_dir / f"{block_name}{ext}"),
                        "extracted_at": datetime.now().isoformat()
                    }
                    
                    # Add to stats
                    stats["file_blocks"][str(file_path)].append(block_info)
                    blocks.append((block_type, block_name, block_content))
                    stats["code_blocks"] = stats.get("code_blocks", 0) + 1
                    
            except Exception as e:
                error_msg = f"Error processing node {child.type}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                continue
                
        # Save blocks to files
        for block_type, block_name, block_content in blocks:
            # Create a sanitized filename
            safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
            block_path = blocks_dir / f"{safe_name}{ext}"
            
            # Ensure unique filenames
            if block_path.exists():
                count = 1
                while block_path.exists():
                    block_path = blocks_dir / f"{safe_name}_{count}{ext}"
                    count += 1
            
            # Write block to file
            with open(block_path, 'w') as f:
                f.write(block_content)
            
        return len(blocks)
    except Exception as e:
        error_msg = f"Error extracting {language} blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _save_block(
    block_name: str,
    block_content: str,
    source_file: Path,
    output_dir: Path,
    start_line: int,
    end_line: int,
    stats: Dict[str, Any],
    block_type: str,
    language: str,
    original_ext: str
) -> None:
    """Save a code block to a file and update stats."""
    # Create a sanitized filename
    safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
    
    # Preserve the original extension for TSX/JSX files
    if original_ext in {'.tsx', '.jsx'}:
        ext = original_ext
    else:
        ext = ".ts" if language == "typescript" else ".js"
    
    block_file = output_dir / f"{safe_name}{ext}"
    
    # Ensure we don't overwrite existing blocks
    if block_file.exists():
        count = 1
        while block_file.exists():
            block_file = output_dir / f"{safe_name}_{count}{ext}"
            count += 1
    
    # Write block to file
    with open(block_file, "w", encoding="utf-8") as f:
        f.write(block_content)
    
    # Create block metadata
    block_info = {
        "type": block_type,  # Keep type for backward compatibility
        "block_type": block_type,  # Add block_type to match test expectations
        "name": block_name,
        "source_file": str(source_file),
        "source_start_line": start_line,
        "source_end_line": end_line,
        "content": block_content,
        "language": language,
        "output_file": str(block_file),
        "extracted_at": datetime.now().isoformat()
    }
    
    # Add to stats
    stats["file_blocks"][str(source_file)].append(block_info)

# Output directories structure
OUTPUT_DIRS = {
    "CODE_FILES": "code",
    "DOC_FILES": "docs",
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs"
}

def _extract_markdown_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> int:
    """
    Extract sections from markdown based on headers.
    
    Args:
        file_path: Path to the markdown file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Initialize stats if needed
        if "errors" not in stats:
            stats["errors"] = []
        
        # Verify file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return 0
        
        # Create output directory for markdown blocks
        blocks_dir = output_dir / "doc_blocks" / "markdown"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Update language statistics
        stats["languages"]["markdown"] = stats["languages"].get("markdown", 0) + 1
        
        # Update file type statistics
        ext = file_path.suffix.lower()
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Update total files count
        stats["total_files"] = stats.get("total_files", 0) + 1
        
        # Initialize file_blocks for this file if not already present
        if str(file_path) not in stats["file_blocks"]:
            stats["file_blocks"][str(file_path)] = []
        
        # Split on headers (lines starting with #)
        sections = []
        current_section = []
        current_level = 0
        current_title = None
        
        # Process code blocks
        code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        code_blocks = []
        
        # First pass: extract code blocks
        for match in code_block_pattern.finditer(content):
            language = match.group(1)
            code_content = match.group(2)
            
            # Map common language aliases
            language_map = {
                '': 'text',
                'sh': 'bash',
                'shell': 'bash',
                'console': 'bash',
                'js': 'javascript',
                'ts': 'typescript',
                'py': 'python',
                'rb': 'ruby',
                'rs': 'rust',
                'cpp': 'cpp',
                'c++': 'cpp',
                'cs': 'csharp',
                'json': 'json',
                'xml': 'xml',
                'html': 'html',
                'css': 'css',
                'yaml': 'yaml',
                'yml': 'yaml',
                'md': 'markdown',
                'sql': 'sql'
            }
            
            # Use mapped language or original if not in map
            block_language = language_map.get(language.lower(), language) or 'text'
            
            code_blocks.append({
                "block_type": "code_block",
                "language": block_language,
                "content": code_content.strip()
            })
        
        # Second pass: process sections
        lines = content.splitlines()
        for line in lines:
            # Check for header
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if header_match:
                # Save previous section if it exists
                if current_section and current_title:
                    section_content = "\n".join(current_section)
                    if section_content.strip():
                        sections.append({
                            "block_type": "section",
                            "title": current_title,
                            "level": current_level,
                            "content": section_content
                        })
                
                # Start new section
                current_level = len(header_match.group(1))
                # Format title to match test expectations (replace spaces with underscores)
                current_title = re.sub(r'\s+', '_', header_match.group(2).strip())
                current_section = [line]
            else:
                if current_title:  # Only append if we have a section
                    current_section.append(line)
        
        # Save the last section
        if current_section and current_title:
            section_content = "\n".join(current_section)
            if section_content.strip():
                sections.append({
                    "block_type": "section",
                    "title": current_title,
                    "level": current_level,
                    "content": section_content
                })
        
        # Add all blocks to stats
        blocks = sections + code_blocks
        stats["file_blocks"][str(file_path)].extend(blocks)
        stats["doc_blocks"] = stats.get("doc_blocks", 0) + len(blocks)
        
        return len(blocks)
            
    except Exception as e:
        error_msg = f"Error extracting markdown blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _extract_generic_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> int:
    """
    Extract blocks from generic code files by splitting on double newlines.
    
    Args:
        file_path: Path to the code file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize list to collect blocks for this file
        if str(file_path) not in stats["file_blocks"]:
            stats["file_blocks"][str(file_path)] = []
        
        # Update language statistics
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        
        # Update file type statistics
        ext = file_path.suffix.lower() if file_path.suffix else '.txt'
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Update total files count
        stats["total_files"] = stats.get("total_files", 0) + 1
        
        # Split by double newlines
        blocks = re.split(r"\n\s*\n", content)
        
        # Process each block
        block_count = 0
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:  # Skip empty blocks
                continue
                
            # Create block metadata
            block_info = {
                "type": "text",
                "block_type": "text",
                "language": language,
                "content": block,
                "file": str(file_path),
                "chunk_index": i,
                "output_file": str(blocks_dir / f"{file_path.stem}_chunk_{i}{ext}")
            }
            
            # Add to stats
            stats["file_blocks"][str(file_path)].append(block_info)
            block_count += 1
            
            # Save block to file
            with open(block_info["output_file"], "w", encoding="utf-8") as f:
                f.write(block)
        
        # Update statistics
        stats["code_blocks"] = stats.get("code_blocks", 0) + block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting generic blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _process_documentation_file(
    file_path: Path,
    output_dir: Path,
    stats: Dict[str, Any],
    extract_blocks: bool = True
) -> None:
    """
    Process a documentation file and update statistics.
    
    Args:
        file_path: Path to the documentation file
        output_dir: Output directory to save processed files
        stats: Statistics dictionary to update
        extract_blocks: Whether to extract blocks from the documentation file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Update statistics
        stats["documentation_files"] += 1
        stats["languages"]["markdown"] = stats["languages"].get("markdown", 0) + 1
        
        # Get file extension
        ext = file_path.suffix.lower()
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Extract blocks if requested
        if extract_blocks:
            _extract_markdown_blocks(file_path, content, output_dir, stats)
        
        logger.debug(f"Processed documentation file: {file_path}")
        
    except Exception as e:
        error_msg = f"Error processing documentation file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)

def _verify_code_block(block, language=None):
    """
    Verify if a code block is valid.
    
    Args:
        block (dict): Code block to verify
        language (str, optional): Language to verify against, defaults to block's language
        
    Returns:
        bool: True if the block is valid, False otherwise
    """
    if not block or not isinstance(block, dict):
        return False
    
    # Get the language from the block if not provided
    if not language:
        language = block.get("language")
    if not language:
        return False
    
    # Get the content
    content = block.get("content")
    if not content:
        return False
    
    try:
        # Language-specific verification
        if language == "python":
            try:
                ast.parse(content)
                return True
            except SyntaxError:
                return False
        elif language in ["javascript", "typescript"]:
            # Use tree-sitter for JS/TS verification
            parser = PARSERS.get(language, PARSERS["javascript"])
            tree = parser.parse(bytes(content, "utf8"))
            return not bool(tree.root_node.has_error)
        elif language in PARSERS:
            # Use tree-sitter for other supported languages
            parser = PARSERS[language]
            tree = parser.parse(bytes(content, "utf8"))
            return not bool(tree.root_node.has_error)
        else:
            # For unsupported languages, just check if content is not empty
            return bool(content.strip())
    except Exception:
        return False

def format_output_as_json(results):
    """
    Format extraction results as JSON.
    
    Args:
        results (dict): Extraction results containing blocks and stats
        
    Returns:
        str: JSON-formatted string
    """
    return json.dumps(results, indent=2, default=_json_serializer)

def format_output_as_md(results):
    """
    Format extraction results as Markdown.
    
    Args:
        results (dict): Extraction results containing blocks and stats
        
    Returns:
        str: Markdown-formatted string
    """
    output = []
    
    # Title
    output.append("# Extraction Results\n")
    
    # Statistics
    output.append("## Statistics\n")
    stats = results.get("stats", {})
    output.append(f"- Total Files: {stats.get('total_files', 0)}")
    output.append(f"- Code Files: {stats.get('code_files', 0)}")
    output.append(f"- Documentation Files: {stats.get('documentation_files', 0)}")
    output.append(f"- Code Blocks: {stats.get('code_blocks', 0)}")
    output.append(f"- Languages: {', '.join(stats.get('languages', {}))}\n")
    
    # Code Blocks
    output.append("## Code Blocks\n")
    for block in results.get("blocks", []):
        output.append(f"### {block.get('name', 'Unnamed Block')} ({block.get('type', 'unknown')})")
        output.append(f"- File: {block.get('path', 'unknown')}")
        output.append(f"- Lines: {block.get('start_line', 0)}-{block.get('end_line', 0)}\n")
        output.append(f"```{block.get('language', '')}")
        output.append(block.get("content", "").strip())
        output.append("```\n")
    
    return "\n".join(output)

def format_output_as_html(results):
    """
    Format extraction results as HTML.
    
    Args:
        results (dict): Extraction results containing blocks and stats
        
    Returns:
        str: HTML-formatted string
    """
    output = []
    
    # HTML header
    output.append("<!DOCTYPE html>")
    output.append("<html lang='en'>")
    output.append("<head>")
    output.append("  <meta charset='UTF-8'>")
    output.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    output.append("  <title>Extraction Results</title>")
    output.append("  <link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css'>")
    output.append("  <script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js'></script>")
    output.append("  <script>hljs.highlightAll();</script>")
    output.append("  <style>")
    output.append("    body { font-family: Arial, sans-serif; margin: 2em; }")
    output.append("    pre { background: #f5f5f5; padding: 1em; border-radius: 4px; }")
    output.append("    .stats { margin: 1em 0; padding: 1em; background: #eef; border-radius: 4px; }")
    output.append("    .block { margin: 2em 0; padding: 1em; background: #fff; border: 1px solid #ddd; border-radius: 4px; }")
    output.append("  </style>")
    output.append("</head>")
    output.append("<body>")
    
    # Title
    output.append("<h1>Extraction Results</h1>")
    
    # Statistics
    output.append("<h2>Statistics</h2>")
    output.append("<div class='stats'>")
    stats = results.get("stats", {})
    output.append(f"<p>Total Files: {stats.get('total_files', 0)}</p>")
    output.append(f"<p>Code Files: {stats.get('code_files', 0)}</p>")
    output.append(f"<p>Documentation Files: {stats.get('documentation_files', 0)}</p>")
    output.append(f"<p>Code Blocks: {stats.get('code_blocks', 0)}</p>")
    output.append(f"<p>Languages: {', '.join(stats.get('languages', {}))}</p>")
    if "repo_url" in stats:
        output.append(f"<p>Repository: <a href='{stats['repo_url']}'>{stats['repo_url']}</a></p>")
    output.append("</div>")
    
    # Code Blocks
    output.append("<h2>Code Blocks</h2>")
    for block in results.get("blocks", []):
        output.append("<div class='block'>")
        output.append(f"<h3>{block.get('name', 'Unnamed Block')} ({block.get('type', 'unknown')})</h3>")
        output.append(f"<p>File: {block.get('path', 'unknown')}</p>")
        output.append(f"<p>Lines: {block.get('start_line', 0)}-{block.get('end_line', 0)}</p>")
        output.append(f"<pre><code class='language-{block.get('language', '')}'>")
        output.append(block.get("content", "").strip())
        output.append("</code></pre>")
        output.append("</div>")
    
    # HTML footer
    output.append("</body>")
    output.append("</html>")
    
    return "\n".join(output)

def _is_code_file(file_path: str) -> bool:
    """
    Determine if a file is a code file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a code file, False otherwise
    """
    code_extensions = {
        '.py', '.js', '.ts', '.tsx', '.jsx', 
        '.java', '.cpp', '.c', '.h', '.hpp',
        '.rs', '.go', '.rb', '.php', '.cs'
    }
    return Path(file_path).suffix.lower() in code_extensions

def _is_documentation_file(file_path: str) -> bool:
    """
    Determine if a file is a documentation file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is a documentation file, False otherwise
    """
    return Path(file_path).suffix.lower() in DOCUMENTATION_FILE_EXTENSIONS

def _save_stats_to_json(stats: Dict[str, Any], output_path: str) -> None:
    """
    Save extraction statistics to a JSON file.
    
    Args:
        stats: Dictionary containing extraction statistics
        output_path: Directory to save the stats file
    """
    stats_file = os.path.join(output_path, "extraction_stats.json")
    try:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved extraction stats to {stats_file}")
    except Exception as e:
        logger.error(f"Failed to save stats to {stats_file}: {e}")

# ... rest of the existing code ...
