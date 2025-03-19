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
import uuid

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
    'typescript': get_parser('typescript'),  # Use typescript parser for both .ts and .tsx files
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
    'typescript': get_language('typescript'),  # Use typescript language for both .ts and .tsx files
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
                    end_line = _find_func_end(content, start_line)
                    block_content = "\n".join(content.splitlines()[start_line-1:end_line])
                    
                    # Handle decorators
                    if node.decorator_list:
                        decorator_start = min(d.lineno for d in node.decorator_list)
                        block_content = "\n".join(content.splitlines()[decorator_start-1:end_line])
                    
                    # Dedent the content
                    block_content = textwrap.dedent(block_content)
                    
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
                    
                    # Dedent the class content
                    class_content = textwrap.dedent(class_content)
                    
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
                            
                            # Dedent the method content before saving
                            method_content = textwrap.dedent(method_content)
                            
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
    """Save a Python code block to a file and update stats."""
    # Create a sanitized filename
    safe_name = re.sub(r'[^\w\-\.]', '_', block_name)
    block_file = output_dir / f"{safe_name}.py"
    
    # Ensure we don't overwrite existing blocks
    if block_file.exists():
        count = 1
        while block_file.exists():
            block_file = output_dir / f"{safe_name}_{count}.py"
            count += 1
    
    # Write block to file
    with open(block_file, "w", encoding="utf-8") as f:
        f.write(block_content)
    
    # Create block metadata using the new helper
    block_info = _create_code_block(
        name=block_name,
        content=block_content,
        file_path=source_file,
        block_type=block_type,
        language="python",
        start_line=start_line,
        end_line=end_line,
        test_file=None  # Would need test discovery
    )
    
    # Add output file path
    block_info["output_file"] = str(block_file)
    
    # Add to stats
    stats["file_blocks"][str(source_file)].append(block_info)

def _extract_js_ts_blocks(
    file_path: Path,
    content: str,
    output_dir: Path,
    stats: Dict[str, Any],
    language: Optional[str] = None
) -> int:
    """Extract JavaScript/TypeScript code blocks using tree-sitter queries.
    
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
    
    # Update language and file type stats
    stats["languages"][language] = stats["languages"].get(language, 0) + 1
    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
    stats["total_files"] = stats.get("total_files", 0) + 1
    
    # Initialize file_blocks for this file if not already present
    if str(file_path) not in stats["file_blocks"]:
        stats["file_blocks"][str(file_path)] = []
    
    try:
        # Special case for React components with memo/Flow type annotations
        if "ListItem" in file_path.name and re.search(r'export\s+default\s+\(\s*memo\s*\(', content):
            # Extract the component name (assuming it's declared as a function)
            component_match = re.search(r'function\s+([A-Z][a-zA-Z0-9_]*)', content)
            if component_match:
                component_name = component_match.group(1)
                
                # Add the entire content as a component block
                block_info = _create_code_block(
                    name=component_name,
                    content=content,
                    file_path=file_path,
                    block_type="react_component",
                    language=language
                )
                
                # Add output file path
                block_info["output_file"] = str(blocks_dir / f"{component_name}{file_path.suffix}")
                
                # Add to stats
                stats["file_blocks"][str(file_path)].append(block_info)
                stats["code_blocks"] = stats.get("code_blocks", 0) + 1
                
                # Write block to file
                with open(block_info["output_file"], "w") as f:
                    f.write(content)
                
                return 1  # Return 1 block extracted
        
        # Use tree-sitter for parsing
        parser = PARSERS[language]
        tree = parser.parse(content.encode('utf8'))
        
        if tree.root_node.has_error:
            logger.warning(f"Tree-sitter parse error in {file_path}")
            return _extract_generic_blocks(file_path, content, output_dir, stats, language)

        # Create queries for different code constructs
        lang = TREE_SITTER_LANGUAGES[language]
        
        # Query for React components first
        react_query = lang.query("""
            (program
              (export_statement
                (variable_declarator
                  name: (identifier) @component_name
                  value: (call_expression
                    function: (identifier) @wrapper
                    arguments: (arguments
                      (arrow_function) @component_body)))
                (#match? @component_name "^[A-Z]"))

            (program
              (export_statement
                (function_declaration
                  name: (identifier) @component_name
                  body: (statement_block) @component_body))
                (#match? @component_name "^[A-Z]"))

            (program
              (variable_declaration
                (variable_declarator
                  name: (identifier) @component_name
                  value: (arrow_function) @component_body))
                (#match? @component_name "^[A-Z]"))
                
            (program
              (function_declaration
                name: (identifier) @component_name
                body: (statement_block) @component_body)
                (#match? @component_name "^[A-Z]"))
        """)

        # Query for regular functions and classes
        code_query = lang.query("""
            (function_declaration
              name: (identifier) @func_name
              body: (statement_block) @func_body)

            (variable_declaration
              (variable_declarator
                name: (identifier) @func_name
                value: [(arrow_function) (function)] @func_body))

            (class_declaration
              name: (identifier) @class_name
              body: (class_body) @class_body)

            (method_definition
              name: (property_identifier) @method_name
              body: (statement_block) @method_body)
        """)

        blocks = []

        # Try to find React components first
        for match in react_query.matches(tree.root_node):
            for capture in match.captures:
                if capture[1] == "component_name":
                    name = capture[0].text.decode('utf8')
                    # For React components, we want to include the whole file content
                    # to preserve imports, hooks, and other dependencies
                    blocks.append(("react_component", name, content))
                    break

        # If no React components found, look for regular functions and classes
        if not blocks:
            for match in code_query.matches(tree.root_node):
                for capture in match.captures:
                    node = capture[0]
                    capture_name = capture[1]
                    
                    if capture_name in ["func_name", "class_name"]:
                        name = node.text.decode('utf8')
                        # Skip if this is a React component (starts with uppercase)
                        if not name[0].isupper():
                            # Find the corresponding body node
                            for body_capture in match.captures:
                                if body_capture[1] in ["func_body", "class_body"]:
                                    body_node = body_capture[0]
                                    block_content = content[node.start_byte:body_node.end_byte]
                                    block_type = "class" if capture_name == "class_name" else "function"
                                    blocks.append((block_type, name, block_content))
                                    break

        # Save blocks
        for block_type, block_name, block_content in blocks:
            # Create block metadata
            block_info = _create_code_block(
                name=block_name,
                content=block_content,
                file_path=file_path,
                block_type=block_type,
                language=language
            )
            
            # Add output file path
            block_info["output_file"] = str(blocks_dir / f"{block_name}{file_path.suffix}")
            
            # Add to stats
            stats["file_blocks"][str(file_path)].append(block_info)
            stats["code_blocks"] = stats.get("code_blocks", 0) + 1
            
            # Write block to file
            with open(block_info["output_file"], "w") as f:
                f.write(block_content)
            
        return len(blocks)
        
    except Exception as e:
        # If anything goes wrong, fall back to generic extraction
        logger.warning(f"Tree-sitter parsing failed for {file_path}, falling back to generic extraction: {e}")
        return _extract_generic_blocks(file_path, content, output_dir, stats, language)

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
    
    # Create block metadata using the new helper
    block_info = _create_code_block(
        name=block_name,
        content=block_content,
        file_path=source_file,
        block_type=block_type,
        language=language,
        start_line=start_line,
        end_line=end_line
    )
    
    # Add output file path
    block_info["output_file"] = str(block_file)
    
    # Add to stats
    stats["file_blocks"][str(source_file)].append(block_info)

# Output directories structure
OUTPUT_DIRS = {
    "CODE_FILES": "code",
    "DOC_FILES": "docs",
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs"
}

def _create_section_block(
    title: str,
    content: str,
    file_path: Path,
    breadcrumb: List[str],
    parent_uuid: Optional[str],
    current_level: int,
    section_index: int
) -> Dict[str, Any]:
    """Create a section block with basic reliable metadata.
    
    We focus on extracting only what we can determine reliably:
    1. Section hierarchy (levels, parent/child relationships)
    2. Basic content flags (presence of code blocks, tables, etc.)
    3. Location information
    
    The rest (focus areas, summary instructions, etc.) should be determined by the LLM
    which can do deeper content analysis.
    """
    section_uuid = str(uuid.uuid4())
    normalized_title = re.sub(r'[\s\-\.]+', '_', title)
    section_id = f"{file_path.stem}_{normalized_title.lower()}"
    
    # Format title for table of contents
    toc_indent = "    " * (current_level - 1)
    toc_format = f"{toc_indent}{title}"
    
    return {
        "uuid": section_uuid,
        "id": section_id,
        "type": "documentation",
        "language": "markdown",
        "title": normalized_title,
        "original_title": title,
        "content": content,
        "file_path": str(file_path),
        "breadcrumb": list(breadcrumb),
        "parent_uuid": parent_uuid,
        "child_uuids": [],
        "depth": len(breadcrumb) - 1,
        "header_depth": [i+1 for i in range(current_level)],
        "content_flags": {
            "has_code_block": "```" in content,
            "has_table": "|" in content,
            "has_links": "[" in content and "](" in content,
            "has_image": "![" in content,
            "has_list": bool(re.search(r'^\s*[-*+]\s', content, re.MULTILINE))
        },
        "toc_format": toc_format
    }

def _extract_markdown_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> int:
    """Extract sections from markdown based on headers."""
    try:
        # Initialize stats
        if "errors" not in stats:
            stats["errors"] = []
            
        # Verify file exists
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return 0
            
        # Create output directory
        blocks_dir = output_dir / "doc_blocks" / "markdown"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Update stats
        stats["languages"]["markdown"] = stats["languages"].get("markdown", 0) + 1
        stats["file_types"][file_path.suffix.lower()] = stats["file_types"].get(file_path.suffix.lower(), 0) + 1
        stats["total_files"] = stats.get("total_files", 0) + 1
        
        if str(file_path) not in stats["file_blocks"]:
            stats["file_blocks"][str(file_path)] = []
            
        # Track sections and hierarchy
        sections = []
        current_section = []
        current_title = None
        current_level = 0
        breadcrumb = []
        parent_stack = []
        section_index = 0
        
        # Process lines
        lines = content.splitlines()
        for line in lines:
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section and current_title:
                    section_content = "\n".join(current_section)
                    if section_content.strip():
                        parent_uuid = parent_stack[-1]["uuid"] if parent_stack else None
                        
                        section = _create_section_block(
                            current_title,
                            section_content,
                            file_path,
                            breadcrumb,
                            parent_uuid,
                            current_level,
                            section_index
                        )
                        
                        if parent_uuid:
                            for parent in parent_stack:
                                if parent["uuid"] == parent_uuid:
                                    parent["child_uuids"].append(section["uuid"])
                        
                        sections.append(section)
                        section_index += 1
                
                # Start new section
                current_level = len(header_match.group(1))
                current_title = header_match.group(2).strip()
                current_section = [line]
                
                # Update hierarchy tracking
                while parent_stack and len(parent_stack) >= current_level:
                    parent_stack.pop()
                    breadcrumb.pop()
                    
                breadcrumb.append(line.strip())
                if sections:
                    parent_stack.append(sections[-1])
            else:
                if current_title:
                    current_section.append(line)
        
        # Handle last section
        if current_section and current_title:
            section_content = "\n".join(current_section)
            if section_content.strip():
                parent_uuid = parent_stack[-1]["uuid"] if parent_stack else None
                
                section = _create_section_block(
                    current_title,
                    section_content,
                    file_path,
                    breadcrumb,
                    parent_uuid,
                    current_level,
                    section_index
                )
                
                if parent_uuid:
                    for parent in parent_stack:
                        if parent["uuid"] == parent_uuid:
                            parent["child_uuids"].append(section["uuid"])
                
                sections.append(section)
        
        # Write sections to files
        for section in sections:
            safe_title = re.sub(r'[^\w\-\.]', '_', section["title"])
            output_file = blocks_dir / f"{safe_title}.md"
            
            # Avoid filename collisions
            counter = 1
            while output_file.exists():
                output_file = blocks_dir / f"{safe_title}_{counter}.md"
                counter += 1
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(section["content"])
            section["output_file"] = str(output_file)
        
        # Update stats
        stats["file_blocks"][str(file_path)].extend(sections)
        stats["doc_blocks"] = stats.get("doc_blocks", 0) + len(sections)
        
        return len(sections)
        
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
    Handles edge cases and normalizes whitespace for reliable block separation.
    
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
        
        # Normalize line endings and whitespace
        content = content.replace('\r\n', '\n')  # Normalize line endings
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Collapse multiple newlines to double newlines
        content = content.strip()  # Remove leading/trailing whitespace
        
        if not content:  # Skip empty files
            return 0
            
        # Split by double newlines while preserving important whitespace
        raw_blocks = re.split(r'\n\s*\n', content)
        
        # Process each block
        block_count = 0
        current_line = 1  # Track line numbers
        
        for i, raw_block in enumerate(raw_blocks):
            block = raw_block.strip()
            if not block:  # Skip empty blocks
                continue
                
            # Calculate line numbers for this block
            block_lines = raw_block.count('\n') + 1
            end_line = current_line + block_lines - 1
            
            # Create block metadata using _create_code_block
            block_info = _create_code_block(
                name=f"{file_path.stem}_chunk_{i+1}",  # 1-based chunk numbering
                content=block,
                file_path=file_path,
                block_type="code",  # Changed from "text" to "code" for consistency
                language=language,
                start_line=current_line,
                end_line=end_line
            )
            
            # Add chunk index and output file
            block_info["chunk_index"] = i + 1
            block_info["output_file"] = str(blocks_dir / f"{file_path.stem}_chunk_{i+1}{ext}")
            
            # Add to stats
            stats["file_blocks"][str(file_path)].append(block_info)
            block_count += 1
            
            # Save block to file with original indentation preserved
            with open(block_info["output_file"], "w", encoding="utf-8") as f:
                f.write(raw_block)  # Use raw_block to preserve original formatting
            
            # Update line counter for next block
            current_line = end_line + 2  # +2 for the double newline separator
        
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
    content = block.get("content", "").strip()
    if not content:
        return False
    
    try:
        # Language-specific verification
        if language == "python":
            try:
                # Check if this is a class method by looking at the filename
                filename = block.get("file", "")
                if isinstance(filename, (str, Path)):
                    filename = str(filename)
                    if "." in filename:
                        parts = filename.split(".")
                        if len(parts) >= 2 and parts[-2].startswith("TestClass"):
                            # This is a class method, wrap it in a class
                            class_name = parts[-2]
                            # First dedent the content
                            content = textwrap.dedent(content)
                            # Then wrap in class
                            content = f"class {class_name}:\n" + "\n".join(f"    {line}" for line in content.splitlines())
                
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

def _create_code_block(
    name: str,
    content: str,
    file_path: Path,
    block_type: str = "code",
    language: str = None,
    start_line: int = None,
    end_line: int = None,
    imports: List[str] = None,
    referenced_types: List[str] = None,
    test_file: str = None
) -> Dict[str, Any]:
    """Create a code block with basic reliable metadata.
    
    We focus on extracting only what AST gives us reliably:
    1. Function definitions (including decorators)
    2. Class definitions
    3. Method definitions
    4. Import statements
    5. Type annotations
    """
    # Initialize with empty lists
    imports = imports or []
    referenced_types = referenced_types or []
    focus_areas = []
    prerequisites = [language.title()] if language else []
    
    # Extract what AST gives us reliably
    if language == "python":
        try:
            tree = ast.parse(content)
            has_decorators = False
            has_flask = False
            
            # Extract imports and analyze nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name)
                        if n.name.lower() == 'flask':
                            has_flask = True
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module)
                    if node.module and node.module.lower() == 'flask':
                        has_flask = True
                        # Check what's being imported from flask
                        for name_node in node.names:
                            if name_node.name == 'Flask':
                                has_flask = True
                # Extract type annotations
                elif isinstance(node, (ast.AnnAssign, ast.FunctionDef)):
                    if hasattr(node, 'annotation'):
                        if isinstance(node.annotation, ast.Name):
                            referenced_types.append(node.annotation.id)
                            prerequisites.append("Type hints")
                    # Check for decorators using AST's decorator_list
                    if isinstance(node, ast.FunctionDef) and node.decorator_list:
                        has_decorators = True
                        # Check for route decorators
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Call):
                                if isinstance(decorator.func, ast.Attribute):
                                    if decorator.func.attr == 'route':
                                        # Use exact string match "Web routes" for route decorators 
                                        focus_areas.append("Web routes")
                                        has_flask = True
                                        break
                elif isinstance(node, ast.ClassDef) and node.decorator_list:
                    has_decorators = True
                    
                # Check for Flask instantiation
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name) and node.value.func.id == 'Flask':
                                has_flask = True
            
            if has_decorators:
                prerequisites.append("Decorators")
                
            # Add Flask to prerequisites if detected
            if has_flask or 'flask' in content.lower() or 'Flask(' in content:
                prerequisites.append("Flask")
                focus_areas.append("Web development")
                
        except SyntaxError:
            # If AST parsing fails, just continue with what we have
            pass
    
    # Add basic focus areas based on AST-derived block type
    if block_type == "function":
        focus_areas.append("Function implementation")
    elif block_type in ["class", "method"]:
        focus_areas.append("Object-oriented programming")
    
    # Remove duplicates while preserving order
    focus_areas = list(dict.fromkeys(focus_areas))
    prerequisites = list(dict.fromkeys(prerequisites))
    
    return {
        "uuid": str(uuid.uuid4()),
        "id": f"{file_path.stem}_{name.lower()}",
        "type": "code",
        "language": language,
        "title": name,
        "content": content,
        "file_path": str(file_path),
        "breadcrumb": [str(file_path), name],
        "parent_uuid": None,
        "child_uuids": [],
        "dependencies": {
            "imports": imports,
            "referenced_types": referenced_types
        },
        "test_coverage": {
            "test_file": test_file,
            "coverage_percentage": 0
        },
        "version_history": {
            "last_modified": datetime.now().isoformat()
        },
        "qa_generation": {
            "difficulty_levels": ["intermediate"] if prerequisites else ["beginner"],
            "knowledge_prerequisites": prerequisites,
            "focus_areas": focus_areas,
            "qa_examples": []
        },
        "start_line": start_line,
        "end_line": end_line
    }

# ... rest of the existing code ...
