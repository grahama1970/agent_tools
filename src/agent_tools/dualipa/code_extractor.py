"""
Code extraction module for DuaLipa.

This module provides functions to extract code from repositories,
parse files, and generate datasets for LoRA fine-tuning.

Official Documentation References:
- gitpython: https://gitpython.readthedocs.io/en/stable/tutorial.html
- loguru: https://loguru.readthedocs.io/en/stable/
- tqdm: https://tqdm.github.io/docs/
- pathlib: https://docs.python.org/3/library/pathlib.html
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-python: https://github.com/tree-sitter/tree-sitter-python
- tree-sitter-javascript: https://github.com/tree-sitter/tree-sitter-javascript
- tree-sitter-typescript: https://github.com/tree-sitter/tree-sitter-typescript
"""

import os
import json 
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple, Any
import re
import time
from loguru import logger
from tqdm import tqdm
import shutil
import ast
from collections import defaultdict
import textwrap
import argparse
from datetime import datetime

# Import local modules
try:
    # When running as part of the package
    from .github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
    from .language_detection import detect_language
    from .markdown_parser import extract_code_blocks
    from .utils import format_string
except ImportError:
    # Handle case where this module is run standalone
    logger.warning("Running in standalone mode, using direct imports")
    from github_utils import clone_github_repo, is_github_url, parse_github_url, get_clone_url, download_github_repo
    from language_detection import detect_language
    from markdown_parser import extract_code_blocks
    from utils import format_string

# Try to import tree-sitter for advanced code extraction
TREE_SITTER_AVAILABLE = False
TREE_SITTER_LANGUAGES = {}
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
    
    # Try to import language modules
    try:
        import tree_sitter_python
        TREE_SITTER_LANGUAGES["python"] = tree_sitter_python
    except ImportError:
        logger.debug("tree-sitter-python not available")
        
    try:
        import tree_sitter_javascript
        TREE_SITTER_LANGUAGES["javascript"] = tree_sitter_javascript
        TREE_SITTER_LANGUAGES["js"] = tree_sitter_javascript
    except ImportError:
        logger.debug("tree-sitter-javascript not available")
        
    try:
        import tree_sitter_typescript
        TREE_SITTER_LANGUAGES["typescript"] = tree_sitter_typescript
        TREE_SITTER_LANGUAGES["ts"] = tree_sitter_typescript
    except ImportError:
        logger.debug("tree-sitter-typescript not available")
        
    # Additional languages
    try:
        import tree_sitter_go
        TREE_SITTER_LANGUAGES["go"] = tree_sitter_go
    except ImportError:
        logger.debug("tree-sitter-go not available")
        
    try:
        import tree_sitter_rust
        TREE_SITTER_LANGUAGES["rust"] = tree_sitter_rust
    except ImportError:
        logger.debug("tree-sitter-rust not available")
        
    try:
        import tree_sitter_cpp
        TREE_SITTER_LANGUAGES["cpp"] = tree_sitter_cpp
        TREE_SITTER_LANGUAGES["c++"] = tree_sitter_cpp
    except ImportError:
        logger.debug("tree-sitter-cpp not available")
        
    try:
        import tree_sitter_java
        TREE_SITTER_LANGUAGES["java"] = tree_sitter_java
    except ImportError:
        logger.debug("tree-sitter-java not available")
        
    try:
        import tree_sitter_ruby
        TREE_SITTER_LANGUAGES["ruby"] = tree_sitter_ruby
    except ImportError:
        logger.debug("tree-sitter-ruby not available")
        
    try:
        import tree_sitter_bash
        TREE_SITTER_LANGUAGES["bash"] = tree_sitter_bash
        TREE_SITTER_LANGUAGES["sh"] = tree_sitter_bash
    except ImportError:
        logger.debug("tree-sitter-bash not available")
        
    logger.info(f"Tree-sitter available with {len(TREE_SITTER_LANGUAGES)} language grammars")
except ImportError:
    logger.info("Tree-sitter not available, using fallback extraction methods")

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

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

# Output directories structure
OUTPUT_DIRS = {
    "CODE_FILES": "code",
    "DOC_FILES": "docs",
    "CODE_BLOCKS": "blocks/code",
    "DOC_BLOCKS": "blocks/docs"
}

def _is_code_file(filename: str) -> bool:
    """Check if a file is a code file based on its extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in CODE_FILE_EXTENSIONS


def _is_documentation_file(filename: str) -> bool:
    """Check if a file is a documentation file based on its extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in DOCUMENTATION_FILE_EXTENSIONS


def _should_process_file(
    file_path: str, 
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> bool:
    """
    Determine if a file should be processed based on include/exclude patterns.
    
    Args:
        file_path: Path to the file
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        Boolean indicating whether the file should be processed
    """
    import fnmatch
    
    # Get the filename
    filename = os.path.basename(file_path)
    
    # Skip ignored files
    if filename in IGNORED_FILES:
        return False
    
    # Check exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return False
    
    # Check include patterns
    if include_patterns:
        for pattern in include_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        # If include patterns are specified but none match, skip this file
        return False
    
    # If no include patterns, process the file
    return True


# ---------- Block Extraction Functions ----------

def _extract_with_tree_sitter(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> Optional[int]:
    """
    Extract code blocks using tree-sitter parser.
    
    Args:
        file_path: Path to the code file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier
        
    Returns:
        Number of blocks extracted or None if extraction failed
    """
    if not TREE_SITTER_AVAILABLE or language not in TREE_SITTER_LANGUAGES:
        return None
    
    try:
        # Create tree-sitter parser
        parser = tree_sitter.Parser()
        
        # Load language
        lang_module = TREE_SITTER_LANGUAGES[language]
        try:
            ts_language = tree_sitter.Language(lang_module.language())
            parser.language = ts_language
        except Exception as e:
            logger.debug(f"Could not load tree-sitter language for {language}: {e}")
            return None
            
        # Parse the code
        tree = parser.parse(bytes(content, 'utf8'))
        root = tree.root_node
        
        # Create output directory for blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Define node types to extract based on language
        function_types = []
        class_types = []
        interface_types = []
        
        if language in ["python", "py"]:
            function_types = ["function_definition", "async_function_definition"]
            class_types = ["class_definition"]
        elif language in ["javascript", "js"]:
            function_types = ["function_declaration", "function", "arrow_function", "method_definition"]
            class_types = ["class_declaration"]
        elif language in ["typescript", "ts"]:
            function_types = ["function_declaration", "function", "arrow_function", "method_definition"]
            class_types = ["class_declaration"]
            interface_types = ["interface_declaration", "type_alias_declaration"]
        elif language == "go":
            function_types = ["function_declaration"]
            class_types = ["type_declaration"]
        elif language == "rust":
            function_types = ["function_item"]
            class_types = ["struct_item", "impl_item"]
        elif language in ["cpp", "c++"]:
            function_types = ["function_definition"]
            class_types = ["class_specifier"]
        elif language == "java":
            function_types = ["method_declaration"]
            class_types = ["class_declaration"]
        elif language == "ruby":
            function_types = ["method"]
            class_types = ["class"]
        elif language in ["bash", "sh"]:
            function_types = ["function_definition"]
            
        # Extract blocks
        block_count = 0
        lines = content.splitlines()
        
        # Helper function to extract node text
        def get_node_text(node):
            start_point, end_point = node.start_point, node.end_point
            start_row, start_col = start_point
            end_row, end_col = end_point
            
            if start_row == end_row:
                return lines[start_row][start_col:end_col]
            
            text_lines = [lines[start_row][start_col:]]
            for row in range(start_row + 1, end_row):
                text_lines.append(lines[row])
            text_lines.append(lines[end_row][:end_col])
            
            return "\n".join(text_lines)
            
        # Helper function to get node name
        def get_node_name(node, node_type):
            name_node = None
            
            # Different languages have different ways to get the name
            if node_type in function_types + class_types + interface_types:
                name_node = node.child_by_field_name('name')
            
            if name_node:
                return name_node.text.decode('utf8')
            return f"unnamed_{block_count}"
        
        # Process all child nodes
        def process_nodes(parent_node):
            nonlocal block_count
            
            for node in parent_node.children:
                # Skip comments and imports
                if node.type.endswith('comment') or node.type.endswith('import'):
                    continue
                    
                # Check if this node is a function or class declaration
                node_text = None
                node_type = node.type
                node_name = None
                
                if node_type in function_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "function"
                elif node_type in class_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "class"
                elif node_type in interface_types:
                    node_text = get_node_text(node)
                    node_name = get_node_name(node, node_type)
                    decl_type = "interface"
                    
                if node_text:
                    block_count += 1
                    
                    # Create metadata header based on language
                    if language in ["python", "py"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "go", "rust"]:
                        header = f"// Original file: {file_path}\n"
                        header += f"// Block type: {decl_type}\n"
                        header += f"// Name: {node_name}\n\n"
                    elif language in ["ruby"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    elif language in ["bash", "sh"]:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                    else:
                        header = f"# Original file: {file_path}\n"
                        header += f"# Block type: {decl_type}\n"
                        header += f"# Name: {node_name}\n\n"
                        
                    # Combine header and code
                    block_content = header + node_text
                    
                    # Save to file
                    file_ext = file_path.suffix if file_path.suffix else f".{language}"
                    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', node_name)
                    output_file = blocks_dir / f"{file_path.stem}_{safe_name}_{block_count}{file_ext}"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(block_content)
                
                # Process nested declarations (for classes with methods, etc.)
                if len(node.children) > 0:
                    process_nodes(node)
        
        # Process all nodes
        process_nodes(root)
        
        # Update statistics
        stats["code_blocks"] += block_count
        
        return block_count
        
    except Exception as e:
        logger.debug(f"Error using tree-sitter for {language}: {e}")
        return None

def _extract_python_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any]
) -> None:
    """
    Extract functions and classes from Python code using AST.
    
    Args:
        file_path: Path to the Python file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
    """
    # Try tree-sitter first if available
    if TREE_SITTER_AVAILABLE:
        blocks = _extract_with_tree_sitter(file_path, content, output_dir, stats, "python")
        if blocks is not None and blocks > 0:
            logger.debug(f"Extracted {blocks} Python blocks with tree-sitter from {file_path}")
            return blocks
    
    try:
        # Parse the Python code
        tree = ast.parse(content)
        
        # Create output directory for Python blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / "python"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract blocks from the AST
        block_count = 0
        lines = content.splitlines()
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                block_count += 1
                
                # Get line numbers
                start = node.lineno - 1
                end = getattr(node, "end_lineno", None)
                if end is None:
                    # For Python < 3.8 that doesn't have end_lineno
                    # Find the last line by iterating through children
                    end = start
                    for child in ast.walk(node):
                        if hasattr(child, "lineno"):
                            end = max(end, getattr(child, "lineno", 0))
                
                # Extract the block code
                block_lines = lines[start:end]
                block_code = "\n".join(block_lines)
                
                # Get block metadata
                docstring = ast.get_docstring(node) or "No docstring available."
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                
                # Create block header with metadata
                header = f"# Original file: {file_path}\n"
                header += f"# Block type: {node_type}\n"
                header += f"# Name: {node.name}\n"
                header += f"# Docstring: {docstring}\n"
                
                # Add parameter info for functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [arg.arg for arg in node.args.args]
                    param_info = f"# Parameters: {', '.join(params)}" if params else "# No parameters."
                    header += f"{param_info}\n"
                
                header += "\n"
                
                # Combine header and code
                block_content = header + block_code
                
                # Save the block to a file
                output_file = blocks_dir / f"{file_path.stem}_{node.name}_{block_count}.py"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(block_content)
                
                # Update statistics
                stats["code_blocks"] += 1
                
        # If no blocks were extracted, log a message
        if block_count == 0:
            logger.debug(f"No blocks extracted from Python file: {file_path}")
            
        return block_count
        
    except SyntaxError:
        # Handle Python files that can't be parsed with AST
        logger.warning(f"SyntaxError parsing Python file: {file_path}, falling back to generic splitting")
        return _extract_generic_blocks(file_path, content, output_dir, stats, "python")
        
    except Exception as e:
        error_msg = f"Error extracting Python blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

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
        stats: Statistics dictionary to update
        
    Returns:
        Number of blocks extracted
    """
    try:
        # Create output directory for markdown blocks
        blocks_dir = output_dir / OUTPUT_DIRS["DOC_BLOCKS"] / "markdown"
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Split on headers (lines starting with #)
        sections = re.split(r"(?m)(?=^# )", content)
        
        # Process each section
        block_count = 0
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            
            # Get the section header for naming
            header_match = re.match(r"^(# .+)", section)
            if header_match:
                header_line = header_match.group(1)
                section_title = re.sub(r"\W+", "_", header_line.strip("# ").strip()).strip("_")
            else:
                section_title = f"section{i}"
                
            # Create metadata header
            block_header = f"<!-- Original file: {file_path} -->\n"
            block_header += f"<!-- Section: {section_title} -->\n\n"
                
            # Combine header and content
            block_content = block_header + section
            
            # Save the section to a file
            output_file = blocks_dir / f"{file_path.stem}_{section_title}_{i}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["doc_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting markdown blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

def _extract_js_ts_blocks(
    file_path: Path, 
    content: str, 
    output_dir: Path, 
    stats: Dict[str, Any],
    language: str
) -> int:
    """
    Extract functions and classes from JavaScript/TypeScript.
    Tries tree-sitter first, then falls back to regex approach.
    
    Args:
        file_path: Path to the JS/TS file
        content: Content of the file
        output_dir: Output directory to save blocks
        stats: Statistics dictionary to update
        language: Language identifier (js, ts, etc.)
        
    Returns:
        Number of blocks extracted
    """
    # Try tree-sitter first if available
    if TREE_SITTER_AVAILABLE:
        blocks = _extract_with_tree_sitter(file_path, content, output_dir, stats, language)
        if blocks is not None and blocks > 0:
            logger.debug(f"Extracted {blocks} {language} blocks with tree-sitter from {file_path}")
            return blocks
    
    try:
        # Create output directory for JS/TS blocks
        blocks_dir = output_dir / OUTPUT_DIRS["CODE_BLOCKS"] / language
        blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # Using regex to find function and class definitions
        # This is a simplified approach - a full parser would be better
        function_pattern = r"(function\s+\w+\s*\([^)]*\)\s*{[\s\S]*?})"
        arrow_func_pattern = r"(const\s+\w+\s*=\s*(?:\([^)]*\)|[^=]*)\s*=>\s*{[\s\S]*?})"
        class_pattern = r"(class\s+\w+[\s\S]*?{[\s\S]*?})"
        
        # Combine patterns
        patterns = [function_pattern, arrow_func_pattern, class_pattern]
        
        # Extract blocks
        blocks = []
        for pattern in patterns:
            blocks.extend(re.findall(pattern, content))
            
        # If no blocks found with regex, fall back to splitting by double newlines
        if not blocks:
            return _extract_generic_blocks(file_path, content, output_dir, stats, language)
            
        # Process each block
        block_count = 0
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
                
            # Create metadata header
            block_header = f"// Original file: {file_path}\n"
            block_header += f"// Block index: {i}\n\n"
            
            # Combine header and block
            block_content = block_header + block
            
            # Save the block to a file
            output_file = blocks_dir / f"{file_path.stem}_block_{i}.{language}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["code_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting JS/TS blocks from {file_path}: {str(e)}"
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
        
        # Split by double newlines
        blocks = re.split(r"\n\s*\n", content)
        
        # Process each block
        block_count = 0
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block or len(block.split('\n')) < 3:  # Skip very small blocks
                continue
                
            # Create metadata header
            if language in ["python", "py"]:
                block_header = f"# Original file: {file_path}\n# Block index: {i}\n\n"
            elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "cs", "go", "rust", "swift", "kotlin"]:
                block_header = f"// Original file: {file_path}\n// Block index: {i}\n\n"
            elif language in ["html", "xml"]:
                block_header = f"<!-- Original file: {file_path} -->\n<!-- Block index: {i} -->\n\n"
            else:
                block_header = f"# Original file: {file_path}\n# Block index: {i}\n\n"
                
            # Combine header and block
            block_content = block_header + block
            
            # Save the block to a file
            file_ext = file_path.suffix if file_path.suffix else f".{language}"
            output_file = blocks_dir / f"{file_path.stem}_chunk_{i}{file_ext}"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(block_content)
                
            block_count += 1
            
        # Update statistics
        stats["code_blocks"] += block_count
        
        return block_count
            
    except Exception as e:
        error_msg = f"Error extracting generic blocks from {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        return 0

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
        language: Language of the code file (if known)
        extract_blocks: Whether to extract code blocks from the file
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Detect language if not provided
        if not language:
            language = detect_language(file_path)
        
        # Update statistics
        stats["code_files"] += 1
        stats["languages"][language] = stats["languages"].get(language, 0) + 1
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for code files
        code_dir = output_dir / OUTPUT_DIRS["CODE_FILES"]
        code_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory for the language if it doesn't exist
        lang_dir = code_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        comment_marker = ""
        if language in ["python", "py"]:
            comment_marker = "# "
        elif language in ["javascript", "typescript", "js", "ts", "java", "c", "cpp", "cs", "go", "rust", "swift", "kotlin"]:
            comment_marker = "// "
        elif language in ["html", "xml"]:
            comment_marker = "<!-- "
        else:
            comment_marker = "# "  # Default to # for unknown languages
            
        path_comment = f"{comment_marker}Original file path: {file_path}\n\n"
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
            blocks_extracted = 0
            if language in ["python", "py"]:
                blocks_extracted = _extract_python_blocks(file_path, content, output_dir, stats)
            elif language in ["javascript", "js"]:
                blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, "javascript")
            elif language in ["typescript", "ts"]:
                blocks_extracted = _extract_js_ts_blocks(file_path, content, output_dir, stats, "typescript")
            else:
                # Try tree-sitter for other languages first
                if TREE_SITTER_AVAILABLE and language in TREE_SITTER_LANGUAGES:
                    tree_sitter_blocks = _extract_with_tree_sitter(file_path, content, output_dir, stats, language)
                    if tree_sitter_blocks is not None and tree_sitter_blocks > 0:
                        blocks_extracted = tree_sitter_blocks
                    else:
                        # Fall back to generic extraction if tree-sitter failed
                        blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, language)
                else:
                    # Use generic block extraction for unsupported languages
                    blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, language)
                
            logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
        
        logger.debug(f"Processed code file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing code file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)


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
        
        # Get file extension
        _, ext = os.path.splitext(file_path.name.lower())
        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
        
        # Create output directory for documentation files
        docs_dir = output_dir / OUTPUT_DIRS["DOC_FILES"]
        docs_dir.mkdir(exist_ok=True)
        
        # Create a subdirectory based on the file type (e.g., markdown)
        doc_type = ext.replace('.', '')
        if not doc_type:
            doc_type = "text"
        type_dir = docs_dir / doc_type
        type_dir.mkdir(exist_ok=True)
        
        # Add a comment with the original file path at the beginning of the file
        path_comment = ""
        if ext.lower() in ['.md', '.markdown']:
            path_comment = f"<!-- Original file path: {file_path} -->\n\n"
        else:
            path_comment = f"# Original file path: {file_path}\n\n"
            
        content_with_path = path_comment + content
        
        # Save the file to the output directory with a unique name
        # Add the original path as part of the filename to maintain context
        path_hash = hash(str(file_path)) % 10000
        rel_path = str(file_path).replace('/', '_').replace('\\', '_')
        output_file = type_dir / f"{rel_path}_{path_hash:04d}{ext}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content_with_path)
        
        # Extract blocks if requested
        if extract_blocks:
            blocks_extracted = 0
            if ext.lower() in ['.md', '.markdown']:
                blocks_extracted = _extract_markdown_blocks(file_path, content, output_dir, stats)
            else:
                # For other doc types, use a simple splitting strategy
                blocks_extracted = _extract_generic_blocks(file_path, content, output_dir, stats, doc_type)
                
            logger.debug(f"Extracted {blocks_extracted} blocks from {file_path}")
        
        logger.debug(f"Processed documentation file: {file_path} -> {output_file}")
        
    except Exception as e:
        error_msg = f"Error processing documentation file {file_path}: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)


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
    # Check if source is a GitHub URL
    from agent_tools.dualipa.github_utils import is_github_url, download_github_repo
    is_github_repo = is_github_url(source)
    
    # Set up output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(tempfile.gettempdir(), f"extracted_{timestamp}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {
        "source": source,
        "output_path": str(output_dir),
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "total_files": 0,
        "documentation_files": 0,
        "code_files": 0,
        "code_blocks": 0,
        "skipped_files": 0,
        "error_files": 0,
        "languages": {},
        "file_types": {},
        "errors": []
    }
    
    repo_dir = None
    main_progress = tqdm(total=3, desc="Repository extraction", unit="steps")
    
    try:
        # Step 1: Handle source (clone if GitHub URL, otherwise use local path)
        main_progress.set_description("Step 1: Preparing repository")
        if is_github_repo:
            try:
                from agent_tools.dualipa.github_utils import parse_github_url
                repo_info = parse_github_url(source)
                repo_name = repo_info.get("repo", "unknown")
                logger.info(f"Cloning GitHub repository: {source}")
                
                repo_dir = output_dir / "repo"
                repo_dir.mkdir(exist_ok=True)
                
                with tqdm(total=1, desc=f"Cloning {repo_name}", leave=False) as clone_progress:
                    # Use the improved download_github_repo function which handles errors better
                    download_github_repo(source, str(repo_dir))
                    clone_progress.update(1)
                    
                stats["repo_info"] = repo_info
                logger.info(f"Repository cloned successfully to {repo_dir}")
            except Exception as e:
                error_msg = f"Error cloning repository {source}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                main_progress.close()
                raise ValueError(error_msg)
        else:
            # Local repository
            repo_dir = Path(source)
            if not repo_dir.exists():
                error_msg = f"Repository path does not exist: {source}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                main_progress.close()
                raise ValueError(error_msg)
            
            logger.info(f"Using local repository: {repo_dir}")
        
        main_progress.update(1)
        
        # Step 2: Process files in the repository
        main_progress.set_description("Step 2: Processing files")
        
        # Get all files in the repository
        file_paths = []
        for root, dirs, files in os.walk(repo_dir):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
            
            # Add files to the list
            for file in files:
                file_path = Path(os.path.join(root, file))
                if _should_process_file(str(file_path), include_patterns, exclude_patterns):
                    file_paths.append(file_path)
                else:
                    stats["skipped_files"] += 1
        
        # Set up a progress bar for file processing
        with tqdm(total=len(file_paths), desc="Processing files", unit="files") as file_progress:
            for file_path in file_paths:
                # Check if we've reached the maximum number of files
                if stats["total_files"] >= max_files:
                    logger.info(f"Reached maximum number of files ({max_files}), stopping.")
                    break
                
                # Process the file based on its type
                try:
                    stats["total_files"] += 1
                    
                    if _is_code_file(file_path.name) and extract_code:
                        language = detect_language(file_path)
                        _process_code_file(file_path, output_dir, stats, language, extract_blocks)
                        file_progress.set_postfix_str(f"Processing {language} file: {file_path.name}")
                        file_progress.update(1)
                    elif _is_documentation_file(file_path.name) and extract_documentation:
                        _process_documentation_file(file_path, output_dir, stats, extract_blocks)
                        file_progress.set_postfix_str(f"Processing doc file: {file_path.name}")
                        file_progress.update(1)
                    else:
                        stats["skipped_files"] += 1
                        file_progress.set_postfix_str(f"Skipping unsupported file: {file_path.name}")
                        file_progress.update(1)
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    stats["error_files"] += 1
                    file_progress.set_postfix_str(f"Error: {file_path.name}")
                    file_progress.update(1)
        
        main_progress.update(1)
        
        # Step 3: Generate summary and cleanup
        main_progress.set_description("Step 3: Generating summary")
        
        # Calculate success rate
        total_processed = stats["total_files"] - stats["skipped_files"]
        if total_processed > 0:
            success_rate = ((total_processed - stats["error_files"]) / total_processed) * 100
            stats["success_rate"] = f"{success_rate:.2f}%"
        
        # Record end time and duration
        stats["end_time"] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(stats["start_time"])
        end_time = datetime.fromisoformat(stats["end_time"])
        stats["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # Save statistics to a JSON file in the output directory
        with open(output_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Extraction completed. Statistics saved to {output_dir / 'extraction_stats.json'}")
        main_progress.update(1)
        
        return stats
    except Exception as e:
        error_msg = f"Error during extraction: {str(e)}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        
        # Save statistics even if an error occurred
        try:
            # Record end time and duration
            stats["end_time"] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(stats["start_time"])
            end_time = datetime.fromisoformat(stats["end_time"])
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            
            with open(output_dir / "extraction_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Extraction failed but statistics saved to {output_dir / 'extraction_stats.json'}")
        except Exception as save_error:
            logger.error(f"Failed to save statistics: {save_error}")
        
        return stats
    finally:
        main_progress.close()

def move_extracted_files(source_dir: str, target_dir: str = None) -> int:
    """
    Move extracted files from a source directory to the target directory.
    
    Args:
        source_dir: Path to the source directory containing extracted files
        target_dir: Path to the target directory (defaults to data/files if None)
        
    Returns:
        Number of files moved
    """
    # If no target directory specified, use the default data/files path
    if target_dir is None:
        # Get the path to the dualipa module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(module_dir, "data", "files")
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Track files moved
    files_moved = 0
    
    # Copy the directory structure and files
    for root, dirs, files in os.walk(source_path):
        rel_path = os.path.relpath(root, source_path)
        if rel_path == '.':
            rel_path = ''
        
        # Create corresponding directories in the target
        target_dir_path = target_path / rel_path
        target_dir_path.mkdir(exist_ok=True)
        
        # Copy files
        for file in files:
            source_file = Path(root) / file
            target_file = target_dir_path / file
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            files_moved += 1
            
    logger.info(f"Moved {files_moved} files from {source_dir} to {target_dir}")
    return files_moved

def load_template(filename: str) -> str:
    """Load a template file from the resources directory.
    
    Args:
        filename: Name of the template file to load
        
    Returns:
        String content of the template file
    """
    template_dir = Path(__file__).parent / "resources" / "templates"
    template_path = template_dir / filename
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading template {filename}: {str(e)}")
        return f"Error: Could not load template {filename}"

def demo_code_extractor() -> None:
    """Demonstrate the code extractor functionality with examples.
    
    This function shows how to use the main components of the code extractor:
    1. Extracting code from a Python file
    2. Extracting code blocks from a Markdown file
    3. Extracting a repository (using a local example)
    4. Using the default data directory
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("Code Extractor Demo")
        logger.info("===================")
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. Create a sample Python file
            logger.info("\n1. Extracting code from a Python file:")
            python_file = temp_path / "sample.py"
            with open(python_file, "w") as f:
                f.write(load_template("sample_python.py"))
            
            # 2. Create a sample Markdown file
            logger.info("\n2. Extracting code blocks from a Markdown file:")
            md_file = temp_path / "sample.md"
            with open(md_file, "w") as f:
                f.write(load_template("sample_markdown.md"))
            
            # 3. Create a simple repo structure
            logger.info("\n3. Extracting a repository:")
            repo_dir = temp_path / "sample_repo"
            repo_dir.mkdir(exist_ok=True)
            (repo_dir / "src").mkdir(exist_ok=True)
            
            # Create some files in the repo
            with open(repo_dir / "src" / "main.py", "w") as f:
                f.write(load_template("repo_main.py"))
                
            with open(repo_dir / "README.md", "w") as f:
                f.write(load_template("repo_readme.md"))
            
            # 4. Extract each example
            output_dir = temp_path / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Extract Python file
            python_stats = extract_repository(
                str(python_file),
                output_path=str(output_dir / "python_example"),
                extract_documentation=False
            )
            logger.info(f"Python extraction stats: {json.dumps(python_stats, indent=2)}")
            
            # Extract Markdown file
            md_stats = extract_repository(
                str(md_file),
                output_path=str(output_dir / "markdown_example"),
                extract_code=False
            )
            logger.info(f"Markdown extraction stats: {json.dumps(md_stats, indent=2)}")
            
            # Extract repository
            repo_stats = extract_repository(
                str(repo_dir),
                output_path=str(output_dir / "repo_example"),
                max_files=10
            )
            logger.info(f"Repository extraction stats: {json.dumps(repo_stats, indent=2)}")
            
            logger.info("\nDemo completed successfully!")
            
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function for command-line usage."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Code Extractor - Extract code and documentation from files and repositories")
    parser.add_argument("--verify", action="store_true", help="Verify basic functionality")
    parser.add_argument("--source", help="Source file or directory (for verification)")
    parser.add_argument("--output", help="Output directory (for verification)")
    parser.add_argument("--demo", action="store_true", help="Run the full demonstration with examples")
    args = parser.parse_args()
    
    # If demo is requested, run the demo and exit
    if args.demo:
        demo_code_extractor()
        return
    
    # If verification is requested, run it and exit
    if args.verify:
        verify_basic_functionality(args.source, args.output)
        return
    
    # If no arguments were provided, run a quick self-test
    if len(sys.argv) == 1:
        print("Running quick self-test to verify basic functionality...")
        # Use this file itself as the source
        verify_basic_functionality(__file__, None)
        return
    
    # Otherwise, just print a simple message
    message = format_string("Code Extractor module loaded successfully!")
    print(message)
    print("\nTo verify functionality, run:")
    print("  python -m agent_tools.dualipa.code_extractor --verify")
    print("  python code_extractor.py --verify")
    print("\nTo run the full demo with examples:")
    print("  python -m agent_tools.dualipa.code_extractor --demo")
    print("  python code_extractor.py --demo")

def verify_basic_functionality(source_path=None, output_dir=None):
    """
    Verify the basic functionality of the code extractor.
    
    This function demonstrates the most basic usage of the code extractor.
    It extracts code from the provided source path (file or directory)
    and prints the extraction statistics.
    
    Args:
        source_path: Path to the source file or directory.
                    If None, uses this file (code_extractor.py) itself.
        output_dir: Directory to save extracted files.
                    If None, uses a temporary directory.
    
    Returns:
        None
    """
    import tempfile
    from pathlib import Path
    
    # If no source path provided, use this file
    if source_path is None:
        source_path = __file__
    
    # If no output directory provided, create a temporary one
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    
    try:
        print(f"\n{'='*80}")
        print(f" VERIFYING CODE EXTRACTOR FUNCTIONALITY ".center(80, '='))
        print(f"{'='*80}\n")
        
        print(f"Source: {source_path}")
        print(f"Output directory: {output_dir}")
        
        # Extract code from the source path
        stats = extract_repository(
            source=source_path,
            output_path=output_dir,
            max_files=10,  # Limit to 10 files for quick verification
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Print extraction statistics
        print("\nExtraction completed successfully!")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Code files: {stats['code_files']}")
        print(f"Documentation files: {stats['documentation_files']}")
        print(f"Code blocks extracted: {stats['code_blocks']}")
        print(f"Documentation blocks extracted: {stats['doc_blocks']}")
        
        # List the top-level directories in the output directory
        output_path = Path(output_dir)
        print("\nOutput directory structure:")
        for item in output_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}/")
                # List a few items in each subdirectory
                count = 0
                for subitem in item.iterdir():
                    if count < 3:  # Show only first 3 items
                        print(f"    - {subitem.name}")
                        count += 1
                    else:
                        remaining = sum(1 for _ in item.iterdir()) - 3
                        if remaining > 0:
                            print(f"    - ... and {remaining} more items")
                        break
            else:
                print(f"  - {item.name}")
        
        # Check for errors
        if stats["errors"]:
            print("\nWarning: Some errors occurred during extraction:")
            for i, error in enumerate(stats["errors"][:5]):  # Show only first 5 errors
                print(f"  {i+1}. {error}")
            if len(stats["errors"]) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more errors")
        else:
            print("\nNo errors occurred during extraction.")
        
        print(f"\n{'='*80}")
        print(f" VERIFICATION COMPLETED SUCCESSFULLY ".center(80, '='))
        print(f"{'='*80}\n")
        
    finally:
        # Clean up temporary directory if we created one
        if temp_dir:
            temp_dir.cleanup()

def run_test(source_path=None, output_dir=None, max_files=5):
    """
    Simple function to run a basic test for external scripts.
    
    This is a stripped-down version of verify_basic_functionality that can
    be imported and called by test scripts to verify that code extraction works.
    
    Args:
        source_path: Path to the source file or directory
        output_dir: Directory to save extracted files
        max_files: Maximum number of files to process
        
    Returns:
        Dictionary with statistics from the extraction process
    """
    # If no source path provided, use this file
    if source_path is None:
        source_path = __file__
    
    # Extract code from the source path
    stats = extract_repository(
        source=source_path,
        output_path=output_dir,
        max_files=max_files,
        extract_documentation=True,
        extract_code=True,
        extract_blocks=True
    )
    
    return stats


if __name__ == "__main__":
    main()
    
    # Uncomment the line below to verify basic functionality
    # verify_basic_functionality()
