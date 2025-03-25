"""
JavaScript/TypeScript code extraction for DuaLipa.

This module handles extraction of JavaScript and TypeScript code using a cascading parser approach:
1. First attempts tree-sitter parsing for precise AST-based extraction
2. Falls back to regex-based generic extraction when tree-sitter fails
3. Implements React component detection and extraction

Key Features:
1. Tree-sitter based parsing with fallback mechanisms
2. React component extraction
3. TypeScript type handling
4. Class and method extraction
5. Cascading parser approach for reliability
6. JSDoc and comment extraction for docstrings (/** ... */ and // style)

Dependencies:
- tree-sitter: For JS/TS parsing (https://tree-sitter.github.io/tree-sitter/)
- tree_sitter_languages: For language support (https://github.com/grantjenks/py-tree-sitter-languages)
- loguru: For logging (https://github.com/Delgan/loguru)
- textwrap: For text formatting (https://docs.python.org/3/library/textwrap.html)

Documentation Links:
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Tree-sitter Languages: https://py-tree-sitter-languages.readthedocs.io/
- Loguru: https://loguru.readthedocs.io/
- React Components: https://reactjs.org/docs/components-and-props.html
- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
- JSDoc: https://jsdoc.app/

Input/Output Specifications:

parse_tree_sitter(content: str, language: str = "javascript") -> Optional[Any]:
    Input:
        - content: Source code to parse
        - language: 'javascript' or 'typescript'
    Output:
        - Tree-sitter tree if successful, None if parsing fails
    Example Input:
        content = "function add(a, b) { return a + b; }"
        language = "javascript"
    Example Output:
        <tree-sitter Tree object representing the AST>

extract_js_ts_blocks(file_path: str, output_dir: Path = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    Input:
        - file_path: Path to JS/TS file
        - output_dir: Optional output directory for extracted blocks
    Output:
        - Tuple containing:
        - List of dictionaries containing extracted blocks
        - Statistics dictionary
    Example Input:
        file_path = "src/utils/math.js"
    Example Output:
        ([
            {
                "type": "function",
                "name": "add",
                "content": "function add(a, b) { return a + b; }",
                "doc_string": "Adds two numbers together",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "language": "javascript",
                    "file": "src/utils/math.js",
                    "has_docstring": true
                }
            }
        ],
        {
            "functions": 1,
            "classes": 0,
            "total_blocks": 1,
            "file_blocks": {"src/utils/math.js": [...]}
        })

_extract_docstring_from_js(node: Any, source: str) -> Optional[str]:
    Input:
        - node: Tree-sitter node
        - source: Original source code
    Output:
        - Extracted JSDoc or comment docstring if found, None otherwise
    Example Input:
        node = <tree-sitter node for function>
        source = "/** Adds two numbers\n * @param {number} a First number\n * @param {number} b Second number\n * @returns {number} Sum\n */\nfunction add(a, b) { return a + b; }"
    Example Output:
        "Adds two numbers\n@param {number} a First number\n@param {number} b Second number\n@returns {number} Sum"

_is_react_component(content: str) -> bool:
    Input:
        - content: Source code
    Output:
        - True if content contains a React component, False otherwise
    Example Input:
        content = "import React from 'react'; function Button() { return <button>Click</button>; }"
    Example Output:
        True

_extract_react_component(file_path: str, content: str, language: str) -> Optional[Dict[str, Any]]:
    Input:
        - file_path: Path to source file
        - content: Source code
        - language: 'javascript' or 'typescript'
    Output:
        - Component block dictionary if successful, None otherwise
    Example Input:
        file_path = "src/components/Button.jsx"
        content = "import React from 'react'; export function Button() { return <button>Click</button>; }"
        language = "javascript"
    Example Output:
        {
            "uuid": "a1b2c3d4-...",
            "type": "react_component",
            "name": "Button",
            "content": "import React from 'react'; export function Button() { return <button>Click</button>; }",
            "doc_string": "A button component",
            "metadata": {
                "line_start": 1,
                "line_end": 1,
                "language": "javascript",
                "framework": "react",
                "has_docstring": true
            }
        }

_get_node_text(node: Any, source: str) -> Optional[str]:
    Input:
        - node: Tree-sitter node
        - source: Original source code
    Output:
        - Extracted text if successful, None otherwise
    Example Input:
        node = <tree-sitter node for a function>
        source = "function add(a, b) { return a + b; }"
    Example Output:
        "function add(a, b) { return a + b; }"

_get_class_name(node: Any, source: str) -> Optional[str]:
    Input:
        - node: Tree-sitter node
        - source: Original source code
    Output:
        - Class name if found, None otherwise
    Example Input:
        node = <tree-sitter node for a class declaration>
        source = "class Calculator { ... }"
    Example Output:
        "Calculator"

_get_function_name(node: Any, source: str) -> Optional[str]:
    Input:
        - node: Tree-sitter node
        - source: Original source code
    Output:
        - Function name if found, None otherwise
    Example Input:
        node = <tree-sitter node for a function declaration>
        source = "function add(a, b) { return a + b; }"
    Example Output:
        "add"

_extract_method(node: Any, source: str, class_name: str, language: str, imports: List[str], exports: List[str]) -> Optional[Dict[str, Any]]:
    Input:
        - node: Tree-sitter node
        - source: Original source code
        - class_name: Name of containing class
        - language: 'javascript' or 'typescript'
        - imports: List of import statements
        - exports: List of export statements
    Output:
        - Method block dictionary if successful, None otherwise
    Example Input:
        node = <tree-sitter node for method>
        source = "class Calculator { add(a, b) { return a + b; } }"
        class_name = "Calculator"
        language = "javascript"
        imports = ["import { sum } from './utils';"]
        exports = ["export class Calculator"]
    Example Output:
        {
            "uuid": "a1b2c3d4-...",
            "type": "method",
            "name": "add",
            "content": "add(a, b) { return a + b; }",
            "doc_string": "No documentation provided",
            "metadata": {
                "line_start": 1,
                "line_end": 1,
                "class_name": "Calculator",
                "imports": ["import { sum } from './utils';"],
                "exports": ["export class Calculator"],
                "language": "javascript",
                "has_docstring": false
            }
        }

_fallback_to_generic_extractor(file_path: str, content: str, language: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    Input:
        - file_path: Path to source file
        - content: Source code
        - language: 'javascript' or 'typescript'
    Output:
        - Tuple containing:
        - List of blocks extracted using generic patterns
        - Statistics dictionary
    Example Input:
        file_path = "src/utils/math.js"
        content = "function add(a, b) { return a + b; }"
        language = "javascript"
    Example Output:
        ([
            {
                "uuid": "a1b2c3d4-...",
                "type": "function",
                "name": "add",
                "content": "function add(a, b) { return a + b; }",
                "doc_string": "No documentation provided",
                "metadata": {
                    "line_start": 1,
                    "line_end": 1,
                    "imports": [],
                    "language": "javascript",
                    "has_docstring": false
                }
            }
        ],
        {
            "functions": 1,
            "classes": 0,
            "total_blocks": 1,
            "file_blocks": {"src/utils/math.js": [...]}
        })

Related Files:
- python_extractor.py: Similar extraction for Python
- generic_extractor.py: Fallback extraction methods
"""

import os
import textwrap
import uuid
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats, merge_stats
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_utils import get_parser, extract_js_ts_imports_exports
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_helpers import get_node_text
from agent_tools.dualipa.extraction.extractors.utils.react_extractor import is_react_component, extract_react_component
from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks
from tree_sitter import Parser

def parse_tree_sitter(content: str, language: str) -> Any:
    """Parse content using tree-sitter."""
    parser = get_parser(language)
    if not parser:
        logger.warning(f"Failed to get parser for {language}")
        return None
    try:
        return parser.parse(bytes(content, "utf8"))
    except Exception as e:
        logger.error(f"Error parsing {language} content with tree-sitter: {e}")
        return None
        
def _fallback_to_generic_extractor(file_path: str, content: str, language: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fallback to generic pattern-based extraction when tree-sitter fails.
    
    Args:
        file_path: Path to the source file
        content: Source code content
        language: 'javascript' or 'typescript'
        
    Returns:
        Tuple containing extracted blocks and statistics
    """
    logger.info(f"Using generic extractor fallback for {file_path}")
    
    # Extract imports and exports using regex
    imports, exports = extract_js_ts_imports_exports(content, None)
    logger.debug(f"Extracted {len(imports)} imports and {len(exports)} exports with regex fallback for {file_path}")
    
    # Create a temporary file with the content
    temp_file_path = f"{file_path}.temp"
    try:
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(content)
            
        # Run generic extraction
        blocks, stats = extract_generic_blocks(temp_file_path)
        
        # Add imports, exports, and extraction metadata to blocks
        for block in blocks:
            if "metadata" not in block:
                block["metadata"] = {}
            block["metadata"]["extraction_method"] = "generic_fallback"
            block["metadata"]["extraction_quality"] = "low"
            block["metadata"]["imports"] = imports.copy()
            block["metadata"]["exports"] = exports.copy()
        
        # Update stats
        stats["imports"] = len(imports)
        stats["exports"] = len(exports)
        
        return blocks, stats
    except Exception as e:
        logger.error(f"Error in generic extractor fallback: {e}")
        return [], init_stats()
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {e}")

def extract_js_ts_blocks(file_path: str, output_dir: Path = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract code blocks from a JavaScript/TypeScript file.
    
    Args:
        file_path: Path to the JS/TS file
        output_dir: Optional output directory for extracted blocks
        
    Returns:
        Tuple containing:
        - List of dictionaries containing extracted blocks
        - Statistics dictionary
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Determine language from extension
        ext = Path(file_path).suffix.lower()
        language = 'typescript' if ext in {'.ts', '.tsx'} else 'javascript'
        
        # Check for React components first
        if is_react_component(content):
            react_block = extract_react_component(file_path, content, language)
            if react_block:
                blocks = [react_block]
                stats['react_components'] = 1
                stats['total_blocks'] = 1
                stats['file_blocks'] = {file_path: blocks}
                logger.info(f"Extracted React component from {file_path}")
                return blocks, stats
                
        # Try tree-sitter parsing first (primary approach)
        tree = parse_tree_sitter(content, language)
        if tree and tree.root_node:
            blocks = []
            root_node = tree.root_node
            
            # Extract imports and exports
            imports, exports = extract_js_ts_imports_exports(content, tree)
            logger.debug(f"Extracted {len(imports)} imports and {len(exports)} exports from {file_path}")
            
            # Extract functions and classes
            for node in root_node.children:
                if node.type in ['function_declaration', 'class_declaration', 'interface_declaration']:
                    try:
                        # Get node name
                        name_node = node.child_by_field_name('name')
                        if not name_node:
                            continue
                            
                        name = name_node.text.decode('utf-8')
                        
                        # Determine block type
                        if node.type == 'function_declaration':
                            block_type = 'function'
                        elif node.type == 'class_declaration':
                            block_type = 'class'
                        else:  # interface_declaration
                            block_type = 'interface'
                        
                        # Check if this declaration is exported
                        is_exported = False
                        for child in node.children:
                            if child.type == 'export':
                                is_exported = True
                                break
                        
                        # Create node-specific exports list
                        node_exports = []
                        if is_exported:
                            node_exports.append(f"export {block_type} {name}")
                        
                        # Extract docstring
                        docstring = _extract_docstring_from_js(node, content)
                        
                        # Create block
                        block = {
                            'uuid': str(uuid.uuid4()),
                            'type': block_type,
                            'name': name,
                            'content': content[node.start_byte:node.end_byte],
                            'doc_string': docstring or "No documentation provided",
                            'line_start': node.start_point[0] + 1,
                            'line_end': node.end_point[0] + 1,
                            'metadata': {
                                'language': language,
                                'file': file_path,
                                'extraction_method': 'tree_sitter',
                                'extraction_quality': 'high',
                                'imports': imports.copy(),
                                'exports': exports.copy() if not node_exports else node_exports,
                                'has_docstring': docstring is not None
                            }
                        }
                        blocks.append(block)
                        
                        # Update stats
                        stats[f"{block_type}s"] = stats.get(f"{block_type}s", 0) + 1
                    except Exception as inner_e:
                        logger.warning(f"Error extracting node {node.type}: {inner_e}")
                        continue
                        
            # Update stats
            stats['total_blocks'] = len(blocks)
            stats['file_blocks'] = {file_path: blocks}
            stats['imports'] = len(imports)
            stats['exports'] = len(exports)
            
            # If we successfully extracted blocks with tree-sitter, return them
            if blocks:
                logger.info(f"Successfully extracted {len(blocks)} blocks from {file_path} using tree-sitter")
                return blocks, stats
                
        # If tree-sitter failed or found no blocks, fall back to generic extractor
        logger.warning(f"Tree-sitter extraction failed for {file_path}, falling back to generic extractor")
        generic_blocks, generic_stats = _fallback_to_generic_extractor(file_path, content, language)
        
        # Merge stats
        merged_stats = merge_stats(stats, generic_stats)
        merged_stats['fallback_used'] = True
        
        return generic_blocks, merged_stats
        
    except Exception as e:
        logger.error(f"Error extracting JS/TS blocks: {e}")
        
        # Final fallback - try generic extraction even if there was an exception
        try:
            logger.warning(f"Exception in primary extraction, attempting final generic fallback for {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Determine language from extension
            ext = Path(file_path).suffix.lower()
            language = 'typescript' if ext in {'.ts', '.tsx'} else 'javascript'
                
            # Use generic extractor as last resort
            return _fallback_to_generic_extractor(file_path, content, language)
        except Exception as fallback_e:
            logger.error(f"Final fallback failed: {fallback_e}")
            return [], init_stats()

def _extract_docstring_from_js(node: Any, source: str) -> Optional[str]:
    """
    Extract JSDoc or comment-style documentation for JavaScript/TypeScript nodes.
    
    Args:
        node: Tree-sitter node
        source: Original source code
        
    Returns:
        Extracted docstring if found, None otherwise
    """
    try:
        # Get node position information
        start_byte = node.start_byte
        start_line = node.start_point[0]
        
        # Convert bytes to string index for processing
        # Look at up to 50 lines before the node or 1000 characters, whichever is smaller
        max_lines_to_check = 50
        max_chars_to_check = 1000
        
        # Find the line start
        line_start = source.rfind('\n', max(0, start_byte - max_chars_to_check), start_byte)
        if line_start == -1:  # Handle case where node is at the start of file
            line_start = 0
        else:
            line_start += 1  # Move past the newline
            
        # Extract text before the node
        text_before = source[max(0, line_start - max_chars_to_check):start_byte]
        
        # Check for JSDoc comments (/** ... */)
        jsdoc_pattern = r'/\*\*([^*]|\*[^/])*\*/'
        jsdoc_matches = list(re.finditer(jsdoc_pattern, text_before, re.DOTALL))
        if jsdoc_matches:
            # Get the last JSDoc comment before the node
            jsdoc = jsdoc_matches[-1].group(0)
            
            # Clean up the JSDoc comment
            # 1. Remove the /** and */ markers
            # 2. Remove leading * from each line
            # 3. Trim whitespace
            lines = jsdoc.splitlines()
            cleaned_lines = []
            
            for i, line in enumerate(lines):
                if i == 0:  # First line with /**
                    line = line.replace('/**', '').strip()
                    if line:
                        cleaned_lines.append(line)
                elif i == len(lines) - 1:  # Last line with */
                    line = line.replace('*/', '').strip()
                    if line:
                        cleaned_lines.append(line)
                else:  # Middle lines with *
                    line = line.strip()
                    if line.startswith('*'):
                        line = line[1:].strip()
                    if line:
                        cleaned_lines.append(line)
            
            if cleaned_lines:
                return '\n'.join(cleaned_lines)
        
        # If no JSDoc found, check for single-line comments
        comment_lines = []
        
        # Extract line by line, up to max_lines_to_check
        lines = text_before.splitlines()
        if not lines:
            return None
            
        # Start from the end (closest to the node)
        for line in reversed(lines[-max_lines_to_check:]):
            line = line.strip()
            
            # Check if it's a comment
            if line.startswith('//'):
                # Remove comment marker and add to our collection
                comment_line = line[2:].strip()
                comment_lines.insert(0, comment_line)
            else:
                # If we hit a non-comment line, stop looking
                # Unless it's just whitespace and we're still collecting
                if line and comment_lines:
                    break
        
        if comment_lines:
            return '\n'.join(comment_lines)
            
        return None
    except Exception as e:
        logger.error(f"Error extracting JS docstring: {e}")
        return None

# React component extraction has been moved to react_extractor.py

# Node text extraction has been moved to tree_sitter_helpers.py

# Tree-sitter node handling has been moved to tree_sitter_helpers.py

# Usage examples have been moved to usage_examples.py 