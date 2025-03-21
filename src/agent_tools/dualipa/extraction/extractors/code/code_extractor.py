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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from agent_tools.dualipa.extraction.extractors.code.hierarchy import analyze_code_hierarchy
from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
from agent_tools.dualipa.extraction.extractors.code.generic_extractor import extract_generic_blocks
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info
from agent_tools.dualipa.extraction.extractors.utils.tree_sitter_utils import get_parser
from agent_tools.dualipa.extraction.extractors.utils.validation_utils import validate_block
from agent_tools.dualipa.extraction.extractors.utils.verification_utils import verify_block

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
                        "name": _get_node_text(node.child_by_field_name("name"), content),
                        "content": _get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "is_async": _is_async_node(node),
                            "return_type": _get_ts_return_type(node, content)
                        }
                    })
                    
                elif node.type == "class_declaration":
                    blocks.append({
                        "type": "class",
                        "name": _get_node_text(node.child_by_field_name("name"), content),
                        "content": _get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "extends": _get_extends_class(node, content),
                            "implements": _get_implements_interfaces(node, content)
                        }
                    })
                    
                elif node.type == "method_definition":
                    blocks.append({
                        "type": "method",
                        "name": _get_node_text(node.child_by_field_name("name"), content),
                        "content": _get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "is_static": _is_static_node(node),
                            "is_async": _is_async_node(node),
                            "return_type": _get_ts_return_type(node, content)
                        }
                    })
                    
                elif node.type == "interface_declaration" and language == "typescript":
                    blocks.append({
                        "type": "interface",
                        "name": _get_node_text(node.child_by_field_name("name"), content),
                        "content": _get_node_text(node, content),
                        "line_start": node.start_point[0] + 1,
                        "line_end": node.end_point[0] + 1,
                        "metadata": {
                            "language": language,
                            "file": file_path,
                            "extends": _get_extends_interfaces(node, content),
                            "properties": _get_interface_properties(node, content)
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
                    "content": _get_block_content(content, match),
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
                    "content": _get_block_content(content, match),
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

# Helper functions
def _get_node_text(node: Any, source: str) -> str:
    """Get text from a tree-sitter node."""
    try:
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source[start_byte:end_byte]
    except Exception:
        return ""

def _get_block_content(content: str, match: re.Match) -> str:
    """Get block content from regex match."""
    try:
        # Find block end
        start = match.end()
        brace_count = 1
        end = start
        
        while end < len(content) and brace_count > 0:
            if content[end] == '{':
                brace_count += 1
            elif content[end] == '}':
                brace_count -= 1
            end += 1
            
        return content[match.start():end]
        
    except Exception:
        return match.group(0)

def _is_async_node(node: Any) -> bool:
    """Check if node is async."""
    try:
        for child in node.children:
            if child.type == "async":
                return True
        return False
    except Exception:
        return False

def _is_static_node(node: Any) -> bool:
    """Check if node is static."""
    try:
        for child in node.children:
            if child.type == "static":
                return True
        return False
    except Exception:
        return False

def _get_ts_return_type(node: Any, source: str) -> Optional[str]:
    """Get TypeScript return type."""
    try:
        for child in node.children:
            if child.type == "type_annotation":
                return _get_node_text(child, source).strip(": ")
        return None
    except Exception:
        return None

def _get_extends_class(node: Any, source: str) -> Optional[str]:
    """Get extended class name."""
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        return _get_node_text(extends_child, source)
        return None
    except Exception:
        return None

def _get_implements_interfaces(node: Any, source: str) -> List[str]:
    """Get implemented interface names."""
    interfaces = []
    try:
        for child in node.children:
            if child.type == "implements_clause":
                for impl_child in child.children:
                    if impl_child.type == "identifier":
                        interfaces.append(_get_node_text(impl_child, source))
        return interfaces
    except Exception:
        return []

def _get_extends_interfaces(node: Any, source: str) -> List[str]:
    """Get extended interface names."""
    interfaces = []
    try:
        for child in node.children:
            if child.type == "extends_clause":
                for extends_child in child.children:
                    if extends_child.type == "identifier":
                        interfaces.append(_get_node_text(extends_child, source))
        return interfaces
    except Exception:
        return []

def _get_interface_properties(node: Any, source: str) -> List[Dict[str, str]]:
    """Get interface property definitions."""
    properties = []
    try:
        for child in node.children:
            if child.type == "interface_body":
                for prop in child.children:
                    if prop.type == "property_signature":
                        prop_info = {
                            "name": _get_node_text(prop.child_by_field_name("name"), source),
                            "type": _get_node_text(prop.child_by_field_name("type"), source).strip(": ")
                        }
                        properties.append(prop_info)
        return properties
    except Exception:
        return []

def usage_example() -> None:
    """Example usage of code extraction."""
    # Example Python file
    python_content = textwrap.dedent('''
        class ExampleClass:
            """Example class docstring."""
            
            def __init__(self, name: str):
                self.name = name
                
            def greet(self) -> str:
                return f"Hello, {self.name}!"
                
        def example_function(x: int, y: int) -> int:
            """Example function docstring."""
            return x + y
    ''').strip()
    
    # Save to temp file
    with open('example.py', 'w') as f:
        f.write(python_content)
        
    # Extract blocks
    blocks = extract_code_blocks('example.py', Path('output'))
    
    print("Extracted Blocks:")
    for block in blocks:
        print(f"\nType: {block['type']}")
        print(f"Name: {block['name']}")
        print(f"Lines: {block['line_start']}-{block['line_end']}")
        print("Metadata:", block['metadata'])
        print("Content:")
        print(block['content'])
        
    # Cleanup
    import os
    os.remove('example.py') 