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
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "language": "javascript",
                    "file": "src/utils/math.js"
                }
            }
        ],
        {
            "functions": 1,
            "classes": 0,
            "total_blocks": 1,
            "file_blocks": {"src/utils/math.js": [...]}
        })

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
            "metadata": {
                "line_start": 1,
                "line_end": 1,
                "language": "javascript",
                "framework": "react"
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
            "metadata": {
                "line_start": 1,
                "line_end": 1,
                "class_name": "Calculator",
                "imports": ["import { sum } from './utils';"],
                "exports": ["export class Calculator"],
                "language": "javascript"
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
                "metadata": {
                    "line_start": 1,
                    "line_end": 1,
                    "imports": [],
                    "language": "javascript"
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
        if _is_react_component(content):
            react_block = _extract_react_component(file_path, content, language)
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
                        
                        # Create block
                        block = {
                            'uuid': str(uuid.uuid4()),
                            'type': block_type,
                            'name': name,
                            'content': content[node.start_byte:node.end_byte],
                            'line_start': node.start_point[0] + 1,
                            'line_end': node.end_point[0] + 1,
                            'metadata': {
                                'language': language,
                                'file': file_path,
                                'extraction_method': 'tree_sitter',
                                'extraction_quality': 'high',
                                'imports': imports.copy(),
                                'exports': exports.copy() if not node_exports else node_exports
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

def _is_react_component(content: str) -> bool:
    """Check if content contains a React component."""
    patterns = [
        r'import\s+.*?React.*?\s+from\s+[\'"]react[\'"]',
        r'class\s+\w+\s+extends\s+(?:React\.)?Component',
        r'function\s+[A-Z]\w*\s*\([^)]*\)\s*{',
        r'const\s+[A-Z]\w*\s*=\s*(?:React\.)?memo\(',
        r'export\s+default\s+[A-Z]\w+'
    ]
    return any(re.search(pattern, content) for pattern in patterns)

def _extract_react_component(file_path: str, content: str, language: str) -> Optional[Dict[str, Any]]:
    """Extract a React component as a single block."""
    try:
        # Try to find component name
        name_match = re.search(r'(?:class|function|const)\s+([A-Z]\w*)', content)
        if not name_match:
            return None
            
        component_name = name_match.group(1)
        
        # Try tree-sitter parsing for imports/exports
        tree = None
        imports = []
        exports = []
        
        try:
            parser = get_parser(language)
            if parser:
                tree = parser.parse(bytes(content, "utf8"))
                imports, exports = extract_js_ts_imports_exports(content, tree)
        except Exception as parse_e:
            logger.warning(f"Error parsing React component with tree-sitter: {parse_e}")
            # Fallback to regex for imports/exports
            imports, exports = extract_js_ts_imports_exports(content, None)
        
        # Check if component is exported
        is_exported = False
        export_match = re.search(r'export\s+(?:default\s+)?(?:class|function|const)\s+' + component_name, content)
        if export_match or any(component_name in exp for exp in exports):
            is_exported = True
            
        # Add component-specific export if needed
        component_exports = exports.copy()
        if is_exported and not any(component_name in exp for exp in exports):
            component_exports.append(f"export default {component_name}")
            
        return {
            "uuid": str(uuid.uuid4()),
            "type": "react_component",
            "name": component_name,
            "content": content,  # Keep entire file content for React components
            "line_start": 1,
            "line_end": len(content.splitlines()),
            "metadata": {
                "language": language,
                "framework": "react",
                "extraction_method": "react_detection",
                "extraction_quality": "high",
                "file": file_path,
                "imports": imports,
                "exports": component_exports
            }
        }
    except Exception as e:
        logger.error(f"Error extracting React component: {e}")
        return None

def _get_node_text(node: Any, source: str) -> Optional[str]:
    """Extract text for a tree-sitter node."""
    try:
        start_byte = node.start_byte
        end_byte = node.end_byte
        text = source[start_byte:end_byte]
        return textwrap.dedent(text)
    except Exception as e:
        logger.error(f"Error getting node text: {e}")
        return None

def _get_class_name(node: Any, source: str) -> Optional[str]:
    """Extract class name from a tree-sitter node."""
    try:
        for child in node.children:
            if child.type == "identifier":
                return _get_node_text(child, source)
        return None
    except Exception:
        return None

def _get_function_name(node: Any, source: str) -> Optional[str]:
    """Extract function name from a tree-sitter node."""
    try:
        if node.type == "function_declaration":
            for child in node.children:
                if child.type == "identifier":
                    return _get_node_text(child, source)
        elif node.type == "arrow_function":
            # Try to find variable name for arrow function
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                for child in parent.children:
                    if child.type == "identifier":
                        return _get_node_text(child, source)
        return None
    except Exception:
        return None

def _extract_method(
    node: Any,
    source: str,
    class_name: str,
    language: str,
    imports: List[str],
    exports: List[str]
) -> Optional[Dict[str, Any]]:
    """Extract a class method as a block."""
    try:
        method_name = None
        for child in node.children:
            if child.type == "property_identifier":
                method_name = _get_node_text(child, source)
                break
                
        if not method_name:
            return None
            
        method_content = _get_node_text(node, source)
        if not method_content:
            return None
            
        return {
            "uuid": str(uuid.uuid4()),
            "type": "method",
            "name": method_name,
            "content": method_content,
            "metadata": {
                "line_start": node.start_point[0] + 1,
                "line_end": node.end_point[0] + 1,
                "class_name": class_name,
                "imports": imports,
                "exports": exports,
                "language": language
            }
        }
    except Exception as e:
        logger.error(f"Error extracting method: {e}")
        return None

def usage_example():
    """Example usage of JS/TS code extraction with cascading parser approach"""
    import tempfile
    import os
    
    # Sample TypeScript React component (should work with tree-sitter)
    sample_code_valid = '''
    import React from 'react';
    
    interface Props {
        name: string;
    }
    
    export class Greeter extends React.Component<Props> {
        render() {
            return <div>Hello {this.props.name}!</div>;
        }
    }
    
    function add(a: number, b: number): number {
        return a + b;
    }
    '''
    
    # Sample code with complex structures that might cause tree-sitter to fail
    sample_code_complex = '''
    import React, { useState, useEffect } from 'react';

    // This is a complex TypeScript type that might challenge tree-sitter
    type ComplexGeneric<T extends Record<string, unknown>> = {
        [K in keyof T]: T[K] extends Function ? (...args: any[]) => Promise<any> : never;
    } & {
        readonly [Symbol.iterator]: () => Iterator<any>;
    };

    // Arrow function with complex type
    const processData = async <T extends unknown>(
        data: T[],
        callback: (item: T) => Promise<void>
    ): Promise<ComplexGeneric<T>> => {
        for (const item of data) {
            await callback(item);
        }
        return {} as ComplexGeneric<T>;
    };

    // React component with hooks and complex props
    function DataProcessor<T extends { id: string }>({ 
        items, 
        onProcess 
    }: { 
        items: T[], 
        onProcess: (result: ComplexGeneric<T>) => void 
    }) {
        const [processing, setProcessing] = useState(false);
        
        useEffect(() => {
            let mounted = true;
            
            const process = async () => {
                setProcessing(true);
                try {
                    const result = await processData(items, async (item) => {
                        console.log(`Processing item ${item.id}`);
                    });
                    if (mounted) {
                        onProcess(result);
                    }
                } finally {
                    if (mounted) {
                        setProcessing(false);
                    }
                }
            };
            
            process();
            
            return () => {
                mounted = false;
            };
        }, [items, onProcess]);
        
        return (
            <div>
                {processing ? 'Processing...' : 'Done!'}
            </div>
        );
    }
    '''
    
    try:
        # Test with valid code (tree-sitter should work)
        with tempfile.NamedTemporaryFile(suffix='.tsx', mode='w', delete=False) as f:
            f.write(sample_code_valid)
            valid_path = f.name
            
        # Test with complex code (might fall back to generic extractor)
        with tempfile.NamedTemporaryFile(suffix='.tsx', mode='w', delete=False) as f:
            f.write(sample_code_complex)
            complex_path = f.name
            
        print("\n=== Testing with simple TypeScript code (tree-sitter parsing) ===")
        blocks_valid, stats_valid = extract_js_ts_blocks(valid_path)
        
        print(f"\nExtracted {len(blocks_valid)} blocks:")
        for block in blocks_valid:
            print(f"\nType: {block['type']}")
            print(f"Name: {block['name']}")
            print(f"Lines: {block['line_start']}-{block['line_end']}")
            print(f"Extraction method: {block['metadata'].get('extraction_method', 'unknown')}")
            print(f"Extraction quality: {block['metadata'].get('extraction_quality', 'unknown')}")
            
        print("\n=== Testing with complex TypeScript code (potential fallback) ===")
        blocks_complex, stats_complex = extract_js_ts_blocks(complex_path)
        
        print(f"\nExtracted {len(blocks_complex)} blocks:")
        for block in blocks_complex:
            print(f"\nType: {block['type']}")
            print(f"Name: {block['name']}")
            print(f"Lines: {block['line_start']}-{block['line_end']}")
            print(f"Extraction method: {block['metadata'].get('extraction_method', 'unknown')}")
            print(f"Extraction quality: {block['metadata'].get('extraction_quality', 'unknown')}")
            
        # Check if fallback was used
        fallback_used = stats_complex.get('fallback_used', False)
        print(f"\nFallback used: {fallback_used}")
        
    finally:
        # Clean up
        for path in [valid_path, complex_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                print(f"Error cleaning up {path}: {e}")

if __name__ == '__main__':
    usage_example() 