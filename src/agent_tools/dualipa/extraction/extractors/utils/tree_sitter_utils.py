"""
Tree-sitter Utilities for DuaLipa

Provides common tree-sitter initialization and utilities for syntax-aware code extraction.

Key Features:
1. Centralized tree-sitter parser management using precompiled parsers
2. Parser instance caching for performance
3. Robust error handling and fallbacks
4. Syntax node traversal utilities
5. Import and export statement extraction
6. Regex fallback mechanisms

Example Input:
```python
# Get a parser for TypeScript
parser = get_parser('typescript')

# Parse some TypeScript code
code = '''
import React from 'react';
interface Calculator {
    add(a: number, b: number): number;
}
'''
tree = parser.parse(bytes(code, 'utf8'))

# Extract imports and exports
imports, exports = extract_js_ts_imports_exports(code, tree)
```

Expected Output:
```python
# Extracted imports
['import React from \'react\';']

# Extracted exports
[]
```

Functions:
- get_parser: Get a tree-sitter parser for a specific language
- extract_js_ts_imports_exports: Extract import and export statements from JavaScript/TypeScript code
- _extract_js_ts_imports_exports_regex: Fallback regex extraction for imports/exports

Expected Output:
```python
# Parser initialization
<Parser language=typescript>

# Parsed tree structure
root (program)
└── interface_declaration
    ├── name: Calculator
    └── body
        └── method_signature
            ├── name: add
            ├── parameters
            │   ├── a: number
            │   └── b: number
            └── return_type: number
```

Dependencies:
- tree-sitter (0.24.0): Core parsing library
- tree-sitter-python (0.23.6): Python language parser
- tree-sitter-javascript (0.23.1): JavaScript language parser
- tree-sitter-typescript (0.23.2): TypeScript language parser
- tree-sitter-java (0.23.5): Java language parser
- tree-sitter-languages (1.10.2): Language bindings with precompiled parsers

Documentation Links:
- Tree-sitter Python: https://tree-sitter.github.io/py-tree-sitter/
- Language Support: ../docs/tree_sitter_support.md

Related Files:
- extractors/code/js_ts_extractor.py: JavaScript/TypeScript extraction
- extractors/code/python_extractor.py: Python extraction
- extractors/code/hierarchy.py: Code hierarchy analysis
"""

from typing import Dict, Optional, List, Tuple, Any
from loguru import logger
from tree_sitter import Parser, Node
from tree_sitter_language_pack import get_binding, get_language, get_parser as get_ts_parser
import textwrap
import re

# Cache for parser instances
PARSERS: Dict[str, Parser] = {}

# Complete list of supported languages from tree-sitter-language-pack
SUPPORTED_LANGUAGES = [
    'actionscript', 'ada', 'agda', 'arduino', 'asm', 'astro', 'bash', 'beancount',
    'bibtex', 'bicep', 'bitbake', 'c', 'cairo', 'capnp', 'chatito', 'clarity',
    'clojure', 'cmake', 'comment', 'commonlisp', 'cpon', 'cpp', 'csharp', 'css',
    'csv', 'cuda', 'd', 'dart', 'dockerfile', 'doxygen', 'elisp', 'elixir', 'elm',
    'embeddedtemplate', 'erlang', 'fennel', 'firrtl', 'fish', 'fortran', 'func',
    'gdscript', 'gitattributes', 'gitcommit', 'gitignore', 'gleam', 'glsl', 'gn',
    'go', 'gomod', 'gosum', 'groovy', 'gstlaunch', 'hack', 'hare', 'haskell',
    'haxe', 'hcl', 'heex', 'hlsl', 'html', 'hyprlang', 'ispc', 'janet', 'java',
    'javascript', 'jsdoc', 'json', 'jsonnet', 'julia', 'kconfig', 'kdl', 'kotlin',
    'linkerscript', 'llvm', 'lua', 'luadoc', 'luap', 'luau', 'magik', 'make',
    'markdown', 'matlab', 'mermaid', 'meson', 'ninja', 'nix', 'nqc', 'objc',
    'odin', 'org', 'pascal', 'pem', 'perl', 'pgn', 'php', 'po', 'pony',
    'powershell', 'printf', 'prisma', 'properties', 'proto', 'psv', 'puppet',
    'purescript', 'pymanifest', 'python', 'qmldir', 'query', 'r', 'racket',
    'rbs', 're2c', 'readline', 'requirements', 'ron', 'rst', 'ruby', 'rust',
    'scala', 'scheme', 'scss', 'slang', 'smali', 'smithy', 'solidity', 'sparql',
    'sql', 'squirrel', 'starlark', 'svelte', 'swift', 'tablegen', 'tcl', 'test',
    'thrift', 'toml', 'tsv', 'twig', 'typescript', 'typst', 'udev', 'ungrammar',
    'uxntal', 'v', 'verilog', 'vhdl', 'vim', 'vue', 'wgsl', 'xcompose', 'xml',
    'yaml', 'yuck', 'zig'
]

def get_parser(lang_name: str) -> Optional[Parser]:
    """
    Get or initialize a parser for a given language.

    Args:
        lang_name: The language name (e.g., 'typescript')

    Returns:
        A Parser instance if successful, None otherwise.
    """
    try:
        if lang_name not in SUPPORTED_LANGUAGES:
            logger.error(f"Language {lang_name} not supported. Available languages: {', '.join(SUPPORTED_LANGUAGES)}")
            return None
            
        if lang_name not in PARSERS:
            # Get pre-built parser from tree_sitter_language_pack
            PARSERS[lang_name] = get_ts_parser(lang_name)
            logger.info(f"Successfully initialized parser for: {lang_name}")
        return PARSERS.get(lang_name)
    except Exception as e:
        logger.warning(f"Error getting parser for {lang_name}: {e}")
        return None

def get_supported_languages() -> List[str]:
    """Returns the list of all supported languages."""
    return SUPPORTED_LANGUAGES.copy()

def extract_js_ts_imports_exports(content: str, tree: Any) -> Tuple[List[str], List[str]]:
    """
    Extract import and export statements from JavaScript or TypeScript code
    using tree-sitter parsing.
    
    Args:
        content: Source code content
        tree: Tree-sitter parse tree
        
    Returns:
        Tuple of (imports, exports) lists
    """
    if not tree or not tree.root_node:
        # Fallback to regex if tree-sitter parsing failed
        return _extract_js_ts_imports_exports_regex(content)
    
    imports = []
    exports = []
    
    try:
        # Helper function to extract node text
        def get_node_text(node: Node) -> str:
            return content[node.start_byte:node.end_byte]
        
        # Visit all nodes to find imports and exports
        def visit_node(node: Node) -> None:
            if node.type == 'import_statement':
                imports.append(get_node_text(node).strip())
            elif node.type == 'export_statement':
                exports.append(get_node_text(node).strip())
            # Handle destructured imports
            elif node.type == 'lexical_declaration' and node.children and any(child.type == 'import' for child in node.children):
                imports.append(get_node_text(node).strip())
            # Handle default exports
            elif node.type.startswith('export_'):
                exports.append(get_node_text(node).strip())
            # Handle export keywords on declarations
            elif node.type in ('function_declaration', 'class_declaration'):
                for child in node.children:
                    if child.type == 'export':
                        exports.append(f"export {get_node_text(node)}")
                        break
            
            # Continue traversing
            for child in node.children:
                visit_node(child)
        
        # Start traversal
        visit_node(tree.root_node)
        
        return imports, exports
    except Exception as e:
        logger.error(f"Error extracting imports/exports using tree-sitter: {e}")
        # Fall back to regex as a last resort
        return _extract_js_ts_imports_exports_regex(content)

def _extract_js_ts_imports_exports_regex(content: str) -> Tuple[List[str], List[str]]:
    """
    Fallback method to extract imports and exports using regex patterns.
    
    Args:
        content: Source code content
        
    Returns:
        Tuple of (imports, exports) lists
    """
    imports = []
    exports = []
    
    # Common patterns for JS/TS imports
    import_patterns = [
        r'import\s+.*?from\s+[\'"].*?[\'"];?',  # import X from 'Y'
        r'import\s+[\'"].*?[\'"];?',             # import 'X'
        r'import\s*\(.*?\);?',                   # import()
        r'require\s*\([\'"].*?[\'"]\);?'         # require('X')
    ]
    
    # Common patterns for JS/TS exports
    export_patterns = [
        r'export\s+(?:default\s+)?(?:class|function|const|let|var|interface)\s+\w+',
        r'export\s+default\s+.*?;?',
        r'export\s+\{.*?\}\s+from\s+[\'"].*?[\'"];?',
        r'export\s+\{.*?\};?',
        r'export\s+\*\s+from\s+[\'"].*?[\'"];?'
    ]
    
    # Extract imports
    for pattern in import_patterns:
        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            imports.append(match.group(0).strip())
    
    # Extract exports
    for pattern in export_patterns:
        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            exports.append(match.group(0).strip())
    
    return imports, exports

def usage_example():
    """Simple usage example to verify tree-sitter initialization."""
    # Sample TypeScript code with multiple node types and imports/exports
    ts_code = textwrap.dedent('''
    import React from 'react';
    import { useState, useEffect } from 'react';
    import * as utils from './utils';
    
    export interface Calculator {
        add(a: number, b: number): number;
    }
    
    export class BasicCalculator implements Calculator {
        constructor() {}
        
        add(a: number, b: number): number {
            return a + b;
        }
        
        private subtract(a: number, b: number): number {
            return a - b;
        }
    }
    
    export default BasicCalculator;
    ''')
    
    # Get TypeScript parser and parse the code
    parser = get_parser('typescript')
    if not parser:
        print("Failed to get TypeScript parser")
        return

    try:
        tree = parser.parse(bytes(ts_code, 'utf8'))
        root = tree.root_node
        print(f"Root node type: {root.type}")
        
        # Traverse the tree and print interface and class declarations
        for node in root.children:
            if node.type == 'interface_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    print(f"Found interface: {name_node.text.decode('utf8')}")
            elif node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    print(f"Found class: {name_node.text.decode('utf8')}")
                    
                # Look for method definitions within the class
                for child in node.children:
                    if child.type == 'method_definition':
                        name_node = child.child_by_field_name('name')
                        if name_node:
                            print(f"  Found method: {name_node.text.decode('utf8')}")
        
        # Extract imports and exports
        imports, exports = extract_js_ts_imports_exports(ts_code, tree)
        print("\nExtracted Imports:")
        for imp in imports:
            print(f"  {imp}")
            
        print("\nExtracted Exports:")
        for exp in exports:
            print(f"  {exp}")
    except Exception as e:
        print(f"Error parsing TypeScript code: {e}")

if __name__ == '__main__':
    usage_example()
