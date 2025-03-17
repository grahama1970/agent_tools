#!/usr/bin/env python3
"""
Test script for JavaScript and TypeScript code extraction using tree-sitter.
This script demonstrates how to use tree-sitter to extract functions, classes, 
and other declarations from JavaScript and TypeScript files.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
import traceback

# Add the parent directory to sys.path to access dualipa module
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import tree-sitter modules
import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Parser

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Sample file paths
JS_SAMPLE_PATH = SCRIPT_DIR / "sample_js.js"
TS_SAMPLE_PATH = SCRIPT_DIR / "sample_ts.ts"

# Node types to extract
FUNCTION_TYPES = [
    'function_declaration',
    'method_definition',
    'arrow_function',
    'function'
]

CLASS_TYPES = [
    'class_declaration'
]

INTERFACE_TYPES = [
    'interface_declaration'
]

TYPE_TYPES = [
    'type_alias_declaration'
]

VARIABLE_TYPES = [
    'lexical_declaration', 
    'variable_declaration'
]

EXTRACTABLE_TYPES = FUNCTION_TYPES + CLASS_TYPES + INTERFACE_TYPES + TYPE_TYPES

def setup_parsers():
    """Set up the tree-sitter parsers for JavaScript and TypeScript."""
    try:
        # Load languages directly from the modules
        JS_LANGUAGE = Language(tree_sitter_javascript.language())
        TS_LANGUAGE = Language(tree_sitter_typescript.language())
        
        # Initialize parsers
        js_parser = Parser()
        ts_parser = Parser()
        
        # Set languages
        # Updated for tree-sitter 0.24.0: use property assignment instead of set_language
        js_parser.language = JS_LANGUAGE
        ts_parser.language = TS_LANGUAGE
        
        return js_parser, ts_parser
    except Exception as e:
        print(f"Failed to set up tree-sitter parsers: {e}")
        traceback.print_exc()
        return None, None

def get_node_text(node, source_bytes):
    """Get the text of a node from source bytes."""
    return source_bytes[node.start_byte:node.end_byte].decode('utf8')

def get_docstring(node, source_bytes):
    """
    Extract JSDoc comment above a node if it exists.
    """
    # Get the line of the node
    start_line = node.start_point[0]
    if start_line == 0:
        # No room for a comment above
        return None
    
    # Get the text of the whole file
    text = source_bytes.decode('utf8')
    lines = text.split('\n')
    
    # Look for JSDoc comment above the node
    line_index = start_line - 1
    while line_index >= 0:
        line = lines[line_index].strip()
        if line.startswith('/**') and line_index > 0:
            # Found start of JSDoc comment, collect all lines
            doc_lines = []
            while line_index < start_line:
                doc_line = lines[line_index].strip()
                doc_lines.append(doc_line)
                if doc_line.endswith('*/'):
                    break
                line_index += 1
            
            return '\n'.join(doc_lines)
        elif line == '' or line.startswith('//'):
            # Skip empty lines and single-line comments
            line_index -= 1
        else:
            # Found non-comment content, stop searching
            break
    
    return None

def get_function_params(func_node, source_bytes):
    """Extract function parameters."""
    params = []
    
    # Look for formal_parameters or parameter nodes
    for child in func_node.children:
        if child.type == 'formal_parameters':
            for param_child in child.children:
                if param_child.type == 'identifier' or param_child.type == 'formal_parameter':
                    # For identifiers, get name directly
                    if param_child.type == 'identifier':
                        params.append(get_node_text(param_child, source_bytes))
                    # For formal_parameters, get the identifier child
                    else:
                        for subchild in param_child.children:
                            if subchild.type == 'identifier':
                                params.append(get_node_text(subchild, source_bytes))
                                break
    
    return params

def get_type_info(node, source_bytes):
    """Extract type information from TypeScript nodes."""
    type_info = {}
    
    # For interface declarations, get extended interfaces
    if node.type == 'interface_declaration':
        for child in node.children:
            if child.type == 'extends_clause':
                extends_types = []
                for extends_child in child.children:
                    if extends_child.type == 'type_identifier':
                        extends_types.append(get_node_text(extends_child, source_bytes))
                if extends_types:
                    type_info['extends'] = extends_types
    
    # For class declarations, get extended class
    elif node.type == 'class_declaration':
        for child in node.children:
            if child.type == 'extends_clause':
                for extends_child in child.children:
                    if extends_child.type == 'identifier':
                        type_info['extends'] = get_node_text(extends_child, source_bytes)
                        break
            elif child.type == 'implements_clause':
                implements_types = []
                for impl_child in child.children:
                    if impl_child.type == 'type_identifier':
                        implements_types.append(get_node_text(impl_child, source_bytes))
                if implements_types:
                    type_info['implements'] = implements_types
    
    return type_info

def extract_imports(root_node, source_bytes):
    """Extract import statements."""
    imports = []
    
    def find_imports(node):
        if node.type == 'import_statement':
            imports.append(get_node_text(node, source_bytes))
        
        for child in node.children:
            find_imports(child)
    
    find_imports(root_node)
    return imports

def extract_exports(root_node, source_bytes):
    """Extract export statements."""
    exports = []
    
    def find_exports(node):
        if node.type in ['export_statement', 'export_default_statement']:
            exports.append(get_node_text(node, source_bytes))
        
        for child in node.children:
            find_exports(child)
    
    find_exports(root_node)
    return exports

def extract_declarations(file_path, parser, language_name):
    """
    Extract declarations from a file using tree-sitter.
    
    Args:
        file_path: Path to the file to extract declarations from
        parser: Tree-sitter parser to use
        language_name: Name of the language (javascript or typescript)
    
    Returns:
        List of extracted declarations
    """
    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    
    # Parse the file
    tree = parser.parse(source_bytes)
    root_node = tree.root_node
    
    # Extract declarations
    declarations = []
    imports = extract_imports(root_node, source_bytes)
    exports = extract_exports(root_node, source_bytes)
    
    # First-level traversal to find declarations
    for node in root_node.children:
        # Skip non-declaration nodes
        if node.type not in EXTRACTABLE_TYPES:
            # Check for variable declarations that might contain function expressions
            if node.type in VARIABLE_TYPES:
                for child in node.children:
                    if child.type == 'variable_declarator':
                        # Look for arrow functions or function expressions in the value
                        for value_child in child.children:
                            if value_child.type in ['arrow_function', 'function']:
                                # Found a function expression assigned to a variable
                                # Get the variable name from the declarator
                                var_name = None
                                for id_child in child.children:
                                    if id_child.type == 'identifier':
                                        var_name = get_node_text(id_child, source_bytes)
                                        break
                                
                                if var_name:
                                    declaration = extract_declaration_info(
                                        value_child, source_bytes, file_path, language_name, 
                                        node_type=value_child.type, name=var_name
                                    )
                                    declarations.append(declaration)
            continue
        
        # Extract declaration info
        declaration = extract_declaration_info(node, source_bytes, file_path, language_name)
        if declaration:
            # Add import context
            if imports:
                declaration['context'] = {
                    'imports': imports,
                    'exports': exports
                }
            declarations.append(declaration)
    
    return declarations

def extract_declaration_info(node, source_bytes, file_path, language_name, node_type=None, name=None):
    """
    Extract information about a declaration node.
    
    Args:
        node: Tree-sitter node for the declaration
        source_bytes: Source bytes of the file
        file_path: Path to the file
        language_name: Name of the language (javascript or typescript)
        node_type: Override the node type (for variable declarations with functions)
        name: Override the name (for variable declarations with functions)
    
    Returns:
        Dictionary with declaration information
    """
    # Get the node type if not provided
    if node_type is None:
        node_type = node.type
    
    # Get declaration name
    if name is None:
        name = None
        for child in node.children:
            if child.type == 'identifier':
                name = get_node_text(child, source_bytes)
                break
    
    if not name:
        # Skip anonymous declarations
        return None
    
    # Get start and end lines
    start_line = node.start_point[0] + 1  # 1-indexed
    end_line = node.end_point[0] + 1  # 1-indexed
    
    # Get the text of the declaration
    declaration_text = get_node_text(node, source_bytes)
    
    # Get docstring if available
    docstring = get_docstring(node, source_bytes)
    
    # Get function parameters for function declarations
    params = []
    if node_type in FUNCTION_TYPES:
        params = get_function_params(node, source_bytes)
    
    # Get type information for TypeScript declarations
    type_info = {}
    if language_name == 'typescript':
        type_info = get_type_info(node, source_bytes)
    
    # Create declaration info
    declaration_info = {
        'name': name,
        'type': node_type,
        'language': language_name,
        'file_path': str(file_path),
        'start_line': start_line,
        'end_line': end_line,
        'docstring': docstring,
        'text': declaration_text,
    }
    
    # Add function-specific info
    if node_type in FUNCTION_TYPES:
        declaration_info['parameters'] = params
    
    # Add TypeScript-specific type info
    if type_info:
        declaration_info['type_info'] = type_info
    
    return declaration_info

def save_declarations(declarations, output_dir, file_stem):
    """
    Save extracted declarations to individual files.
    
    Args:
        declarations: List of declaration dictionaries
        output_dir: Directory to save files to
        file_stem: Base name for the files
    
    Returns:
        Dictionary with paths to saved files
    """
    output_files = {}
    
    # Create directories if they don't exist
    blocks_dir = Path(output_dir) / 'js_ts_blocks'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Save declarations
    for i, decl in enumerate(declarations):
        # Create filename with declaration info
        filename = f"{file_stem}_{decl['name']}_{i+1}.{decl['language']}"
        output_file = blocks_dir / filename
        
        # Generate header comments
        if decl['language'] == 'javascript':
            header = "// "
        else:
            header = "// "
        
        # Create header with metadata
        header_lines = [
            f"{header}Original file: {decl['file_path']}",
            f"{header}Block type: {decl['type']}",
            f"{header}Name: {decl['name']}",
        ]
        
        # Add docstring to header if available
        if decl.get('docstring'):
            header_lines.append(f"{header}Docstring: {decl['docstring']}")
        
        # Add parameters to header for functions
        if 'parameters' in decl:
            params_str = ', '.join(decl['parameters'])
            header_lines.append(f"{header}Parameters: {params_str}")
        
        # Add type info to header for TypeScript declarations
        if 'type_info' in decl:
            for k, v in decl['type_info'].items():
                if isinstance(v, list):
                    header_lines.append(f"{header}{k}: {', '.join(v)}")
                else:
                    header_lines.append(f"{header}{k}: {v}")
        
        # Add imports if available in context
        if 'context' in decl and 'imports' in decl['context']:
            header_lines.append(f"\n{header}Required imports:")
            for imp in decl['context']['imports']:
                header_lines.append(f"{header}{imp}")
        
        # Add a blank line after the header
        header_lines.append("\n")
        
        # Write the file with header and declaration text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header_lines))
            f.write(decl['text'])
        
        # Store the output file in the result
        output_files[decl['name']] = str(output_file)
    
    return output_files

def extract_from_file(file_path, parser, language_name, output_dir=None):
    """
    Extract declarations from a file and save them.
    
    Args:
        file_path: Path to the file to extract from
        parser: Tree-sitter parser to use
        language_name: Name of the language (javascript or typescript)
        output_dir: Directory to save extracted declarations to
    
    Returns:
        Tuple of (declarations, output_files)
    """
    # Parse the file
    declarations = extract_declarations(file_path, parser, language_name)
    
    # Use a temporary directory if no output dir specified
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    
    # Save declarations to files
    output_files = save_declarations(
        declarations, 
        output_dir, 
        Path(file_path).stem
    )
    
    return declarations, output_files

def main():
    """Main function to run the extraction on sample files."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract JavaScript and TypeScript declarations')
    parser.add_argument('--js', help='Path to JavaScript file')
    parser.add_argument('--ts', help='Path to TypeScript file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set up the tree-sitter parsers
    print("Setting up tree-sitter parsers...")
    js_parser, ts_parser = setup_parsers()
    
    if js_parser is None or ts_parser is None:
        print("Failed to set up tree-sitter parsers")
        return 1
    
    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process JavaScript file
    js_file = args.js if args.js else JS_SAMPLE_PATH
    ts_file = args.ts if args.ts else TS_SAMPLE_PATH
    
    print(f"\n{'='*80}")
    print(f" JAVASCRIPT AND TYPESCRIPT CODE EXTRACTION ".center(80, '='))
    print(f"{'='*80}\n")
    
    # Extract JavaScript declarations
    try:
        print(f"Processing JavaScript file: {js_file}")
        js_decls, js_files = extract_from_file(js_file, js_parser, 'javascript', output_dir)
        print(f"Extracted {len(js_decls)} JavaScript declarations:")
        for i, decl in enumerate(js_decls):
            print(f"  {i+1}. {decl['type']} '{decl['name']}'")
            if decl.get('parameters'):
                print(f"     Parameters: {', '.join(decl['parameters'])}")
            print(f"     Lines: {decl['start_line']}-{decl['end_line']}")
            print(f"     Saved to: {js_files.get(decl['name'])}")
    except Exception as e:
        print(f"Error processing JavaScript file: {e}")
        traceback.print_exc()
    
    # Extract TypeScript declarations
    try:
        print(f"\nProcessing TypeScript file: {ts_file}")
        ts_decls, ts_files = extract_from_file(ts_file, ts_parser, 'typescript', output_dir)
        print(f"Extracted {len(ts_decls)} TypeScript declarations:")
        for i, decl in enumerate(ts_decls):
            print(f"  {i+1}. {decl['type']} '{decl['name']}'")
            if decl.get('parameters'):
                print(f"     Parameters: {', '.join(decl['parameters'])}")
            if decl.get('type_info'):
                for k, v in decl['type_info'].items():
                    if isinstance(v, list):
                        print(f"     {k.capitalize()}: {', '.join(v)}")
                    else:
                        print(f"     {k.capitalize()}: {v}")
            print(f"     Lines: {decl['start_line']}-{decl['end_line']}")
            print(f"     Saved to: {ts_files.get(decl['name'])}")
    except Exception as e:
        print(f"Error processing TypeScript file: {e}")
        traceback.print_exc()
    
    print(f"\nLLM USE CASES FOR ENHANCED JS/TS EXTRACTION:")
    print("="*80)
    print("The enhanced tree-sitter extraction provides the following benefits for LLMs:")
    print("1. Accurate parsing of complex JavaScript and TypeScript syntax")
    print("2. Type information extraction from TypeScript interfaces and classes")
    print("3. JSDoc comment extraction for function and class documentation")
    print("4. Inheritance relationships for classes and interfaces")
    print("5. Import/export context to understand dependencies")
    print("6. Support for arrow functions and variable-assigned functions")
    print("\nThis information enables LLMs to:")
    print("- Generate more accurate JavaScript and TypeScript code")
    print("- Understand complex type relationships in TypeScript codebases")
    print("- Maintain proper type signatures in generated code")
    print("- Respect class relationships and inheritance patterns")
    print("- Use appropriate imports and module patterns")
    
    # Save debug information if requested
    if args.debug:
        debug_file = Path(output_dir) / "extraction_debug.json"
        debug_data = {
            "javascript": js_decls,
            "typescript": ts_decls
        }
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2)
        print(f"\nDebug information saved to: {debug_file}")
    
    print(f"\n{'='*80}")
    print(f" EXTRACTION COMPLETED SUCCESSFULLY ".center(80, '='))
    print(f"{'='*80}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 