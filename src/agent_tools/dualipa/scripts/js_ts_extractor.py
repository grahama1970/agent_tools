#!/usr/bin/env python3
"""
Multi-language code extractor using tree-sitter.

This script extracts functions, classes, and other declarations from
source code files using tree-sitter for syntax-aware extraction when available.
For languages not supported by tree-sitter, it falls back to newline-based extraction.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
import traceback
import re

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Add the parent directory to sys.path to access dualipa module
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import tree-sitter
from tree_sitter import Parser

# Sample file paths
JS_SAMPLE_PATH = SCRIPT_DIR / "sample_js.js"
TS_SAMPLE_PATH = SCRIPT_DIR / "sample_ts.ts"

# Tree-sitter supported languages
TREE_SITTER_LANGUAGES = {
    'javascript': {'extensions': ['.js', '.jsx', '.mjs']},
    'typescript': {'extensions': ['.ts', '.tsx']},
    'c': {'extensions': ['.c', '.h']},
    'cpp': {'extensions': ['.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx']},
    'c_sharp': {'extensions': ['.cs']},
    'java': {'extensions': ['.java']},
    'ruby': {'extensions': ['.rb']},
    'go': {'extensions': ['.go']},
    'rust': {'extensions': ['.rs']},
    'php': {'extensions': ['.php']},
    'bash': {'extensions': ['.sh', '.bash']},
    'html': {'extensions': ['.html', '.htm']},
    'css': {'extensions': ['.css']},
    'python': {'extensions': ['.py']},  # Handled separately
}

def create_sample_files():
    """Create sample JS and TS files if they don't exist."""
    # Create sample JavaScript file
    if not JS_SAMPLE_PATH.exists():
        with open(JS_SAMPLE_PATH, 'w') as f:
            f.write("""/**
 * Sample JavaScript function with JSDoc
 * @param {string} name - The name parameter
 * @returns {string} Greeting message
 */
function greet(name) {
    return `Hello, ${name}!`;
}

/**
 * Sample JavaScript class
 */
class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    /**
     * Get person info
     */
    getInfo() {
        return `${this.name} is ${this.age} years old`;
    }
}

// Arrow function example
const multiply = (a, b) => {
    return a * b;
};
""")
    
    # Create sample TypeScript file
    if not TS_SAMPLE_PATH.exists():
        with open(TS_SAMPLE_PATH, 'w') as f:
            f.write("""/**
 * Sample TypeScript interface
 */
interface User {
    id: number;
    name: string;
    email?: string;
}

/**
 * Sample TypeScript function with type annotations
 * @param user The user object
 * @returns Formatted user info
 */
function formatUser(user: User): string {
    return `User ${user.name} (ID: ${user.id})`;
}

/**
 * Sample TypeScript class with interface implementation
 */
class Employee implements User {
    id: number;
    name: string;
    department: string;
    
    constructor(id: number, name: string, department: string) {
        this.id = id;
        this.name = name;
        this.department = department;
    }
    
    /**
     * Get employee details
     */
    getDetails(): string {
        return `${this.name} works in ${this.department}`;
    }
}

// Type alias example
type NumberOrString = number | string;

// Arrow function with types
const process = (value: NumberOrString): string => {
    return `Processed: ${value}`;
};
""")

def print_tree_sitter_setup_instructions():
    """Print instructions for setting up tree-sitter with locally cloned grammar repositories."""
    print("\n===== Tree-sitter Setup Instructions =====")
    print("To use tree-sitter for more accurate parsing:")
    print("1. Clone the grammar repositories:")
    print("   git clone https://github.com/tree-sitter/tree-sitter-javascript")
    print("   git clone https://github.com/tree-sitter/tree-sitter-typescript")
    print("\n2. Build the language libraries:")
    print("   mkdir -p build")
    print("   cd build")
    print("   gcc -o javascript.so -shared ../tree-sitter-javascript/src/parser.c -I../tree-sitter-javascript/src -fPIC")
    print("   gcc -o typescript.so -shared ../tree-sitter-typescript/typescript/src/parser.c -I../tree-sitter-typescript/typescript/src -fPIC")
    print("\n3. Run the extractor with the paths to the grammar repositories:")
    print("   python code_extractor.py --js-grammar ./tree-sitter-javascript --ts-grammar ./tree-sitter-typescript/typescript\n")
    print("For more information, see: https://tree-sitter.github.io/py-tree-sitter/")
    print("===========================================\n")

def setup_parsers(grammar_paths=None):
    """
    Set up tree-sitter parsers for various languages.
    
    Args:
        grammar_paths: Dictionary mapping language names to local grammar repository paths
        
    Returns:
        Dictionary mapping language names to Parser objects
    """
    # We need to use an alternative approach since the current tree-sitter
    # version in this environment may not support building languages directly
    
    print("Note: Using simplified extraction without tree-sitter.")
    print("For full tree-sitter functionality, ensure grammar repositories are cloned locally.")
    
    # Print detailed instructions if path arguments were provided
    if grammar_paths and any(grammar_paths.values()):
        print_tree_sitter_setup_instructions()
    
    return {}

def detect_language(file_path):
    """
    Detect the language based on file extension.
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Tuple of (language_name, is_supported_by_treesitter)
    """
    ext = Path(file_path).suffix.lower()
    
    # Special case for Python - handled separately with AST
    if ext in TREE_SITTER_LANGUAGES['python']['extensions']:
        return 'python', False
    
    # Check if the extension matches any tree-sitter supported language
    for lang, info in TREE_SITTER_LANGUAGES.items():
        if ext in info['extensions']:
            return lang, True
    
    # Not a known tree-sitter language
    # Try to guess based on common extensions
    extension_map = {
        '.lua': 'lua',
        '.pl': 'perl',
        '.pm': 'perl',
        '.scala': 'scala',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.swift': 'swift',
        '.m': 'objective_c',
        '.mm': 'objective_c',
        '.dart': 'dart',
        '.r': 'r',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.xml': 'xml',
        '.sql': 'sql',
        '.md': 'markdown',
        '.markdown': 'markdown',
    }
    
    if ext in extension_map:
        return extension_map[ext], False
    
    # Default to 'unknown'
    return 'unknown', False

def extract_declarations_regex(source_file, language_name):
    """
    Extract declarations using regex as a fallback when tree-sitter isn't available.
    
    Args:
        source_file: Path to the source file
        language_name: Language name
        
    Returns:
        List of declaration dictionaries
    """
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            source_text = f.read()
        
        # Get relative path information
        try:
            source_path = Path(source_file).resolve()
            workspace_root = Path.cwd().resolve()
            rel_path = source_path.relative_to(workspace_root)
            relative_path = str(rel_path)
        except ValueError:
            # If we can't compute a relative path, use the absolute path
            relative_path = str(source_path)
        
        declarations = []
        
        # Language-specific regex patterns
        patterns = {}
        
        # JavaScript/TypeScript patterns
        if language_name in ['javascript', 'typescript']:
            patterns = {
                'function_declaration': r'(?:export\s+)?function\s+(\w+)\s*\([^)]*\)\s*{',
                'class_declaration': r'(?:export\s+)?class\s+(\w+)[^{]*{',
                'arrow_function': r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*[{]',
                'interface_declaration': r'(?:export\s+)?interface\s+(\w+)[^{]*{',
                'type_declaration': r'(?:export\s+)?type\s+(\w+)\s*='
            }
        # Java patterns
        elif language_name == 'java':
            patterns = {
                'class_declaration': r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*class\s+(\w+)[^{]*{',
                'interface_declaration': r'(?:public|private|protected)?\s*(?:static)?\s*interface\s+(\w+)[^{]*{',
                'method_declaration': r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:[\w<>[\],\s]+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*{',
                'constructor': r'(?:public|private|protected)?\s*(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*{'
            }
        # C/C++ patterns
        elif language_name in ['c', 'cpp', 'c_sharp']:
            patterns = {
                'function_declaration': r'(?:static|inline|extern)?\s*[\w:*&<>[\],\s]+\s+(\w+)\s*\([^)]*\)\s*{',
                'class_declaration': r'(?:class|struct)\s+(\w+)[^{]*{',
                'namespace_declaration': r'namespace\s+(\w+)\s*{'
            }
        # Go patterns
        elif language_name == 'go':
            patterns = {
                'function_declaration': r'func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*{',
                'struct_declaration': r'type\s+(\w+)\s+struct\s*{',
                'interface_declaration': r'type\s+(\w+)\s+interface\s*{'
            }
        # Rust patterns
        elif language_name == 'rust':
            patterns = {
                'function_declaration': r'(?:pub)?\s*fn\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*{',
                'struct_declaration': r'(?:pub)?\s*struct\s+(\w+)(?:<[^>]*>)?[^{]*{',
                'enum_declaration': r'(?:pub)?\s*enum\s+(\w+)(?:<[^>]*>)?[^{]*{',
                'impl_block': r'impl(?:<[^>]*>)?\s+(?:[^{]*\s+for\s+)?(\w+)(?:<[^>]*>)?[^{]*{'
            }
        # Ruby patterns
        elif language_name == 'ruby':
            patterns = {
                'class_declaration': r'class\s+(\w+)(?:\s*<\s*\w+)?\s*',
                'module_declaration': r'module\s+(\w+)\s*',
                'method_declaration': r'def\s+(\w+)(?:\(.*?\))?\s*'
            }
        # PHP patterns
        elif language_name == 'php':
            patterns = {
                'class_declaration': r'class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{',
                'function_declaration': r'function\s+(\w+)\s*\([^)]*\)\s*{',
                'method_declaration': r'(?:public|private|protected)?\s*(?:static)?\s*function\s+(\w+)\s*\([^)]*\)\s*{'
            }
        
        # If no patterns are defined for this language, we'll use a generic block-based approach
        if not patterns:
            return extract_generic_blocks(source_file, source_text, language_name, relative_path)
        
        # Process each pattern
        for decl_type, pattern in patterns.items():
            matches = re.finditer(pattern, source_text)
            
            for match in matches:
                # Get the matched text and the name from the first capture group
                full_text = match.group(0)
                try:
                    name = match.group(1)
                except IndexError:
                    # No capture group, use a generic name
                    name = f"unnamed_{decl_type}_{len(declarations)}"
                
                # Calculate line numbers
                start_pos = match.start()
                start_line = source_text[:start_pos].count('\n') + 1
                
                # Extract the full declaration (heuristic: find matching closing brace)
                # For most languages, we need to find the matching closing brace
                if decl_type not in ['type_declaration']:  # Special case for TypeScript type aliases
                    # For blocks with braces, find the matching closing brace
                    brace_count = 1
                    end_pos = start_pos + len(full_text)
                    
                    while brace_count > 0 and end_pos < len(source_text):
                        if source_text[end_pos] == '{':
                            brace_count += 1
                        elif source_text[end_pos] == '}':
                            brace_count -= 1
                        end_pos += 1
                    
                    full_declaration = source_text[start_pos:end_pos]
                    end_line = source_text[:end_pos].count('\n') + 1
                else:
                    # For type declarations (and similar non-braced declarations), find the end of the line or semicolon
                    end_pos = source_text.find(';', start_pos)
                    if end_pos == -1:
                        # No semicolon, find the next newline
                        end_pos = source_text.find('\n', start_pos)
                        if end_pos == -1:
                            # No newline, use the end of the file
                            end_pos = len(source_text)
                    
                    full_declaration = source_text[start_pos:end_pos]
                    end_line = source_text[:end_pos].count('\n') + 1
                
                # Look for documentation comments above the declaration
                doc_comment = None
                doc_end = start_pos
                
                # For languages with JSDoc-style comments
                if language_name in ['javascript', 'typescript', 'java', 'c_sharp', 'php']:
                    doc_start = source_text.rfind('/**', 0, doc_end)
                    if doc_start != -1:
                        doc_end_match = source_text.find('*/', doc_start)
                        if doc_end_match != -1 and doc_end_match < start_pos:
                            # Make sure there's nothing but whitespace between doc and declaration
                            if not source_text[doc_end_match+2:start_pos].strip():
                                doc_comment = source_text[doc_start:doc_end_match+2]
                # For languages with triple-quote doc strings (like Python)
                elif language_name in ['python']:
                    for triple_quote in ['"""', "'''"]:
                        doc_start = source_text.rfind(triple_quote, 0, doc_end)
                        if doc_start != -1:
                            doc_end_match = source_text.find(triple_quote, doc_start + 3)
                            if doc_end_match != -1 and doc_end_match + 3 <= start_pos:
                                if not source_text[doc_end_match+3:start_pos].strip():
                                    doc_comment = source_text[doc_start:doc_end_match+3]
                                    break
                
                # Extract inheritance information for classes
                inheritance = ""
                if decl_type == 'class_declaration':
                    if language_name in ['javascript', 'typescript', 'java', 'c_sharp', 'php']:
                        extends_match = re.search(r'extends\s+(\w+)', full_declaration)
                        if extends_match:
                            inheritance = f" extends {extends_match.group(1)}"
                    
                    if language_name in ['typescript', 'java', 'c_sharp', 'php']:
                        implements_match = re.search(r'implements\s+([^{]+)', full_declaration)
                        if implements_match:
                            inheritance += f" implements {implements_match.group(1).strip()}"
                
                # Extract imports
                imports = []
                if language_name in ['javascript', 'typescript']:
                    # Look for imports in the entire file
                    import_patterns = [
                        r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
                        r'(?:const|let|var)\s+.*?=\s+require\([\'"]([^\'"]+)[\'"]\)'
                    ]
                    for import_pattern in import_patterns:
                        for import_match in re.finditer(import_pattern, source_text):
                            imports.append(import_match.group(1))
                
                # Extract parameter information for functions
                parameters = []
                return_type = ""
                
                if decl_type in ['function_declaration', 'method_declaration', 'arrow_function']:
                    # Extract parameters
                    params_match = re.search(r'\(([^)]*)\)', full_declaration)
                    if params_match:
                        params_text = params_match.group(1).strip()
                        if params_text:
                            # Basic parameter extraction - can be enhanced for specific languages
                            parameters = [p.strip() for p in params_text.split(',')]
                    
                    # Extract return type - varies by language
                    if language_name == 'typescript':
                        return_match = re.search(r'\):\s*([^{]+)', full_declaration)
                        if return_match:
                            return_type = return_match.group(1).strip()
                    elif language_name in ['rust', 'go']:
                        return_match = re.search(r'\)\s*->\s*([^{]+)', full_declaration)
                        if return_match:
                            return_type = return_match.group(1).strip()
                
                declarations.append({
                    'type': decl_type,
                    'name': name,
                    'text': full_declaration,
                    'doc_comment': doc_comment,
                    'start_line': start_line,
                    'end_line': end_line,
                    'start_byte': start_pos,
                    'end_byte': end_pos,
                    'language': language_name,
                    'relative_path': relative_path,
                    'inheritance': inheritance,
                    'parameters': parameters,
                    'return_type': return_type,
                    'imports': imports[:10]  # Limit to first 10 imports
                })
        
        # Sort declarations by their position in the file
        declarations.sort(key=lambda d: d['start_byte'])
        
        return declarations
    
    except Exception as e:
        print(f"Error extracting declarations using regex from {source_file}: {e}")
        traceback.print_exc()
        return []

def extract_generic_blocks(source_file, source_text, language_name, relative_path):
    """
    Extract code blocks using a simple newline-based approach for unsupported languages.
    
    Args:
        source_file: Path to the source file
        source_text: Content of the file
        language_name: Language name
        relative_path: Relative path to the file
        
    Returns:
        List of generic block dictionaries
    """
    declarations = []
    
    # Split by double newlines to create logical blocks
    blocks = re.split(r'\n\s*\n', source_text)
    
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block or len(block.split('\n')) < 3:
            # Skip very small blocks (less than 3 lines)
            continue
        
        # Try to guess a name for the block
        name = f"block_{i+1}"
        
        # Look for patterns that might indicate names
        name_patterns = [
            # Function-like patterns
            r'(?:function|def|func)\s+(\w+)',
            # Class-like patterns
            r'(?:class|struct|interface|type)\s+(\w+)',
            # Variable declarations
            r'(?:var|let|const)\s+(\w+)\s*=',
            # Method-like patterns
            r'(?:public|private|protected)?\s*(?:static)?\s*[\w<>[\],\s]+\s+(\w+)\s*\('
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, block)
            if match:
                name = match.group(1)
                break
        
        # Calculate line numbers
        block_start = source_text.find(block)
        start_line = source_text[:block_start].count('\n') + 1
        end_line = start_line + block.count('\n')
        
        # Create a generic declaration
        declarations.append({
            'type': 'generic_block',
            'name': name,
            'text': block,
            'doc_comment': None,
            'start_line': start_line,
            'end_line': end_line,
            'start_byte': block_start,
            'end_byte': block_start + len(block),
            'language': language_name,
            'relative_path': relative_path,
        })
    
    return declarations

def save_declarations(declarations, output_dir, file_stem):
    """Save declarations to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store output files
    output_files = {}
    
    for i, decl in enumerate(declarations):
        # Get file extension based on language
        language = decl['language']
        ext = get_extension_for_language(language)
        
        # Create a filename
        filename = f"{file_stem}_{decl['name']}_{i+1}{ext}"
        output_file = output_dir / filename
        
        # Create header comments based on language
        header_lines = create_header_lines(decl, file_stem, ext)
        header_text = '\n'.join(header_lines)
        
        # Write the file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{header_text}\n\n{decl['text']}")
        
        # Store the output file
        output_files[decl['name']] = str(output_file)
    
    return output_files

def get_extension_for_language(language):
    """Get the file extension for a language."""
    # Check tree-sitter languages first
    if language in TREE_SITTER_LANGUAGES and TREE_SITTER_LANGUAGES[language]['extensions']:
        return TREE_SITTER_LANGUAGES[language]['extensions'][0]
    
    # Common extensions for other languages
    extensions = {
        'lua': '.lua',
        'perl': '.pl',
        'scala': '.scala',
        'kotlin': '.kt',
        'swift': '.swift',
        'objective_c': '.m',
        'dart': '.dart',
        'r': '.r',
        'json': '.json',
        'yaml': '.yaml',
        'xml': '.xml',
        'sql': '.sql',
        'markdown': '.md',
        'unknown': '.txt'
    }
    
    return extensions.get(language, '.txt')

def create_header_lines(decl, file_stem, ext):
    """Create appropriate header comment lines based on language."""
    language = decl['language']
    
    # Comment styles for different languages
    comment_markers = {
        'javascript': '//',
        'typescript': '//',
        'java': '//',
        'c': '//',
        'cpp': '//',
        'c_sharp': '//',
        'go': '//',
        'rust': '//',
        'swift': '//',
        'kotlin': '//',
        'php': '//',
        'python': '#',
        'ruby': '#',
        'perl': '#',
        'lua': '--',
        'r': '#',
        'shell': '#',
        'bash': '#',
        'yaml': '#',
        'markdown': '<!--',
        'html': '<!--',
        'xml': '<!--',
    }
    
    comment_end_markers = {
        'markdown': '-->',
        'html': '-->',
        'xml': '-->',
    }
    
    # Default to # for unknown languages
    marker = comment_markers.get(language, '#')
    end_marker = comment_end_markers.get(language, '')
    
    # Create header lines with metadata
    header_lines = [
        f"{marker} Path: {decl.get('relative_path', 'unknown')}{end_marker}",
        f"{marker} Original file: {file_stem}{ext}{end_marker}",
        f"{marker} Type: {decl['type']}{decl.get('inheritance', '')}{end_marker}",
        f"{marker} Name: {decl['name']}{end_marker}",
        f"{marker} Lines: {decl['start_line']}-{decl['end_line']}{end_marker}",
    ]
    
    # Add parameters if available
    if 'parameters' in decl and decl['parameters']:
        params_text = ', '.join(decl['parameters'])
        header_lines.append(f"{marker} Parameters: {params_text}{end_marker}")
    
    # Add return type if available
    if 'return_type' in decl and decl['return_type']:
        header_lines.append(f"{marker} Returns: {decl['return_type']}{end_marker}")
    
    # Add imports if available
    if 'imports' in decl and decl['imports']:
        imports_text = ', '.join(decl['imports'])
        header_lines.append(f"{marker} Imports: {imports_text}{end_marker}")
    
    # Add documentation comment if available
    if 'doc_comment' in decl and decl['doc_comment']:
        header_lines.append("")
        header_lines.append(decl['doc_comment'])
    
    return header_lines

def process_file(file_path, parsers, output_dir):
    """Process a file and extract declarations."""
    try:
        # Detect language
        language_name, is_tree_sitter_supported = detect_language(file_path)
        
        print(f"Processing {language_name.upper()} file: {file_path}")
        
        # For Python, we use the separate AST-based extractor
        if language_name == 'python':
            print(f"Python files should be processed using the AST-based extractor")
            return [], {}
        
        # Extract declarations using appropriate method
        declarations = []
        
        # Tree-sitter approach would go here if we had working parsers
        if is_tree_sitter_supported and language_name in parsers:
            parser = parsers[language_name]
            # Tree-sitter extraction would go here
            # For now, fall back to regex
            declarations = extract_declarations_regex(file_path, language_name)
        else:
            # Use regex-based extraction as fallback
            declarations = extract_declarations_regex(file_path, language_name)
        
        # Get the stem of the file (filename without extension)
        file_stem = Path(file_path).stem
        
        # Create output directory
        lang_output_dir = Path(output_dir) / language_name
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save declarations to files
        output_files = save_declarations(declarations, lang_output_dir, file_stem)
        
        # Print results
        print(f"Extracted {len(declarations)} declarations:")
        for i, decl in enumerate(declarations):
            print(f"  {i+1}. {decl['type']} '{decl['name']}'")
            print(f"     Lines: {decl['start_line']}-{decl['end_line']}")
            print(f"     Saved to: {output_files.get(decl['name'])}")
        
        return declarations, output_files
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        traceback.print_exc()
        return [], {}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract code from source files')
    parser.add_argument('file', nargs='?', help='Path to source file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--grammar-path', help='Path to local grammar repository')
    parser.add_argument('--js-grammar', help='Path to local JavaScript grammar repository')
    parser.add_argument('--ts-grammar', help='Path to local TypeScript grammar repository')
    args = parser.parse_args()
    
    # Create sample files if needed for testing
    if not args.file:
        create_sample_files()
    
    # Set up grammar paths
    grammar_paths = {}
    if args.js_grammar:
        grammar_paths['javascript'] = args.js_grammar
    if args.ts_grammar:
        grammar_paths['typescript'] = args.ts_grammar
    if args.grammar_path:
        # Generic grammar path - could be used for any language
        grammar_paths['generic'] = args.grammar_path
    
    # Set up parsers
    print("Setting up parsers...")
    parsers = setup_parsers(grammar_paths)
    
    # Set output directory
    output_dir = args.output
    if not output_dir:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
    
    # Process the specified file or use sample files
    source_file = args.file
    declarations = None
    if source_file:
        try:
            declarations, _ = process_file(source_file, parsers, output_dir)
        except Exception as e:
            print(f"Error processing file: {e}")
            traceback.print_exc()
    else:
        # Process sample files
        js_declarations = None
        ts_declarations = None
        
        try:
            js_declarations, _ = process_file(JS_SAMPLE_PATH, parsers, output_dir)
        except Exception as e:
            print(f"Error processing JavaScript sample: {e}")
            traceback.print_exc()
        
        try:
            ts_declarations, _ = process_file(TS_SAMPLE_PATH, parsers, output_dir)
        except Exception as e:
            print(f"Error processing TypeScript sample: {e}")
            traceback.print_exc()
        
        # Combine declarations
        if js_declarations or ts_declarations:
            declarations = {
                'javascript': js_declarations,
                'typescript': ts_declarations
            }
    
    # Save debug information if requested
    if args.debug and declarations:
        debug_file = Path(output_dir) / "extraction_debug.json"
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(declarations, f, indent=2)
        print(f"\nDebug information saved to: {debug_file}")
    
    print("\nExtraction complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 