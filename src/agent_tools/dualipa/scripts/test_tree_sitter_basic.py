#!/usr/bin/env python3
"""
Basic test script to understand how to use tree-sitter with JavaScript and TypeScript.
"""

import sys
import importlib
from pathlib import Path

# Try to import the necessary modules
try:
    from tree_sitter import Language, Parser
    print("Successfully imported tree_sitter module")
    
    # List all installed modules
    print("\nChecking installed tree-sitter modules:")
    for module_name in ['tree_sitter', 'tree_sitter_javascript', 'tree_sitter_typescript', 'tree_sitter_python']:
        try:
            module = importlib.import_module(module_name)
            print(f"  - {module_name}: {module}")
            # Print module attributes
            print(f"    Attributes: {dir(module)[:10]}...")
        except ImportError as e:
            print(f"  - {module_name}: Not installed ({e})")
            
    # Try to parse a simple JavaScript file
    print("\nAttempting to parse JavaScript:")
    js_code = b"""
    function hello(name) {
        console.log(`Hello, ${name}!`);
    }
    
    class Person {
        constructor(name) {
            this.name = name;
        }
        
        greet() {
            console.log(`Hi, I'm ${this.name}`);
        }
    }
    """
    
    # Try different approaches to set up the JavaScript parser
    try:
        # Approach 1: Try direct import
        import tree_sitter_javascript
        print("  Approach 1: Using direct import")
        print(f"  Module attributes: {dir(tree_sitter_javascript)}")
        
        # Look for language-related attributes
        language_attrs = [attr for attr in dir(tree_sitter_javascript) if 'language' in attr.lower()]
        print(f"  Language-related attributes: {language_attrs}")
        
    except Exception as e:
        print(f"  Error in Approach 1: {e}")
    
    # Approach 2: Try to build from sources
    try:
        print("\n  Approach 2: Building from sources")
        # This is the approach shown in some examples
        Language.build_library(
            'build/languages.so',
            [
                # Try different paths that might work
                'tree-sitter-javascript',
                # Maybe these are installed in site-packages?
                str(Path(sys.prefix) / 'lib' / 'python3.10' / 'site-packages' / 'tree_sitter_javascript'),
            ]
        )
        print("  Successfully built language library")
    except Exception as e:
        print(f"  Error in Approach 2: {e}")
    
    # Approach 3: Try more direct instantiation
    try:
        print("\n  Approach 3: Direct instantiation")
        parser = Parser()
        
        # Try to inspect the tree_sitter_javascript module more thoroughly
        import tree_sitter_javascript
        
        # Check if there's anything callable
        callables = [attr for attr in dir(tree_sitter_javascript) 
                    if callable(getattr(tree_sitter_javascript, attr)) 
                    and not attr.startswith('__')]
        print(f"  Callable attributes: {callables}")
        
        # Try various ways to get language data
        try:
            if hasattr(tree_sitter_javascript, '_language_path'):
                lang_path = getattr(tree_sitter_javascript, '_language_path')
                print(f"  Found language path: {lang_path}")
                if Path(lang_path).exists():
                    print(f"  Language file exists at: {lang_path}")
                    # Try to load from this path
                    js_lang = Language(tree_sitter_javascript.language())
                    parser.language = js_lang
                    tree = parser.parse(js_code)
                    print(f"  Parse successful. Root node type: {tree.root_node.type}")
                else:
                    print("  Language file does not exist")
        except Exception as e:
            print(f"  Error with language path: {e}")
            
    except Exception as e:
        print(f"  Error in Approach 3: {e}")
        
except ImportError as e:
    print(f"Failed to import tree_sitter: {e}")
    sys.exit(1)

print("\nTest complete!") 