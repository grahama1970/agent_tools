#!/usr/bin/env python3
"""
Simple test script for tree-sitter functionality.
This verifies that we can extract code from various languages using tree-sitter.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the tree-sitter modules
try:
    import tree_sitter
    from tree_sitter import Language
    print("✅ tree-sitter is installed correctly")
except ImportError as e:
    print(f"❌ Failed to import tree-sitter: {e}")
    sys.exit(1)

# Try to import language modules
languages = {
    'javascript': 'tree_sitter_javascript',
    'typescript': 'tree_sitter_typescript',
    'python': 'tree_sitter_python'
}

for lang_name, module_name in languages.items():
    try:
        module = __import__(module_name)
        print(f"✅ {lang_name} grammar is available ({module_name})")
    except ImportError as e:
        print(f"❌ {lang_name} grammar failed to import: {e}")

# Test JavaScript parsing
js_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    getInfo() {
        return `${this.name} is ${this.age} years old`;
    }
}
"""

# Test TypeScript parsing
ts_code = """
interface User {
    id: number;
    name: string;
}

function formatUser(user: User): string {
    return `User ${user.name} (ID: ${user.id})`;
}

class Employee implements User {
    id: number;
    name: string;
    department: string;
    
    constructor(id: number, name: string, department: string) {
        this.id = id;
        this.name = name;
        this.department = department;
    }
}
"""

def parse_code(language_module, code):
    try:
        # Load from the module directly
        language = Language(language_module.language())
        parser = tree_sitter.Parser()
        parser.set_language(language)
        
        # Parse the code
        tree = parser.parse(bytes(code, "utf8"))
        
        # Get the root node
        root_node = tree.root_node
        
        # Count declarations (functions, classes, interfaces)
        declarations = []
        
        # Inspect child nodes
        for child in root_node.children:
            if child.type in ['function_declaration', 'class_declaration', 'interface_declaration']:
                declarations.append({
                    'type': child.type,
                    'start_line': child.start_point[0] + 1,
                    'end_line': child.end_point[0] + 1
                })
        
        return True, declarations
    except Exception as e:
        return False, f"Error parsing code: {e}"

# Test JavaScript parsing
print("\n--- Testing JavaScript parsing ---")
try:
    import tree_sitter_javascript
    success, result = parse_code(tree_sitter_javascript, js_code)
    if success:
        print(f"✅ JavaScript parsing successful. Found {len(result)} declarations:")
        for i, decl in enumerate(result):
            print(f"  {i+1}. {decl['type']} (lines {decl['start_line']}-{decl['end_line']})")
    else:
        print(f"❌ JavaScript parsing failed: {result}")
except ImportError:
    print("❌ Cannot test JavaScript parsing - tree_sitter_javascript not available")

# Test TypeScript parsing
print("\n--- Testing TypeScript parsing ---")
try:
    import tree_sitter_typescript
    success, result = parse_code(tree_sitter_typescript, ts_code)
    if success:
        print(f"✅ TypeScript parsing successful. Found {len(result)} declarations:")
        for i, decl in enumerate(result):
            print(f"  {i+1}. {decl['type']} (lines {decl['start_line']}-{decl['end_line']})")
    else:
        print(f"❌ TypeScript parsing failed: {result}")
except ImportError:
    print("❌ Cannot test TypeScript parsing - tree_sitter_typescript not available")

print("\n--- Test complete ---") 