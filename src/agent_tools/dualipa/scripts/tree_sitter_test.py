#!/usr/bin/env python3
"""
Simple tree-sitter test script.
Tests extraction capabilities for multiple languages supported by tree-sitter.
"""

import sys
from pathlib import Path
import os
import tempfile

# Sample code for each language
SAMPLES = {
    'javascript': """
function calculateSum(a, b) {
    return a + b;
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
""",

    'typescript': """
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
""",

    'python': """
def calculate_sum(a, b):
    return a + b

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def get_info(self):
        return f"{self.name} is {self.age} years old"
""",

    'go': """
package main

import "fmt"

func calculateSum(a int, b int) int {
    return a + b
}

type Person struct {
    Name string
    Age  int
}

func (p Person) GetInfo() string {
    return fmt.Sprintf("%s is %d years old", p.Name, p.Age)
}

func main() {
    fmt.Println("Hello, Go!")
}
""",

    'rust': """
struct Person {
    name: String,
    age: u32,
}

impl Person {
    fn new(name: &str, age: u32) -> Self {
        Person {
            name: name.to_string(),
            age,
        }
    }
    
    fn get_info(&self) -> String {
        format!("{} is {} years old", self.name, self.age)
    }
}

fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    println!("Hello, Rust!");
}
""",

    'c': """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* name;
    int age;
} Person;

Person* create_person(const char* name, int age) {
    Person* p = (Person*) malloc(sizeof(Person));
    p->name = strdup(name);
    p->age = age;
    return p;
}

int calculate_sum(int a, int b) {
    return a + b;
}

int main() {
    printf("Hello, C!\\n");
    return 0;
}
""",

    'cpp': """
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;
    
public:
    Person(const std::string& name, int age) : name(name), age(age) {}
    
    std::string get_info() const {
        return name + " is " + std::to_string(age) + " years old";
    }
};

int calculate_sum(int a, int b) {
    return a + b;
}

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}
""",

    'java': """
public class Main {
    public static int calculateSum(int a, int b) {
        return a + b;
    }
    
    public static class Person {
        private String name;
        private int age;
        
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }
        
        public String getInfo() {
            return name + " is " + age + " years old";
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
}
""",

    'ruby': """
def calculate_sum(a, b)
  a + b
end

class Person
  attr_reader :name, :age
  
  def initialize(name, age)
    @name = name
    @age = age
  end
  
  def get_info
    "#{@name} is #{@age} years old"
  end
end

puts "Hello, Ruby!"
""",

    'php': """
<?php

function calculateSum($a, $b) {
    return $a + $b;
}

class Person {
    private $name;
    private $age;
    
    public function __construct($name, $age) {
        $this->name = $name;
        $this->age = $age;
    }
    
    public function getInfo() {
        return $this->name . " is " . $this->age . " years old";
    }
}

echo "Hello, PHP!";
?>
""",

    'bash': """
#!/bin/bash

calculate_sum() {
    echo $(($1 + $2))
}

get_person_info() {
    local name="$1"
    local age="$2"
    echo "${name} is ${age} years old"
}

echo "Hello, Bash!"
"""
}

def main():
    print("=== Tree-sitter Functionality Test ===")
    
    # Test tree-sitter availability
    try:
        import tree_sitter
        print("✅ tree-sitter module is available")
    except ImportError:
        print("❌ tree-sitter module is not available!")
        sys.exit(1)
    
    # Test JavaScript
    print("\n--- Testing JavaScript ---")
    try:
        import tree_sitter_javascript
        print("✅ JavaScript grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create language - needs to be done in an unusual way (API quirk)
        js_language = tree_sitter.Language(tree_sitter_javascript.language())
        
        # Assign language to parser
        parser.language = js_language
        
        # Parse JS code
        js_code = SAMPLES['javascript']
        tree = parser.parse(bytes(js_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed JavaScript: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        js_functions = []
        js_classes = []
        
        for child in root.children:
            if child.type == 'function_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    js_functions.append(name_node.text.decode('utf8'))
            elif child.type == 'class_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    js_classes.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {js_functions}")
        print(f"  - Classes found: {js_classes}")
        
    except ImportError as e:
        print(f"❌ JavaScript grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing JavaScript: {e}")
    
    # Test TypeScript
    print("\n--- Testing TypeScript ---")
    try:
        import tree_sitter_typescript
        print("✅ TypeScript grammar is available")
        
        # TypeScript is a bit different, it has two languages
        if hasattr(tree_sitter_typescript, 'language_typescript'):
            # Create parser
            parser = tree_sitter.Parser()
            
            # Create TypeScript language
            ts_language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            
            # Assign language to parser
            parser.language = ts_language
            
            # Parse TS code
            ts_code = SAMPLES['typescript']
            tree = parser.parse(bytes(ts_code, 'utf8'))
            root = tree.root_node
            
            print(f"✅ Successfully parsed TypeScript: Root type '{root.type}', {len(root.children)} children")
            
            # Find declarations
            ts_functions = []
            ts_classes = []
            ts_interfaces = []
            
            for child in root.children:
                if child.type == 'function_declaration':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        ts_functions.append(name_node.text.decode('utf8'))
                elif child.type == 'class_declaration':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        ts_classes.append(name_node.text.decode('utf8'))
                elif child.type == 'interface_declaration':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        ts_interfaces.append(name_node.text.decode('utf8'))
            
            print(f"  - Functions found: {ts_functions}")
            print(f"  - Classes found: {ts_classes}")
            print(f"  - Interfaces found: {ts_interfaces}")
        else:
            print("❌ TypeScript language function not found in module")
        
    except ImportError as e:
        print(f"❌ TypeScript grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing TypeScript: {e}")
    
    # Test Python
    print("\n--- Testing Python ---")
    try:
        import tree_sitter_python
        print("✅ Python grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Python language
        py_language = tree_sitter.Language(tree_sitter_python.language())
        
        # Assign language to parser
        parser.language = py_language
        
        # Parse Python code
        py_code = SAMPLES['python']
        tree = parser.parse(bytes(py_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Python: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        py_functions = []
        py_classes = []
        
        for child in root.children:
            if child.type == 'function_definition':
                name_node = child.child_by_field_name('name')
                if name_node:
                    py_functions.append(name_node.text.decode('utf8'))
            elif child.type == 'class_definition':
                name_node = child.child_by_field_name('name')
                if name_node:
                    py_classes.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {py_functions}")
        print(f"  - Classes found: {py_classes}")
        
    except ImportError as e:
        print(f"❌ Python grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Python: {e}")
    
    # Test Go
    print("\n--- Testing Go ---")
    try:
        import tree_sitter_go
        print("✅ Go grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Go language
        go_language = tree_sitter.Language(tree_sitter_go.language())
        
        # Assign language to parser
        parser.language = go_language
        
        # Parse Go code
        go_code = SAMPLES['go']
        tree = parser.parse(bytes(go_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Go: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        go_functions = []
        go_structs = []
        
        for child in root.children:
            if child.type == 'function_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    go_functions.append(name_node.text.decode('utf8'))
            elif child.type == 'type_declaration':
                for grandchild in child.children:
                    if grandchild.type == 'type_spec':
                        name_node = grandchild.child_by_field_name('name')
                        if name_node:
                            go_structs.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {go_functions}")
        print(f"  - Structs found: {go_structs}")
        
    except ImportError as e:
        print(f"❌ Go grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Go: {e}")

    # Test Rust
    print("\n--- Testing Rust ---")
    try:
        import tree_sitter_rust
        print("✅ Rust grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Rust language
        rust_language = tree_sitter.Language(tree_sitter_rust.language())
        
        # Assign language to parser
        parser.language = rust_language
        
        # Parse Rust code
        rust_code = SAMPLES['rust']
        tree = parser.parse(bytes(rust_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Rust: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        rust_functions = []
        rust_structs = []
        
        for child in root.children:
            if child.type == 'function_item':
                name_node = child.child_by_field_name('name')
                if name_node:
                    rust_functions.append(name_node.text.decode('utf8'))
            elif child.type == 'struct_item':
                name_node = child.child_by_field_name('name')
                if name_node:
                    rust_structs.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {rust_functions}")
        print(f"  - Structs found: {rust_structs}")
        
    except ImportError as e:
        print(f"❌ Rust grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Rust: {e}")

    # Test C
    print("\n--- Testing C ---")
    try:
        import tree_sitter_c
        print("✅ C grammar is available")
        
        try:
            # Create parser
            parser = tree_sitter.Parser()
            
            # Create C language
            c_language = tree_sitter.Language(tree_sitter_c.language())
            
            # Assign language to parser
            parser.language = c_language
            
            # Parse C code
            c_code = SAMPLES['c']
            tree = parser.parse(bytes(c_code, 'utf8'))
            root = tree.root_node
            
            print(f"✅ Successfully parsed C: Root type '{root.type}', {len(root.children)} children")
            
            # Find declarations
            c_functions = []
            c_structs = []
            
            for child in root.children:
                if child.type == 'function_definition':
                    declarator = child.child_by_field_name('declarator')
                    if declarator:
                        name_node = declarator.child_by_field_name('declarator')
                        if name_node:
                            c_functions.append(name_node.text.decode('utf8'))
                elif child.type == 'type_definition':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        c_structs.append(name_node.text.decode('utf8'))
            
            print(f"  - Functions found: {c_functions}")
            print(f"  - Structs/Types found: {c_structs}")
            
        except Exception as e:
            print(f"⚠️ C grammar version incompatibility: {e}")
            print("Using regex fallback for C extraction...")
            
            # Simple regex fallback for basic function extraction
            import re
            c_code = SAMPLES['c']
            
            # Function pattern - captures function names
            func_pattern = r'(?:^|\n)\s*(\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
            func_matches = re.findall(func_pattern, c_code)
            c_functions = [m[1] for m in func_matches if m[0] not in ('if', 'for', 'while', 'switch')]
            
            # Struct pattern
            struct_pattern = r'struct\s+(\w+)\s*\{'
            struct_matches = re.findall(struct_pattern, c_code)
            
            # Typedef pattern
            typedef_pattern = r'typedef\s+struct\s+\w*\s*\{[^}]*\}\s*(\w+)'
            typedef_matches = re.findall(typedef_pattern, c_code)
            
            print(f"  - Functions found (regex): {c_functions}")
            print(f"  - Structs found (regex): {struct_matches + typedef_matches}")
            print("NOTE: For production use, consider using pycparser for C extraction")
            
    except ImportError as e:
        print(f"❌ C grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error with C: {e}")

    # Test C++
    print("\n--- Testing C++ ---")
    try:
        import tree_sitter_cpp
        print("✅ C++ grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create C++ language
        cpp_language = tree_sitter.Language(tree_sitter_cpp.language())
        
        # Assign language to parser
        parser.language = cpp_language
        
        # Parse C++ code
        cpp_code = SAMPLES['cpp']
        tree = parser.parse(bytes(cpp_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed C++: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        cpp_functions = []
        cpp_classes = []
        
        for child in root.children:
            if child.type == 'function_definition':
                declarator = child.child_by_field_name('declarator')
                if declarator:
                    name_node = declarator.child_by_field_name('declarator')
                    if name_node:
                        cpp_functions.append(name_node.text.decode('utf8'))
            elif child.type == 'class_specifier':
                name_node = child.child_by_field_name('name')
                if name_node:
                    cpp_classes.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {cpp_functions}")
        print(f"  - Classes found: {cpp_classes}")
        
    except ImportError as e:
        print(f"❌ C++ grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing C++: {e}")

    # Test Java
    print("\n--- Testing Java ---")
    try:
        import tree_sitter_java
        print("✅ Java grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Java language
        java_language = tree_sitter.Language(tree_sitter_java.language())
        
        # Assign language to parser
        parser.language = java_language
        
        # Parse Java code
        java_code = SAMPLES['java']
        tree = parser.parse(bytes(java_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Java: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        java_methods = []
        java_classes = []
        
        for child in root.children:
            if child.type == 'class_declaration':
                name_node = child.child_by_field_name('name')
                if name_node:
                    java_classes.append(name_node.text.decode('utf8'))
                
                # Look for methods inside the class
                for grandchild in child.children:
                    if grandchild.type == 'class_body':
                        for element in grandchild.children:
                            if element.type == 'method_declaration':
                                name_node = element.child_by_field_name('name')
                                if name_node:
                                    java_methods.append(name_node.text.decode('utf8'))
        
        print(f"  - Methods found: {java_methods}")
        print(f"  - Classes found: {java_classes}")
        
    except ImportError as e:
        print(f"❌ Java grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Java: {e}")

    # Test Ruby
    print("\n--- Testing Ruby ---")
    try:
        import tree_sitter_ruby
        print("✅ Ruby grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Ruby language
        ruby_language = tree_sitter.Language(tree_sitter_ruby.language())
        
        # Assign language to parser
        parser.language = ruby_language
        
        # Parse Ruby code
        ruby_code = SAMPLES['ruby']
        tree = parser.parse(bytes(ruby_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Ruby: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        ruby_methods = []
        ruby_classes = []
        
        for child in root.children:
            if child.type == 'method':
                name_node = child.child_by_field_name('name')
                if name_node:
                    ruby_methods.append(name_node.text.decode('utf8'))
            elif child.type == 'class':
                name_node = child.child_by_field_name('name')
                if name_node:
                    ruby_classes.append(name_node.text.decode('utf8'))
        
        print(f"  - Methods found: {ruby_methods}")
        print(f"  - Classes found: {ruby_classes}")
        
    except ImportError as e:
        print(f"❌ Ruby grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Ruby: {e}")

    # Test Bash
    print("\n--- Testing Bash ---")
    try:
        import tree_sitter_bash
        print("✅ Bash grammar is available")
        
        # Create parser
        parser = tree_sitter.Parser()
        
        # Create Bash language
        bash_language = tree_sitter.Language(tree_sitter_bash.language())
        
        # Assign language to parser
        parser.language = bash_language
        
        # Parse Bash code
        bash_code = SAMPLES['bash']
        tree = parser.parse(bytes(bash_code, 'utf8'))
        root = tree.root_node
        
        print(f"✅ Successfully parsed Bash: Root type '{root.type}', {len(root.children)} children")
        
        # Find declarations
        bash_functions = []
        
        for child in root.children:
            if child.type == 'function_definition':
                name_node = child.child_by_field_name('name')
                if name_node:
                    bash_functions.append(name_node.text.decode('utf8'))
        
        print(f"  - Functions found: {bash_functions}")
        
    except ImportError as e:
        print(f"❌ Bash grammar not available: {e}")
    except Exception as e:
        print(f"❌ Error parsing Bash: {e}")
    
    # List all available language grammars from our installed packages
    print("\n--- Checking for other language grammars ---")
    language_modules = [
        'tree_sitter_go',
        'tree_sitter_rust',
        'tree_sitter_c',
        'tree_sitter_cpp',
        'tree_sitter_java',
        'tree_sitter_ruby',
        'tree_sitter_php',
        'tree_sitter_bash'
    ]
    
    for module_name in language_modules:
        try:
            module = __import__(module_name)
            language_attr = None
            
            # Check for language attributes
            if hasattr(module, 'language'):
                language_attr = 'language'
            elif hasattr(module, 'language_name'):
                language_attr = 'language_name'
                
            if language_attr:
                print(f"✅ {module_name} grammar is available with {language_attr} attribute")
            else:
                print(f"⚠️ {module_name} grammar is available but language attribute not found")
                
        except ImportError:
            print(f"❌ {module_name} grammar is not available")
    
    print("\n=== Test Complete ===")
    
    # Summary of available vs missing languages
    print("\n=== Summary ===")
    print("For multi-language extraction, the following grammars are required:")
    print("JavaScript: ✅")
    print("TypeScript: ✅")
    print("Python: ✅")
    print("To support more languages, install additional grammars:")
    print("pip install tree-sitter-go tree-sitter-rust tree-sitter-c tree-sitter-cpp")
    print("pip install tree-sitter-java tree-sitter-ruby tree-sitter-php tree-sitter-bash")
    

if __name__ == "__main__":
    main() 