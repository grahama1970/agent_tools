#!/usr/bin/env python3
"""
Test multilanguage extraction using code_extractor with tree-sitter.

This script creates test files for multiple languages and extracts 
code blocks from them to verify the tree-sitter integration.

Official Documentation References:
- tree-sitter: https://tree-sitter.github.io/tree-sitter/
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

import os
import sys
import tempfile
from pathlib import Path
import argparse
import json

# Add parent directory to path to import code_extractor
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent.parent
sys.path.append(str(parent_dir))

try:
    from agent_tools.dualipa.code_extractor import extract_repository
except ImportError:
    print("Failed to import code_extractor. Make sure you're running from the correct directory.")
    sys.exit(1)

def create_test_files(test_dir):
    """Create test files for multiple languages."""
    # JavaScript sample
    js_file = test_dir / "sample.js"
    with open(js_file, "w") as f:
        f.write("""
function calculateSum(a, b) {
    return a + b;
}

class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }
    
    greet() {
        return `Hello, my name is ${this.name}`;
    }
}
""")
    
    # TypeScript sample
    ts_file = test_dir / "sample.ts"
    with open(ts_file, "w") as f:
        f.write("""
interface User {
    id: number;
    name: string;
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

function formatUser(user: User): string {
    return `User: ${user.name} (ID: ${user.id})`;
}
""")
    
    # Python sample
    py_file = test_dir / "sample.py"
    with open(py_file, "w") as f:
        f.write("""
def calculate_sum(a, b):
    \"\"\"Add two numbers together.\"\"\"
    return a + b

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, my name is {self.name}"
""")

    # Go sample
    go_file = test_dir / "sample.go"
    with open(go_file, "w") as f:
        f.write("""
package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func calculateSum(a, b int) int {
    return a + b
}

func main() {
    fmt.Println("Hello, Go!")
}
""")

    # Rust sample
    rust_file = test_dir / "sample.rs"
    with open(rust_file, "w") as f:
        f.write("""
struct Person {
    name: String,
    age: u32,
}

fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    println!("Hello, Rust!");
}
""")

    # C++ sample
    cpp_file = test_dir / "sample.cpp"
    with open(cpp_file, "w") as f:
        f.write("""
#include <iostream>
#include <string>

class Person {
public:
    std::string name;
    int age;
    
    Person(std::string name, int age) : name(name), age(age) {}
    
    void greet() {
        std::cout << "Hello, my name is " << name << std::endl;
    }
};

int calculate_sum(int a, int b) {
    return a + b;
}

int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}
""")

    # Java sample
    java_file = test_dir / "Main.java"
    with open(java_file, "w") as f:
        f.write("""
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
    
    public static int calculateSum(int a, int b) {
        return a + b;
    }
}
""")

    # Ruby sample
    ruby_file = test_dir / "sample.rb"
    with open(ruby_file, "w") as f:
        f.write("""
def calculate_sum(a, b)
  a + b
end

class Person
  attr_accessor :name, :age
  
  def initialize(name, age)
    @name = name
    @age = age
  end
  
  def greet
    "Hello, my name is #{@name}"
  end
end
""")

    # Bash sample
    bash_file = test_dir / "sample.sh"
    with open(bash_file, "w") as f:
        f.write("""
#!/bin/bash

function calculate_sum() {
    echo $(($1 + $2))
}

function get_person_info() {
    echo "Name: $1, Age: $2"
}

# Main script
echo "Hello, Bash!"
calculate_sum 5 7
get_person_info "John" 30
""")
    
    print(f"Created test files in {test_dir}")
    print(f"  - JavaScript: {js_file}")
    print(f"  - TypeScript: {ts_file}")
    print(f"  - Python: {py_file}")
    print(f"  - Go: {go_file}")
    print(f"  - Rust: {rust_file}")
    print(f"  - C++: {cpp_file}")
    print(f"  - Java: {java_file}")
    print(f"  - Ruby: {ruby_file}")
    print(f"  - Bash: {bash_file}")

def main():
    """Run the test for multiple languages."""
    parser = argparse.ArgumentParser(description="Test multi-language extraction")
    parser.add_argument("--output", help="Output directory for extracted files")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        print(f"Created temporary directory: {test_dir}")
        
        # Create test files
        create_test_files(test_dir)
        
        # Set up output directory
        output_dir = Path(args.output) if args.output else test_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract code
        print(f"Extracting code to {output_dir}...")
        stats = extract_repository(
            source=str(test_dir),
            output_path=str(output_dir),
            extract_documentation=False
        )
        
        # Print results
        print("\n=== Extraction Results ===")
        print(f"Total files processed: {stats['total_files']}")
        print(f"Code files: {stats['code_files']}")
        print(f"Code blocks extracted: {stats['code_blocks']}")
        print(f"Languages: {list(stats['languages'].keys())}")
        
        # Print detailed statistics if debug is enabled
        if args.debug:
            print("\n=== Detailed Statistics ===")
            print(json.dumps(stats, indent=2))
            
            print("\n=== Output Directory Structure ===")
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(str(output_dir), '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{sub_indent}{f}")

if __name__ == "__main__":
    main() 