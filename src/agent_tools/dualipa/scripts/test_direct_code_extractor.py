#!/usr/bin/env python3
"""
Test direct code extractor import without loading train_lora or unsloth.

This script demonstrates importing and using code_extractor directly without
importing any training-related modules to avoid the unsloth import overhead.
"""

import os
import sys
import tempfile
from pathlib import Path
import json
import time

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent.parent
sys.path.append(str(parent_dir))

# Direct import of code_extractor only
# We avoid importing from agent_tools.dualipa directly to prevent __init__.py loading everything
from src.agent_tools.dualipa.code_extractor import extract_repository
from src.agent_tools.dualipa.github_utils import is_github_url
from src.agent_tools.dualipa.language_detection import detect_language

# Create test directory with sample files
def setup_test_files(test_dir):
    """Create test files for different languages."""
    # Create language directories
    langs = ["python", "javascript", "typescript", "go", "rust", "cpp", "java"]
    
    for lang in langs:
        os.makedirs(os.path.join(test_dir, lang), exist_ok=True)
    
    # Python test file
    with open(os.path.join(test_dir, "python", "test.py"), "w") as f:
        f.write("""
def hello_world():
    \"\"\"Print hello world.\"\"\"
    print("Hello, Python!")
    
class TestClass:
    def test_method(self):
        return "Testing"
""")
    
    # JavaScript test file
    with open(os.path.join(test_dir, "javascript", "test.js"), "w") as f:
        f.write("""
/**
 * Say hello in JavaScript
 */
function helloWorld() {
    console.log("Hello, JavaScript!");
}

class TestClass {
    constructor() {
        this.name = "test";
    }
    
    sayHello() {
        return `Hello from ${this.name}`;
    }
}
""")
    
    # TypeScript test file
    with open(os.path.join(test_dir, "typescript", "test.ts"), "w") as f:
        f.write("""
/**
 * Say hello in TypeScript
 * @param name Optional name parameter
 */
function helloWorld(name?: string): void {
    console.log(`Hello, ${name || 'TypeScript'}!`);
}

interface Person {
    name: string;
    age: number;
}

class TypedClass {
    private data: string;
    
    constructor(data: string) {
        this.data = data;
    }
    
    getData(): string {
        return this.data;
    }
}
""")
    
    # Go test file
    with open(os.path.join(test_dir, "go", "test.go"), "w") as f:
        f.write("""
package main

import "fmt"

// Main function
func main() {
    fmt.Println("Hello, Go!")
}

// Greeter struct
type Greeter struct {
    Message string
}

// SayHello method
func (g Greeter) SayHello() string {
    return g.Message
}
""")

    # Rust test file
    with open(os.path.join(test_dir, "rust", "test.rs"), "w") as f:
        f.write("""
/// Main function
fn main() {
    println!("Hello, Rust!");
}

/// A simple structure
struct Person {
    name: String,
    age: u32,
}

impl Person {
    /// Create a new person
    fn new(name: &str, age: u32) -> Person {
        Person {
            name: name.to_string(),
            age,
        }
    }
    
    /// Get person's greeting
    fn greet(&self) -> String {
        format!("Hello, my name is {} and I am {}", self.name, self.age)
    }
}
""")

    # C++ test file
    with open(os.path.join(test_dir, "cpp", "test.cpp"), "w") as f:
        f.write("""
#include <iostream>
#include <string>

/**
 * Main function
 */
int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}

/**
 * Person class
 */
class Person {
private:
    std::string name;
    int age;
    
public:
    Person(std::string name, int age) : name(name), age(age) {}
    
    std::string greet() {
        return "Hello, my name is " + name;
    }
};
""")

    # Java test file
    with open(os.path.join(test_dir, "java", "Test.java"), "w") as f:
        f.write("""
package com.example;

/**
 * Main Test class
 */
public class Test {
    /**
     * Main method
     */
    public static void main(String[] args) {
        System.out.println("Hello, Java!");
    }
    
    /**
     * Greeter class
     */
    public static class Greeter {
        private String message;
        
        public Greeter(String message) {
            this.message = message;
        }
        
        public String greet() {
            return message;
        }
    }
}
""")

    return test_dir

def main():
    """Test code extraction without loading unsloth."""
    print("Starting direct code extractor test...")
    start_time = time.time()
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as test_dir:
        print(f"Created test directory: {test_dir}")
        
        # Setup test files
        setup_test_files(test_dir)
        print(f"Created test files for multiple languages")
        
        # Extract repository
        print("Extracting code from test files...")
        result = extract_repository(
            source=test_dir,
            output_path=None,
            max_files=1000,
            include_patterns=None,
            exclude_patterns=None,
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Print statistics
        print(f"\nExtraction completed in {time.time() - start_time:.2f} seconds")
        
        # Examine result structure
        print("\nResult structure:")
        for key in result:
            print(f"  {key}")
        
        # Count declarations by language
        languages = {}
        for decl in result.get('declarations', []):
            lang = decl.get('language', 'unknown')
            if lang not in languages:
                languages[lang] = 0
            languages[lang] += 1
        
        # Print extraction by language
        print("\nExtraction by language:")
        for lang, count in languages.items():
            print(f"  {lang}: {count} declarations")
        
        # Save result to file
        output_file = Path(test_dir) / "extraction_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print first few declarations for demonstration
        print("\nSample extracted declarations:")
        for i, decl in enumerate(result.get('declarations', [])[:5]):
            print(f"\n{i+1}. {decl.get('language', 'unknown')} {decl.get('type', 'unknown')} from {decl.get('relative_path', 'unknown')}")
            print(f"   Name: {decl.get('name', 'unnamed')}")
            if 'docstring' in decl and decl['docstring']:
                print(f"   Doc: {decl['docstring'].strip()[:50]}...")
            if 'start_line' in decl and 'end_line' in decl:
                print(f"   Lines: {decl['start_line']}-{decl['end_line']}")

if __name__ == "__main__":
    main() 