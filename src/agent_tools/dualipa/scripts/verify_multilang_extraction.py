#!/usr/bin/env python3
"""
Verification script for multilanguage extraction.

This standalone script verifies that code extraction works correctly
for all supported languages without using pytest's framework.
"""

import os
import sys
import tempfile
from pathlib import Path
import json
import shutil
import time

# Add parent directory to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent.parent
sys.path.append(str(parent_dir))

# Import code_extractor
from src.agent_tools.dualipa.code_extractor import extract_repository
try:
    # Try to import TREE_SITTER_LANGUAGES to check available grammars
    from src.agent_tools.dualipa.code_extractor import TREE_SITTER_LANGUAGES
    AVAILABLE_LANGUAGES = list(TREE_SITTER_LANGUAGES.keys())
except ImportError:
    AVAILABLE_LANGUAGES = []

# Sample code snippets for various languages
LANGUAGE_SAMPLES = {
    "python": """
def hello_world():
    \"\"\"Print hello world.\"\"\"
    print("Hello, Python!")
    
class TestClass:
    def test_method(self):
        return "Testing"
""",
    "javascript": """
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
""",
    "typescript": """
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
""",
    "go": """
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
""",
    "rust": """
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
""",
    "cpp": """
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
""",
    "c": """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Main function
 */
int main() {
    printf("Hello, C!\\n");
    return 0;
}

/**
 * A simple structure
 */
typedef struct {
    char* name;
    int age;
} Person;

/**
 * Create a greeting for the person
 */
char* get_greeting(Person* p) {
    char* greeting = malloc(100);
    sprintf(greeting, "Hello, my name is %s", p->name);
    return greeting;
}
""",
    "java": """
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
""",
    "ruby": """
# A simple function
def hello_world
  puts "Hello, Ruby!"
end

# Person class
class Person
  attr_reader :name, :age
  
  def initialize(name, age)
    @name = name
    @age = age
  end
  
  def greet
    "Hello, my name is #{@name}"
  end
end

# Call the function
hello_world()
""",
    "bash": """
#!/bin/bash

# Say hello function
function hello_world() {
    echo "Hello, Bash!"
}

# Person info function
function get_person_info() {
    local name=$1
    local age=$2
    echo "Name: $name, Age: $age"
}

# Call the function
hello_world
"""
}

def setup_test_files(test_dir):
    """Create test files for different languages."""
    # Create directories for each language
    for lang in LANGUAGE_SAMPLES:
        lang_dir = os.path.join(test_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Create sample file
        with open(os.path.join(lang_dir, f"test.{lang}"), "w") as f:
            f.write(LANGUAGE_SAMPLES[lang])
        
        # Special case for languages with different extensions
        if lang == "typescript":
            with open(os.path.join(lang_dir, "test.ts"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "javascript":
            with open(os.path.join(lang_dir, "test.js"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "cpp":
            with open(os.path.join(lang_dir, "test.cpp"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "c":
            with open(os.path.join(lang_dir, "test.c"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "java":
            # Java files need to match class name
            with open(os.path.join(lang_dir, "Test.java"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "ruby":
            with open(os.path.join(lang_dir, "test.rb"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "bash":
            with open(os.path.join(lang_dir, "test.sh"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "rust":
            with open(os.path.join(lang_dir, "test.rs"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])
        elif lang == "go":
            with open(os.path.join(lang_dir, "test.go"), "w") as f:
                f.write(LANGUAGE_SAMPLES[lang])

def verify_extraction_results(result, test_dir):
    """Verify the extraction results for various languages."""
    print("\n=== Testing Extraction Results ===")
    
    # Check overall structure
    required_keys = ["total_files", "code_files", "code_blocks", "languages"]
    for key in required_keys:
        if key not in result:
            print(f"❌ Missing key in result: {key}")
            return False
    
    print(f"✅ Result contains all required keys")
    
    # Check languages extracted
    languages_detected = set()
    declarations_by_language = {}
    
    for decl in result.get("declarations", []):
        lang = decl.get("language", "unknown")
        languages_detected.add(lang)
        
        if lang not in declarations_by_language:
            declarations_by_language[lang] = []
        declarations_by_language[lang].append(decl)
    
    # Print found languages
    print("\nLanguages detected in extraction:")
    for lang in sorted(languages_detected):
        count = len(declarations_by_language.get(lang, []))
        print(f"  - {lang}: {count} declarations")
    
    # Check for specific language features
    language_checks = {
        "python": lambda decls: any(d.get("type") == "class" for d in decls),
        "javascript": lambda decls: any(d.get("type") == "function" for d in decls),
        "typescript": lambda decls: any(d.get("type") in ["interface", "class"] for d in decls),
        "go": lambda decls: any(d.get("name") == "main" or d.get("type") == "type" for d in decls),
        "rust": lambda decls: any(d.get("name") == "main" or "struct" in d.get("type", "") for d in decls),
        "c": lambda decls: any(d.get("name") == "main" for d in decls),
        "cpp": lambda decls: any(d.get("type") == "class" for d in decls),
        "java": lambda decls: any("class" in d.get("type", "") for d in decls),
        "ruby": lambda decls: any(d.get("type") == "class" for d in decls),
        "bash": lambda decls: any("function" in d.get("type", "") for d in decls)
    }
    
    # Check each language
    print("\nVerifying language-specific extractions:")
    all_passed = True
    for lang, check_fn in language_checks.items():
        if lang in languages_detected:
            decls = declarations_by_language.get(lang, [])
            if decls and check_fn(decls):
                print(f"✅ {lang.capitalize()}: Found expected declarations")
            else:
                print(f"❌ {lang.capitalize()}: Failed to find expected declarations")
                all_passed = False
        else:
            print(f"⚠️ {lang.capitalize()}: No declarations detected")
            # Only mark as failure if tree-sitter should support this language
            if lang in AVAILABLE_LANGUAGES:
                print(f"  - Expected {lang} to be supported by tree-sitter")
                all_passed = False
    
    # Check code blocks directory structure
    blocks_dir = Path(test_dir) / "extraction" / "blocks" / "code"
    if blocks_dir.exists():
        print("\nChecking block extraction directories:")
        for lang in languages_detected:
            lang_block_dir = blocks_dir / lang
            if lang_block_dir.exists():
                num_files = len(list(lang_block_dir.glob("*")))
                print(f"✅ {lang.capitalize()}: Found {num_files} extracted block files")
            else:
                print(f"❌ {lang.capitalize()}: Block directory missing")
                all_passed = False
    
    return all_passed

def main():
    """Test multilanguage extraction without invoking pytest."""
    print("=== Multilanguage Extraction Verification ===")
    
    print(f"\nTree-sitter languages available: {len(AVAILABLE_LANGUAGES)}")
    if AVAILABLE_LANGUAGES:
        print(f"Supported languages: {', '.join(sorted(AVAILABLE_LANGUAGES))}")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        print(f"\nCreated test directory: {test_dir}")
        
        # Setup test files
        setup_test_files(temp_dir)
        print("Created test files for multiple languages")
        
        # Create extraction output directory
        extraction_dir = test_dir / "extraction"
        extraction_dir.mkdir()
        
        # Extract code
        print("\nExtracting code...")
        start_time = time.time()
        
        result = extract_repository(
            source=str(test_dir),
            output_path=str(extraction_dir),
            max_files=100,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        end_time = time.time()
        print(f"Extraction completed in {end_time - start_time:.2f} seconds")
        
        # Print basic stats
        print(f"\nTotal files processed: {result.get('total_files', 0)}")
        print(f"Code files: {result.get('code_files', 0)}")
        print(f"Code blocks: {result.get('code_blocks', 0)}")
        
        # Save result to file for inspection
        with open(test_dir / "extraction_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Verify extraction results
        success = verify_extraction_results(result, test_dir)
        
        if success:
            print("\n✅ All languages were extracted correctly!")
        else:
            print("\n⚠️ Some languages had extraction issues. Check the details above.")
            
        # Prompt to keep temp directory for debugging
        keep_temp = input("\nKeep temporary test directory for inspection? (y/n): ")
        if keep_temp.lower() == 'y':
            # Copy to a persistent location
            persistent_dir = "/tmp/multilang_test"
            if os.path.exists(persistent_dir):
                shutil.rmtree(persistent_dir)
            shutil.copytree(test_dir, persistent_dir)
            print(f"Test files copied to {persistent_dir}")

if __name__ == "__main__":
    main() 