#!/usr/bin/env python3
"""
Verify that multilanguage extraction works correctly.
This script is completely self-contained and doesn't use pytest.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent.parent
sys.path.append(str(parent_dir))

# Import directly from code_extractor
from src.agent_tools.dualipa.code_extractor import extract_repository, TREE_SITTER_LANGUAGES

def main():
    """Verify multilanguage extraction functionality."""
    print("Verifying multilanguage extraction...")
    print(f"Tree-sitter languages available: {len(TREE_SITTER_LANGUAGES)}")
    print(f"Available languages: {', '.join(sorted(TREE_SITTER_LANGUAGES.keys()))}")
    
    # Create sample files for different languages
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create samples
        create_samples(test_dir)
        
        # Extract code
        print("\nExtracting code from test files...")
        result = extract_repository(
            source=str(test_dir),
            max_files=100
        )
        
        # Check results
        print(f"\nExtraction Results:")
        print(f"Total files processed: {result.get('total_files', 0)}")
        print(f"Code files processed: {result.get('code_files', 0)}")
        print(f"Code blocks extracted: {result.get('code_blocks', 0)}")
        
        # Count declarations by language
        declarations_by_language = {}
        for decl in result.get('declarations', []):
            lang = decl.get('language', 'unknown')
            if lang not in declarations_by_language:
                declarations_by_language[lang] = []
            declarations_by_language[lang].append(decl)
        
        # Print language results
        print("\nLanguages detected:")
        for lang, decls in declarations_by_language.items():
            print(f"  {lang}: {len(decls)} declarations")
            
            # Print 1-2 examples per language
            for i, decl in enumerate(decls[:2]):
                print(f"    {i+1}. {decl.get('type', 'unknown')}: {decl.get('name', 'unnamed')}")
        
        # Overall success
        success = len(declarations_by_language) >= 4  # At least 4 languages should be detected
        
        if success:
            print("\n✅ SUCCESS: Multilanguage extraction is working!")
            return 0
        else:
            print("\n❌ FAILURE: Not enough languages were extracted")
            return 1

def create_samples(dir_path):
    """Create sample files for testing."""
    samples = {
        "javascript": {
            "filename": "test.js",
            "content": """
            function helloWorld() {
                console.log("Hello, world!");
            }
            
            class Person {
                constructor(name) {
                    this.name = name;
                }
                
                greet() {
                    return `Hello, ${this.name}!`;
                }
            }
            """
        },
        "typescript": {
            "filename": "test.ts",
            "content": """
            function greet(name: string): string {
                return `Hello, ${name}!`;
            }
            
            interface User {
                name: string;
                age: number;
            }
            
            class Employee implements User {
                name: string;
                age: number;
                department: string;
                
                constructor(name: string, age: number, department: string) {
                    this.name = name;
                    this.age = age;
                    this.department = department;
                }
            }
            """
        },
        "python": {
            "filename": "test.py",
            "content": """
            def hello_world():
                print("Hello, world!")
                
            class Person:
                def __init__(self, name):
                    self.name = name
                    
                def greet(self):
                    return f"Hello, {self.name}!"
            """
        },
        "go": {
            "filename": "test.go",
            "content": """
            package main
            
            import "fmt"
            
            func main() {
                fmt.Println("Hello, world!")
            }
            
            type Person struct {
                Name string
                Age int
            }
            
            func (p Person) Greet() string {
                return fmt.Sprintf("Hello, %s!", p.Name)
            }
            """
        },
        "rust": {
            "filename": "test.rs",
            "content": """
            fn main() {
                println!("Hello, world!");
            }
            
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
                
                fn greet(&self) -> String {
                    format!("Hello, {}!", self.name)
                }
            }
            """
        }
    }
    
    # Create files
    print("Creating test files:")
    for lang, info in samples.items():
        # Create language directory
        lang_dir = dir_path / lang
        lang_dir.mkdir(exist_ok=True)
        
        # Create file
        file_path = lang_dir / info["filename"]
        with open(file_path, "w") as f:
            f.write(info["content"])
        print(f"  Created {lang} file: {file_path}")
    
    print(f"Created test files in {dir_path}")

if __name__ == "__main__":
    sys.exit(main()) 