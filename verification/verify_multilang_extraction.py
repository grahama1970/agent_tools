#!/usr/bin/env python3
"""
Verify multi-language code extraction functionality.

This script tests the multi-language code extraction functionality in the DuaLipa library,
which extracts code blocks from files of various programming languages.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the required modules
try:
    from agent_tools.dualipa.code_extractor import extract_blocks_from_file, extract_blocks_from_text
    from agent_tools.dualipa.languages import get_language_by_extension
    print("Successfully imported code extraction modules")
except ImportError as e:
    print(f"Error importing code extraction modules: {e}")
    sys.exit(1)

def print_header(text, underline='='):
    """Print a header with underline."""
    print(f"\n{text}")
    print(underline * len(text))

def get_test_files():
    """Return a dictionary of test files for various languages."""
    return {
        # Python test file
        "python.py": """
# This is a Python file

def hello_world():
    \"\"\"This is a simple function that prints Hello World\"\"\"
    print("Hello, World!")
    return "Hello, World!"

class Calculator:
    \"\"\"A simple calculator class\"\"\"
    
    def __init__(self):
        self.result = 0
    
    def add(self, a, b=None):
        \"\"\"Add a number to the result or add two numbers\"\"\"
        if b is None:
            self.result += a
            return self.result
        return a + b
""",
        
        # JavaScript test file
        "javascript.js": """
// This is a JavaScript file

function helloWorld() {
    console.log("Hello, World!");
    return "Hello, World!";
}

class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(a, b = null) {
        // Add a number to the result or add two numbers
        if (b === null) {
            this.result += a;
            return this.result;
        }
        return a + b;
    }
}
""",
        
        # Java test file
        "java.java": """
// This is a Java file

public class HelloWorld {
    /**
     * Main method that prints Hello World
     */
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
    
    /**
     * A simple add method
     */
    public static int add(int a, int b) {
        return a + b;
    }
}

class Calculator {
    private int result = 0;
    
    /**
     * Add a number to the result
     */
    public int add(int a) {
        result += a;
        return result;
    }
    
    /**
     * Add two numbers
     */
    public int add(int a, int b) {
        return a + b;
    }
}
""",
        
        # C# test file
        "csharp.cs": """
// This is a C# file

using System;

namespace HelloWorld
{
    /// <summary>
    /// Main program class
    /// </summary>
    class Program
    {
        /// <summary>
        /// Main method that prints Hello World
        /// </summary>
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }
        
        /// <summary>
        /// A simple add method
        /// </summary>
        static int Add(int a, int b)
        {
            return a + b;
        }
    }
    
    /// <summary>
    /// A simple calculator class
    /// </summary>
    class Calculator
    {
        private int result = 0;
        
        /// <summary>
        /// Add a number to the result
        /// </summary>
        public int Add(int a)
        {
            result += a;
            return result;
        }
        
        /// <summary>
        /// Add two numbers
        /// </summary>
        public int Add(int a, int b)
        {
            return a + b;
        }
    }
}
""",
        
        # Ruby test file
        "ruby.rb": """
# This is a Ruby file

# A simple hello world method
def hello_world
  puts "Hello, World!"
  return "Hello, World!"
end

# A simple calculator class
class Calculator
  def initialize
    @result = 0
  end
  
  # Add a number to the result or add two numbers
  def add(a, b = nil)
    if b.nil?
      @result += a
      return @result
    else
      return a + b
    end
  end
end
"""
    }

def verify_language_detection():
    """Verify language detection by file extension."""
    print_header("Testing language detection", "-")
    
    # Test file extensions
    extensions = {
        ".py": "python",
        ".js": "javascript",
        ".java": "java",
        ".cs": "csharp",
        ".rb": "ruby",
        ".ts": "typescript",
        ".go": "golang",
        ".rs": "rust",
        ".cpp": "cpp",
        ".md": "markdown",
        ".unknown": None
    }
    
    try:
        print("Testing language detection for various file extensions...")
        for ext, expected in extensions.items():
            language = get_language_by_extension(ext)
            success = language == expected
            print(f"  {ext}: {language} {'✅' if success else '❌'}")
        
        all_success = all(get_language_by_extension(ext) == expected for ext, expected in extensions.items())
        return all_success
    except Exception as e:
        print(f"❌ Error during language detection: {str(e)}")
        return False

def verify_multilang_extraction():
    """Verify code extraction from multiple languages."""
    print_header("Testing multi-language code extraction", "-")
    
    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = get_test_files()
        
        # Write test files
        for filename, content in test_files.items():
            file_path = temp_path / filename
            file_path.write_text(content)
        
        # Track overall success
        overall_success = True
        
        # Process each file
        for filename in test_files.keys():
            try:
                file_path = temp_path / filename
                extension = os.path.splitext(filename)[1]
                language = get_language_by_extension(extension)
                
                print(f"\nProcessing {filename} ({language})...")
                blocks = extract_blocks_from_file(str(file_path))
                
                # Print block information
                print(f"Extracted {len(blocks)} blocks:")
                for i, block in enumerate(blocks):
                    print(f"  Block {i+1}:")
                    print(f"    Type: {block.get('type', 'unknown')}")
                    print(f"    Language: {block.get('language', 'unknown')}")
                    print(f"    Content length: {len(block.get('content', ''))}")
                    first_line = (block.get('content', '') or '').splitlines()
                    if first_line:
                        print(f"    First line: {first_line[0]}")
                
                # Check if we found any blocks
                if not blocks:
                    print(f"❌ No blocks found in {filename}")
                    overall_success = False
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                overall_success = False
        
        return overall_success

def verify_language_specific_extraction():
    """Verify language-specific code extraction features."""
    print_header("Testing language-specific extraction features", "-")
    
    test_files = get_test_files()
    overall_success = True
    
    try:
        # Check Python docstring extraction
        python_file = test_files["python.py"]
        print("\nTesting Python docstring extraction...")
        python_blocks = extract_blocks_from_text(python_file, "python.py")
        has_docstrings = any(block.get('type') == 'docstring' for block in python_blocks)
        print(f"Found docstrings: {'✅' if has_docstrings else '❌'}")
        if not has_docstrings:
            overall_success = False
        
        # Check Java/C# documentation comment extraction
        java_file = test_files["java.java"]
        print("\nTesting Java documentation comment extraction...")
        java_blocks = extract_blocks_from_text(java_file, "java.java")
        has_doc_comments = any(block.get('type') == 'documentation' for block in java_blocks)
        print(f"Found documentation comments: {'✅' if has_doc_comments else '❌'}")
        if not has_doc_comments:
            overall_success = False
        
        # Check function/method extraction
        print("\nTesting function/method extraction...")
        for lang, file_content in [("Python", test_files["python.py"]), 
                                  ("JavaScript", test_files["javascript.js"])]:
            blocks = extract_blocks_from_text(file_content, f"{lang.lower()}.{lang.lower()[:2]}")
            has_functions = any(block.get('type') == 'function' for block in blocks)
            print(f"Found functions in {lang}: {'✅' if has_functions else '❌'}")
            if not has_functions:
                overall_success = False
        
        # Check class extraction
        print("\nTesting class extraction...")
        for lang, file_content in [("Python", test_files["python.py"]), 
                                  ("JavaScript", test_files["javascript.js"]),
                                  ("Java", test_files["java.java"]),
                                  ("C#", test_files["csharp.cs"])]:
            blocks = extract_blocks_from_text(file_content, f"{lang.lower().replace('#', 'sharp')}.{lang.lower().replace('#', 's')[:2]}")
            has_classes = any(block.get('type') == 'class' for block in blocks)
            print(f"Found classes in {lang}: {'✅' if has_classes else '❌'}")
            if not has_classes:
                overall_success = False
        
        return overall_success
    except Exception as e:
        print(f"❌ Error during language-specific extraction tests: {str(e)}")
        return False

def main():
    """Run all verification tests."""
    print_header("Multi-Language Code Extraction Verification")
    
    # Run all verification tests
    language_detection_success = verify_language_detection()
    multilang_extraction_success = verify_multilang_extraction()
    language_specific_extraction_success = verify_language_specific_extraction()
    
    # Calculate overall success
    all_success = (
        language_detection_success and
        multilang_extraction_success and
        language_specific_extraction_success
    )
    
    # Print summary
    print_header("Verification Summary")
    print(f"Language Detection: {'✅' if language_detection_success else '❌'}")
    print(f"Multi-Language Extraction: {'✅' if multilang_extraction_success else '❌'}")
    print(f"Language-Specific Features: {'✅' if language_specific_extraction_success else '❌'}")
    print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 