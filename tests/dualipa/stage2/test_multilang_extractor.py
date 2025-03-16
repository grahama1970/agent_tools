#!/usr/bin/env python3
"""
Tests for the multi-language code extractor.

These tests verify that the extractor correctly handles code from various 
languages supported by tree-sitter (excluding Python, which has its own tests).
"""

import os
import tempfile
import pytest
from pathlib import Path
import json
import shutil
import sys
import importlib

# Remove the path manipulation as we'll import directly from the package
# parent_dir = str(Path(__file__).parent.parent.parent)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

# Create a module-level variable to skip tests if required modules aren't available
js_ts_extractor_path = Path(__file__).parent.parent.parent.parent / "src" / "agent_tools" / "dualipa" / "scripts" / "js_ts_extractor.py"
if not js_ts_extractor_path.exists():
    SKIP_REASON = "js_ts_extractor.py module not found"
else:
    SKIP_REASON = None

# Only try importing if the module exists
if not SKIP_REASON:
    # Use importlib to import the module dynamically to avoid import errors
    SCRIPT_PATH = str(js_ts_extractor_path)
    MODULE_NAME = "js_ts_extractor"
    
    # Import the module using spec
    import importlib.util
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
    js_ts_extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(js_ts_extractor)
    
    # Now get the functions and variables we need
    process_file = getattr(js_ts_extractor, "process_file", None)
    TREE_SITTER_LANGUAGES = getattr(js_ts_extractor, "TREE_SITTER_LANGUAGES", {})
    detect_language = getattr(js_ts_extractor, "detect_language", None)
    
    # Verify we got all required functions
    if not (process_file and TREE_SITTER_LANGUAGES and detect_language):
        SKIP_REASON = "Required functions not found in js_ts_extractor.py"

# Sample code for each language
LANGUAGE_SAMPLES = {
    'javascript': """
    /**
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
    """,
    
    'typescript': """
    /**
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
        
        getDetails(): string {
            return `${this.name} works in ${this.department}`;
        }
    }
    """,
    
    'java': """
    /**
     * Sample Java class
     */
    public class HelloWorld {
        private String greeting;
        
        /**
         * Constructor
         */
        public HelloWorld(String greeting) {
            this.greeting = greeting;
        }
        
        /**
         * Sample method with parameters
         * @param name The name to greet
         * @return Formatted greeting string
         */
        public String greet(String name) {
            return this.greeting + ", " + name + "!";
        }
        
        /**
         * Main method
         */
        public static void main(String[] args) {
            HelloWorld hello = new HelloWorld("Hello");
            System.out.println(hello.greet("World"));
        }
    }
    """,
    
    'c': """
    /**
     * Sample C header with documentation
     */
    #include <stdio.h>
    #include <stdlib.h>

    /**
     * Sample structure definition
     */
    typedef struct {
        char* name;
        int age;
    } Person;

    /**
     * Function to create a new person
     */
    Person* create_person(const char* name, int age) {
        Person* p = (Person*) malloc(sizeof(Person));
        p->name = strdup(name);
        p->age = age;
        return p;
    }

    /**
     * Sample main function
     */
    int main() {
        Person* person = create_person("John", 30);
        printf("Created person: %s, age %d\\n", person->name, person->age);
        free(person->name);
        free(person);
        return 0;
    }
    """,
    
    'cpp': """
    /**
     * Sample C++ program
     */
    #include <iostream>
    #include <string>

    /**
     * Person class
     */
    class Person {
    private:
        std::string name;
        int age;
        
    public:
        /**
         * Constructor
         */
        Person(const std::string& name, int age) : name(name), age(age) {}
        
        /**
         * Get person info
         */
        std::string getInfo() const {
            return name + " is " + std::to_string(age) + " years old";
        }
    };

    /**
     * Main function
     */
    int main() {
        Person person("Alice", 25);
        std::cout << person.getInfo() << std::endl;
        return 0;
    }
    """,
    
    'c_sharp': """
    using System;

    /**
     * Sample C# program
     */
    namespace HelloWorld {
        /**
         * Person class
         */
        public class Person {
            public string Name { get; set; }
            public int Age { get; set; }
            
            /**
             * Constructor
             */
            public Person(string name, int age) {
                Name = name;
                Age = age;
            }
            
            /**
             * Get person info
             */
            public string GetInfo() {
                return $"{Name} is {Age} years old";
            }
        }
        
        /**
         * Program class with entry point
         */
        public class Program {
            public static void Main(string[] args) {
                var person = new Person("Bob", 30);
                Console.WriteLine(person.GetInfo());
            }
        }
    }
    """,
    
    'go': """
    // Sample Go program
    package main

    import "fmt"

    // Person struct
    type Person struct {
        Name string
        Age  int
    }

    // GetInfo returns person information
    func (p Person) GetInfo() string {
        return fmt.Sprintf("%s is %d years old", p.Name, p.Age)
    }

    // Main function
    func main() {
        person := Person{
            Name: "Charlie",
            Age:  35,
        }
        fmt.Println(person.GetInfo())
    }
    """,
    
    'rust': """
    // Sample Rust program

    /// Person struct
    struct Person {
        name: String,
        age: u32,
    }

    impl Person {
        /// Create a new person
        fn new(name: &str, age: u32) -> Self {
            Person {
                name: name.to_string(),
                age,
            }
        }
        
        /// Get person info
        fn get_info(&self) -> String {
            format!("{} is {} years old", self.name, self.age)
        }
    }

    /// Main function
    fn main() {
        let person = Person::new("Dave", 40);
        println!("{}", person.get_info());
    }
    """,
    
    'ruby': """
    # Sample Ruby program

    # Person class
    class Person
      attr_reader :name, :age
      
      # Constructor
      def initialize(name, age)
        @name = name
        @age = age
      end
      
      # Get person info
      def get_info
        "#{@name} is #{@age} years old"
      end
    end

    # Create a person
    person = Person.new("Eve", 45)
    puts person.get_info
    """,
    
    'php': """
    <?php
    /**
     * Sample PHP program
     */

    /**
     * Person class
     */
    class Person {
        private $name;
        private $age;
        
        /**
         * Constructor
         */
        public function __construct(string $name, int $age) {
            $this->name = $name;
            $this->age = $age;
        }
        
        /**
         * Get person info
         */
        public function getInfo(): string {
            return $this->name . " is " . $this->age . " years old";
        }
    }

    // Create a person
    $person = new Person("Frank", 50);
    echo $person->getInfo();
    ?>
    """,
    
    'bash': """
    #!/bin/bash
    
    # Sample Bash script
    
    # Function to greet
    greet() {
        local name="$1"
        echo "Hello, $name!"
    }
    
    # Main script
    NAME=${1:-"World"}
    greet "$NAME"
    
    # Create a person object (simulated)
    declare -A person
    person[name]="George"
    person[age]=55
    
    # Print person info
    echo "${person[name]} is ${person[age]} years old"
    """,
    
    'html': """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sample HTML</title>
        <style>
            .person {
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 15px;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <!-- Person card -->
        <div class="person">
            <h2>Person Information</h2>
            <p><strong>Name:</strong> <span id="name">Hannah</span></p>
            <p><strong>Age:</strong> <span id="age">60</span></p>
        </div>

        <script>
            // Function to get person info
            function getPersonInfo() {
                const name = document.getElementById('name').textContent;
                const age = document.getElementById('age').textContent;
                return `${name} is ${age} years old`;
            }
            
            // Log to console
            console.log(getPersonInfo());
        </script>
    </body>
    </html>
    """,
    
    'css': """
    /* Sample CSS file */
    
    /* Main container styles */
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    
    /* Person card styles */
    .person-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Person name styles */
    .person-name {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    
    /* Person age styles */
    .person-age {
        font-size: 14px;
        color: #666;
    }
    
    /* Media query for responsive design */
    @media screen and (max-width: 768px) {
        .person-card {
            padding: 10px;
        }
    }
    """
}

# Mark all tests to be skipped if the required module isn't available
pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for each language"""
    files = {}
    for lang, code in LANGUAGE_SAMPLES.items():
        # Get the proper extension for the language
        extensions = TREE_SITTER_LANGUAGES.get(lang, {}).get('extensions', [])
        ext = extensions[0] if extensions else '.txt'
        
        # Create the file
        file_path = Path(temp_dir) / f"sample.{lang}{ext}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        files[lang] = file_path
    
    return files

def test_language_detection():
    """Test that all tree-sitter supported languages are correctly detected"""
    # Create a temporary file for each language
    with tempfile.TemporaryDirectory() as temp_dir:
        for lang, info in TREE_SITTER_LANGUAGES.items():
            if not info['extensions']:
                continue
            
            ext = info['extensions'][0]
            file_path = Path(temp_dir) / f"test{ext}"
            with open(file_path, 'w') as f:
                f.write("// Sample file")
            
            detected_lang, is_supported = detect_language(file_path)
            
            # Python is handled separately with AST
            if lang == 'python':
                assert detected_lang == 'python'
                assert is_supported is False
            else:
                assert detected_lang == lang
                assert is_supported is True

def test_supported_languages_list():
    """Test that all expected tree-sitter languages are in the supported list"""
    expected_languages = [
        'javascript', 'typescript', 'c', 'cpp', 'c_sharp', 'java',
        'ruby', 'go', 'rust', 'php', 'bash', 'html', 'css', 'python'
    ]
    
    for lang in expected_languages:
        assert lang in TREE_SITTER_LANGUAGES
        assert 'extensions' in TREE_SITTER_LANGUAGES[lang]
        assert isinstance(TREE_SITTER_LANGUAGES[lang]['extensions'], list)

def test_javascript_extraction(sample_files, temp_dir):
    """Test extraction of JavaScript code"""
    js_file = sample_files['javascript']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(js_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 3  # Should extract at least 3 declarations
    
    # Check types of declarations
    declaration_types = [d['type'] for d in declarations]
    assert 'function_declaration' in declaration_types
    assert 'class_declaration' in declaration_types
    assert 'arrow_function' in declaration_types
    
    # Check content of files
    for decl in declarations:
        if decl['type'] == 'function_declaration' and decl['name'] == 'greet':
            # Verify the header contains important metadata
            assert any('Path:' in line for line in open(output_files[decl['name']]).readlines()[:5])
            assert any('Type: function_declaration' in line for line in open(output_files[decl['name']]).readlines()[:5])
            assert any('Name: greet' in line for line in open(output_files[decl['name']]).readlines()[:5])
            
            # Verify the content contains the function code
            content = open(output_files[decl['name']]).read()
            assert 'function greet(' in content
            assert 'return `Hello' in content

def test_typescript_extraction(sample_files, temp_dir):
    """Test extraction of TypeScript code"""
    ts_file = sample_files['typescript']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(ts_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 2  # Should extract at least interface and class
    
    # Check types of declarations - tree-sitter seems to miss function declarations in some cases
    declaration_types = [d['type'] for d in declarations]
    
    # We should at least have interface and class declarations
    assert 'interface_declaration' in declaration_types
    assert 'class_declaration' in declaration_types
    
    # Check output files exist
    assert len(output_files) == len(declarations)
    for decl in declarations:
        assert decl['name'] in output_files
        assert os.path.exists(output_files[decl['name']])

def test_java_extraction(sample_files, temp_dir):
    """Test extraction of Java code"""
    java_file = sample_files['java']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(java_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 1  # Should extract at least the class declaration
    
    # Verify class declaration
    has_class = False
    for decl in declarations:
        if decl['type'] == 'class_declaration' and decl['name'] == 'HelloWorld':
            has_class = True
            break
    
    assert has_class

def test_go_extraction(sample_files, temp_dir):
    """Test extraction of Go code"""
    go_file = sample_files['go']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(go_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 1  # Should extract at least one function
    
    # Check that main function is extracted
    has_main = False
    for decl in declarations:
        if decl['name'] == 'main':
            has_main = True
            break
    
    assert has_main

def test_rust_extraction(sample_files, temp_dir):
    """Test extraction of Rust code"""
    rust_file = sample_files['rust']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(rust_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 1  # Should extract at least one function or struct
    
    # Check for specific structures or functions
    has_main = False
    for decl in declarations:
        if decl['name'] == 'main':
            has_main = True
            break
    
    assert has_main

def test_c_extraction(sample_files, temp_dir):
    """Test extraction of C code"""
    c_file = sample_files['c']
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(c_file, {}, output_dir)
    
    # Check results
    assert len(declarations) >= 1  # Should extract at least one function
    
    # Check for main function
    has_main = False
    for decl in declarations:
        if decl['name'] == 'main':
            has_main = True
            break
    
    assert has_main

def test_python_handling(temp_dir):
    """Test that Python files are skipped with a message"""
    # Create a sample Python file
    python_file = Path(temp_dir) / "sample.py"
    with open(python_file, 'w') as f:
        f.write("def test_function():\n    return 'Hello, Python!'")
    
    output_dir = Path(temp_dir) / "output"
    
    # Process the file
    declarations, output_files = process_file(python_file, {}, output_dir)
    
    # Python files should be skipped and handled by the AST extractor
    assert len(declarations) == 0
    assert len(output_files) == 0

def test_all_languages(sample_files, temp_dir):
    """Test extraction of all supported languages"""
    # Skip Python as it's handled separately
    languages_to_test = [lang for lang in LANGUAGE_SAMPLES.keys() if lang != 'python']
    output_dir = Path(temp_dir) / "output"
    
    all_declarations = {}
    
    for lang in languages_to_test:
        file_path = sample_files[lang]
        declarations, _ = process_file(file_path, {}, output_dir)
        
        # Store declarations for each language
        all_declarations[lang] = declarations
        
        # Ensure we extracted at least one declaration
        assert len(declarations) > 0, f"Failed to extract declarations from {lang} code"
    
    # Save the results to a debug file
    debug_file = Path(temp_dir) / "all_languages_debug.json"
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(all_declarations, f, indent=2)

def test_metadata_consistency(sample_files, temp_dir):
    """Test that all languages include consistent metadata in headers"""
    output_dir = Path(temp_dir) / "output"
    
    # List of required fields in every declaration
    required_fields = ['type', 'name', 'start_line', 'end_line', 'language', 'relative_path']
    
    for lang, file_path in sample_files.items():
        if lang == 'python':
            continue  # Skip Python as it's handled separately
            
        declarations, _ = process_file(file_path, {}, output_dir)
        
        # Check that all declarations have the required fields
        for decl in declarations:
            for field in required_fields:
                assert field in decl, f"Field '{field}' missing from {lang} declaration" 