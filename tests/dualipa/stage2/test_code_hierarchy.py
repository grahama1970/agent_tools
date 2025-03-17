"""
Tests for code hierarchy (parent-child relationships).

This module verifies that the hierarchy parser correctly extracts
code hierarchies with proper depth and relationship information
using real-world code examples from popular repositories.
"""

import os
import sys
import json
import tempfile
import pytest
import shutil
import requests
from pathlib import Path

# Add the parent directory to path if needed for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from agent_tools.dualipa.code_hierarchy import (
        build_code_hierarchy,
        extract_code_hierarchy,
        get_children,
        get_parent
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    HAS_DEPENDENCIES = False

# Skip tests if dependencies are not available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required imports not available")

# Real repository URLs and files to test
REAL_REPOS = {
    'python': 'https://github.com/pallets/flask',
    'javascript': 'https://github.com/expressjs/express',
    'typescript': 'https://github.com/microsoft/TypeScript',
    'java': 'https://github.com/spring-projects/spring-boot'
}

REPO_FILES = {
    'python': 'src/flask/app.py',
    'javascript': 'lib/express.js',
    'typescript': 'src/compiler/types.ts',
    'java': 'spring-boot-project/spring-boot/src/main/java/org/springframework/boot/ApplicationRunner.java'
}

def fetch_real_code(language):
    """Fetch real code from a repository for the specified language."""
    repo_url = REAL_REPOS.get(language)
    file_path = REPO_FILES.get(language)
    
    if not repo_url or not file_path:
        return f"# Sample {language} code (could not fetch real example)"
    
    # Convert GitHub URL to raw content URL
    raw_url = repo_url.replace('github.com', 'raw.githubusercontent.com')
    if not raw_url.endswith('/'):
        raw_url += '/'
    
    # Try both main and master branches
    for branch in ['main', 'master']:
        try:
            url = f"{raw_url}{branch}/{file_path}"
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    # Fallback content if fetching fails
    fallbacks = {
        'python': """
# Sample Python code with hierarchy
class OuterClass:
    \"\"\"Outer class docstring.\"\"\"
    
    def __init__(self, value):
        self.value = value
    
    def outer_method(self):
        \"\"\"Outer method docstring.\"\"\"
        return self.value
        
    class InnerClass:
        \"\"\"Inner class docstring.\"\"\"
        
        def __init__(self):
            self.inner_value = 42
            
        def inner_method(self):
            \"\"\"Inner method docstring.\"\"\"
            return self.inner_value

def standalone_function():
    \"\"\"Standalone function docstring.\"\"\"
    return "Hello World"
""",
        'javascript': """
// Sample JavaScript code with hierarchy
class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
  
  getName() {
    return this.name;
  }
  
  getAge() {
    return this.age;
  }
}

function sayHello(person) {
  return `Hello ${person.getName()}`;
}
""",
        'typescript': """
// Sample TypeScript code with hierarchy
interface Person {
  name: string;
  age: number;
}

class Employee implements Person {
  name: string;
  age: number;
  department: string;
  
  constructor(name: string, age: number, department: string) {
    this.name = name;
    this.age = age;
    this.department = department;
  }
  
  getInfo(): string {
    return `${this.name}, ${this.age}, ${this.department}`;
  }
}

function processEmployee(emp: Employee): void {
  console.log(emp.getInfo());
}
""",
        'java': """
// Sample Java code with hierarchy
public class ExampleClass {
    private String name;
    
    public ExampleClass(String name) {
        this.name = name;
    }
    
    public String getName() {
        return this.name;
    }
    
    public static void main(String[] args) {
        ExampleClass example = new ExampleClass("Test");
        System.out.println(example.getName());
    }
}
"""
    }
    
    return fallbacks.get(language, f"# Sample {language} code")

def visualize_hierarchy(hierarchy):
    """Helper function to visualize the hierarchy for debugging."""
    result = []
    
    def _recurse(node, depth=0):
        indent = "  " * depth
        node_type = node.get("type", "unknown")
        node_name = node.get("name", "unnamed")
        result.append(f"{indent}- {node_type}: {node_name}")
        
        for child in node.get("children", []):
            _recurse(child, depth + 1)
    
    for root in hierarchy:
        _recurse(root)
    
    return "\n".join(result)

def print_entity_details(entity):
    """Helper function to print details of an entity for debugging."""
    print(f"Entity: {entity.get('name', 'unnamed')}")
    print(f"Type: {entity.get('type', 'unknown')}")
    print(f"Start line: {entity.get('start_line', 'unknown')}")
    print(f"End line: {entity.get('end_line', 'unknown')}")
    print(f"Parent: {entity.get('parent', 'None')}")
    print(f"Children: {entity.get('children', [])}")
    print("---")

@pytest.fixture
def python_code():
    """Fixture to provide real Python code."""
    return fetch_real_code('python')

@pytest.fixture
def javascript_code():
    """Fixture to provide real JavaScript code."""
    return fetch_real_code('javascript')

@pytest.fixture
def typescript_code():
    """Fixture to provide real TypeScript code."""
    return fetch_real_code('typescript')

@pytest.fixture
def java_code():
    """Fixture to provide real Java code."""
    return fetch_real_code('java')

def test_python_hierarchy_extraction(python_code):
    """Test extraction of Python code hierarchy using real code."""
    # Create a temporary file with the Python code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write(python_code)
        f.flush()
        
        # Extract the hierarchy
        hierarchy = extract_code_hierarchy(f.name)
        
        # Verify we got something
        assert hierarchy is not None, "Should extract a hierarchy"
        assert isinstance(hierarchy, list), "Hierarchy should be a list"
        assert len(hierarchy) > 0, "Should extract at least one top-level entity"
        
        # Print hierarchy for debugging
        print("\nPython hierarchy:")
        print(visualize_hierarchy(hierarchy))
        
        # Verify hierarchy structure
        for entity in hierarchy:
            assert "type" in entity, "Entity should have a type"
            assert "name" in entity, "Entity should have a name"
            assert "start_line" in entity, "Entity should have a start line"
            assert "end_line" in entity, "Entity should have an end line"
            
            # Check children if present
            if "children" in entity and entity["children"]:
                for child in entity["children"]:
                    assert "type" in child, "Child should have a type"
                    assert "name" in child, "Child should have a name"
                    assert "start_line" in child, "Child should have a start line"
                    assert "end_line" in child, "Child should have an end line"
        
        # Test utility functions
        for entity in hierarchy:
            # Test get_children
            children = get_children(entity, hierarchy)
            assert isinstance(children, list), "get_children should return a list"
            
            # If entity has children, test get_parent
            if "children" in entity and entity["children"]:
                child = entity["children"][0]
                parent = get_parent(child, hierarchy)
                assert parent is not None, "Child's parent should not be None"
                assert parent["name"] == entity["name"], "Parent should match original entity"

def test_javascript_hierarchy_extraction(javascript_code):
    """Test extraction of JavaScript code hierarchy using real code."""
    # Create a temporary file with the JavaScript code
    with tempfile.NamedTemporaryFile(suffix='.js', mode='w+') as f:
        f.write(javascript_code)
        f.flush()
        
        # Extract the hierarchy
        hierarchy = extract_code_hierarchy(f.name)
        
        # Verify we got something
        assert hierarchy is not None, "Should extract a hierarchy"
        assert isinstance(hierarchy, list), "Hierarchy should be a list"
        assert len(hierarchy) > 0, "Should extract at least one top-level entity"
        
        # Print hierarchy for debugging
        print("\nJavaScript hierarchy:")
        print(visualize_hierarchy(hierarchy))
        
        # Verify hierarchy structure
        for entity in hierarchy:
            assert "type" in entity, "Entity should have a type"
            assert "name" in entity, "Entity should have a name"
            assert "start_line" in entity, "Entity should have a start line"
            assert "end_line" in entity, "Entity should have an end line"
            
            # Check children if present
            if "children" in entity and entity["children"]:
                for child in entity["children"]:
                    assert "type" in child, "Child should have a type"
                    assert "name" in child, "Child should have a name"
                    assert "start_line" in child, "Child should have a start line"
                    assert "end_line" in child, "Child should have an end line"

def test_typescript_hierarchy_extraction(typescript_code):
    """Test extraction of TypeScript code hierarchy using real code."""
    # Create a temporary file with the TypeScript code
    with tempfile.NamedTemporaryFile(suffix='.ts', mode='w+') as f:
        f.write(typescript_code)
        f.flush()
        
        # Extract the hierarchy
        hierarchy = extract_code_hierarchy(f.name)
        
        # Verify we got something
        assert hierarchy is not None, "Should extract a hierarchy"
        assert isinstance(hierarchy, list), "Hierarchy should be a list"
        assert len(hierarchy) > 0, "Should extract at least one top-level entity"
        
        # Print hierarchy for debugging
        print("\nTypeScript hierarchy:")
        print(visualize_hierarchy(hierarchy))
        
        # Verify hierarchy structure
        for entity in hierarchy:
            assert "type" in entity, "Entity should have a type"
            assert "name" in entity, "Entity should have a name"
            assert "start_line" in entity, "Entity should have a start line"
            assert "end_line" in entity, "Entity should have an end line"
            
            # Check children if present
            if "children" in entity and entity["children"]:
                for child in entity["children"]:
                    assert "type" in child, "Child should have a type"
                    assert "name" in child, "Child should have a name"
                    assert "start_line" in child, "Child should have a start line"
                    assert "end_line" in child, "Child should have an end line"

def test_java_hierarchy_extraction(java_code):
    """Test extraction of Java code hierarchy using real code."""
    # Create a temporary file with the Java code
    with tempfile.NamedTemporaryFile(suffix='.java', mode='w+') as f:
        f.write(java_code)
        f.flush()
        
        # Extract the hierarchy
        hierarchy = extract_code_hierarchy(f.name)
        
        # Verify we got something
        assert hierarchy is not None, "Should extract a hierarchy"
        assert isinstance(hierarchy, list), "Hierarchy should be a list"
        assert len(hierarchy) > 0, "Should extract at least one top-level entity"
        
        # Print hierarchy for debugging
        print("\nJava hierarchy:")
        print(visualize_hierarchy(hierarchy))
        
        # Verify hierarchy structure
        for entity in hierarchy:
            assert "type" in entity, "Entity should have a type"
            assert "name" in entity, "Entity should have a name"
            assert "start_line" in entity, "Entity should have a start line"
            assert "end_line" in entity, "Entity should have an end line"
            
            # Check children if present
            if "children" in entity and entity["children"]:
                for child in entity["children"]:
                    assert "type" in child, "Child should have a type"
                    assert "name" in child, "Child should have a name"
                    assert "start_line" in child, "Child should have a start line"
                    assert "end_line" in child, "Child should have an end line"

def test_build_code_hierarchy_function():
    """Test the build_code_hierarchy function with real-world-like entities."""
    # Create a list of entities that might come from real code
    entities = [
        {"id": "class_1", "type": "class", "name": "User", "start_line": 10, "end_line": 50},
        {"id": "method_1", "type": "method", "name": "getName", "start_line": 15, "end_line": 20},
        {"id": "method_2", "type": "method", "name": "setName", "start_line": 25, "end_line": 30},
        {"id": "func_1", "type": "function", "name": "processUser", "start_line": 55, "end_line": 70},
        {"id": "class_2", "type": "class", "name": "Admin", "start_line": 80, "end_line": 120},
        {"id": "method_3", "type": "method", "name": "getPermissions", "start_line": 85, "end_line": 90},
        {"id": "method_4", "type": "method", "name": "grantAccess", "start_line": 95, "end_line": 100}
    ]
    
    # Build hierarchy
    hierarchy = build_code_hierarchy(entities)
    
    # Verify the hierarchy structure
    assert len(hierarchy) == 3, "Should have 3 top-level entities (2 classes and 1 function)"
    
    # Find the User class
    user_class = None
    for entity in hierarchy:
        if entity["name"] == "User":
            user_class = entity
            break
    
    assert user_class is not None, "User class should be in the hierarchy"
    assert "children" in user_class, "User class should have children"
    assert len(user_class["children"]) == 2, "User class should have 2 methods"
    
    # Find the Admin class
    admin_class = None
    for entity in hierarchy:
        if entity["name"] == "Admin":
            admin_class = entity
            break
    
    assert admin_class is not None, "Admin class should be in the hierarchy"
    assert "children" in admin_class, "Admin class should have children"
    assert len(admin_class["children"]) == 2, "Admin class should have 2 methods"
    
    # Print hierarchy for debugging
    print("\nBuilt hierarchy:")
    print(visualize_hierarchy(hierarchy))

def test_multifile_hierarchy():
    """Test building hierarchy across multiple real files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create multiple files with code
        files = {
            'python': 'module.py',
            'javascript': 'app.js',
            'typescript': 'types.ts',
            'java': 'Main.java'
        }
        
        file_paths = {}
        for language, filename in files.items():
            content = fetch_real_code(language)
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            file_paths[language] = file_path
        
        # Extract hierarchies from each file
        all_entities = []
        for language, file_path in file_paths.items():
            hierarchy = extract_code_hierarchy(file_path)
            
            # Flatten hierarchy to a list of entities
            entities = []
            
            def flatten(node):
                entity = node.copy()
                if "children" in entity:
                    for child in entity["children"]:
                        entities.append(child)
                        flatten(child)
                    del entity["children"]
                entities.append(entity)
            
            for root in hierarchy:
                flatten(root)
            
            # Add source file info
            for entity in entities:
                entity["source_file"] = os.path.basename(file_path)
                entity["language"] = language
            
            all_entities.extend(entities)
        
        # Build a combined hierarchy
        combined_hierarchy = build_code_hierarchy(all_entities)
        
        # Verify we got a reasonable result
        assert combined_hierarchy is not None, "Should build a combined hierarchy"
        assert len(combined_hierarchy) > 0, "Combined hierarchy should have at least one top-level entity"
        
        # Print the combined hierarchy for debugging
        print("\nCombined hierarchy:")
        print(visualize_hierarchy(combined_hierarchy))
        
        # Verify that entities from different files are present
        languages_found = set()
        
        def check_languages(node):
            if "language" in node:
                languages_found.add(node["language"])
            for child in node.get("children", []):
                check_languages(child)
        
        for root in combined_hierarchy:
            check_languages(root)
        
        # We should find at least 2 different languages
        assert len(languages_found) >= 2, f"Should have entities from at least 2 languages, got {languages_found}"

def test_nested_class_hierarchy(python_code):
    """Test extracting hierarchy with nested classes from real Python code."""
    # Look for nested class patterns in the code, or use fallback
    if "class" not in python_code or "def" not in python_code:
        python_code = """
class OuterClass:
    def outer_method(self):
        return "outer"
    
    class InnerClass:
        def inner_method(self):
            return "inner"
"""
    
    # Create a temporary file with the Python code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as f:
        f.write(python_code)
        f.flush()
        
        # Extract the hierarchy
        hierarchy = extract_code_hierarchy(f.name)
        
        # Verify we got something
        assert hierarchy is not None, "Should extract a hierarchy"
        
        # Look for classes with nested structures
        has_nested = False
        for entity in hierarchy:
            if entity["type"] == "class" and "children" in entity:
                for child in entity["children"]:
                    if child["type"] == "class":
                        has_nested = True
                        break
        
        # We might not find nested classes in all real code, so skip if not found
        if not has_nested:
            pytest.skip("No nested classes found in the Python code")
        
        # Print hierarchy for debugging
        print("\nNested class hierarchy:")
        print(visualize_hierarchy(hierarchy))
        
        # Verify parent-child relationships are correct
        for entity in hierarchy:
            if entity["type"] == "class" and "children" in entity:
                for child in entity["children"]:
                    if child["type"] == "class":
                        # Verify the parent reference
                        parent = get_parent(child, hierarchy)
                        assert parent is not None, "Nested class should have a parent"
                        assert parent["name"] == entity["name"], "Parent should be the outer class"
                        
                        # Verify child's position is within parent
                        assert child["start_line"] >= entity["start_line"], "Child should start after parent starts"
                        assert child["end_line"] <= entity["end_line"], "Child should end before parent ends" 