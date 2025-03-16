"""
Tests for the code hierarchy parser.

These tests verify that the hierarchy parser correctly extracts code hierarchies
with proper depth and relationship information.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path

from agent_tools.dualipa.code_hierarchy import (
    slugify, 
    extract_code_structure,
    write_code_entities,
    build_code_repository_hierarchy,
    process_code_repository
)

# Set to True to see detailed output of extracted hierarchies
VERBOSE = os.environ.get('VERBOSE_CODE_TESTS', 'False').lower() in ('true', '1', 't')

def visualize_hierarchy(entities, indent=0):
    """Helper function to visualize a code hierarchy."""
    result = []
    for entity in entities:
        entity_type = entity['type'].upper() if entity['type'] == 'class' else entity['type']
        params = f"({', '.join(entity.get('parameters', []))})" if entity.get('parameters') else ""
        line = f"{'  ' * indent}{entity_type}: {entity['name']}{params}"
        result.append(line)
        
        # Recursively visualize children if they exist
        if 'children' in entity and entity['children']:
            result.extend(visualize_hierarchy(entity['children'], indent + 1))
    
    return result

def print_entity_details(entity, show_content=False):
    """Print details of an entity for debugging."""
    print(f"\nEntity: {entity['name']} ({entity['type']})")
    print(f"Depth: {entity['depth']}")
    print(f"Path: {[(p['type'], p['name']) for p in entity['path']]}")
    print(f"File paths: {entity['file_paths']}")
    if 'parameters' in entity:
        print(f"Parameters: {entity['parameters']}")
    if show_content and 'content' in entity:
        print(f"Content snippet: {entity['content'][:100]}...")

def test_slugify():
    """Test the code slugify function."""
    assert slugify("MyClassName") == "myclassname"
    assert slugify("get_user_by_id") == "get-user-by-id"
    assert slugify("__init__") == "init"
    assert slugify("camelCaseMethod") == "camelcasemethod"
    assert slugify("API_KEY_VALUE") == "api-key-value"
    assert slugify("special$characters%^&") == "special-characters"
    
    if VERBOSE:
        print("\nSlugify examples:")
        examples = [
            "MyClassName", 
            "get_user_by_id", 
            "__init__", 
            "camelCaseMethod",
            "API_KEY_VALUE",
            "special$characters%^&"
        ]
        for example in examples:
            print(f"  {example} -> {slugify(example)}")

def test_extract_code_structure_classes():
    """Test extraction of class definitions."""
    code = """
class SimpleClass:
    \"\"\"A simple class.\"\"\"
    
    def __init__(self):
        self.value = 42
        
    def get_value(self):
        return self.value
        
class AnotherClass:
    \"\"\"Another class definition.\"\"\"
    
    def method(self):
        \"\"\"A method.\"\"\"
        pass
"""
    
    entities = extract_code_structure(code)
    
    if VERBOSE:
        print("\nExtracted class hierarchy:")
        
        # Build a visualization-friendly hierarchy
        classes = [e for e in entities if e['type'] == 'class']
        methods = [e for e in entities if e['type'] == 'method']
        
        # Group methods under their classes
        class_hierarchy = []
        for cls in classes:
            cls_copy = cls.copy()
            cls_copy['children'] = [
                m for m in methods if m['path'] and 
                m['path'][0]['name'] == cls['name']
            ]
            class_hierarchy.append(cls_copy)
        
        # Print visualization
        hierarchy_lines = visualize_hierarchy(class_hierarchy)
        for line in hierarchy_lines:
            print(line)
    
    # Verify entity count (2 classes, 3 methods)
    assert len(entities) == 5
    
    # Verify class extraction
    classes = [e for e in entities if e['type'] == 'class']
    assert len(classes) == 2
    assert classes[0]['name'] == 'SimpleClass'
    assert classes[1]['name'] == 'AnotherClass'
    
    # Verify method extraction
    methods = [e for e in entities if e['type'] == 'method']
    assert len(methods) == 3
    method_names = [m['name'] for m in methods]
    assert '__init__' in method_names
    assert 'get_value' in method_names
    assert 'method' in method_names
    
    # Verify paths (parent-child relationships)
    init_method = next(m for m in methods if m['name'] == '__init__')
    assert len(init_method['path']) == 1
    assert init_method['path'][0]['name'] == 'SimpleClass'
    assert init_method['path'][0]['type'] == 'class'

def test_extract_code_structure_functions():
    """Test extraction of function definitions."""
    code = """
def top_level_function(param1, param2):
    \"\"\"A top level function.\"\"\"
    return param1 + param2
    
def another_function():
    \"\"\"Another function.\"\"\"
    
    def nested_function():
        \"\"\"A nested function.\"\"\"
        return 42
        
    return nested_function()
"""
    
    entities = extract_code_structure(code)
    
    # Verify entity count (2 top-level functions, 1 nested function)
    assert len(entities) == 3
    
    # Verify function extraction
    functions = [e for e in entities if e['type'] == 'function']
    assert len(functions) == 3
    
    # Verify function names
    function_names = [f['name'] for f in functions]
    assert 'top_level_function' in function_names
    assert 'another_function' in function_names
    assert 'nested_function' in function_names
    
    # Verify parameters
    top_function = next(f for f in functions if f['name'] == 'top_level_function')
    assert 'parameters' in top_function
    assert top_function['parameters'] == ['param1', 'param2']
    
    # Verify nesting
    nested_function = next(f for f in functions if f['name'] == 'nested_function')
    assert len(nested_function['path']) == 1
    assert nested_function['path'][0]['name'] == 'another_function'
    assert nested_function['path'][0]['type'] == 'function'
    assert nested_function['depth'] == 2  # Nested one level deep

def test_extract_code_structure_complex():
    """Test extraction of complex code with nested structures."""
    code = """
class OuterClass:
    \"\"\"An outer class.\"\"\"
    
    class InnerClass:
        \"\"\"A nested class.\"\"\"
        
        def inner_method(self):
            \"\"\"A method in a nested class.\"\"\"
            return 42
    
    def outer_method(self):
        \"\"\"A method in the outer class.\"\"\"
        
        def local_function():
            \"\"\"A local function in a method.\"\"\"
            return 21
        
        return local_function() * 2

def standalone_function():
    \"\"\"A standalone function.\"\"\"
    pass
"""
    
    entities = extract_code_structure(code)
    
    # Count entities by type
    classes = [e for e in entities if e['type'] == 'class']
    methods = [e for e in entities if e['type'] == 'method']
    functions = [e for e in entities if e['type'] == 'function']
    
    # Verify counts
    assert len(classes) == 2
    assert len(methods) == 2
    assert len(functions) == 2  # standalone_function and local_function
    
    # Verify nesting structure for InnerClass
    inner_class = next(c for c in classes if c['name'] == 'InnerClass')
    assert len(inner_class['path']) == 1
    assert inner_class['path'][0]['name'] == 'OuterClass'
    
    # Verify nesting for inner_method
    inner_method = next(m for m in methods if m['name'] == 'inner_method')
    assert len(inner_method['path']) == 1  # Should only track direct parent
    assert inner_method['path'][0]['name'] == 'InnerClass'
    
    # Verify local function in method
    local_function = next(f for f in functions if f['name'] == 'local_function')
    assert len(local_function['path']) == 1
    assert local_function['path'][0]['type'] == 'function'  # Not method since we track closest parent
    assert local_function['depth'] == 2

def test_extract_code_structure_file_paths():
    """Test file paths generated for code entities."""
    code = """
class ParentClass:
    def parent_method(self):
        pass
        
    class NestedClass:
        def nested_method(self):
            pass
            
def standalone_function():
    pass
"""
    
    entities = extract_code_structure(code)
    
    # Get entities by name
    parent_class = next(e for e in entities if e['name'] == 'ParentClass')
    parent_method = next(e for e in entities if e['name'] == 'parent_method')
    nested_class = next(e for e in entities if e['name'] == 'NestedClass')
    nested_method = next(e for e in entities if e['name'] == 'nested_method')
    standalone = next(e for e in entities if e['name'] == 'standalone_function')
    
    # Verify file paths
    assert parent_class['file_paths'] == ['parentclass.py']
    assert 'parentclass/parent-method.py' in parent_method['file_paths']
    assert 'parentclass/nestedclass.py' in nested_class['file_paths']
    assert 'parentclass/nestedclass/nested-method.py' in nested_method['file_paths']
    assert standalone['file_paths'] == ['standalone-function.py']

def test_write_code_entities():
    """Test writing code entities to files."""
    code = """
class TestClass:
    \"\"\"Test class docstring.\"\"\"
    
    def test_method(self, param):
        \"\"\"Test method docstring.\"\"\"
        return param * 2
        
def test_function(a, b=None):
    \"\"\"Test function docstring.\"\"\"
    return a + (b or 0)
"""
    
    entities = extract_code_structure(code)
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write entities to files
        output_files = write_code_entities(entities, temp_dir)
        
        # Verify output files
        assert 'class:TestClass' in output_files
        assert 'method:test_method' in output_files
        assert 'function:test_function' in output_files
        
        # Get file paths
        class_file = Path(output_files['class:TestClass'])
        method_file = Path(output_files['method:test_method'])
        function_file = Path(output_files['function:test_function'])
        
        # Verify files exist
        assert class_file.exists()
        assert method_file.exists()
        assert function_file.exists()
        
        # Verify metadata in files
        with open(class_file, 'r') as f:
            class_content = f.read()
            assert "name: 'TestClass'" in class_content
            assert "type: 'class'" in class_content
            assert "depth: 1" in class_content
        
        with open(method_file, 'r') as f:
            method_content = f.read()
            assert "name: 'test_method'" in method_content
            assert "type: 'method'" in method_content
            assert "parameters: ['self', 'param']" in method_content
        
        with open(function_file, 'r') as f:
            function_content = f.read()
            assert "name: 'test_function'" in function_content
            assert "type: 'function'" in function_content
            assert "parameters: ['a', 'b']" in function_content
        
        # Verify directory structure
        assert (Path(temp_dir) / "testclass").is_dir()

def test_handle_syntax_errors():
    """Test handling of syntax errors in code."""
    code_with_error = """
def function_with_error(
    missing_parenthesis
    return "oops"
"""
    
    entities = extract_code_structure(code_with_error)
    
    # Should have one entity with error info
    assert len(entities) == 1
    assert entities[0]['type'] == 'error'
    assert 'error' in entities[0]
    assert 'Syntax error' in entities[0]['content']

def test_complex_repository_structure():
    """Test building a repository hierarchy with complex structure."""
    with tempfile.TemporaryDirectory() as repo_dir:
        # Create nested directory structure
        os.makedirs(os.path.join(repo_dir, "package/subpackage"))
        
        # Create Python files at different levels
        with open(os.path.join(repo_dir, "main.py"), "w") as f:
            f.write("""
def main():
    \"\"\"Main function.\"\"\"
    print("Hello world")
    
if __name__ == "__main__":
    main()
""")
        
        with open(os.path.join(repo_dir, "package/models.py"), "w") as f:
            f.write("""
class User:
    \"\"\"User model.\"\"\"
    
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_display_name(self):
        return f"{self.name} <{self.email}>"
""")
        
        with open(os.path.join(repo_dir, "package/subpackage/utils.py"), "w") as f:
            f.write("""
def validate_email(email):
    \"\"\"Validate email address.\"\"\"
    return "@" in email

class EmailValidator:
    \"\"\"Email validator class.\"\"\"
    
    @staticmethod
    def is_valid(email):
        return validate_email(email)
""")
        
        # Build the repository hierarchy
        hierarchy = build_code_repository_hierarchy(repo_dir)
        
        if VERBOSE:
            print("\nRepository hierarchy:")
            print(f"Files found: {len(hierarchy)}")
            
            for file_info in hierarchy:
                print(f"\nFile: {file_info['path']} (depth: {file_info['depth']})")
                print(f"  Directory hierarchy: {file_info['dir_hierarchy']}")
                
                if 'internal_entities' in file_info and file_info['internal_entities']:
                    print("  Entities:")
                    hierarchy_lines = visualize_hierarchy(file_info['internal_entities'])
                    for line in hierarchy_lines:
                        print(f"    {line}")
        
        # Verify structure
        assert len(hierarchy) == 3  # 3 Python files
        
        # Find files by path
        main_file = next(f for f in hierarchy if f["path"] == "main.py")
        models_file = next(f for f in hierarchy if f["path"] == "package/models.py")
        utils_file = next(f for f in hierarchy if f["path"] == "package/subpackage/utils.py")
        
        # Verify file attributes
        assert main_file["depth"] == 0
        assert models_file["depth"] == 1
        assert utils_file["depth"] == 2
        
        # Verify directory hierarchies
        assert main_file["dir_hierarchy"] == []
        assert models_file["dir_hierarchy"] == ["package"]
        assert utils_file["dir_hierarchy"] == ["package", "subpackage"]
        
        # Verify internal entities
        assert len(main_file["internal_entities"]) == 1  # main function
        assert main_file["internal_entities"][0]["name"] == "main"
        
        assert len(models_file["internal_entities"]) == 1  # User class
        user_class = models_file["internal_entities"][0]
        assert user_class["name"] == "User"
        assert len(user_class["children"]) == 2  # __init__ and get_display_name methods
        
        # Verify nested structures in utils.py
        utils_entities = utils_file["internal_entities"]
        assert len(utils_entities) == 2  # validate_email function and EmailValidator class
        
        validator_class = next(e for e in utils_entities if e["name"] == "EmailValidator")
        assert len(validator_class["children"]) == 1  # is_valid method
        
        # Test output generation
        with tempfile.TemporaryDirectory() as output_dir:
            result = process_code_repository(repo_dir, output_dir)
            
            if VERBOSE:
                print("\nGenerated output files:")
                for key, path in result["output_files"].items():
                    print(f"  {key} -> {path}")
                
                # Print a sample of file contents
                if result["output_files"]:
                    sample_key = list(result["output_files"].keys())[0]
                    sample_path = result["output_files"][sample_key]
                    print(f"\nSample file content ({sample_key}):")
                    try:
                        with open(sample_path, 'r') as f:
                            content = f.read()
                            print("---")
                            print(content[:500] + ("..." if len(content) > 500 else ""))
                            print("---")
                    except Exception as e:
                        print(f"Error reading sample file: {e}")
            
            # Verify output files
            assert len(result["output_files"]) > 0
            
            # Check some key files exist
            assert os.path.exists(os.path.join(output_dir, "main.py"))
            assert os.path.exists(os.path.join(output_dir, "user.py"))
            assert os.path.exists(os.path.join(output_dir, "user/init.py"))  # __init__ becomes init.py
            assert os.path.exists(os.path.join(output_dir, "emailvalidator.py"))

# Run this to demo the code hierarchy extractors
if __name__ == "__main__":
    print("Code hierarchy extractor demo")
    print("=============================")
    
    # Set VERBOSE globally for this run
    globals()['VERBOSE'] = True
    
    # Create a test case with the sample code
    sample_code = """
class DataProcessor:
    \"\"\"Process data from various sources.\"\"\"
    
    def __init__(self, source):
        \"\"\"Initialize with data source.\"\"\"
        self.source = source
        self._cache = {}
    
    def get_data(self, key=None):
        \"\"\"Retrieve data, optionally filtered by key.\"\"\"
        if key and key in self._cache:
            return self._cache[key]
            
        class DataResult:
            \"\"\"Represents a result from the data source.\"\"\"
            def __init__(self, data):
                self.data = data
                
            def transform(self):
                \"\"\"Transform the data.\"\"\"
                return [x * 2 for x in self.data]
        
        result = DataResult([1, 2, 3])  # Example data
        
        if key:
            self._cache[key] = result
            
        return result
        
def process_and_print(processor, key=None):
    \"\"\"Process data and print results.\"\"\"
    result = processor.get_data(key)
    
    def format_result():
        \"\"\"Format the result for display.\"\"\"
        return f"Result: {result.transform()}"
    
    print(format_result())
    return result
"""

    print("\nSample code structure:")
    entities = extract_code_structure(sample_code)
    
    # Build hierarchical structure
    top_level = []
    entity_map = {}
    
    # First create all entities with empty children
    for entity in entities:
        entity_copy = entity.copy()
        entity_copy['children'] = []
        entity_key = (entity['name'], entity['type'])
        entity_map[entity_key] = entity_copy
    
    # Then build the hierarchy
    for entity in entities:
        entity_key = (entity['name'], entity['type'])
        entity_obj = entity_map[entity_key]
        
        if entity['path']:
            parent_info = entity['path'][-1]
            parent_key = (parent_info['name'], parent_info['type'])
            if parent_key in entity_map:
                entity_map[parent_key]['children'].append(entity_obj)
            else:
                top_level.append(entity_obj)
        else:
            top_level.append(entity_obj)
    
    # Visualize the hierarchy
    hierarchy_lines = visualize_hierarchy(top_level)
    for line in hierarchy_lines:
        print(line)
    
    print("\nFile paths for each entity:")
    for entity in entities:
        print(f"{entity['type'].upper()}: {entity['name']}")
        for i, path in enumerate(entity['file_paths']):
            print(f"  {i}: {path}")
        print()
    
    print("\nOutput directory structure that would be created:")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_files = write_code_entities(entities, temp_dir)
        
        # Show all created directories and files
        print(f"Directory structure in {temp_dir}:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}") 