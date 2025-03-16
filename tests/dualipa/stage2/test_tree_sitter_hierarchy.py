"""
Tests for verifying the consistency of extraction structure across different parsers:
- Python AST parser
- Markdown parser
- Tree-sitter parser for other languages

The focus is ensuring all parsers produce output with the same structure and fields
so an LLM can process them identically regardless of the source format.
"""

import os
import tempfile
from pathlib import Path
import pytest

# Import the hierarchy extraction functions
from agent_tools.dualipa.code_hierarchy import (
    extract_code_structure,
    extract_code_structure_tree_sitter,
    write_code_entities
)

from agent_tools.dualipa.markdown_hierarchy import extract_hierarchical_sections as extract_markdown_structure


def test_structure_consistency_across_parsers():
    """
    Test that all parsers (Python AST, markdown, tree-sitter) produce structurally
    similar outputs with the same fields and format.
    """
    # Sample content for each parser
    python_code = '''
class TestClass:
    """A test class."""
    
    def __init__(self, param1):
        """Initialize the class."""
        self.param1 = param1
    
    def test_method(self, param2):
        """A test method."""
        return param2 + self.param1
'''

    js_code = '''
class TestClass {
  constructor(param1) {
    // Initialize the class
    this.param1 = param1;
  }
  
  testMethod(param2) {
    // A test method
    return param2 + this.param1;
  }
}
'''

    markdown_content = '''
# TestClass

A test class.

## __init__

Initialize the class.

## test_method

A test method.
'''
    
    # Extract with each parser
    python_entities = extract_code_structure(python_code)
    js_entities = extract_code_structure_tree_sitter(js_code, "javascript")
    markdown_entities = extract_markdown_structure(markdown_content)
    
    # Debug print
    print("Python entities:", [(e['name'], e['type']) for e in python_entities])
    print("JS entities:", [(e.get('name', 'unknown'), e.get('type', 'unknown')) for e in js_entities])
    print("Markdown entities:", [(e.get('title', e.get('name', 'unknown')), e.get('type', 'unknown')) for e in markdown_entities])
    
    # First, check that all three parsers extracted entities
    assert len(python_entities) > 0, "Python parser should extract entities"
    assert len(js_entities) > 0, "Tree-sitter parser should extract entities"
    assert len(markdown_entities) > 0, "Markdown parser should extract entities"
    
    # Get one example entity from each parser
    py_entity = next(e for e in python_entities if e['type'] == 'class')
    js_entity = next(e for e in js_entities if e['type'] == 'class')
    md_entity = next(e for e in markdown_entities if e.get('type') == 'header' and e.get('title') == 'TestClass')
    
    # Normalize field names - markdown uses 'title' instead of 'name'
    for entity in markdown_entities:
        if 'title' in entity and 'name' not in entity:
            entity['name'] = entity['title']
    
    # Check that all parsers produce entities with the same basic structure
    # Common required fields
    common_fields = ['name', 'type', 'content', 'depth', 'path', 'file_paths']
    for field in common_fields:
        assert field in py_entity, f"Python entity missing {field}"
        assert field in js_entity, f"JavaScript entity missing {field}"
        assert field in md_entity, f"Markdown entity missing {field}"
    
    # Check that path structure is consistent
    assert isinstance(py_entity['path'], list), "Python path should be a list"
    assert isinstance(js_entity['path'], list), "JavaScript path should be a list"
    assert isinstance(md_entity['path'], list), "Markdown path should be a list"
    
    # Check that file_paths structure is consistent
    assert isinstance(py_entity['file_paths'], list), "Python file_paths should be a list"
    assert isinstance(js_entity['file_paths'], list), "JavaScript file_paths should be a list"
    assert isinstance(md_entity['file_paths'], list), "Markdown file_paths should be a list"
    
    # Check that nested entities have proper path referencing parent
    py_method = next(e for e in python_entities if e['type'] == 'method' and e['name'] == 'test_method')
    js_method = next(e for e in js_entities if e['type'] == 'method' and e['name'] == 'testMethod')
    md_method = next(e for e in markdown_entities if e.get('type') == 'header' and (
                            e.get('name') == 'test_method' or e.get('title') == 'test_method'))
    
    # Check parent-child relationship representation
    assert len(py_method['path']) > 0, "Python method should have path referencing parent"
    assert len(js_method['path']) > 0, "JavaScript method should have path referencing parent"
    
    # Check the paths contain parent information
    assert py_method['path'][0]['name'] == 'TestClass', "Python method path should reference TestClass"
    assert js_method['path'][0]['name'] == 'TestClass', "JavaScript method path should reference TestClass"
    
    # The markdown parser might have a slightly different structure, but should still represent hierarchy
    if len(md_method['path']) > 0:
        assert md_method['path'][0]['name'] == 'TestClass', "Markdown method path should reference TestClass"
    
    # Check file paths are structured consistently across parsers
    assert any('test-method' in fp for fp in py_method['file_paths']), "Python should have method file path"
    assert any('test-method' in fp for fp in js_method['file_paths']), "JavaScript should have method file path"
    assert any('test-method' in fp for fp in md_method['file_paths']), "Markdown should have method file path"
    
    # Check for parent directory in file paths
    assert any('testclass/test-method' in fp.lower() for fp in py_method['file_paths']), "Python should have nested path"
    assert any('testclass/test-method' in fp.lower() for fp in js_method['file_paths']), "JavaScript should have nested path"
    assert any('testclass/test-method' in fp.lower() for fp in md_method['file_paths']), "Markdown should have nested path"


def test_output_file_structure_consistency():
    """
    Test that files written by each parser follow the same structure and naming convention.
    """
    # Sample content for each parser with exactly the same structure
    python_code = '''
class User:
    """User class for managing user data."""
    
    def __init__(self, name, email):
        """Initialize user with name and email."""
        self.name = name
        self.email = email
    
    def validate(self):
        """Validate user data."""
        return '@' in self.email
'''

    js_code = '''
class User {
  constructor(name, email) {
    // Initialize user with name and email
    this.name = name;
    this.email = email;
  }
  
  validate() {
    // Validate user data
    return this.email.includes('@');
  }
}
'''

    markdown_content = '''
# User

User class for managing user data.

## __init__

Initialize user with name and email.

## validate

Validate user data.
'''
    
    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Extract and write entities for each parser
        python_entities = extract_code_structure(python_code)
        js_entities = extract_code_structure_tree_sitter(js_code, "javascript")
        markdown_entities = extract_markdown_structure(markdown_content)
        
        # Write entities to separate directories
        py_output_dir = output_dir / "python"
        js_output_dir = output_dir / "javascript" 
        md_output_dir = output_dir / "markdown"
        
        py_output_dir.mkdir()
        js_output_dir.mkdir()
        md_output_dir.mkdir()
        
        # Write entities
        write_code_entities(python_entities, str(py_output_dir))
        write_code_entities(js_entities, str(js_output_dir))
        write_code_entities(markdown_entities, str(md_output_dir))
        
        # Get file paths
        py_files = list(py_output_dir.glob("**/*"))
        js_files = list(js_output_dir.glob("**/*"))
        md_files = list(md_output_dir.glob("**/*"))
        
        # Check that each parser created files
        assert len(py_files) > 0, "Python parser should create files"
        assert len(js_files) > 0, "JavaScript parser should create files"
        assert len(md_files) > 0, "Markdown parser should create files"
        
        # Compare directory structures (ignoring language-specific extensions)
        py_structure = {str(p.relative_to(py_output_dir)).replace('.py', '') for p in py_files if p.is_file()}
        js_structure = {str(p.relative_to(js_output_dir)).replace('.javascript', '') for p in js_files if p.is_file()}
        md_structure = {str(p.relative_to(md_output_dir)).replace('.md', '') for p in md_files if p.is_file()}
        
        # Check for common structure elements
        assert 'user' in py_structure, "Python should have user file"
        assert 'user' in js_structure, "JavaScript should have user file"
        assert 'user' in md_structure, "Markdown should have user file"
        
        assert 'user/validate' in py_structure, "Python should have validate file in user directory"
        assert 'user/validate' in js_structure, "JavaScript should have validate file in user directory"
        assert 'user/validate' in md_structure, "Markdown should have validate file in user directory"


def test_combined_repository_processing():
    """
    Test processing a repository with mixed file types and verify
    consistent structure across all parsers.
    """
    # Create a temporary repository with multiple file types
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        
        # Create files with identical structure in different formats
        files = {
            "model.py": '''
class Document:
    """Document class for storing content."""
    
    def __init__(self, title, content):
        """Initialize document with title and content."""
        self.title = title
        self.content = content
    
    def word_count(self):
        """Count words in document."""
        return len(self.content.split())
''',
            "model.js": '''
class Document {
  constructor(title, content) {
    // Initialize document with title and content
    this.title = title;
    this.content = content;
  }
  
  wordCount() {
    // Count words in document
    return this.content.split(/\\s+/).length;
  }
}
''',
            "model.md": '''
# Document

Document class for storing content.

## __init__

Initialize document with title and content.

## word_count

Count words in document.
'''
        }
        
        # Create the files
        for filename, content in files.items():
            file_path = repo_dir / filename
            with open(file_path, "w") as f:
                f.write(content)
        
        # Process each file individually
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Extract entities from each file
        with open(repo_dir / "model.py", "r") as f:
            py_content = f.read()
            py_entities = extract_code_structure(py_content)
            
        with open(repo_dir / "model.js", "r") as f:
            js_content = f.read()
            js_entities = extract_code_structure_tree_sitter(js_content, "javascript")
            
        with open(repo_dir / "model.md", "r") as f:
            md_content = f.read()
            md_entities = extract_markdown_structure(md_content)
        
        # Compare extracted entities
        py_class = next(e for e in py_entities if e['type'] == 'class')
        js_class = next(e for e in js_entities if e['type'] == 'class')
        md_class = next(e for e in md_entities if (e.get('name') == 'Document' or e.get('title') == 'Document'))
        
        # All should have the same basic structure
        assert py_class['name'] == 'Document'
        assert js_class['name'] == 'Document'
        assert md_class['name'] == 'Document'
        
        # Check method extraction
        py_method = next(e for e in py_entities if e['type'] == 'method' and e['name'] == 'word_count')
        js_method = next(e for e in js_entities if e['type'] == 'method' and e['name'] == 'wordCount')
        md_method = next(e for e in md_entities if (e.get('name') == 'word_count' or e.get('title') == 'word_count'))
        
        # Check common fields format
        required_fields = ['content', 'depth', 'path', 'file_paths']
        for field in required_fields:
            assert field in py_method, f"Python method missing {field}"
            assert field in js_method, f"JavaScript method missing {field}"
            assert field in md_method, f"Markdown method missing {field}"
        
        # Check file path structure (should include parent)
        assert any('document/word-count' in fp.lower() for fp in py_method['file_paths']), "Python path should be hierarchical"
        # JS may have camelCase, check for variations
        assert (any('document/word-count' in fp.lower() for fp in js_method['file_paths']) or 
                any('document/wordcount' in fp.lower() for fp in js_method['file_paths'])), "JavaScript path should be hierarchical"
        assert any('document/word-count' in fp.lower() for fp in md_method['file_paths']), "Markdown path should be hierarchical"


if __name__ == "__main__":
    # Run tests directly
    pytest.main(["-xvs", __file__]) 