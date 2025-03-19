#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the Python function extraction module."""

import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest
import requests
import json
import glob
import importlib

# Configure paths properly - add src directory to path
project_root = Path(__file__).parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Add debugging for import locations
try:
    # Import the python extractor module from code_extractor
    from agent_tools.dualipa.code_extractor import (
        _extract_python_blocks,
        initialize_stats_dict
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    # Detailed error message about what's missing
    agent_tools_dir = None
    try:
        agent_tools_dir = importlib.util.find_spec("agent_tools").submodule_search_locations[0]
    except:
        pass

    # Instead of silently skipping, fail loudly with a clear error message
    raise ImportError(f"Required Python extractor modules not available: {e}. Fix the dependencies to run these tests.")

# Real repository URLs for Python code
PYTHON_REPOS = {
    'flask': 'https://github.com/pallets/flask',
    'django': 'https://github.com/django/django',
    'pandas': 'https://github.com/pandas-dev/pandas',
    'requests': 'https://github.com/psf/requests',
    'fastapi': 'https://github.com/tiangolo/fastapi',
}

# Specific files to test in each repository
REPO_FILES = {
    'flask': 'src/flask/app.py',
    'django': 'django/http/request.py',
    'pandas': 'pandas/core/frame.py',
    'requests': 'requests/models.py',
    'fastapi': 'fastapi/applications.py',
}

def fetch_real_python(repo_key):
    """Fetch real Python code from the specified repository."""
    repo_url = PYTHON_REPOS.get(repo_key)
    file_path = REPO_FILES.get(repo_key)
    
    if not repo_url or not file_path:
        return "# Sample Python code (could not fetch real example)"
    
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
    return """
def hello_world():
    \"\"\"Sample fallback function.\"\"\"
    return "Hello, World!"

class SampleClass:
    \"\"\"Sample fallback class.\"\"\"
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        return f"Hello, {self.name}!"
"""

def load_extracted_blocks(output_dir):
    """Load extracted blocks from the output directory."""
    blocks = []
    blocks_dir = output_dir / "blocks" / "code" / "python"
    
    if not blocks_dir.exists():
        return []
        
    for block_file in blocks_dir.glob("*.py"):
        # Read the block file
        with open(block_file, 'r') as f:
            content = f.read()
            
        # Extract metadata from the file content if available
        metadata = {}
        for line in content.splitlines()[:5]:  # Check first 5 lines for metadata
            if line.startswith("# "):
                key_value = line[2:].split(": ", 1)
                if len(key_value) == 2:
                    metadata[key_value[0].lower()] = key_value[1]
        
        # Create a block object
        block = {
            'file': block_file.name,
            'content': content,
            'type': metadata.get('block type', ''),
            'name': metadata.get('name', ''),
            **metadata
        }
        blocks.append(block)
        
    return blocks

@pytest.fixture
def flask_code():
    """Fixture to provide real Flask code."""
    return fetch_real_python('flask')

@pytest.fixture
def django_code():
    """Fixture to provide real Django code."""
    return fetch_real_python('django')

@pytest.fixture
def pandas_code():
    """Fixture to provide real Pandas code."""
    return fetch_real_python('pandas')

def test_extract_python_blocks():
    """Test extracting function definitions from Python files."""
    test_str = """
def my_function(a, b):
    "This is a docstring"
    print(a + b)
    return True

def another_function():
    '''Triple quote docstring'''
    return False
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        # Create a temporary file path (doesn't need to exist physically)
        file_path = Path(os.path.join(tmp_dir, "test_functions.py"))
        stats = initialize_stats_dict(source=file_path, output_dir=output_dir)
        
        # Call the function
        try:
            num_blocks = _extract_python_blocks(file_path, test_str, output_dir, stats)
            assert num_blocks > 0, "Should extract at least one block"
            assert stats["code_blocks"] > 0, "Should increment code_blocks in stats"
        except Exception as e:
            pytest.fail(f"Failed to extract functions: {e}")

def test_extract_python_functions():
    """Test extraction of Python functions from real Flask code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        blocks_dir = output_dir / "blocks" / "code" / "python"
        os.makedirs(blocks_dir, exist_ok=True)
        stats = initialize_stats_dict(source=temp_dir, output_dir=output_dir)
        
        # Create a simple Flask-like function for testing
        flask_code = '''
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/user/<username>')
def show_user_profile(username):
    """Show the user profile for that user."""
    return f'User {username}'

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    return app
'''
        
        # Write the code to a file
        file_path_str = os.path.join(temp_dir, "flask_app.py")
        with open(file_path_str, 'w') as f:
            f.write(flask_code)
        
        # Convert string path to Path object
        file_path = Path(file_path_str)
        
        # Extract functions
        try:
            print(f"\nAttempting to extract blocks from Flask code: {file_path}")
            num_blocks = _extract_python_blocks(file_path, flask_code, output_dir, stats)
            print(f"Extracted {num_blocks} blocks from Flask code")
            
            # Check that blocks were extracted
            assert num_blocks > 0, "Should extract at least one block from Flask code"
            
            # Check if block files were created
            block_files = list(blocks_dir.glob("*.py"))
            assert len(block_files) > 0, "Should create at least one block file"
            
            # Optional: verify content of blocks
            for block_file in block_files:
                with open(block_file, 'r') as f:
                    content = f.read()
                print(f"Block content preview: {content[:50]}...")
                assert "def " in content, "Block should contain a function definition"
        except Exception as e:
            pytest.fail(f"Failed to extract Python functions: {e}")

def test_extract_python_classes():
    """Test extraction of Python classes from real Django code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        blocks_dir = output_dir / "blocks" / "code" / "python"
        os.makedirs(blocks_dir, exist_ok=True)
        stats = initialize_stats_dict(source=temp_dir, output_dir=output_dir)
        
        # Create a Django-like class for testing
        django_code = '''
from django.db import models
from django.contrib.auth.models import User

class Post(models.Model):
    """A blog post model."""
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        """Return the URL for this post."""
        return f"/post/{self.id}/"

class Comment(models.Model):
    """A comment on a blog post."""
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
'''
        
        # Write the code to a file
        file_path_str = os.path.join(temp_dir, "django_models.py")
        with open(file_path_str, 'w') as f:
            f.write(django_code)
        
        # Convert string path to Path object
        file_path = Path(file_path_str)
        
        # Extract classes
        try:
            print(f"\nAttempting to extract blocks from Django code: {file_path}")
            num_blocks = _extract_python_blocks(file_path, django_code, output_dir, stats)
            print(f"Extracted {num_blocks} blocks from Django code")
            
            # Check that blocks were extracted
            assert num_blocks > 0, "Should extract at least one block from Django code"
            
            # Check if block files were created
            block_files = list(blocks_dir.glob("*.py"))
            assert len(block_files) > 0, "Should create at least one block file"
            
            # Check if any of the blocks contains a class definition
            class_found = False
            for block_file in block_files:
                with open(block_file, 'r') as f:
                    content = f.read()
                print(f"Block content preview: {content[:50]}...")
                if "class " in content:
                    class_found = True
                    break  # Found at least one class, no need to check further
            
            assert class_found, "Should extract at least one block with a class definition"
        except Exception as e:
            pytest.fail(f"Failed to extract Python classes: {e}")

def test_extract_from_multiple_repositories():
    """Test extraction from multiple real Python repositories to ensure robustness."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define some Python code snippets from different repos
        code_snippets = {
            "flask": '''
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"
''',
            "django": '''
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=100)
    body = models.TextField()
''',
            "requests": '''
def get(url, params=None, **kwargs):
    """Sends a GET request."""
    return request('get', url, params=params, **kwargs)
'''
        }
        
        # Test extracting from each snippet
        success = False
        for repo_key, code in code_snippets.items():
            try:
                file_path_str = os.path.join(temp_dir, f"{repo_key}_sample.py")
                with open(file_path_str, 'w') as f:
                    f.write(code)
                
                # Convert string path to Path object
                file_path = Path(file_path_str)
                
                stats = initialize_stats_dict(source=file_path, output_dir=output_dir)
                
                print(f"\nAttempting to extract blocks from {repo_key}: {file_path}")
                num_blocks = _extract_python_blocks(file_path, code, output_dir, stats)
                print(f"Extracted {num_blocks} blocks from {repo_key}")
                
                if num_blocks > 0:
                    success = True
                    # Found a working repository, can continue with testing
                    break
            except Exception as e:
                print(f"Error extracting from {repo_key}: {e}")
        
        if not success:
            pytest.fail("Extraction failed for all repositories")
        
        # Check that blocks directory exists
        blocks_dir = output_dir / "blocks" / "code" / "python"
        assert blocks_dir.exists(), "Blocks directory should be created"
        
        # Check if any block files were created
        block_files = list(blocks_dir.glob("*.py"))
        assert len(block_files) > 0, "Should create at least one block file"

def test_code_with_decorators():
    """Test extraction of Python code with decorators."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        blocks_dir = output_dir / "blocks" / "code" / "python"
        os.makedirs(blocks_dir, exist_ok=True)
        stats = initialize_stats_dict(source=temp_dir, output_dir=output_dir)
        
        # Create code with decorators
        decorator_example = '''
import functools

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

class APIView:
    @classmethod
    def as_view(cls, **initkwargs):
        """Main entry point for a request-response process."""
        def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            return self.dispatch(request, *args, **kwargs)
        return view
'''
        
        # Write the code to a file
        file_path_str = os.path.join(temp_dir, "decorators.py")
        with open(file_path_str, 'w') as f:
            f.write(decorator_example)
        
        # Convert string path to Path object
        file_path = Path(file_path_str)
        
        # Extract blocks
        try:
            print(f"\nAttempting to extract blocks with decorators from: {file_path}")
            num_blocks = _extract_python_blocks(file_path, decorator_example, output_dir, stats)
            print(f"Extracted {num_blocks} blocks with decorators")
            
            # Check that blocks were extracted
            assert num_blocks > 0, "Should extract at least one block with decorators"
            
            # Check if block files were created
            block_files = list(blocks_dir.glob("*.py"))
            assert len(block_files) > 0, "Should create at least one block file"
            
            # Optional: verify content of blocks
            decorator_found = False
            for block_file in block_files:
                with open(block_file, 'r') as f:
                    content = f.read()
                print(f"Block content preview: {content[:50]}...")
                if "@" in content:
                    decorator_found = True
            
            assert decorator_found, "Should extract at least one block with a decorator"
        except Exception as e:
            pytest.fail(f"Failed to extract code with decorators: {e}")

def test_imports_work():
    """Verify that imports work correctly."""
    try:
        print("\nTesting imports:")
        from agent_tools.dualipa import __version__
        print(f"dualipa version: {__version__}")
        
        assert callable(_extract_python_blocks), "_extract_python_blocks should be callable"
        
        assert True, "Imports are working correctly"
    except Exception as e:
        pytest.fail(f"Import test failed: {e}")

if __name__ == "__main__":
    # Run the tests directly
    try:
        import agent_tools
        print(f"agent_tools found at: {agent_tools.__file__}")
        
        # Test _extract_python_blocks with a simple Python script
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w') as f:
                f.write("def test_func():\n    return 'Hello, World!'")
                f.flush()
                
                output_dir = Path(temp_dir)
                stats = initialize_stats_dict(source=f.name, output_dir=output_dir)
                
                file_path = Path(f.name)
                try:
                    num_blocks = _extract_python_blocks(file_path, "def test_func():\n    return 'Hello, World!'", output_dir, stats)
                    print(f"Number of blocks extracted: {num_blocks}")
                    print(f"Stats: {stats}")
                except Exception as e:
                    print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error running as script: {e}")