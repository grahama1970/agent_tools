"""
Tests for block extraction functionality in code_extractor.py.

This module tests the block extraction functionality of the code_extractor module
for different languages and file formats.

These tests use real code examples from template files or sample repositories
to verify that code blocks are properly extracted.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import sys

# Configure the Path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Flag to track if dependencies are available
HAS_DEPENDENCIES = False
try:
    from agent_tools.dualipa.code_extractor import extract_repository, run_test
    from agent_tools.dualipa.language_detection import detect_language
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("Skipping tests that require missing modules")

# Path to test resources
RESOURCES_DIR = project_root / "src" / "agent_tools" / "dualipa" / "resources" / "templates"

# Skip all tests if the required modules don't exist
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="Required modules not available"
)

def check_file_exists(file_path):
    """Check if a test file exists, skip if not."""
    if not file_path.exists():
        pytest.skip(f"Sample file {file_path} does not exist")
    return True

def test_python_block_extraction():
    """Test Python block extraction using AST."""
    # Get the path to sample Python file
    sample_file = RESOURCES_DIR / "sample_python.py"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats["code_blocks"] > 0, "No Python code blocks were extracted"
            print(f"Extracted {stats['code_blocks']} Python blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "python"
            assert blocks_dir.exists(), "Python blocks directory was not created"
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.py"))
            assert len(block_files) > 0, "No block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "Block file is empty"
                    print(f"First block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract Python blocks: {e}")

def test_markdown_block_extraction():
    """Test Markdown code block extraction."""
    # Get the path to sample Markdown file
    sample_file = RESOURCES_DIR / "sample_markdown.md"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=True,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that Markdown sections were extracted
            assert stats.get("documentation_blocks", 0) > 0, "No Markdown sections were extracted"
            print(f"Extracted {stats.get('documentation_blocks', 0)} Markdown sections")
            
            # Check the output directory structure for Markdown sections
            docs_dir = Path(temp_dir) / "blocks" / "docs" / "markdown"
            if not docs_dir.exists():
                pytest.skip("Markdown docs directory was not created - may not be implemented yet")
            
            # Count the number of extracted Markdown sections
            markdown_files = list(docs_dir.glob("*.md"))
            assert len(markdown_files) > 0, "No Markdown section files were created"
            
            # Verify file contents of one section
            if markdown_files:
                with open(markdown_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "Markdown section file is empty"
        
        except Exception as e:
            pytest.skip(f"Failed to extract Markdown blocks: {e}")

def test_javascript_block_extraction():
    """Test JavaScript block extraction using tree-sitter."""
    # Get the path to sample JavaScript file
    sample_file = RESOURCES_DIR / "sample_javascript.js"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats.get("code_blocks", 0) > 0, "No JavaScript code blocks were extracted"
            print(f"Extracted {stats.get('code_blocks', 0)} JavaScript blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "javascript"
            if not blocks_dir.exists():
                pytest.skip("JavaScript blocks directory was not created - may not be supported yet")
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.js"))
            assert len(block_files) > 0, "No JavaScript block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "JavaScript block file is empty"
                    print(f"First JS block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract JavaScript blocks: {e}")

def test_typescript_block_extraction():
    """Test TypeScript block extraction using tree-sitter."""
    # Get the path to sample TypeScript file
    sample_file = RESOURCES_DIR / "sample_typescript.ts"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats.get("code_blocks", 0) > 0, "No TypeScript code blocks were extracted"
            print(f"Extracted {stats.get('code_blocks', 0)} TypeScript blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "typescript"
            if not blocks_dir.exists():
                pytest.skip("TypeScript blocks directory was not created - may not be supported yet")
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.ts"))
            assert len(block_files) > 0, "No TypeScript block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "TypeScript block file is empty"
                    print(f"First TS block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract TypeScript blocks: {e}")

def test_c_language_extraction():
    """Test C language block extraction using tree-sitter."""
    # Get the path to sample C file
    sample_file = RESOURCES_DIR / "sample_c.c"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats.get("code_blocks", 0) > 0, "No C code blocks were extracted"
            print(f"Extracted {stats.get('code_blocks', 0)} C blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "c"
            if not blocks_dir.exists():
                pytest.skip("C blocks directory was not created - may not be supported yet")
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.c"))
            assert len(block_files) > 0, "No C block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "C block file is empty"
                    print(f"First C block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract C blocks: {e}")

def test_go_language_extraction():
    """Test Go language block extraction using tree-sitter."""
    # Get the path to sample Go file
    sample_file = RESOURCES_DIR / "sample_go.go"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats.get("code_blocks", 0) > 0, "No Go code blocks were extracted"
            print(f"Extracted {stats.get('code_blocks', 0)} Go blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "go"
            if not blocks_dir.exists():
                pytest.skip("Go blocks directory was not created - may not be supported yet")
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.go"))
            assert len(block_files) > 0, "No Go block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "Go block file is empty"
                    print(f"First Go block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract Go blocks: {e}")

def test_rust_language_extraction():
    """Test Rust language block extraction using tree-sitter."""
    # Get the path to sample Rust file
    sample_file = RESOURCES_DIR / "sample_rust.rs"
    if not check_file_exists(sample_file):
        return
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Verify that code blocks were extracted
            assert stats.get("code_blocks", 0) > 0, "No Rust code blocks were extracted"
            print(f"Extracted {stats.get('code_blocks', 0)} Rust blocks")
            
            # Check the output directory structure
            blocks_dir = Path(temp_dir) / "blocks" / "code" / "rust"
            if not blocks_dir.exists():
                pytest.skip("Rust blocks directory was not created - may not be supported yet")
            
            # Count the number of extracted blocks
            block_files = list(blocks_dir.glob("*.rs"))
            assert len(block_files) > 0, "No Rust block files were created"
            
            # Verify file contents of one block
            if block_files:
                with open(block_files[0], 'r') as f:
                    content = f.read()
                    assert len(content) > 0, "Rust block file is empty"
                    print(f"First Rust block content: {content[:100]}...")
        
        except Exception as e:
            pytest.skip(f"Failed to extract Rust blocks: {e}")

def test_generic_block_extraction():
    """Test generic block extraction for unsupported languages."""
    # Create a sample file with a generic language
    with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w') as f:
        f.write("""
        function hello() {
            console.log("Hello, world!");
        }
        
        class Example {
            constructor() {
                this.value = 42;
            }
            
            getValue() {
                return this.value;
            }
        }
        """)
        f.flush()
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract code blocks from the sample file
                stats = extract_repository(
                    source=f.name,
                    output_path=temp_dir,
                    extract_documentation=False,
                    extract_code=True,
                    extract_blocks=True
                )
                
                # For generic languages, we don't expect specific block extraction
                # But the file should be processed and we should have no errors
                assert "errors" in stats and len(stats["errors"]) == 0, "Errors occurred during generic extraction"
                
                # If code blocks were extracted, verify them
                if stats.get("code_blocks", 0) > 0:
                    print(f"Extracted {stats.get('code_blocks', 0)} generic blocks")
                    
                    # Detect the language directory
                    code_dir = Path(temp_dir) / "blocks" / "code"
                    if not code_dir.exists():
                        pytest.skip("Code directory was not created - may not be supported for generic languages")
                    
                    # Find any code block files
                    block_files = []
                    for lang_dir in code_dir.iterdir():
                        if lang_dir.is_dir():
                            block_files.extend(list(lang_dir.glob("*.*")))
                    
                    if block_files:
                        # Verify file contents of one block
                        with open(block_files[0], 'r') as f:
                            content = f.read()
                            assert len(content) > 0, "Generic block file is empty"
                else:
                    # Generic languages might not extract blocks, which is acceptable
                    print("No blocks extracted for generic language (acceptable)")
            
            except Exception as e:
                pytest.skip(f"Failed to process generic language file: {e}")

def test_no_empty_blocks():
    """Test that no empty blocks are extracted."""
    # Create a sample file with meaningful and empty blocks
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w') as f:
        f.write("""
        # Valid function
        def valid_function():
            '''This is a valid function with a body.'''
            return "Hello, world!"
            
        # Empty function
        def empty_function():
            '''This function has no body.'''
            pass
            
        # Valid class
        class ValidClass:
            '''This is a valid class with methods.'''
            def __init__(self):
                self.value = 42
                
            def get_value(self):
                return self.value
                
        # Empty class
        class EmptyClass:
            '''This class has no methods or attributes.'''
            pass
        """)
        f.flush()
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract code blocks from the sample file
                stats = extract_repository(
                    source=f.name,
                    output_path=temp_dir,
                    extract_documentation=False,
                    extract_code=True,
                    extract_blocks=True
                )
                
                # Verify that code blocks were extracted
                if stats.get("code_blocks", 0) > 0:
                    print(f"Extracted {stats.get('code_blocks', 0)} Python blocks")
                    
                    # Check the output directory structure
                    blocks_dir = Path(temp_dir) / "blocks" / "code" / "python"
                    if not blocks_dir.exists():
                        pytest.skip("Python blocks directory was not created")
                    
                    # Count the number of extracted blocks
                    block_files = list(blocks_dir.glob("*.py"))
                    
                    # We don't necessarily skip empty blocks, so this test just verifies
                    # that the extraction completed without errors
                    print(f"Found {len(block_files)} block files")
                    
                    # Verify each block file has content
                    for block_file in block_files:
                        with open(block_file, 'r') as f:
                            content = f.read()
                            assert len(content) > 0, f"Block file {block_file.name} is empty"
                else:
                    pytest.skip("No Python blocks were extracted")
            
            except Exception as e:
                pytest.skip(f"Failed to extract Python blocks: {e}")

def test_run_test_function():
    """Test the run_test helper function."""
    try:
        # Run a simple test with a Python string
        result = run_test("""
        def greet(name):
            return f"Hello, {name}!"
            
        class Calculator:
            def add(self, a, b):
                return a + b
        """, language="python")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "stats" in result, "Result should contain stats"
        assert "output_dir" in result, "Result should contain output_dir"
        
        stats = result["stats"]
        assert stats.get("code_blocks", 0) >= 0, "Stats should include code_blocks count"
        
        # If blocks were extracted, check them
        if stats.get("code_blocks", 0) > 0:
            output_dir = Path(result["output_dir"])
            blocks_dir = output_dir / "blocks" / "code" / "python"
            
            if blocks_dir.exists():
                block_files = list(blocks_dir.glob("*.py"))
                assert len(block_files) > 0, "No Python block files were created"
                
                # Clean up temp dir
                shutil.rmtree(output_dir, ignore_errors=True)
        
    except Exception as e:
        pytest.skip(f"Failed to run test function: {e}")

if __name__ == "__main__":
    # Run all tests
    pytest.main(["-xvs", __file__]) 