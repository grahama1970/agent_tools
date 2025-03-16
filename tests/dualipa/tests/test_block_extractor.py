"""
Tests for block extraction functionality in code_extractor.py.

This module tests the block extraction functionality of the code_extractor module
for different languages and file formats.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest

from agent_tools.dualipa.code_extractor import extract_repository, run_test
from agent_tools.dualipa.language_detection import detect_language

# Path to test resources
RESOURCES_DIR = Path(__file__).parent.parent / "resources" / "templates"

def test_python_block_extraction():
    """Test Python block extraction using AST."""
    # Get the path to sample Python file
    sample_file = RESOURCES_DIR / "sample_python.py"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
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
        assert len(block_files) == stats["code_blocks"], "Number of block files doesn't match stats"
        
        # Check for specific functions and classes
        block_names = [f.stem for f in block_files]
        assert any("greet" in name for name in block_names), "Function 'greet' not found in extracted blocks"
        assert any("Calculator" in name for name in block_names), "Class 'Calculator' not found in extracted blocks"
        
        # Verify blocks have metadata
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "# Original file" in content, "Block metadata missing original file info"
                assert "# Block type" in content, "Block metadata missing block type info"
                assert "# Name" in content, "Block metadata missing name info"
                assert "# Docstring" in content, "Block metadata missing docstring info"

def test_markdown_block_extraction():
    """Test Markdown block extraction."""
    # Get the path to sample Markdown file
    sample_file = RESOURCES_DIR / "sample_markdown.md"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=True,
            extract_code=False,
            extract_blocks=True
        )
        
        # Verify that documentation blocks were extracted
        assert stats["doc_blocks"] > 0, "No Markdown blocks were extracted"
        print(f"Extracted {stats['doc_blocks']} Markdown blocks")
        
        # Check the output directory structure
        blocks_dir = Path(temp_dir) / "blocks" / "docs" / "markdown"
        assert blocks_dir.exists(), "Markdown blocks directory was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob("*.md"))
        assert len(block_files) > 0, "No Markdown block files were found"
        
        # Check for specific sections
        sections_found = False
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                if "# Sample Documentation" in content or "## Python Example" in content:
                    sections_found = True
                assert "<!-- Original file" in content, "Block metadata missing original file info"
                assert "<!-- Section" in content, "Block metadata missing section info"
        
        assert sections_found, "Expected Markdown sections not found in extracted blocks"

def test_javascript_block_extraction():
    """Test JavaScript block extraction."""
    # Get the path to sample JavaScript file
    sample_file = RESOURCES_DIR / "sample_javascript.js"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that code blocks were extracted
        assert stats["code_blocks"] > 0, "No JavaScript code blocks were extracted"
        print(f"Extracted {stats['code_blocks']} JavaScript blocks")
        
        # Check the output directory structure
        blocks_dir = Path(temp_dir) / "blocks" / "code" / "javascript"
        assert blocks_dir.exists(), "JavaScript blocks directory was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob("*.javascript"))
        assert len(block_files) > 0, "No JavaScript block files were found"
        
        # Verify each block has metadata
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "// Original file" in content, "Block metadata missing original file info"

def test_typescript_block_extraction():
    """Test TypeScript block extraction."""
    # Get the path to sample TypeScript file
    sample_file = RESOURCES_DIR / "sample_typescript.ts"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that code blocks were extracted
        assert stats["code_blocks"] > 0, "No TypeScript code blocks were extracted"
        print(f"Extracted {stats['code_blocks']} TypeScript blocks")
        
        # Check the output directory structure
        blocks_dir = Path(temp_dir) / "blocks" / "code" / "typescript"
        assert blocks_dir.exists(), "TypeScript blocks directory was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob("*.typescript"))
        assert len(block_files) > 0, "No TypeScript block files were found"
        
        # Verify each block has metadata
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "// Original file" in content, "Block metadata missing original file info"

def test_c_language_extraction():
    """Test C language block extraction."""
    # Get the path to sample C file
    sample_file = RESOURCES_DIR / "sample_c.c"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that code blocks were extracted
        assert stats["code_blocks"] > 0, "No C code blocks were extracted"
        print(f"Extracted {stats['code_blocks']} C blocks")
        
        # Check the output directory structure
        language = detect_language(sample_file)
        blocks_dir = Path(temp_dir) / "blocks" / "code" / language
        assert blocks_dir.exists(), f"C blocks directory ({language}) was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob(f"*{sample_file.suffix}"))
        assert len(block_files) > 0, "No C block files were found"
        
        # Verify each block has metadata and is non-empty
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "// Original file" in content, "Block metadata missing original file info"
                assert len(content.split('\n')) > 3, "Block content is too short"

def test_go_language_extraction():
    """Test Go language block extraction."""
    # Get the path to sample Go file
    sample_file = RESOURCES_DIR / "sample_go.go"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that code blocks were extracted
        assert stats["code_blocks"] > 0, "No Go code blocks were extracted"
        print(f"Extracted {stats['code_blocks']} Go blocks")
        
        # Check the output directory structure
        language = detect_language(sample_file)
        blocks_dir = Path(temp_dir) / "blocks" / "code" / language
        assert blocks_dir.exists(), f"Go blocks directory ({language}) was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob(f"*{sample_file.suffix}"))
        assert len(block_files) > 0, "No Go block files were found"
        
        # Verify each block has metadata and is non-empty
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "// Original file" in content, "Block metadata missing original file info"
                assert len(content.split('\n')) > 3, "Block content is too short"

def test_rust_language_extraction():
    """Test Rust language block extraction."""
    # Get the path to sample Rust file
    sample_file = RESOURCES_DIR / "sample_rust.rs"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that code blocks were extracted
        assert stats["code_blocks"] > 0, "No Rust code blocks were extracted"
        print(f"Extracted {stats['code_blocks']} Rust blocks")
        
        # Check the output directory structure
        language = detect_language(sample_file)
        blocks_dir = Path(temp_dir) / "blocks" / "code" / language
        assert blocks_dir.exists(), f"Rust blocks directory ({language}) was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob(f"*{sample_file.suffix}"))
        assert len(block_files) > 0, "No Rust block files were found"
        
        # Verify each block has metadata and is non-empty
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "// Original file" in content, "Block metadata missing original file info"
                assert len(content.split('\n')) > 3, "Block content is too short"

def test_generic_block_extraction():
    """Test generic text file block extraction."""
    # Get the path to sample generic file
    sample_file = RESOURCES_DIR / "sample_generic.txt"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=True,
            extract_code=True,
            extract_blocks=True
        )
        
        # Verify that blocks were extracted
        assert stats["code_blocks"] > 0, "No generic blocks were extracted"
        print(f"Extracted {stats['code_blocks']} generic blocks")
        
        # Check the output directory structure
        language = "txt"  # Generic text files are treated as txt
        blocks_dir = Path(temp_dir) / "blocks" / "code" / language
        assert blocks_dir.exists(), f"Generic blocks directory ({language}) was not created"
        
        # Count the number of extracted blocks
        block_files = list(blocks_dir.glob(f"*{sample_file.suffix}"))
        assert len(block_files) > 0, "No generic block files were found"
        
        # Verify each block has metadata and is non-empty
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                assert "# Original file" in content, "Block metadata missing original file info"
                assert len(content.strip()) > 0, "Block content is empty"

def test_no_empty_blocks():
    """Test that no empty blocks are generated."""
    # Test with multiple file types
    for filename in [
        "sample_python.py", 
        "sample_javascript.js", 
        "sample_typescript.ts",
        "sample_c.c",
        "sample_go.go",
        "sample_rust.rs",
        "sample_generic.txt"
    ]:
        sample_file = RESOURCES_DIR / filename
        if not sample_file.exists():
            print(f"Skipping {filename} - file not found")
            continue
            
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract code blocks from the sample file
            stats = extract_repository(
                source=str(sample_file),
                output_path=temp_dir,
                extract_documentation=False,
                extract_code=True,
                extract_blocks=True
            )
            
            # Get the language
            language = detect_language(sample_file)
            blocks_dir = Path(temp_dir) / "blocks" / "code" / language
            
            # Skip if no blocks directory was created
            if not blocks_dir.exists():
                print(f"Skipping {filename} - no blocks directory created")
                continue
                
            # Check that no empty blocks were created
            for block_file in blocks_dir.glob(f"*{sample_file.suffix}"):
                with open(block_file, 'r') as f:
                    # Get the content after the metadata lines
                    content = f.read()
                    # Skip metadata lines (lines starting with # or //)
                    lines = [line.strip() for line in content.split('\n') 
                             if not (line.strip().startswith('#') or 
                                    line.strip().startswith('//') or 
                                    line.strip().startswith('<!--'))]
                    non_empty_lines = [line for line in lines if line]
                    
                    # Assert that there is at least some non-empty content
                    assert len(non_empty_lines) > 0, f"Empty block found in {block_file}"

def test_run_test_function():
    """Test the run_test utility function."""
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the test with a sample file
        sample_file = RESOURCES_DIR / "sample_python.py"
        stats = run_test(
            source_path=str(sample_file),
            output_dir=temp_dir,
            max_files=1
        )
        
        # Verify that the test ran successfully
        assert stats["code_files"] == 1, "Expected exactly one code file to be processed"
        assert stats["code_blocks"] > 0, "Expected at least one code block to be extracted"
        assert "python" in stats["languages"], "Python language not detected"
        
        # Check the output directory structure
        assert Path(temp_dir).exists(), "Output directory does not exist"
        assert (Path(temp_dir) / "code").exists(), "Code directory was not created"
        assert (Path(temp_dir) / "blocks").exists(), "Blocks directory was not created"

if __name__ == "__main__":
    # Run all tests
    pytest.main(["-xvs", __file__]) 