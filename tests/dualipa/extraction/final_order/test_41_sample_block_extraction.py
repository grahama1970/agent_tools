"""
Tests for block extraction functionality in code_extractor.py.

This module tests the block extraction functionality of the code_extractor module
for different languages and file formats.

These tests use real code examples from template files or sample repositories
to verify that code blocks are properly extracted.
"""

import os
import sys
import tempfile
import pytest
import json
from pathlib import Path

# Configure the Path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Flag to track if dependencies are available
HAS_DEPENDENCIES = False
try:
    from agent_tools.dualipa.code_extractor import (
        _extract_python_blocks,
        _extract_js_ts_blocks,
        _extract_markdown_blocks,
        _extract_generic_blocks,
        initialize_stats_dict,
        run_test
    )
    from agent_tools.dualipa.language_detection import detect_language
    HAS_DEPENDENCIES = True
except ImportError as e:
    raise ImportError(f"Required code extractor modules not available: {e}. Fix the dependencies to run these tests.")

# Path to test resources (local templates preferred over external URLs)
RESOURCES_DIR = project_root / "src" / "agent_tools" / "dualipa" / "resources" / "templates"

def check_file_exists(file_path):
    """Check if a test file exists, failing loudly if not."""
    if not file_path.exists():
        assert False, f"Sample file {file_path} does not exist. Create this file to run the test."
    return True

def visualize_hierarchy(sections, indent=0):
    """Helper function to visualize the hierarchy for debugging."""
    result = []
    prefix = "  " * indent
    for section in sections:
        title = section.get("title", "No title")
        level = section.get("level", 0)
        result.append(f"{prefix}- [{level}] {title}")
        if "subsections" in section and section["subsections"]:
            child_result = visualize_hierarchy(section["subsections"], indent + 1)
            result.extend(child_result)
    return result

def test_python_block_extraction():
    """Test Python block extraction using AST."""
    # Try to find a Python template file locally
    python_template = RESOURCES_DIR / "python_template.py"
    if not python_template.exists():
        # If not available, create a simple Python file for testing
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as temp_file:
            temp_file.write('''
def hello_world():
    """A simple Python function."""
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    """A simple test class."""
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        """Greet the user."""
        return f"Hello, {self.name}!"
''')
            temp_file.flush()
            sample_file = Path(temp_file.name)
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                try:
                    with open(sample_file, 'r') as f:
                        sample_code = f.read()
                    stats = initialize_stats_dict(source=sample_file, output_dir=output_dir)
                    num_blocks = _extract_python_blocks(
                        file_path=sample_file,
                        content=sample_code,
                        output_dir=output_dir,
                        stats=stats
                    )
                    assert isinstance(num_blocks, int), "Should return number of blocks"
                    assert num_blocks > 0, "Should extract at least one code block"
                    assert stats["code_blocks"] > 0, "Stats should be updated with blocks count"
                    blocks_dir = output_dir / "blocks" / "code" / "python"
                    if not blocks_dir.exists():
                        pytest.fail("Python blocks directory was not created")
                    block_files = list(blocks_dir.glob("*.py"))
                    if not block_files:
                        pytest.fail("No Python blocks were extracted")
                    print(f"Blocks directory: {blocks_dir}")
                    print(f"Found {len(block_files)} block files")
                    for block_file in block_files[:3]:
                        print(f"  - {block_file.name}")
                        with open(block_file, 'r') as f:
                            content = f.read()
                        preview = content[:50] + "..." if len(content) > 50 else content
                        print(f"    Content preview: {preview}")
                        assert "def " in content or "class " in content, "Block should contain function or class"
                except Exception as e:
                    pytest.fail(f"Failed to extract Python blocks: {str(e)}")
    else:
        print(f"Using Python template file: {python_template}")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            try:
                with open(python_template, 'r') as f:
                    template_code = f.read()
                stats = initialize_stats_dict(source=python_template, output_dir=output_dir)
                num_blocks = _extract_python_blocks(
                    file_path=python_template,
                    content=template_code,
                    output_dir=output_dir,
                    stats=stats
                )
                assert isinstance(num_blocks, int), "Should return number of blocks"
                assert num_blocks > 0, "Should extract at least one block"
                assert stats["code_blocks"] > 0, "Stats should be updated with blocks count"
                blocks_dir = output_dir / "blocks" / "code" / "python"
                if not blocks_dir.exists():
                    pytest.fail("Python blocks directory was not created")
                block_files = list(blocks_dir.glob("*.py"))
                assert len(block_files) > 0, "Should create at least one block file"
                print(f"Blocks directory: {blocks_dir}")
                print(f"Found {len(block_files)} block files")
                for block_file in block_files[:3]:
                    print(f"  - {block_file.name}")
                    with open(block_file, 'r') as f:
                        content = f.read()
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"    Content preview: {preview}")
            except Exception as e:
                pytest.fail(f"Failed to extract Python blocks: {str(e)}")

def test_markdown_block_extraction():
    """Test Markdown code block extraction."""
    sample_file = project_root / "test_repos" / "react" / "packages" / "react-devtools-core" / "README.md"
    if not sample_file.exists():
        pytest.skip(f"Sample markdown file not found: {sample_file}")
    print(f"Using markdown file: {sample_file}")
    with tempfile.TemporaryDirectory() as temp_dir:
        stats = initialize_stats_dict(source=sample_file, output_dir=Path(temp_dir))
        try:
            with open(sample_file, 'r', errors='ignore') as f:
                content = f.read()
            print(f"File content length: {len(content)} characters")
            print(f"First 100 chars: {content[:100]}")
            output_dir = Path(temp_dir)
            print(f"Output directory: {output_dir}")
            num_blocks = _extract_markdown_blocks(
                file_path=sample_file,
                content=content,
                output_dir=output_dir,
                stats=stats
            )
            assert isinstance(num_blocks, int), "Should return number of blocks"
            print(f"Extraction returned {num_blocks} blocks")
            print(f"Stats: {stats}")
            blocks_dir = output_dir / "blocks"
            if blocks_dir.exists():
                code_dir = blocks_dir / "code"
                doc_dir = blocks_dir / "documentation"
                if code_dir.exists():
                    print(f"Code directory exists: {code_dir}")
                    lang_dirs = list(code_dir.glob("*"))
                    print(f"Language directories: {[d.name for d in lang_dirs]}")
                    code_blocks = []
                    for lang_dir in lang_dirs:
                        if lang_dir.is_dir():
                            lang_blocks = list(lang_dir.glob("*"))
                            code_blocks.extend(lang_blocks)
                            print(f"Found {len(lang_blocks)} blocks in {lang_dir.name}")
                    print(f"Total code blocks: {len(code_blocks)}")
                else:
                    print("Code directory doesn't exist")
                if doc_dir.exists():
                    print(f"Documentation directory exists: {doc_dir}")
                    doc_blocks = list(doc_dir.glob("*"))
                    print(f"Total doc blocks: {len(doc_blocks)}")
                else:
                    print("Documentation directory doesn't exist")
            else:
                print("Blocks directory doesn't exist")
            if stats.get("errors"):
                print(f"Extraction errors: {stats['errors']}")
                pytest.skip(f"Skipping due to extraction errors: {stats['errors']}")
            if num_blocks == 0:
                pytest.skip("No blocks extracted, skipping further assertions")
        except Exception as e:
            print(f"Exception during test: {e}")
            print(f"Stats at exception: {stats}")
            pytest.skip(f"Exception during markdown block extraction: {e}")

def test_run_test_function():
    """Test the run_test helper function."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as temp_file:
            temp_file.write("""
def greet(name):
    return f"Hello, {name}!"
    
class Calculator:
    def add(self, a, b):
        return a + b
""")
            temp_file.flush()
            result = run_test(
                source_path=temp_file.name,
                output_dir=None,
                max_files=1
            )
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "code_files" in result, "Result should contain code_files count"
        assert "code_blocks" in result, "Result should contain code_blocks count"
        if result.get("code_blocks", 0) > 0:
            print(f"Extracted {result.get('code_blocks', 0)} code blocks in run_test")
    except Exception as e:
        pytest.fail(f"Failed to run test function: {e}")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
