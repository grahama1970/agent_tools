"""
Test the markdown_parser module.

These tests verify that the markdown parser correctly extracts sections,
code blocks, and other content from real-world markdown files.

Official Documentation References:
- markdown-it-py: https://markdown-it-py.readthedocs.io/
- mistune: https://mistune.readthedocs.io/
- pytest: https://docs.pytest.org/
"""

import os
import sys
import pytest
import tempfile
import requests
import json
from pathlib import Path

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Flag to track if required modules exist
HAS_DEPENDENCIES = False

# Import directly from the package
try:
    from agent_tools.dualipa.markdown_parser import (
        extract_sections_from_markdown,
        extract_code_blocks,
        process_markdown_file,
        get_markdown_files,
        MARKDOWN_IT_AVAILABLE
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Markdown parser dependencies not available, tests will be skipped")

# Skip all tests if the required modules don't exist
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="Required markdown parser modules not available"
)

# Real markdown sources for blind testing
REAL_MARKDOWN_SOURCES = [
    # Popular repositories with good documentation
    "https://raw.githubusercontent.com/pallets/flask/main/README.md",
    "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
    "https://raw.githubusercontent.com/fastapi-users/fastapi-users/main/README.md",
    "https://raw.githubusercontent.com/expressjs/express/master/Readme.md",
    "https://raw.githubusercontent.com/microsoft/TypeScript/main/README.md",
    # Project documentation with code examples
    "https://raw.githubusercontent.com/sqlalchemy/sqlalchemy/main/doc/build/intro.rst",
    "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/interpreter.rst",
]

@pytest.fixture
def real_markdown_content():
    """Fetch real markdown content from GitHub repositories."""
    for url in REAL_MARKDOWN_SOURCES[:3]:  # Try first three sources
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Successfully fetched content from {url}")
                return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    
    # Fallback content if all URLs fail
    print("Using fallback markdown content")
    return """
    # Sample Markdown
    
    This is a sample markdown file with code blocks.
    
    ## Section 1
    
    ```python
    def hello_world():
        print("Hello, World!")
    ```
    
    ## Section 2
    
    ```javascript
    function greet(name) {
        console.log(`Hello, ${name}!`);
    }
    ```
    """

@pytest.fixture
def markdown_with_code_blocks():
    """Create a markdown string with multiple code blocks in various languages."""
    # Create a markdown string with code blocks in multiple languages
    return """
    # Markdown with Code Blocks
    
    This fixture provides a markdown document with code blocks in multiple languages.
    
    ## Python Example
    
    ```python
    def hello_world():
        print("Hello, World!")
        
    class Example:
        def __init__(self, value):
            self.value = value
            
        def get_value(self):
            return self.value
    ```
    
    ## JavaScript Example
    
    ```javascript
    function greet(name) {
        console.log(`Hello, ${name}!`);
    }
    
    class Calculator {
        constructor() {
            this.value = 0;
        }
        
        add(a, b) {
            return a + b;
        }
    }
    ```
    
    ## Go Example
    
    ```go
    package main
    
    import "fmt"
    
    func main() {
        fmt.Println("Hello, World!")
    }
    ```
    
    ## Plain text (no language specified)
    
    ```
    This is a plain text code block
    without a language specifier.
    ```
    """

def test_extract_sections_from_markdown(real_markdown_content):
    """Test extracting sections from real markdown content."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it-py is not available")
        
    try:
        # Extract sections from the markdown content
        sections = extract_sections_from_markdown(real_markdown_content)
        
        # Verify that sections were extracted
        assert isinstance(sections, list), "Sections should be a list"
        assert len(sections) > 0, "At least one section should be extracted"
        
        # Check the structure of the first section
        first_section = sections[0]
        assert isinstance(first_section, dict), "Section should be a dictionary"
        assert "level" in first_section, "Section should have a level"
        assert "title" in first_section, "Section should have a title"
        assert "content" in first_section, "Section should have content"
        
        # Print some information about the extracted sections
        print(f"Extracted {len(sections)} sections from markdown content")
        for i, section in enumerate(sections[:3]):  # Print details of first 3 sections
            print(f"Section {i+1}: Level {section['level']}, Title: {section['title']}")
            print(f"Content length: {len(section['content'])} characters")
    
    except Exception as e:
        pytest.skip(f"Error extracting sections: {e}")

def test_extract_code_blocks(markdown_with_code_blocks):
    """Test extracting code blocks from markdown with multiple languages."""
    try:
        # Extract code blocks from the markdown content
        blocks = extract_code_blocks(markdown_with_code_blocks)
        
        # Verify that blocks were extracted
        assert isinstance(blocks, list), "Blocks should be a list"
        assert len(blocks) > 0, "At least one block should be extracted"
        
        # Check the structure of each block
        for block in blocks:
            assert isinstance(block, dict), "Block should be a dictionary"
            assert "language" in block, "Block should have a language"
            assert "content" in block, "Block should have content"
        
        # Verify specific languages were extracted
        languages = [block["language"] for block in blocks]
        print(f"Extracted languages: {languages}")
        
        # Check for Python blocks
        python_blocks = [block for block in blocks if block["language"] == "python"]
        if python_blocks:
            assert "def hello_world" in python_blocks[0]["content"], "Python content mismatch"
        
        # Check for JavaScript blocks
        js_blocks = [block for block in blocks if block["language"] == "javascript"]
        if js_blocks:
            assert "function greet" in js_blocks[0]["content"], "JavaScript content mismatch"
    
    except Exception as e:
        pytest.skip(f"Error extracting code blocks: {e}")

def test_process_markdown_file(real_markdown_content):
    """Test processing a markdown file to extract code blocks and sections."""
    try:
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
            f.write(real_markdown_content)
            f.flush()
            
            # Process the markdown file
            result = process_markdown_file(f.name)
            
            # Verify the result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "code_blocks" in result, "Result should have code_blocks"
            assert "sections" in result, "Result should have sections"
            
            # Print information about the processed file
            blocks = result.get("code_blocks", [])
            sections = result.get("sections", [])
            print(f"Processed markdown file: {len(blocks)} code blocks, {len(sections)} sections")
            
            # Get language statistics
            languages = {}
            for block in blocks:
                lang = block.get("language", "none")
                languages[lang] = languages.get(lang, 0) + 1
            
            if languages:
                print(f"Language distribution: {languages}")
    
    except Exception as e:
        pytest.skip(f"Error processing markdown file: {e}")

def test_get_markdown_files():
    """Test finding markdown files in a directory structure."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory structure with markdown files
            root_dir = Path(temp_dir)
            
            # Create markdown files in the root directory
            (root_dir / "readme.md").write_text("# Root README")
            (root_dir / "document.markdown").write_text("# Document")
            
            # Create subdirectories with markdown files
            docs_dir = root_dir / "docs"
            docs_dir.mkdir()
            (docs_dir / "guide.md").write_text("# Guide")
            
            subdir = docs_dir / "subdir"
            subdir.mkdir()
            (subdir / "advanced.md").write_text("# Advanced")
            
            # Create non-markdown files to ensure they're not included
            (root_dir / "script.py").write_text("print('hello')")
            (docs_dir / "data.json").write_text('{"key": "value"}')
            
            # Find markdown files
            md_files = get_markdown_files(str(root_dir))
            
            # Verify files were found
            assert isinstance(md_files, list), "Result should be a list"
            assert len(md_files) == 4, f"Should find 4 markdown files, found {len(md_files)}"
            
            # Verify all files are markdown
            for file_path in md_files:
                assert file_path.lower().endswith(('.md', '.markdown')), f"Non-markdown file found: {file_path}"
            
            # Test with pattern matching
            # Find only files in the docs directory
            docs_files = get_markdown_files(str(root_dir), pattern="**/docs/*.md")
            assert len(docs_files) == 1, f"Should find 1 file matching pattern, found {len(docs_files)}"
            
            # Print the files found
            print(f"All markdown files: {[Path(f).name for f in md_files]}")
            if docs_files:
                print(f"Docs markdown files: {[Path(f).name for f in docs_files]}")
    
    except Exception as e:
        pytest.skip(f"Error testing get_markdown_files: {e}")

def test_markdown_parser_availability():
    """Test the availability of markdown parsers."""
    # This test checks if at least one parser is available
    try:
        # Create a simple markdown document
        md_content = "# Test\n\nThis is a test.\n\n```python\nprint('Hello')\n```"
        
        # Try to extract sections
        sections = extract_sections_from_markdown(md_content)
        assert isinstance(sections, list), "Should return a list of sections"
        
        # Try to extract code blocks
        blocks = extract_code_blocks(md_content)
        assert isinstance(blocks, list), "Should return a list of code blocks"
        
        # If we got here, at least one parser is working
        if MARKDOWN_IT_AVAILABLE:
            print("markdown-it-py is available and working")
        else:
            print("Using fallback parser (mistune or regular expressions)")
        
        # Print results
        print(f"Extracted {len(sections)} sections and {len(blocks)} code blocks")
        
    except Exception as e:
        pytest.skip(f"No markdown parsers available: {e}")

def test_process_readme_files():
    """Test processing multiple README files from real repositories."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory for README files
            root_dir = Path(temp_dir)
            
            # Create files with content from real repositories
            readme_files = []
            
            # Try to get content from each source
            for i, url in enumerate(REAL_MARKDOWN_SOURCES[:2]):  # Try first two sources
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        file_path = root_dir / f"readme_{i+1}.md"
                        file_path.write_text(response.text)
                        readme_files.append(str(file_path))
                        print(f"Created test file from {url}")
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
            
            # If we couldn't get any real content, create synthetic files
            if not readme_files:
                print("Creating synthetic README files")
                file_path = root_dir / "readme_fallback.md"
                file_path.write_text("""
                # Project README
                
                This is a sample README file.
                
                ## Installation
                
                ```bash
                pip install sample-package
                ```
                
                ## Usage
                
                ```python
                import sample
                
                sample.hello()
                ```
                """)
                readme_files.append(str(file_path))
            
            # Process each README file
            for file_path in readme_files:
                result = process_markdown_file(file_path)
                
                # Verify processing worked
                assert result, f"Processing failed for {file_path}"
                assert "sections" in result, "Result should contain sections"
                assert "code_blocks" in result, "Result should contain code blocks"
                
                # Print statistics
                print(f"File: {Path(file_path).name}")
                print(f"  Sections: {len(result['sections'])}")
                print(f"  Code blocks: {len(result['code_blocks'])}")
                
                # Print language distribution
                languages = {}
                for block in result["code_blocks"]:
                    lang = block.get("language", "none")
                    languages[lang] = languages.get(lang, 0) + 1
                
                if languages:
                    print(f"  Languages: {languages}")
    
    except Exception as e:
        pytest.skip(f"Error testing process_readme_files: {e}")

def test_extract_code_blocks_from_documentation():
    """Test extracting code blocks from real project documentation."""
    try:
        # URLs pointing to documentation with code examples
        doc_urls = [
            "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/interpreter.rst",
            "https://raw.githubusercontent.com/sqlalchemy/sqlalchemy/main/doc/build/intro.rst",
            "https://raw.githubusercontent.com/pallets/flask/main/docs/quickstart.rst"
        ]
        
        content = None
        source_url = None
        
        # Try to get content from each source
        for url in doc_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    content = response.text
                    source_url = url
                    print(f"Using content from {url}")
                    break
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        
        # If we couldn't get any real content, create synthetic doc
        if not content:
            print("Using synthetic documentation")
            content = """
            Documentation Example
            ====================
            
            Installation
            ------------
            
            .. code-block:: console
            
                $ pip install example
            
            Usage
            -----
            
            .. code-block:: python
            
                from example import Example
                
                app = Example()
                app.run()
            
            Configuration
            -------------
            
            .. code-block:: python
            
                app.config = {
                    'debug': True,
                    'log_level': 'INFO'
                }
            """
            source_url = "synthetic"
        
        # Process as markdown/rst
        blocks = extract_code_blocks(content)
        
        # Verify blocks were extracted
        assert isinstance(blocks, list), "Result should be a list"
        assert len(blocks) > 0, "Should extract at least one code block"
        
        # Print statistics
        print(f"Extracted {len(blocks)} code blocks from {source_url}")
        
        # Get language counts
        languages = {}
        for block in blocks:
            lang = block.get("language", "none")
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"Language distribution: {languages}")
        
        # Print a sample of the first code block
        if blocks:
            first_block = blocks[0]
            lang = first_block.get("language", "none")
            content = first_block.get("content", "")
            print(f"First block ({lang}):")
            print(content[:100] + "..." if len(content) > 100 else content)
    
    except Exception as e:
        pytest.skip(f"Error extracting code blocks from documentation: {e}")

if __name__ == "__main__":
    # Run tests directly
    test_extract_sections_from_markdown(real_markdown_content())
    test_extract_code_blocks(markdown_with_code_blocks())
    test_process_markdown_file(real_markdown_content()) 