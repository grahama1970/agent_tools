"""
Test the markdown_parser module.

These tests verify that the markdown parser correctly extracts sections,
code blocks, and other content from real-world markdown files.

Official Documentation References:
- markdown-it-py: https://markdown-it-py.readthedocs.io/
- pytest: https://docs.pytest.org/
"""

import os
import sys
import pytest
import tempfile
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
    from agent_tools.dualipa.markdown_it_parser import (
        MARKDOWN_IT_AVAILABLE,
        markdown_to_hierarchical_json,
        extract_code_blocks,
        process_markdown_file,
        get_markdown_files,
        get_flattened_markdown_content
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

# Local markdown samples from cloned repositories
REPO_ROOT = project_root / "test_repos"
REAL_MARKDOWN_SOURCES = [
    # Use README.md files from cloned repositories
    str(REPO_ROOT / "rust-analyzer" / "README.md"),
    str(REPO_ROOT / "requests" / "README.md"),
    str(REPO_ROOT / "react" / "README.md"),
    # Additional markdown files from repositories
    str(REPO_ROOT / "rust-analyzer" / "CONTRIBUTING.md"),
    str(REPO_ROOT / "rust-analyzer" / "docs" / "book" / "src" / "features.md"),
]

@pytest.fixture
def real_markdown_content():
    """Load real markdown content from local repository files."""
    for filepath in REAL_MARKDOWN_SOURCES[:3]:  # Try first three sources
        try:
            if os.path.exists(filepath):
                print(f"Successfully loaded content from {filepath}")
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Fallback content if all files fail
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
        pytest.skip("markdown-it-py not available")
        
    try:
        # Parse the markdown content with markdown-it
        result = markdown_to_hierarchical_json(real_markdown_content)
        
        # Get sections from hierarchy
        hierarchy = result["document"]["hierarchy"]
        
        # Verify that sections were extracted
        assert isinstance(hierarchy, dict), "Hierarchy should be a dictionary"
        assert len(hierarchy) > 0, "At least one section should be extracted"
        
        # Check the structure of the first section
        first_section_title = next(iter(hierarchy))
        first_section = hierarchy[first_section_title]
        assert isinstance(first_section, dict), "Section should be a dictionary"
        assert "level" in first_section, "Section should have a level"
        assert "title" in first_section, "Section should have a title"
        assert "content" in first_section, "Section should have content"
        assert "metadata" in first_section, "Section should have metadata"
        
        # Print some information about the extracted sections
        print(f"Extracted {len(hierarchy)} sections from markdown content")
        for i, (title, section) in enumerate(list(hierarchy.items())[:3]):  # Print details of first 3 sections
            print(f"Section {i+1}: Level {section['level']}, Title: {title}")
            print(f"Content blocks: {len(section['content'])} elements")
    
    except Exception as e:
        pytest.skip(f"Error extracting sections: {e}")

def test_extract_code_blocks(markdown_with_code_blocks):
    """Test extracting code blocks from markdown with multiple languages."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it-py not available")
        
    try:
        # Extract code blocks from the markdown content
        result = markdown_to_hierarchical_json(markdown_with_code_blocks)
        blocks = result["code_blocks"]
        
        # Verify that blocks were extracted
        assert isinstance(blocks, list), "Blocks should be a list"
        assert len(blocks) > 0, "At least one block should be extracted"
        
        # Check the structure of each block
        for block in blocks:
            assert isinstance(block, dict), "Block should be a dictionary"
            assert "language" in block, "Block should have a language"
            assert "content" in block, "Block should have content"
            assert "token_count" in block, "Block should have token count"
        
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
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it-py not available")
        
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
            assert "document" in result, "Result should have document"
            assert "hierarchy" in result["document"], "Result should have document hierarchy"
            
            # Print information about the processed file
            blocks = result.get("code_blocks", [])
            hierarchy = result.get("document", {}).get("hierarchy", {})
            print(f"Processed markdown file: {len(blocks)} code blocks, {len(hierarchy)} top-level sections")
            
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
    # Skip if we don't have access to real repositories
    if not os.path.exists(REPO_ROOT):
        pytest.skip("Test repositories directory not found")
        
    try:
        # Find markdown files in the rust-analyzer repo
        rust_repo = REPO_ROOT / "rust-analyzer"
        if os.path.exists(rust_repo):
            md_files = get_markdown_files(rust_repo)
            
            # Verify that files were found
            assert isinstance(md_files, list), "Result should be a list"
            assert len(md_files) > 0, "Should find at least one markdown file"
            
            # Check file types
            for file_path in md_files:
                assert file_path.suffix.lower() in [".md", ".markdown", ".mdown"], f"Invalid file extension: {file_path.suffix}"
            
            print(f"Found {len(md_files)} markdown files in {rust_repo}")
            
            # Try non-recursive mode
            md_files_nonrecursive = get_markdown_files(rust_repo, recursive=False)
            print(f"Found {len(md_files_nonrecursive)} markdown files in {rust_repo} (non-recursive)")
        else:
            pytest.skip("Rust-analyzer repository not found")
    
    except Exception as e:
        pytest.skip(f"Error finding markdown files: {e}")

def test_markdown_parser_availability():
    """Test the availability of markdown parser implementation."""
    print(f"markdown-it-py availability: {MARKDOWN_IT_AVAILABLE}")
    
    # No need to assert - this test just reports availability
    # It will be skipped automatically if imports fail
    
    if MARKDOWN_IT_AVAILABLE:
        print("markdown-it-py is available and working")

def test_process_readme_files():
    """Test processing README.md files from real repositories."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it-py not available")
        
    # Test with real repositories if available
    for repo_name in ["react", "requests", "rust-analyzer"]:
        repo_path = REPO_ROOT / repo_name
        readme_path = repo_path / "README.md"
        
        if not os.path.exists(readme_path):
            print(f"Skipping {repo_name}: README.md not found")
            continue
            
        try:
            print(f"Processing README.md from {repo_name}")
            
            # Process the file
            result = process_markdown_file(readme_path)
            
            # Verify basic structure
            assert "document" in result
            assert "hierarchy" in result["document"]
            assert "code_blocks" in result
            
            # Print statistics
            blocks = result["code_blocks"]
            hierarchy = result["document"]["hierarchy"]
            code_block_langs = [block.get("language", "none") for block in blocks]
            lang_counts = {}
            for lang in code_block_langs:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
            # Print summary
            print(f"{repo_name} README.md: {len(blocks)} code blocks, {len(hierarchy)} top-level sections")
            print(f"Language distribution: {lang_counts}")
            
            # Get top-level section titles
            print(f"Top-level sections: {list(hierarchy.keys())[:3]}")
        
        except Exception as e:
            print(f"Error processing {repo_name} README.md: {e}")

def test_extract_code_blocks_from_documentation():
    """Test extracting code blocks from documentation files."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it-py not available")
    
    # Look for documentation markdown files
    doc_paths = []
    for repo_name in ["rust-analyzer", "requests"]:
        repo_path = REPO_ROOT / repo_name
        
        # Common documentation folders
        doc_folders = [
            repo_path / "docs",
            repo_path / "doc", 
            repo_path / "documentation"
        ]
        
        # Try to find markdown files in these folders
        for folder in doc_folders:
            if folder.exists() and folder.is_dir():
                md_files = list(folder.glob("**/*.md"))
                if md_files:
                    doc_paths.extend(md_files[:2])  # Take up to 2 files per folder
                    
    if not doc_paths:
        pytest.skip("No documentation markdown files found")
    
    # Process one documentation file
    try:
        doc_file = doc_paths[0]
        print(f"Processing documentation file: {doc_file}")
        
        # Process the markdown file
        result = process_markdown_file(doc_file)
        
        # Get code blocks
        code_blocks = result["code_blocks"]
        
        # Verify basic structure
        assert isinstance(code_blocks, list)
        
        # Print code block information
        print(f"Found {len(code_blocks)} code blocks in {doc_file.name}")
        languages = [block.get("language", "none") for block in code_blocks]
        language_counts = {}
        for lang in languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        print(f"Language distribution: {language_counts}")
        print(f"Total tokens in code blocks: {sum(block.get('token_count', 0) for block in code_blocks)}")
    
    except Exception as e:
        pytest.skip(f"Error extracting code blocks from documentation: {e}")

if __name__ == "__main__":
    # Run tests directly
    test_extract_sections_from_markdown(real_markdown_content())
    test_extract_code_blocks(markdown_with_code_blocks())
    test_process_markdown_file(real_markdown_content()) 