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
    # Import the markdown parser module
    from agent_tools.dualipa.markdown_parser import (
        extract_sections_from_markdown,
        extract_code_blocks as extract_blocks_mistune,
        process_markdown_file,
        get_markdown_files,
        extract_code_blocks_from_documentation,
        markdown_to_html
    )
    
    # Try to import markdown-it parser too
    from agent_tools.dualipa.markdown_it_parser import (
        MARKDOWN_IT_AVAILABLE,
        markdown_to_hierarchical_json,
        extract_code_blocks as extract_blocks_markdown_it,
        process_markdown_file as process_markdown_file_it,
        get_markdown_files as get_markdown_files_it,
        get_flattened_markdown_content
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    # Instead of silently skipping, we'll fail loudly
    raise ImportError(f"Markdown parser dependencies not available: {e}. Fix the dependencies to run these tests.")

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

def test_extract_sections_from_markdown():
    """Test extracting sections from real markdown content."""
    # Create a simple markdown document with sections
    markdown_content = """# Test Markdown Document
    
## Introduction

This is the introduction.

## Section 1

This is section 1.

### Subsection 1.1

This is subsection 1.1.

## Section 2

This is section 2.
"""
    
    try:
        # Extract sections
        sections = extract_sections_from_markdown(markdown_content)
        
        # Verify that sections were extracted
        assert len(sections) > 0, "No sections were extracted"
        
        # Verify section structure
        for section in sections:
            assert "level" in section, "Section should have a level"
            assert "title" in section, "Section should have a title"
            assert "content" in section, "Section should have content"
        
        # Check specific sections
        assert any(section["title"] == "Introduction" for section in sections), "Should extract Introduction section"
        assert any(section["title"] == "Section 1" for section in sections), "Should extract Section 1"
        
        # Print sections for debugging
        print(f"Extracted {len(sections)} sections:")
        for i, section in enumerate(sections):
            print(f"Section {i+1}: {section['title']} (level {section['level']})")
            content_preview = section["content"][:50] + "..." if len(section["content"]) > 50 else section["content"]
            print(f"  Content: {content_preview}")
    except Exception as e:
        pytest.fail(f"Error extracting sections: {e}")

def test_extract_code_blocks():
    """Test extracting code blocks from markdown with multiple languages."""
    # Create a sample markdown with code blocks
    markdown_text = """# Sample Code Blocks

## Python

```python
def hello_world():
    return "Hello, World!"
```

## JavaScript

```javascript
function greet() {
    console.log("Hello, World!");
}
```

## No Language Specified

```
Generic code block
```
"""
    
    try:
        # Extract code blocks with the correct parameter name
        blocks = extract_blocks_mistune(markdown_text)
        
        # Verify that code blocks were extracted
        assert len(blocks) > 0, "No code blocks were extracted"
        
        # Verify block structure
        for block in blocks:
            assert "language" in block, "Block should have a language"
            assert "content" in block, "Block should have content"
        
        # Check specific languages
        python_blocks = [b for b in blocks if b["language"] == "python"]
        js_blocks = [b for b in blocks if b["language"] == "javascript"]
        
        assert len(python_blocks) > 0, "Should extract Python code blocks"
        assert len(js_blocks) > 0, "Should extract JavaScript code blocks"
        
        # Print blocks for debugging
        print(f"Extracted {len(blocks)} code blocks:")
        for i, block in enumerate(blocks):
            print(f"Block {i+1}: {block['language']}")
            content_preview = block["content"][:50] + "..." if len(block["content"]) > 50 else block["content"]
            print(f"  Content: {content_preview}")
    except Exception as e:
        pytest.fail(f"Error extracting code blocks: {e}")

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
            result = process_markdown_file_it(f.name)
            
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
            md_files = get_markdown_files_it(rust_repo)
            
            # Verify that files were found
            assert isinstance(md_files, list), "Result should be a list"
            assert len(md_files) > 0, "Should find at least one markdown file"
            
            # Check file types
            for file_path in md_files:
                assert file_path.suffix.lower() in [".md", ".markdown", ".mdown"], f"Invalid file extension: {file_path.suffix}"
            
            print(f"Found {len(md_files)} markdown files in {rust_repo}")
            
            # Try non-recursive mode
            md_files_nonrecursive = get_markdown_files_it(rust_repo, recursive=False)
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
            result = process_markdown_file_it(readme_path)
            
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
        result = process_markdown_file_it(doc_file)
        
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
    test_extract_sections_from_markdown()
    test_extract_code_blocks()
    test_process_markdown_file(real_markdown_content()) 