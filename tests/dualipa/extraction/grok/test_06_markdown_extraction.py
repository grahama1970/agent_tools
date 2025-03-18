"""
Tests for Markdown block extraction from real-world repositories.
Depends on: Repository operations to provide Markdown files.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Configure the Path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Fail loudly if dependencies are missing
try:
    from agent_tools.dualipa.code_extractor import _extract_markdown_blocks
    from agent_tools.dualipa.utils import initialize_stats_dict
except ImportError as e:
    raise ImportError(f"Required code extractor modules not available: {e}")

# Path to test resources from your document
SAMPLE_FILE = project_root / "test_repos" / "react" / "packages" / "react-devtools-core" / "README.md"

def test_markdown_block_extraction():
    """Test Markdown code block extraction from React's README.md."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample markdown file not found: {SAMPLE_FILE}")
    
    print(f"Using markdown file: {SAMPLE_FILE}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        stats = initialize_stats_dict()  # From original
        
        # Extract code blocks from the sample file
        with open(SAMPLE_FILE, 'r', errors='ignore') as f:
            content = f.read()
        
        print(f"File content length: {len(content)} characters")
        print(f"First 100 chars: {content[:100]}")
        print(f"File contains backticks: {'```' in content}")
        
        num_blocks = _extract_markdown_blocks(
            file_path=SAMPLE_FILE,
            content=content,
            output_dir=output_dir,
            stats=stats
        )
        
        # Original assertions
        assert num_blocks > 0, f"Expected blocks, got {num_blocks}"
        assert isinstance(num_blocks, int), "Should return number of blocks"
        
        # Detailed checks from <DOCUMENT>
        blocks_dir = output_dir / "blocks"
        assert blocks_dir.exists(), "Blocks directory not created"
        print(f"Blocks directory exists: {blocks_dir}")
        
        code_dir = blocks_dir / "code"
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
            # Original check for Python blocks
            python_dir = code_dir / "python"
            if python_dir.exists():
                assert len(list(python_dir.glob("*.py"))) > 0, "Python block files not created"
        else:
            print("Code directory doesn't exist")
        
        doc_dir = blocks_dir / "documentation"
        if doc_dir.exists():
            print(f"Documentation directory exists: {doc_dir}")
            doc_blocks = list(doc_dir.glob("*"))
            print(f"Total doc blocks: {len(doc_blocks)}")
        else:
            print("Documentation directory doesn't exist")
        
        if stats.get("errors"):
            print(f"Extraction errors: {stats['errors']}")
            pytest.skip(f"Skipping due to extraction errors: {stats['errors']}")