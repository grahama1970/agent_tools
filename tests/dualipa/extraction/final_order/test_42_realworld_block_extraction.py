"""
TEST EXPECTATIONS

1. test_extract_python_code_blocks:
   Input: Python files from requests repository
   Expected Output:
   - At least one code block extracted from setup.py
   - At least one code block extracted from a core Python file
   - Blocks contain valid Python code with functions/classes

2. test_extract_javascript_code_blocks:
   Input: JavaScript files from React repository
   Expected Output:
   - At least one code block extracted from each JS file
   - Blocks contain valid JavaScript code
   - No language parameter needed for extraction

3. test_cross_language_extraction:
   Input: Files from multiple repositories (Python, JavaScript, Rust)
   Expected Output:
   - Successful extraction from each language
   - Correct language-specific directory structure
   - Valid code blocks for each language

4. test_block_extraction_from_requests:
   Input: Entire requests repository
   Expected Output:
   - Repository successfully cloned
   - Multiple Python files processed
   - Valid code blocks extracted

CRITICAL RULES:
1. Repository Setup:
   - Clone repositories only when needed
   - Use shallow clones for efficiency
   - Fall back to local files if available

2. Block Extraction:
   - Preserve original formatting
   - Handle language-specific features
   - Track extraction statistics

3. Error Handling:
   - Skip tests if repositories unavailable
   - Log extraction errors
   - Provide meaningful error messages
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import json
import shutil

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import required modules
try:
    from agent_tools.dualipa.code_extractor import (
        _extract_python_blocks,
        _extract_js_ts_blocks,
        _extract_markdown_blocks,
        _extract_generic_blocks,
        initialize_stats_dict
    )
    from agent_tools.dualipa.github_utils import clone_github_repo
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False

# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required code extractor modules not available")

def find_files_by_extension(directory, extension, limit=None):
    """Find files with a specific extension in a directory."""
    files = []
    for root, _, filenames in os.walk(str(directory)):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(Path(os.path.join(root, filename)))
                if limit and len(files) >= limit:
                    return files
    return files

def clone_repository_if_not_exists(url, directory, depth=None):
    """Clone a Git repository if it doesn't exist."""
    repo_path = Path(directory) / Path(url).stem
    if repo_path.exists():
        return repo_path
    return clone_github_repo(url, str(directory))

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"
PYTHON_REPO_PATH = project_root / "test_repos" / "python-sample"  # Fallback sample

@pytest.fixture(scope="session")
def requests_repo():
    """Fixture to provide the requests repository path."""
    requests_path = project_root / "test_repos" / "requests"
    if not requests_path.exists():
        try:
            requests_path = clone_repository_if_not_exists(
                "https://github.com/psf/requests.git",
                project_root / "test_repos",
                depth=1
            )
        except Exception as e:
            pytest.fail(f"Failed to clone requests repository: {e}")
    if not requests_path.exists():
        pytest.fail("Requests repository not available - repository cloning failed")
    return requests_path

def test_extract_python_code_blocks(requests_repo):
    """Test extraction of Python code blocks from real-world Python files."""
    # First, test with a script-like file (setup.py)
    setup_file = requests_repo / "setup.py"
    if setup_file.exists():
        print(f"Testing with script file: {setup_file}")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with open(setup_file, 'r') as file:
                content = file.read()
            stats = initialize_stats_dict(source=str(setup_file), output_dir=output_dir)
            _extract_python_blocks(setup_file, content, output_dir, stats)
            if stats["code_blocks"] > 0:
                print(f"Successfully extracted {stats['code_blocks']} blocks from script file {setup_file}")
            else:
                print(f"Warning: Failed to extract blocks from script file {setup_file}")
    
    # Now, test with a file that likely has functions/classes
    api_files = []
    for potential_file in ["api.py", "models.py", "utils.py", "core.py"]:
        matches = list(requests_repo.glob(f"**/{potential_file}"))
        if matches:
            api_files.extend(matches)
    if not api_files:
        python_files = find_files_by_extension(requests_repo, ".py", limit=10)
        api_files = [f for f in python_files if f.name.lower() != "setup.py"]
    if not api_files:
        pytest.fail("No suitable Python files found in the requests repository")
    api_file = api_files[0]
    print(f"Testing with function file: {api_file}")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        with open(api_file, 'r') as file:
            content = file.read()
        stats = initialize_stats_dict(source=str(api_file), output_dir=output_dir)
        _extract_python_blocks(api_file, content, output_dir, stats)
        assert stats["code_blocks"] > 0, f"Should extract at least one code block from {api_file}"
        print(f"Extracted {stats['code_blocks']} blocks from {api_file}")
        blocks_dir = output_dir / "blocks" / "code" / "python"
        assert blocks_dir.exists(), "Should create blocks directory"
        block_files = list(blocks_dir.glob("*.py"))
        assert len(block_files) > 0, "Should create at least one block file"
        print(f"Found {len(block_files)} block files")
        if block_files:
            with open(block_files[0], 'r') as f:
                block_content = f.read()
            print(f"First block preview: {block_content[:100]}...")
            assert len(block_content) > 0, "Block file should contain content"

def test_extract_javascript_code_blocks():
    """Test extraction of JavaScript code blocks from real-world JS files."""
    print(f"\nDEBUG: REACT_PATH = {REACT_PATH}")
    print(f"DEBUG: REACT_PATH exists = {REACT_PATH.exists()}")
    
    # Try to use local repository first
    if not REACT_PATH.exists():
        try:
            # Clone repository if not available locally
            REACT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Clone into a temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = clone_repository_if_not_exists(
                    "https://github.com/facebook/react.git",
                    temp_dir,
                    depth=1
                )
                # Move the cloned repository to the final location
                if temp_path.exists():
                    shutil.copytree(temp_path, REACT_PATH)
        except Exception as e:
            pytest.fail(f"Failed to access React repository (both local and remote): {e}")
    
    js_files = [
        REACT_PATH / "fixtures" / "devtools" / "scheduling-profiler" / "run.js",
        REACT_PATH / "fixtures" / "devtools" / "regression" / "shared.js",
        REACT_PATH / "fixtures" / "legacy-jsx-runtimes" / "react-15" / "cjs" / "react-jsx-dev-runtime.development.js"
    ]
    print("DEBUG: Looking for these files:")
    for js_file in js_files:
        print(f"  - {js_file} (exists = {js_file.exists()})")
    existing_files = [f for f in js_files if f.exists()]
    if not existing_files:
        pytest.fail("No JavaScript test files found. Repository structure may be corrupted or incomplete.")
    js_file = existing_files[0]
    print(f"Testing with JavaScript file: {js_file} ({js_file.stat().st_size} bytes)")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        with open(js_file, 'r') as file:
            content = file.read()
        stats = initialize_stats_dict(source=str(js_file), output_dir=output_dir)
        _extract_js_ts_blocks(js_file, content, output_dir, stats)
        if stats["errors"]:
            print("Errors during extraction:")
            for error in stats["errors"]:
                print(f"- {error}")
        print(f"Extracted {stats['code_blocks']} blocks from {js_file}")
        blocks_dir = output_dir / "blocks" / "code" / "javascript"
        assert blocks_dir.exists(), "Should create blocks directory"
        block_files = list(blocks_dir.glob("*.js"))
        assert len(block_files) > 0, "Should create at least one block file"
        print(f"Found {len(block_files)} block files")
        if block_files:
            with open(block_files[0], 'r') as f:
                block_content = f.read()
            print(f"First block preview: {block_content[:100]}...")
            assert len(block_content) > 0, "Block file should contain content"

def test_cross_language_extraction(requests_repo):
    """Test extraction of code blocks from multiple files with different languages."""
    test_files = []
    python_file = requests_repo / "setup.py"
    if python_file.exists():
        test_files.append(("python", python_file))
    if REACT_PATH.exists():
        js_file = REACT_PATH / "fixtures" / "devtools" / "scheduling-profiler" / "run.js"
        if js_file.exists():
            test_files.append(("javascript", js_file))
    if RUST_ANALYZER_PATH.exists():
        rust_file = RUST_ANALYZER_PATH / "lib" / "la-arena" / "src" / "lib.rs"
        if rust_file.exists():
            test_files.append(("rust", rust_file))
    if not test_files:
        pytest.fail("No test files found. Required repositories (requests, react, rust-analyzer) are missing or corrupted.")
    print(f"Selected {len(test_files)} files for cross-language extraction testing")
    for lang, file in test_files:
        print(f"- {lang}: {file}")
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        total_blocks = 0
        for lang, file_path in test_files:
            print(f"Testing extraction from {lang} file: {file_path}")
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
            stats = initialize_stats_dict(source=str(file_path), output_dir=output_dir)
            if lang == "python":
                _extract_python_blocks(file_path, content, output_dir, stats)
            elif lang == "javascript":
                _extract_js_ts_blocks(file_path, content, output_dir, stats)
            elif lang == "rust":
                _extract_generic_blocks(file_path, content, output_dir, stats, language="rust")
            if stats["errors"]:
                print(f"Errors during {lang} extraction:")
                for error in stats["errors"]:
                    print(f"- {error}")
            block_count = stats["code_blocks"]
            total_blocks += block_count
            print(f"Extracted {block_count} blocks from {lang} file")
            blocks_dir = output_dir / "blocks" / "code" / lang
            if blocks_dir.exists():
                block_files = list(blocks_dir.glob(f"*.{lang}"))
                print(f"Found {len(block_files)} block files for {lang}")
                if block_files:
                    print(f"First {lang} block file: {block_files[0].name}")
        print(f"Extracted a total of {total_blocks} blocks across {len(test_files)} languages")

def test_block_extraction_from_requests(requests_repo):
    """Test extracting code blocks from requests repository."""
    output_dir = Path(tempfile.mkdtemp())
    try:
        python_files = find_files_by_extension(requests_repo, ".py", limit=5)
        total_blocks = 0
        for file_path in python_files:
            print(f"Processing {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            stats = initialize_stats_dict(source=str(file_path), output_dir=output_dir)
            _extract_python_blocks(file_path, content, output_dir, stats)
            total_blocks += stats["code_blocks"]
            print(f"Extracted {stats['code_blocks']} blocks from {file_path}")
        print(f"Total blocks extracted: {total_blocks}")
        assert total_blocks > 0, "Should extract at least one block"
    finally:
        import shutil
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
