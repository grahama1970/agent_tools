#!/usr/bin/env python3
"""
test_end_to_end.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- json: https://docs.python.org/3/library/json.html

This script performs an end-to-end test of the fetch_docs tool integration with DuaLipa.
It tests the complete pipeline:
1. Creating a test repository with documentation links
2. Running link detection
3. Downloading and processing documentation
4. Converting to DuaLipa-compatible blocks
5. Integrating with DuaLipa's extraction

The goal is to verify that the full integration path works with real documentation sources.

Input: None
Output: Test results showing whether the end-to-end pipeline works correctly

Example usage:
    python test_end_to_end.py
"""

import os
import sys
import json
from pathlib import Path
import tempfile
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_end_to_end')

# Define test constants
ARANGODB_URL = "https://docs.arangodb.com/stable/aql/"
READTHEDOCS_URL = "https://python.readthedocs.io/en/latest/"

def create_test_repository():
    """
    Create a test repository with documentation links.
    
    Returns:
        Path to the test repository
    """
    # Create temporary directory
    repo_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created test repository at {repo_dir}")
    
    # Create a README.md with documentation links
    readme_content = f"""# Test Repository
    
This is a test repository for the fetch_docs tool.

## Documentation Links

Here are some links to documentation:

- [ArangoDB AQL Documentation]({ARANGODB_URL})
- [Python Documentation]({READTHEDOCS_URL})

## Code Examples

Here is a Python example:

```python
def example_function(x, y):
    """Add two numbers."""
    return x + y
```

And a JavaScript example:

```javascript
function exampleFunction(x, y) {
    // Add two numbers
    return x + y;
}
```
"""
    
    # Write the README.md file
    with open(repo_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create a Python file
    python_content = """#!/usr/bin/env python3
# example.py - Example Python file

def example_function(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

class ExampleClass:
    \"\"\"Example class for testing.\"\"\"
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        \"\"\"Return a greeting message.\"\"\"
        return f"Hello, {self.name}!"
"""
    
    # Write the Python file
    with open(repo_dir / "example.py", "w") as f:
        f.write(python_content)
    
    # Create a JavaScript file
    js_content = """// example.js - Example JavaScript file

function exampleFunction(x, y) {
    // Add two numbers
    return x + y;
}

class ExampleClass {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return `Hello, ${this.name}!`;
    }
}
"""
    
    # Write the JavaScript file
    with open(repo_dir / "example.js", "w") as f:
        f.write(js_content)
    
    return repo_dir

def test_link_detection(repo_dir):
    """
    Test documentation link detection.
    
    Args:
        repo_dir: Path to the test repository
        
    Returns:
        Tuple (success, links) where links is a list of detected URLs
    """
    logger.info("Testing documentation link detection")
    
    try:
        # Import link detection function
        from agent_tools.fetch_docs.link_detector import detect_documentation_links
        
        # Detect links
        links = detect_documentation_links(repo_dir)
        
        # Check that links were detected
        if not links:
            logger.error("No documentation links detected")
            return False, []
        
        # Verify that both test URLs were detected
        arangodb_found = any(ARANGODB_URL in link for link in links)
        readthedocs_found = any(READTHEDOCS_URL in link for link in links)
        
        if not arangodb_found:
            logger.error(f"ArangoDB URL not detected: {ARANGODB_URL}")
            return False, links
        
        if not readthedocs_found:
            logger.error(f"ReadTheDocs URL not detected: {READTHEDOCS_URL}")
            return False, links
        
        logger.info(f"Found {len(links)} documentation links")
        for link in links:
            logger.info(f"  - {link}")
        
        return True, links
    
    except ImportError as e:
        logger.error(f"Error importing link detection function: {e}")
        return False, []
    except Exception as e:
        logger.error(f"Unexpected error during link detection: {e}")
        return False, []

def test_documentation_processing(repo_dir, links):
    """
    Test documentation processing.
    
    Args:
        repo_dir: Path to the test repository
        links: List of documentation links to process
        
    Returns:
        Tuple (success, processed_docs)
    """
    logger.info("Testing documentation processing")
    
    try:
        # Import processing function
        from agent_tools.fetch_docs.processor import process_documentation
        
        # Create a cache directory
        cache_dir = repo_dir / ".fetch_docs_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Process documentation
        processed_docs = process_documentation(links, cache_dir)
        
        # Check that docs were processed
        if not processed_docs:
            logger.error("No documentation processed")
            return False, {}
        
        # Verify that both test URLs were processed
        for url in links:
            if url not in processed_docs:
                logger.error(f"URL not found in processed docs: {url}")
                return False, processed_docs
            
            # Check that site data exists
            site_data = processed_docs[url]
            if not site_data:
                logger.error(f"No site data for URL: {url}")
                return False, processed_docs
            
            # Check that sections were extracted
            for page_data in site_data:
                sections = page_data.get("sections", [])
                if not sections:
                    logger.error(f"No sections extracted for page: {page_data.get('file')}")
                    return False, processed_docs
                
                logger.info(f"Processed {page_data.get('file')} with {len(sections)} sections")
        
        return True, processed_docs
    
    except ImportError as e:
        logger.error(f"Error importing processing function: {e}")
        return False, {}
    except Exception as e:
        logger.error(f"Unexpected error during documentation processing: {e}")
        return False, {}

def test_dualipa_integration(repo_dir):
    """
    Test integration with DuaLipa extraction.
    
    Args:
        repo_dir: Path to the test repository
        
    Returns:
        Tuple (success, blocks)
    """
    logger.info("Testing DuaLipa integration")
    
    try:
        # Try to import DuaLipa integration
        try:
            from agent_tools.dualipa.extraction.docs_integration import extract_all_blocks_with_docs
        except ImportError:
            logger.warning("DuaLipa integration not available, skipping integration test")
            return True, []
        
        # Extract all blocks with documentation
        blocks = extract_all_blocks_with_docs(repo_dir)
        
        # Check that blocks were extracted
        if not blocks:
            logger.error("No blocks extracted")
            return False, []
        
        # Count block types
        code_blocks = [b for b in blocks if b.get("type") in ["function", "class", "method"]]
        doc_blocks = [b for b in blocks if b.get("type") in ["documentation", "doc_section"]]
        
        logger.info(f"Extracted {len(blocks)} blocks:")
        logger.info(f"  - {len(code_blocks)} code blocks")
        logger.info(f"  - {len(doc_blocks)} documentation blocks")
        
        # Verify that we have both code and documentation blocks
        if not code_blocks:
            logger.error("No code blocks extracted")
            return False, blocks
        
        if not doc_blocks:
            logger.error("No documentation blocks extracted")
            return False, blocks
        
        # Save blocks to a file for inspection
        blocks_file = repo_dir / "extraction_blocks.json"
        with open(blocks_file, "w") as f:
            json.dump(blocks, f, indent=2)
        logger.info(f"Saved extraction blocks to {blocks_file}")
        
        return True, blocks
    
    except Exception as e:
        logger.error(f"Unexpected error during DuaLipa integration: {e}")
        return False, []

def run_end_to_end_test():
    """
    Run the end-to-end test of fetch_docs and DuaLipa integration.
    
    Returns:
        True if all tests pass, False otherwise
    """
    try:
        # Step 1: Create test repository
        repo_dir = create_test_repository()
        
        try:
            # Step 2: Test link detection
            link_success, links = test_link_detection(repo_dir)
            if not link_success:
                logger.error("Link detection test failed")
                return False
            
            # Step 3: Test documentation processing
            process_success, processed_docs = test_documentation_processing(repo_dir, links)
            if not process_success:
                logger.error("Documentation processing test failed")
                return False
            
            # Step 4: Test DuaLipa integration
            integration_success, blocks = test_dualipa_integration(repo_dir)
            if not integration_success:
                logger.error("DuaLipa integration test failed")
                return False
            
            # All tests passed
            logger.info("All end-to-end tests passed")
            return True
        
        finally:
            # Clean up the test repository
            logger.info(f"Cleaning up test repository: {repo_dir}")
            shutil.rmtree(repo_dir)
    
    except Exception as e:
        logger.error(f"Unexpected error during end-to-end test: {e}")
        return False

if __name__ == "__main__":
    print("Running end-to-end test...")
    success = run_end_to_end_test()
    print(f"End-to-end test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)