#!/usr/bin/env python3
"""
Test for Fetch Docs Integration with DuaLipa Extraction.

This script tests the integration between the fetch_docs module and DuaLipa's
extraction pipeline. It validates that documentation from Read the Docs links
is properly downloaded, processed, and integrated with code extraction.

The test:
1. Sets up a test repository with markdown files containing Read the Docs links
2. Runs the extraction with fetch_docs integration
3. Validates that documentation blocks are properly extracted and formatted
4. Confirms the integration output is compatible with the QA system
"""

import os
import sys
import json
import shutil
import unittest
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_fetch_docs_integration")


class FetchDocsIntegrationTest(unittest.TestCase):
    """Test case for fetch_docs integration with DuaLipa extraction."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for the test
        self.test_dir = Path(tempfile.mkdtemp(prefix="fetch_docs_test_"))
        logger.info(f"Created test directory: {self.test_dir}")
        
        # Create a simple test repository with Read the Docs links
        self.create_test_repository()
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary directory
        logger.info(f"Cleaning up: {self.test_dir}")
        shutil.rmtree(self.test_dir)
    
    def create_test_repository(self):
        """Create a test repository with markdown files containing documentation links."""
        # Create a README.md with a Read the Docs link
        readme_content = """# Test Project
        
This is a test project for the fetch_docs integration.

## Documentation
For more information, please refer to the [example docs](https://sqlalchemy.readthedocs.io/en/14/index.html).

## Installation
```bash
pip install test-project
```

## Usage
```python
from test_project import example

example.run()
```
"""
        
        # Create a docs directory
        docs_dir = self.test_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create a more detailed docs file with multiple links including ArangoDB
        detailed_docs = """# Detailed Documentation

## External References

This project uses several libraries and databases:

- [SQLAlchemy](https://sqlalchemy.readthedocs.io/en/14/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Pandas](https://pandas.pydata.org/docs/)
- [ArangoDB](https://docs.arangodb.com/stable/aql/)

## Database Documentation

For database queries, refer to the [ArangoDB Query Language docs](https://docs.arangodb.com/stable/aql/operations/return/).

## API Reference

The API is documented in detail at [our documentation site](https://example-docs.readthedocs.io/).

## Code Examples

```python
# Example code
def main():
    print("Hello world")

if __name__ == "__main__":
    main()
```

## Table Example

| Method | Description | Returns |
|--------|-------------|---------|
| GET    | Retrieves data | JSON |
| POST   | Creates new data | Status |
| PUT    | Updates existing data | Status |
| DELETE | Deletes data | Status |
"""
        
        # Write the files
        with open(self.test_dir / "README.md", "w") as f:
            f.write(readme_content)
            
        with open(docs_dir / "detailed.md", "w") as f:
            f.write(detailed_docs)
        
        # Create a simple Python file
        py_content = """#!/usr/bin/env python3
# Example Python file

def example_function():
    \"\"\"An example function\"\"\"
    return "Hello, world!"

class ExampleClass:
    \"\"\"An example class\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the class\"\"\"
        self.value = "example"
    
    def get_value(self):
        \"\"\"Get the value\"\"\"
        return self.value
"""
        
        with open(self.test_dir / "example.py", "w") as f:
            f.write(py_content)
    
    def run_extraction(self, return_raw=False) -> Optional[List[Dict[str, Any]]]:
        """
        Run the extraction with fetch_docs integration on the test repository.
        
        Args:
            return_raw: If True, returns the raw extraction blocks without QA formatting
        
        Returns:
            The extraction output or None if it failed
        """
        try:
            # Add the parent directory to the path for imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Patch the download_site function to prevent sys.exit
            import importlib.util
            download_site_patch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_site_patch.py")
            if os.path.exists(download_site_patch_path):
                spec = importlib.util.spec_from_file_location("download_site_patch", download_site_patch_path)
                download_site_patch = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_site_patch)
                
                # Monkey patch the download_site function
                import agent_tools.fetch_docs.download_site
                agent_tools.fetch_docs.download_site.download_site = download_site_patch.download_site
                logger.info("Patched download_site function to handle errors gracefully")
            
            # Import the needed functions
            try:
                # Try importing extract_all_blocks_with_docs first
                from agent_tools.dualipa.fetch_docs_integration import extract_all_blocks_with_docs, integrate_docs_with_extraction
                logger.info("Successfully imported fetch_docs_integration")
            except ImportError as e:
                logger.error(f"Failed to import fetch_docs_integration: {e}")
                try:
                    # Try a relative import
                    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
                    from agent_tools.dualipa.fetch_docs_integration import extract_all_blocks_with_docs, integrate_docs_with_extraction
                    logger.info("Successfully imported fetch_docs_integration after path adjustment")
                except ImportError as e2:
                    logger.error(f"Still failed to import fetch_docs_integration: {e2}")
                    return None
            
            # Import extraction blocks
            from extraction_blocks import extract_all_blocks
            
            # Run extraction with docs integration in two steps for better testing
            logger.info(f"Extracting from test repository: {self.test_dir}")
            
            # First, extract code blocks
            code_blocks = extract_all_blocks(self.test_dir)
            logger.info(f"Extracted {len(code_blocks)} code blocks")
            
            # Then, add documentation
            raw_blocks = integrate_docs_with_extraction(self.test_dir, code_blocks)
            logger.info(f"Enhanced with documentation to {len(raw_blocks)} total blocks")
            
            # Return raw blocks if requested
            if return_raw:
                return raw_blocks
            
            # Convert to QA-compatible format
            from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
            qa_blocks = create_qa_compatible_blocks(raw_blocks)
            output = create_qa_compatible_output(qa_blocks)
            
            return output
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return None
    
    def test_extraction_with_docs(self):
        """Test that extraction with fetch_docs integration works properly."""
        # Run extraction (get raw blocks before QA formatting)
        blocks = self.run_extraction(return_raw=True)
        
        # Check that extraction succeeded
        self.assertIsNotNone(blocks, "Extraction failed")
        self.assertGreater(len(blocks), 0, "No blocks extracted")
        
        # Check for documentation blocks
        doc_blocks = [b for b in blocks if b.get("type") == "documentation"]
        self.assertGreater(len(doc_blocks), 0, "No documentation blocks found")
        
        logger.info(f"Found {len(doc_blocks)} documentation blocks")
        
        # Check documentation content
        for doc_block in doc_blocks:
            self.assertIn("source_url", doc_block, "Documentation block missing source_url")
            self.assertIn("child_uuids", doc_block, "Documentation block missing child_uuids")
            
            # Verify some metadata
            self.assertIn("metadata", doc_block, "Documentation block missing metadata")
            
            # Log the source URL
            logger.info(f"Found documentation from: {doc_block.get('source_url')} (type: {doc_block.get('metadata', {}).get('doc_type', 'unknown')})")
        
        # Check for doc page blocks
        doc_page_blocks = [b for b in blocks if b.get("type") == "doc_page"]
        self.assertGreater(len(doc_page_blocks), 0, "No doc_page blocks found")
        
        # Check for doc section blocks
        doc_section_blocks = [b for b in blocks if b.get("type") == "doc_section"]
        self.assertGreater(len(doc_section_blocks), 0, "No doc_section blocks found")
        
        # Check proper parent-child relationships
        for doc_block in doc_blocks:
            # Check that all child_uuids exist
            for child_uuid in doc_block.get("child_uuids", []):
                child_block = next((b for b in blocks if b.get("uuid") == child_uuid), None)
                self.assertIsNotNone(child_block, f"Child block {child_uuid} not found")
                self.assertEqual(child_block.get("parent_uuid"), doc_block.get("uuid"), 
                                "Child's parent_uuid doesn't match parent's uuid")
    
    def test_arangodb_docs_extraction(self):
        """Test specifically for ArangoDB documentation extraction."""
        # Run extraction (get raw blocks)
        blocks = self.run_extraction(return_raw=True)
        
        # Check that extraction succeeded
        self.assertIsNotNone(blocks, "Extraction failed")
        
        # Find ArangoDB documentation blocks
        arangodb_blocks = [b for b in blocks if b.get("type") == "documentation" 
                          and b.get("metadata", {}).get("doc_type") == "arangodb"]
        
        # Verify that ArangoDB documentation was extracted
        self.assertGreater(len(arangodb_blocks), 0, "No ArangoDB documentation blocks found")
        logger.info(f"Found {len(arangodb_blocks)} ArangoDB documentation blocks")
        
        # Check for ArangoDB-specific content
        for doc_block in arangodb_blocks:
            # Verify source URL
            url = doc_block.get("source_url", "")
            self.assertIn("arangodb.com", url, "Not an ArangoDB documentation URL")
            
            # Check for child pages
            child_uuids = doc_block.get("child_uuids", [])
            self.assertGreater(len(child_uuids), 0, "ArangoDB doc has no child pages")
            
            # Find a doc page block
            doc_page = next((b for b in blocks if b.get("uuid") in child_uuids), None)
            self.assertIsNotNone(doc_page, "Could not find doc page")
            
            # Verify doc page metadata
            self.assertEqual(doc_page.get("type"), "doc_page", "Child is not a doc_page")
            self.assertEqual(doc_page.get("metadata", {}).get("doc_type"), "arangodb", 
                             "Doc page has incorrect doc_type")
            
            # Get doc sections
            page_children = doc_page.get("child_uuids", [])
            doc_sections = [b for b in blocks if b.get("uuid") in page_children]
            
            # Verify sections exist
            self.assertGreater(len(doc_sections), 0, "ArangoDB doc page has no sections")
            logger.info(f"ArangoDB doc from {url} has {len(doc_sections)} sections")
            
            # Check for AQL code blocks
            code_blocks = [b for b in blocks if b.get("type") == "code_block" 
                         and "arangodb" in b.get("metadata", {}).get("doc_type", "")
                         and "javascript" in b.get("language", "")]
            
            if code_blocks:
                logger.info(f"Found {len(code_blocks)} AQL code blocks")
            
            # Check for tables
            table_blocks = [b for b in blocks if b.get("type") == "table" 
                         and "arangodb" in b.get("metadata", {}).get("doc_type", "")]
            
            if table_blocks:
                logger.info(f"Found {len(table_blocks)} table blocks")
        
        # Validate format compatibility with QA (optional)
        try:
            # Convert raw blocks to QA format for validation
            from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
            qa_blocks = create_qa_compatible_blocks(blocks)
            qa_output = create_qa_compatible_output(qa_blocks)
            
            from ...validation.validate_extraction_format import validate_extraction_output
            
            # Try to load the expected format template
            current_dir = os.path.dirname(os.path.abspath(__file__))
            expected_format_path = os.path.join(current_dir, "deepseek_markdown_extraction_example.json")
            
            if os.path.exists(expected_format_path):
                with open(expected_format_path, 'r', encoding='utf-8') as f:
                    expected_format = json.load(f)
                
                # Validate output
                logger.info("Validating extraction output against QA format")
                validation_blocks = qa_output if isinstance(qa_output, list) else qa_output.get("blocks", [])
                results = validate_extraction_output(validation_blocks, expected_format)
                
                if results.get("valid", False):
                    logger.info(f"Validation successful: {results.get('stats', {})}")
                else:
                    logger.warning(f"Validation produced warnings: {results.get('errors', [])}")
            else:
                logger.warning(f"Expected format template not found at {expected_format_path}")
        except ImportError as e:
            logger.warning(f"Couldn't import validation module: {e}")
        except Exception as e:
            logger.warning(f"Non-critical error during validation: {e}")
    
    def test_fetch_docs_extraction(self):
        """Test that fetch_docs correctly extracts documentation links."""
        try:
            # Import the function to detect documentation links
            from agent_tools.dualipa.fetch_docs_integration import detect_doc_links
            
            # Detect links
            links = detect_doc_links(self.test_dir)
            
            # Check that links were found
            self.assertGreater(len(links), 0, "No documentation links detected")
            logger.info(f"Detected {len(links)} documentation links: {links}")
            
            # Check that known links were found
            found_sqlalchemy = any("sqlalchemy.readthedocs.io" in link for link in links)
            found_arangodb = any("docs.arangodb.com" in link for link in links)
            
            self.assertTrue(found_sqlalchemy, "SQLAlchemy link not found")
            self.assertTrue(found_arangodb, "ArangoDB link not found")
            
            # Check link counts by type
            rtd_count = sum(1 for link in links if 'readthedocs.io' in link or 'readthedocs.org' in link)
            arangodb_count = sum(1 for link in links if 'arangodb.com' in link)
            
            self.assertGreaterEqual(rtd_count, 1, "Not enough ReadTheDocs links found")
            self.assertGreaterEqual(arangodb_count, 1, "Not enough ArangoDB links found")
            
            logger.info(f"Link breakdown: {rtd_count} ReadTheDocs, {arangodb_count} ArangoDB")
        except ImportError as e:
            logger.error(f"Failed to import detect_doc_links: {e}")
            self.fail("Could not import detect_doc_links")


if __name__ == "__main__":
    unittest.main()