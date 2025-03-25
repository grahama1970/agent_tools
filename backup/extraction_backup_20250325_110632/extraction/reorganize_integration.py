#!/usr/bin/env python3
"""
Reorganize integration points between the extraction module and other modules.

This script implements Phase 4 of the reorganization plan by:
1. Creating a clear integration directory structure
2. Defining and documenting integration points
3. Creating adapter modules for external systems
"""

import os
import shutil
from pathlib import Path
import re

# Define paths
BASE_DIR = Path(__file__).parent
INTEGRATION_DIR = BASE_DIR / "integration"
DOCS_DIR = BASE_DIR / "docs" / "integration"
EXAMPLES_DIR = BASE_DIR / "examples" / "integration"
FETCH_DOCS_DIR = Path(BASE_DIR.parent.parent.parent) / "fetch_docs"

# Define integration points to document
integration_points = {
    "fetch_docs": {
        "description": "Integration with the fetch_docs module for downloading and processing HTML documentation",
        "adapter": "fetch_docs_adapter.py",
        "interfaces": ["DocumentationDownloader", "HTMLProcessor"],
        "config": "fetch_docs_config.py"
    },
    "qa_system": {
        "description": "Integration with the QA system for answer generation based on extracted content",
        "adapter": "qa_adapter.py",
        "interfaces": ["QAIntegration", "QuestionGenerator"],
        "config": "qa_config.py"
    },
    "validation": {
        "description": "Integration with validation systems for verifying extraction quality",
        "adapter": "validation_adapter.py",
        "interfaces": ["ExtractionValidator", "QualityChecker"],
        "config": "validation_config.py"
    }
}

def create_integration_directory():
    """Create the integration directory structure."""
    print("Creating integration directory structure...")
    
    # Ensure the main integration directory exists
    INTEGRATION_DIR.mkdir(exist_ok=True)
    
    # Create __init__.py
    with open(INTEGRATION_DIR / "__init__.py", "w") as f:
        f.write('"""Integration modules for the extraction system."""\n\n')
        f.write('from .fetch_docs_adapter import DocumentationDownloader, HTMLProcessor\n')
        f.write('from .qa_adapter import QAIntegration, QuestionGenerator\n')
        f.write('from .validation_adapter import ExtractionValidator, QualityChecker\n\n')
        f.write('__all__ = [\n')
        f.write('    "DocumentationDownloader", "HTMLProcessor",\n')
        f.write('    "QAIntegration", "QuestionGenerator",\n')
        f.write('    "ExtractionValidator", "QualityChecker",\n')
        f.write(']\n')
    
    # Create README
    with open(INTEGRATION_DIR / "README.md", "w") as f:
        f.write('# Extraction Module Integration\n\n')
        f.write('This directory contains adapter modules for integrating the extraction module with other systems.\n\n')
        f.write('## Available Integrations\n\n')
        
        for name, details in integration_points.items():
            f.write(f'### {name.title()}\n\n')
            f.write(f'{details["description"]}\n\n')
            f.write(f'- Adapter: `{details["adapter"]}`\n')
            f.write(f'- Interfaces: {", ".join([f"`{i}`" for i in details["interfaces"]])}\n')
            f.write(f'- Configuration: `{details["config"]}`\n\n')
    
    print("Integration directory structure created successfully.")

def create_fetch_docs_adapter():
    """Create the fetch_docs adapter module."""
    adapter_path = INTEGRATION_DIR / "fetch_docs_adapter.py"
    
    adapter_content = '''"""
Integration adapter for the fetch_docs module.

This module provides interfaces for downloading and processing HTML documentation
using the fetch_docs module.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import from fetch_docs module
try:
    from agent_tools.fetch_docs.download_site import download_site, download_site_with_playwright
    from agent_tools.fetch_docs.extract_sections import extract_sections_from_html
    FETCH_DOCS_AVAILABLE = True
except ImportError:
    FETCH_DOCS_AVAILABLE = False
    print("Warning: fetch_docs module not available.")

class DocumentationDownloader:
    """
    Interface for downloading documentation using fetch_docs.
    
    This class provides a simplified interface for downloading documentation
    using either wget or Playwright.
    """
    
    def __init__(self, output_dir: str, use_playwright: bool = False):
        """
        Initialize the DocumentationDownloader.
        
        Args:
            output_dir: Directory to save downloaded files
            use_playwright: Whether to use Playwright for JavaScript rendering
        """
        self.output_dir = output_dir
        self.use_playwright = use_playwright
        
        if not FETCH_DOCS_AVAILABLE:
            raise ImportError("fetch_docs module is required for DocumentationDownloader")
    
    def download(self, url: str, recursive: bool = True) -> bool:
        """
        Download documentation from the specified URL.
        
        Args:
            url: URL to download
            recursive: Whether to download recursively
            
        Returns:
            bool: True if successful, False otherwise
        """
        return download_site(url, self.output_dir, recursive, self.use_playwright)
    
    def download_with_playwright(self, url: str, recursive: bool = True, 
                                max_depth: int = 2) -> Dict[str, Any]:
        """
        Download documentation using Playwright for JavaScript rendering.
        
        Args:
            url: URL to download
            recursive: Whether to download recursively
            max_depth: Maximum recursion depth
            
        Returns:
            Dict containing download statistics
        """
        return download_site_with_playwright(
            url, self.output_dir, recursive=recursive, max_depth=max_depth
        )

class HTMLProcessor:
    """
    Interface for processing HTML content from downloaded documentation.
    
    This class provides methods for extracting structured content from HTML files.
    """
    
    def __init__(self, content_dir: str):
        """
        Initialize the HTMLProcessor.
        
        Args:
            content_dir: Directory containing downloaded HTML files
        """
        self.content_dir = content_dir
        
        if not FETCH_DOCS_AVAILABLE:
            raise ImportError("fetch_docs module is required for HTMLProcessor")
    
    def extract_sections(self, html_file: str) -> List[Dict[str, Any]]:
        """
        Extract sections from an HTML file.
        
        Args:
            html_file: Path to the HTML file
            
        Returns:
            List of extracted sections with hierarchical structure
        """
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return extract_sections_from_html(html_content)
    
    def process_directory(self, doc_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all HTML files in the content directory.
        
        Args:
            doc_type: Optional documentation type (e.g., "readthedocs", "arangodb")
            
        Returns:
            Dictionary mapping URLs to lists of processed documents
        """
        processed_docs = {}
        
        # Get all HTML files
        for root, _, files in os.walk(self.content_dir):
            for file in files:
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.content_dir)
                    
                    # Extract URL
                    url_parts = relative_path.split(os.sep)
                    if len(url_parts) >= 1:
                        domain = url_parts[0]
                        url = f"https://{domain}"
                        
                        # Extract sections
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                html_content = f.read()
                            
                            sections = extract_sections_from_html(html_content)
                            
                            if url not in processed_docs:
                                processed_docs[url] = []
                            
                            processed_docs[url].append({
                                "file": file_path,
                                "relative_path": relative_path,
                                "sections": sections,
                                "doc_type": doc_type or self._detect_doc_type(domain)
                            })
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        
        return processed_docs
    
    def _detect_doc_type(self, domain: str) -> str:
        """
        Detect documentation type from domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Detected documentation type
        """
        if "readthedocs.io" in domain or "readthedocs.org" in domain:
            return "readthedocs"
        elif "arangodb.com" in domain:
            return "arangodb"
        else:
            return "unknown"
'''
    
    # Write the adapter file
    with open(adapter_path, "w") as f:
        f.write(adapter_content)
    
    # Create config file
    config_path = INTEGRATION_DIR / "fetch_docs_config.py"
    config_content = '''"""
Configuration for fetch_docs integration.

This module contains configuration options for the fetch_docs integration.
"""

# Default options for documentation download
DEFAULT_DOWNLOAD_OPTIONS = {
    "recursive": True,
    "max_depth": 2,
    "use_playwright": False,
    "timeout": 30000  # 30 seconds
}

# Doc type detection mapping
DOC_TYPE_MAPPING = {
    "readthedocs.io": "readthedocs",
    "readthedocs.org": "readthedocs",
    "arangodb.com": "arangodb",
    "docs.python.org": "python",
    "developer.mozilla.org": "mdn"
}

# HTML processing options
HTML_PROCESSING_OPTIONS = {
    "extract_code_blocks": True,
    "extract_tables": True,
    "extract_images": True,
    "min_section_length": 50,  # Minimum content length for a section to be extracted
    "max_section_length": 10000  # Maximum content length for a section
}
'''
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Created fetch_docs adapter at {adapter_path}")

def create_qa_adapter():
    """Create the QA system adapter module."""
    adapter_path = INTEGRATION_DIR / "qa_adapter.py"
    
    adapter_content = '''"""
Integration adapter for QA systems.

This module provides interfaces for integrating the extraction output
with question-answering systems.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import uuid

class QAIntegration:
    """
    Interface for integrating extraction output with QA systems.
    
    This class provides methods for converting extraction output to
    formats suitable for QA systems.
    """
    
    def __init__(self):
        """Initialize the QA integration interface."""
        pass
    
    def format_for_qa(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format extraction output for QA system consumption.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Formatted data suitable for QA system
        """
        # Create a dictionary mapping UUIDs to blocks for easy lookup
        blocks_by_uuid = {block["uuid"]: block for block in extraction_output}
        
        # Get root blocks (those without parent_uuid or with parent_uuid not in blocks)
        root_blocks = [
            block for block in extraction_output 
            if "parent_uuid" not in block or block["parent_uuid"] not in blocks_by_uuid
        ]
        
        # Format for QA
        qa_data = {
            "documents": [],
            "metadata": {
                "total_blocks": len(extraction_output),
                "root_blocks": len(root_blocks)
            }
        }
        
        # Process each root block and its children
        for root_block in root_blocks:
            qa_data["documents"].append(self._process_block_hierarchy(root_block, blocks_by_uuid))
        
        return qa_data
    
    def _process_block_hierarchy(self, block: Dict[str, Any], 
                                blocks_by_uuid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a block and its children for QA format.
        
        Args:
            block: The current block
            blocks_by_uuid: Dictionary mapping UUIDs to blocks
            
        Returns:
            Processed block with children for QA
        """
        qa_block = {
            "id": block["uuid"],
            "text": block["content"],
            "metadata": {
                "name": block["name"],
                "type": block["type"],
                "language": block["language"],
                "file_path": block["file_path"]
            },
            "children": []
        }
        
        # Include additional metadata if available
        if "metadata" in block:
            qa_block["metadata"].update(block["metadata"])
        
        # Process children
        if "child_uuids" in block:
            for child_uuid in block["child_uuids"]:
                if child_uuid in blocks_by_uuid:
                    child_block = blocks_by_uuid[child_uuid]
                    qa_block["children"].append(
                        self._process_block_hierarchy(child_block, blocks_by_uuid)
                    )
        
        return qa_block
    
    def save_qa_format(self, extraction_output: List[Dict[str, Any]], 
                      output_file: str) -> None:
        """
        Format extraction output and save to a file for QA system.
        
        Args:
            extraction_output: List of extracted blocks
            output_file: Path to save the formatted output
        """
        qa_data = self.format_for_qa(extraction_output)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)

class QuestionGenerator:
    """
    Generate sample questions from extraction output.
    
    This class provides methods for generating sample questions from
    extracted content to test QA system integration.
    """
    
    def __init__(self):
        """Initialize the question generator."""
        pass
    
    def generate_questions(self, extraction_output: List[Dict[str, Any]], 
                          num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Generate sample questions from extraction output.
        
        Args:
            extraction_output: List of extracted blocks
            num_questions: Number of questions to generate
            
        Returns:
            List of questions with expected answers
        """
        questions = []
        
        # Extract content sections for question generation
        content_sections = []
        for block in extraction_output:
            if len(block.get("content", "")) > 100:
                content_sections.append({
                    "content": block["content"],
                    "name": block["name"],
                    "type": block["type"],
                    "uuid": block["uuid"]
                })
        
        # Generate questions (simplified example approach)
        for i in range(min(num_questions, len(content_sections))):
            section = content_sections[i]
            
            # Generate a simple question (in a real implementation, this would use NLP)
            question = f"What is described in the section '{section['name']}'?"
            
            # Truncate content for answer preview
            answer_preview = section["content"][:100] + "..." if len(section["content"]) > 100 else section["content"]
            
            questions.append({
                "id": str(uuid.uuid4()),
                "question": question,
                "answer_source_id": section["uuid"],
                "expected_answer_preview": answer_preview,
                "type": "content_summary"
            })
        
        return questions
'''
    
    # Write the adapter file
    with open(adapter_path, "w") as f:
        f.write(adapter_content)
    
    # Create config file
    config_path = INTEGRATION_DIR / "qa_config.py"
    config_content = '''"""
Configuration for QA system integration.

This module contains configuration options for QA system integration.
"""

# QA format options
QA_FORMAT_OPTIONS = {
    "include_metadata": True,
    "flatten_hierarchy": False,
    "max_content_length": 10000
}

# Question generation options
QUESTION_GENERATION_OPTIONS = {
    "question_types": ["definition", "explanation", "example", "comparison"],
    "max_questions_per_section": 3,
    "min_content_length": 100,
    "excluded_section_types": ["image", "table", "code_block"]
}
'''
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Created QA system adapter at {adapter_path}")

def create_validation_adapter():
    """Create the validation adapter module."""
    adapter_path = INTEGRATION_DIR / "validation_adapter.py"
    
    adapter_content = '''"""
Integration adapter for validation systems.

This module provides interfaces for validating extraction output quality
and compatibility with other systems.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple

class ExtractionValidator:
    """
    Interface for validating extraction output.
    
    This class provides methods for validating extraction output against
    expected formats and quality standards.
    """
    
    def __init__(self, schema_file: Optional[str] = None):
        """
        Initialize the extraction validator.
        
        Args:
            schema_file: Optional path to JSON schema file for validation
        """
        self.schema = None
        if schema_file and os.path.exists(schema_file):
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
    
    def validate_extraction(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extraction output against schema and quality standards.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Validation results
        """
        # Initialize validation results
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_blocks": len(extraction_output),
                "blocks_by_type": {},
                "blocks_by_language": {}
            }
        }
        
        # Collect statistics
        for block in extraction_output:
            # Count by type
            block_type = block.get("type", "unknown")
            if block_type not in results["stats"]["blocks_by_type"]:
                results["stats"]["blocks_by_type"][block_type] = 0
            results["stats"]["blocks_by_type"][block_type] += 1
            
            # Count by language
            language = block.get("language", "unknown")
            if language not in results["stats"]["blocks_by_language"]:
                results["stats"]["blocks_by_language"][language] = 0
            results["stats"]["blocks_by_language"][language] += 1
            
            # Validate required fields
            for field in ["uuid", "type", "name", "content", "language", "file_path", "metadata"]:
                if field not in block:
                    results["valid"] = False
                    results["errors"].append(f"Block {block.get('name', 'Unknown')} is missing required field: {field}")
            
            # Validate relationships
            if "parent_uuid" in block and "parent_uuid" is not None:
                parent_exists = any(p["uuid"] == block["parent_uuid"] for p in extraction_output)
                if not parent_exists:
                    results["warnings"].append(f"Block {block.get('name', 'Unknown')} references non-existent parent: {block['parent_uuid']}")
            
            # Validate child_uuids if present
            if "child_uuids" in block:
                for child_uuid in block["child_uuids"]:
                    child_exists = any(c["uuid"] == child_uuid for c in extraction_output)
                    if not child_exists:
                        results["warnings"].append(f"Block {block.get('name', 'Unknown')} references non-existent child: {child_uuid}")
        
        return results
    
    def save_validation_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save validation results to a file.
        
        Args:
            results: Validation results
            output_file: Path to save results
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

class QualityChecker:
    """
    Check extraction quality metrics.
    
    This class provides methods for checking extraction quality metrics
    such as content coverage, hierarchy correctness, and metadata completeness.
    """
    
    def __init__(self):
        """Initialize the quality checker."""
        pass
    
    def check_quality(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check extraction quality metrics.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Quality metrics and issues
        """
        # Initialize quality metrics
        metrics = {
            "content_quality": {
                "empty_blocks": 0,
                "short_blocks": 0,
                "long_blocks": 0
            },
            "hierarchy_quality": {
                "orphaned_blocks": 0,
                "missing_children": 0,
                "circular_references": 0
            },
            "metadata_quality": {
                "missing_metadata": 0,
                "incomplete_metadata": 0
            },
            "issues": []
        }
        
        # Create a map of blocks by UUID for quick lookup
        blocks_by_uuid = {block["uuid"]: block for block in extraction_output}
        
        # Check content quality
        for block in extraction_output:
            content = block.get("content", "")
            
            # Check for empty or short content
            if not content:
                metrics["content_quality"]["empty_blocks"] += 1
                metrics["issues"].append({
                    "type": "empty_content",
                    "block_uuid": block["uuid"],
                    "block_name": block.get("name", "Unknown")
                })
            elif len(content) < 50:
                metrics["content_quality"]["short_blocks"] += 1
            elif len(content) > 10000:
                metrics["content_quality"]["long_blocks"] += 1
            
            # Check hierarchy quality
            if "parent_uuid" in block and block["parent_uuid"] is not None:
                if block["parent_uuid"] not in blocks_by_uuid:
                    metrics["hierarchy_quality"]["orphaned_blocks"] += 1
                    metrics["issues"].append({
                        "type": "orphaned_block",
                        "block_uuid": block["uuid"],
                        "block_name": block.get("name", "Unknown"),
                        "parent_uuid": block["parent_uuid"]
                    })
            
            # Check metadata quality
            if "metadata" not in block or not block["metadata"]:
                metrics["metadata_quality"]["missing_metadata"] += 1
                metrics["issues"].append({
                    "type": "missing_metadata",
                    "block_uuid": block["uuid"],
                    "block_name": block.get("name", "Unknown")
                })
            elif isinstance(block["metadata"], dict):
                # Check for important metadata fields
                important_fields = []
                if block["type"] == "doc_section":
                    important_fields = ["doc_type", "section_hierarchy", "source_url"]
                elif block["type"] == "code_block":
                    important_fields = ["language", "source_file"]
                
                for field in important_fields:
                    if field not in block["metadata"]:
                        metrics["metadata_quality"]["incomplete_metadata"] += 1
                        metrics["issues"].append({
                            "type": "incomplete_metadata",
                            "block_uuid": block["uuid"],
                            "block_name": block.get("name", "Unknown"),
                            "missing_field": field
                        })
                        break
        
        return metrics
'''
    
    # Write the adapter file
    with open(adapter_path, "w") as f:
        f.write(adapter_content)
    
    # Create config file
    config_path = INTEGRATION_DIR / "validation_config.py"
    config_content = '''"""
Configuration for validation integration.

This module contains configuration options for validation.
"""

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "min_content_length": 50,
    "max_content_length": 10000,
    "max_empty_blocks_percent": 0.05,
    "max_orphaned_blocks_percent": 0.02,
    "max_missing_metadata_percent": 0.1
}

# Required fields by block type
REQUIRED_FIELDS = {
    "doc_section": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "parent_uuid"],
    "code_block": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "parent_uuid"],
    "documentation": ["uuid", "type", "name", "content", "language", "file_path", "metadata", "child_uuids"]
}

# Required metadata by block type
REQUIRED_METADATA = {
    "doc_section": ["doc_type", "section_hierarchy", "source_url"],
    "code_block": ["language", "source_file"],
    "documentation": ["doc_type", "source_url"]
}
'''
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Created validation adapter at {adapter_path}")

def create_integration_documentation():
    """Create or update integration documentation."""
    print("Creating integration documentation...")
    
    # Ensure the docs/integration directory exists
    DOCS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create main integration guide
    guide_content = '''# Integration Guide

This document provides guidance on integrating the extraction module with other systems.

## Available Integrations

The extraction module can be integrated with the following systems:

### Fetch Docs Integration

The extraction module integrates with the fetch_docs module to download and process HTML documentation.

```python
from agent_tools.dualipa.extraction.integration import DocumentationDownloader, HTMLProcessor

# Download documentation
downloader = DocumentationDownloader("output_dir", use_playwright=True)
success = downloader.download("https://docs.arangodb.com/stable/aql/")

# Process HTML content
processor = HTMLProcessor("output_dir")
processed_docs = processor.process_directory(doc_type="arangodb")

# Convert to extraction format
from agent_tools.dualipa.extraction.docs_integration import convert_to_dualipa_format
blocks = convert_to_dualipa_format(processed_docs, "output_dir")
```

### QA System Integration

The extraction module can format its output for consumption by QA systems.

```python
from agent_tools.dualipa.extraction.integration import QAIntegration

# Format extraction output for QA
qa_integration = QAIntegration()
qa_data = qa_integration.format_for_qa(blocks)

# Save QA-formatted data
qa_integration.save_qa_format(blocks, "qa_data.json")
```

### Validation Integration

The extraction module provides tools for validating extraction quality.

```python
from agent_tools.dualipa.extraction.integration import ExtractionValidator, QualityChecker

# Validate extraction output
validator = ExtractionValidator("schema.json")
validation_results = validator.validate_extraction(blocks)

# Check extraction quality
checker = QualityChecker()
quality_metrics = checker.check_quality(blocks)
```

## Integration Best Practices

1. **Use Adapters**: Always use the provided adapter modules rather than directly importing from external modules.
2. **Validate Input/Output**: Validate input and output data to ensure compatibility.
3. **Handle Errors**: Implement proper error handling for integration failures.
4. **Configure Appropriately**: Use the provided configuration options to customize integration behavior.
5. **Check Dependencies**: Verify that required dependencies are available before attempting integration.
'''
    
    # Write the guide
    with open(DOCS_DIR / "integration_guide.md", "w") as f:
        f.write(guide_content)
    
    # Create fetch_docs integration documentation
    fetch_docs_content = '''# Fetch Docs Integration

This document provides detailed information on integrating with the fetch_docs module.

## Overview

The fetch_docs module provides functionality for downloading and processing HTML documentation from websites. The extraction module integrates with fetch_docs to download documentation and convert it to the extraction format.

## Components

### DocumentationDownloader

The `DocumentationDownloader` class provides a simplified interface for downloading documentation using either wget or Playwright.

```python
from agent_tools.dualipa.extraction.integration import DocumentationDownloader

# Initialize the downloader
downloader = DocumentationDownloader("output_dir", use_playwright=True)

# Download documentation
success = downloader.download("https://docs.arangodb.com/stable/aql/")

# Download with Playwright directly
stats = downloader.download_with_playwright(
    "https://docs.arangodb.com/stable/aql/",
    recursive=True,
    max_depth=2
)
```

### HTMLProcessor

The `HTMLProcessor` class provides methods for extracting structured content from HTML files.

```python
from agent_tools.dualipa.extraction.integration import HTMLProcessor

# Initialize the processor
processor = HTMLProcessor("output_dir")

# Extract sections from a specific file
sections = processor.extract_sections("output_dir/docs.arangodb.com/stable/aql/index.html")

# Process an entire directory
processed_docs = processor.process_directory(doc_type="arangodb")
```

## Integration with Extraction

After downloading and processing documentation, you can convert it to the extraction format:

```python
from agent_tools.dualipa.extraction.docs_integration import convert_to_dualipa_format

# Convert processed docs to extraction format
blocks = convert_to_dualipa_format(processed_docs, "output_dir")
```

## Configuration Options

You can customize the behavior of the fetch_docs integration by modifying the configuration options in `fetch_docs_config.py`:

```python
# Default options for documentation download
DEFAULT_DOWNLOAD_OPTIONS = {
    "recursive": True,
    "max_depth": 2,
    "use_playwright": False,
    "timeout": 30000  # 30 seconds
}

# Doc type detection mapping
DOC_TYPE_MAPPING = {
    "readthedocs.io": "readthedocs",
    "readthedocs.org": "readthedocs",
    "arangodb.com": "arangodb",
    "docs.python.org": "python",
    "developer.mozilla.org": "mdn"
}

# HTML processing options
HTML_PROCESSING_OPTIONS = {
    "extract_code_blocks": True,
    "extract_tables": True,
    "extract_images": True,
    "min_section_length": 50,  # Minimum content length for a section to be extracted
    "max_section_length": 10000  # Maximum content length for a section
}
```
'''
    
    # Write the fetch_docs integration documentation
    with open(DOCS_DIR / "fetch_docs_integration.md", "w") as f:
        f.write(fetch_docs_content)
    
    print("Integration documentation created successfully.")

def create_integration_examples():
    """Create or update integration examples."""
    print("Creating integration examples...")
    
    # Ensure the examples/integration directory exists
    EXAMPLES_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create fetch_docs integration example
    fetch_docs_example = '''"""
Example demonstrating integration with fetch_docs module.

This example shows how to download documentation using fetch_docs and
convert it to the extraction format.
"""

import os
from pathlib import Path

# Import integration components
from agent_tools.dualipa.extraction.integration import DocumentationDownloader, HTMLProcessor
from agent_tools.dualipa.extraction.docs_integration import convert_to_dualipa_format

def download_and_extract_docs(url: str, output_dir: str, use_playwright: bool = False):
    """
    Download and extract documentation from a URL.
    
    Args:
        url: URL to download
        output_dir: Directory to save output
        use_playwright: Whether to use Playwright for JavaScript rendering
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Download documentation
    print(f"Downloading documentation from {url}...")
    downloader = DocumentationDownloader(output_dir, use_playwright=use_playwright)
    success = downloader.download(url, recursive=True)
    
    if not success:
        print("Download failed.")
        return None
    
    print("Download completed successfully.")
    
    # Process HTML content
    print("Processing HTML content...")
    processor = HTMLProcessor(output_dir)
    processed_docs = processor.process_directory()
    
    # Convert to extraction format
    print("Converting to extraction format...")
    blocks = convert_to_dualipa_format(processed_docs, output_dir)
    
    # Save extraction output
    extraction_output_file = os.path.join(output_dir, "extraction_output.json")
    import json
    with open(extraction_output_file, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)
    
    print(f"Extraction completed. Output saved to {extraction_output_file}")
    print(f"Extracted {len(blocks)} blocks.")
    
    return blocks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and extract documentation")
    parser.add_argument("url", help="URL to download")
    parser.add_argument("--output-dir", default="docs_output", help="Output directory")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for JavaScript rendering")
    
    args = parser.parse_args()
    
    download_and_extract_docs(args.url, args.output_dir, args.playwright)
'''
    
    # Write the fetch_docs example
    with open(EXAMPLES_DIR / "fetch_docs_example.py", "w") as f:
        f.write(fetch_docs_example)
    
    # Create QA integration example
    qa_example = '''"""
Example demonstrating integration with QA systems.

This example shows how to format extraction output for QA systems.
"""

import os
import json
from pathlib import Path

# Import integration components
from agent_tools.dualipa.extraction.integration import QAIntegration, QuestionGenerator

def format_for_qa_system(extraction_file: str, output_dir: str):
    """
    Format extraction output for QA system consumption.
    
    Args:
        extraction_file: Path to extraction output JSON file
        output_dir: Directory to save QA-formatted output
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load extraction output
    print(f"Loading extraction output from {extraction_file}...")
    with open(extraction_file, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    
    # Format for QA system
    print("Formatting for QA system...")
    qa_integration = QAIntegration()
    qa_data = qa_integration.format_for_qa(blocks)
    
    # Save QA-formatted output
    qa_output_file = os.path.join(output_dir, "qa_formatted_output.json")
    with open(qa_output_file, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2)
    
    print(f"QA formatting completed. Output saved to {qa_output_file}")
    
    # Generate sample questions
    print("Generating sample questions...")
    question_generator = QuestionGenerator()
    questions = question_generator.generate_questions(blocks, num_questions=5)
    
    # Save sample questions
    questions_file = os.path.join(output_dir, "sample_questions.json")
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated {len(questions)} sample questions. Saved to {questions_file}")
    
    return qa_data, questions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format extraction output for QA systems")
    parser.add_argument("extraction_file", help="Path to extraction output JSON file")
    parser.add_argument("--output-dir", default="qa_output", help="Output directory")
    
    args = parser.parse_args()
    
    format_for_qa_system(args.extraction_file, args.output_dir)
'''
    
    # Write the QA example
    with open(EXAMPLES_DIR / "qa_system_example.py", "w") as f:
        f.write(qa_example)
    
    # Create README for examples
    readme_content = '''# Integration Examples

This directory contains examples demonstrating integration with external systems.

## Available Examples

### Fetch Docs Integration

`fetch_docs_example.py` demonstrates how to download and extract documentation using the fetch_docs module.

Usage:
```bash
python fetch_docs_example.py https://docs.arangodb.com/stable/aql/ --output-dir docs_output --playwright
```

### QA System Integration

`qa_system_example.py` demonstrates how to format extraction output for QA systems.

Usage:
```bash
python qa_system_example.py docs_output/extraction_output.json --output-dir qa_output
```

## Running the Examples

1. Install required dependencies:
   ```bash
   pip install playwright
   playwright install
   ```

2. Run the fetch_docs example to download and extract documentation:
   ```bash
   python fetch_docs_example.py https://docs.arangodb.com/stable/aql/ --playwright
   ```

3. Run the QA system example to format the extracted content:
   ```bash
   python qa_system_example.py docs_output/extraction_output.json
   ```
'''
    
    # Write the README
    with open(EXAMPLES_DIR / "README.md", "w") as f:
        f.write(readme_content)
    
    print("Integration examples created successfully.")

def main():
    """Execute the integration reorganization process."""
    print("Starting integration reorganization...")
    
    # Create the integration directory
    create_integration_directory()
    
    # Create adapter modules
    create_fetch_docs_adapter()
    create_qa_adapter()
    create_validation_adapter()
    
    # Create documentation
    create_integration_documentation()
    
    # Create examples
    create_integration_examples()
    
    print("Integration reorganization complete!")

if __name__ == "__main__":
    main()