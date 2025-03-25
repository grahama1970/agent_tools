# Integration Guide

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
