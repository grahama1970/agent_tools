# Extraction Module Integration

This directory contains adapter modules for integrating the extraction module with other systems.

## Available Integrations

### Fetch_Docs

Integration with the fetch_docs module for downloading and processing HTML documentation

- Adapter: `fetch_docs_adapter.py`
- Interfaces: `DocumentationDownloader`, `HTMLProcessor`
- Configuration: `fetch_docs_config.py`

### Qa_System

Integration with the QA system for answer generation based on extracted content

- Adapter: `qa_adapter.py`
- Interfaces: `QAIntegration`, `QuestionGenerator`
- Configuration: `qa_config.py`

### Validation

Integration with validation systems for verifying extraction quality

- Adapter: `validation_adapter.py`
- Interfaces: `ExtractionValidator`, `QualityChecker`
- Configuration: `validation_config.py`

