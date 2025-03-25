# Integration Examples

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
