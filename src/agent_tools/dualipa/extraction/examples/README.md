# DuaLipa Extraction Examples

This directory contains examples of how to use the DuaLipa extraction modules.

## End-to-End Extraction

The `end_to_end_extraction.py` script demonstrates the complete extraction pipeline from code files to QA-compatible output format. It shows how to:

1. Extract code blocks from source files
2. Analyze code hierarchies
3. Enrich blocks with hierarchical information
4. Create QA-compatible output

### Usage

```bash
python end_to_end_extraction.py <source_dir> <output_file>
```

Example:
```bash
python end_to_end_extraction.py ./test_repos/python-sample ./output.json
```

### Output Format

The script produces a JSON file with the following structure:

```json
{
  "sections": [
    {
      "uuid": "unique-id-1",
      "type": "code",
      "language": "python",
      "title": "sample_file.py",
      "content": "def example(): pass",
      "file_path": "/path/to/sample_file.py",
      "breadcrumb": ["sample_file.py"],
      "parent_uuid": null,
      "child_uuids": ["unique-id-2"],
      "depth": 0,
      "extraction_focus": "code structure",
      "summary_instructions": "Generate QA pairs about the overall code structure",
      "dependencies": {
        "imports": ["import os", "import json"]
      },
      ...
    },
    {
      "uuid": "unique-id-2",
      "type": "code",
      "language": "python",
      "title": "Function example",
      "content": "def example(): pass",
      "file_path": "/path/to/sample_file.py",
      "breadcrumb": ["sample_file.py", "example"],
      "parent_uuid": "unique-id-1",
      "child_uuids": [],
      "depth": 1,
      "extraction_focus": "function implementation",
      "summary_instructions": "Generate QA pairs about the example function implementation",
      "dependencies": {
        "imports": ["import os", "import json"]
      },
      ...
    }
  ],
  "extraction_metadata": {
    "model_used": "extraction-model",
    "timestamp": "2025-03-21T12:34:56Z",
    "version": "1.0.0",
    "purpose": "Code block extraction for QA generation",
    "supported_languages": ["python", "javascript", "typescript", "java", "cpp", "c", "go", "ruby"],
    "extraction_focus_options": ["Code Structure", "API Usage", "Implementation Details"],
    "expected_output_structure": {
      "question": "string",
      "answer": "string",
      "reasoning": "string"
    }
  }
}
```

### Integration with QA Module

The output format is compatible with the QA module requirements:

- Each section has the required fields: `uuid`, `type`, `content`, `extraction_focus`, `summary_instructions`
- Code blocks have additional fields: `language`, `file_path`, `dependencies`
- Hierarchical relationships are maintained through `parent_uuid`, `child_uuids`, `breadcrumb`, and `depth`

### Testing

You can test the end-to-end extraction process using the test script:

```bash
python -m pytest tests/dualipa/extraction/test_end_to_end.py -v
```

## Known Limitations

- Tree-sitter parent-child tracking is limited (see `/hierarchy/TECHNICAL_DEBT.md`)
- Python AST visitor pattern not fully implemented
- Large files (>1MB) are skipped to avoid memory issues
- Advanced relationship tracking (e.g., inherited methods) is limited