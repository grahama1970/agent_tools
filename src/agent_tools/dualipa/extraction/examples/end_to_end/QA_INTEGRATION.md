# QA Integration Guide for Extraction Module

This document outlines how to integrate the extraction module with the QA generation module to create a complete pipeline for extracting code structures and generating question-answer pairs.

## Overview

The extraction pipeline produces structured output from code repositories, which can then be used by the QA module to generate question-answer pairs. This integration enables the creation of a complete pipeline for knowledge extraction and query generation.

```
┌─────────────────┐       ┌───────────────────┐       ┌──────────────────┐
│                 │       │                   │       │                  │
│  Source Files   │─────> │  Extraction Module│─────> │    QA Module     │─────> Q&A Pairs
│  (Repository)   │       │                   │       │                  │
│                 │       │                   │       │                  │
└─────────────────┘       └───────────────────┘       └──────────────────┘
```

## Output Format Compatibility

The extraction module can produce output in two formats:

1. **Standard Format**: A JSON object with sections, metadata, and relationships
2. **DeepSeek Format**: A specialized list format specifically designed for the deepseek.md file

Our integration tools can handle both formats and convert them to the QA-compatible format expected by the QA module.

### DeepSeek Format

The DeepSeek format is a list of section objects with the following structure:

```json
[
  {
    "uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
    "section_hierarchy_depth": ["DeepSeek Usage", "Launch DeepSeek V3 with SGLang"],
    "title": "Launch DeepSeek V3 with SGLang",
    "content": "SGLang provides several optimizations...",
    "images": [
      {
        "uuid": "b7c8d9e0-f1a2-4b3c-95d6-e7f8a9b0c1d2",
        "src": "image.png",
        "alt": "Architecture diagram"
      }
    ],
    "tables": [
      {
        "uuid": "c8d9e0f1-a2b3-4c5d-6e7f-8a9b0c1d2e3f",
        "content": {
          "headers": ["Model", "Tokens/sec"],
          "rows": [["DeepSeek-7B", "120"]]
        }
      }
    ],
    "code": [
      {
        "uuid": "d9e0f1a2-b3c4-5d6e-7f8a-9b0c1d2e3f4g",
        "language": "python",
        "content": "import sglang as sgl\n..."
      }
    ],
    "tests": []
  }
]
```

### QA-Compatible Format

The QA module expects input in the following format:

```json
{
  "sections": [
    {
      "uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
      "type": "documentation",
      "content": "SGLang provides several optimizations...",
      "title": "Launch DeepSeek V3 with SGLang",
      "extraction_focus": "technical details",
      "summary_instructions": "Generate QA pairs about 'Launch DeepSeek V3 with SGLang'",
      "breadcrumb": ["DeepSeek Usage", "Launch DeepSeek V3 with SGLang"]
    }
  ],
  "extraction_metadata": {
    "model_used": "extraction-model",
    "timestamp": "2025-03-21T00:00:00Z",
    "statistics": {
      "total_sections": 1
    }
  }
}
```

## Integration Tools

We provide two tools to help with integrating the extraction module with the QA module:

1. `test_extraction_qa_integration.py`: A test script that validates the integration
2. `validate_qa_compatibility.py`: A validation tool that checks and converts extraction output

### Validation Tool Usage

```bash
# Validate extraction output for QA compatibility
python validate_qa_compatibility.py /path/to/extraction_output.json

# Convert extraction output to QA-compatible format
python validate_qa_compatibility.py /path/to/extraction_output.json --convert --output /path/to/qa_input.json
```

### Integration Test

The integration test verifies that the extraction module produces output that can be correctly processed by the QA module. It covers:

1. Running the extraction pipeline on a repository
2. Validating the extraction output format
3. Converting the output to QA-compatible format
4. Checking that the QA module can process the converted output

```bash
# Run the integration test
python test_extraction_qa_integration.py
```

## Integration Workflow

To integrate the extraction module with the QA module:

1. Run the extraction pipeline:
```bash
python real_world_test.py /path/to/output.json
```

2. Convert the extraction output to QA-compatible format:
```bash
python validate_qa_compatibility.py /path/to/output.json --convert --output /path/to/qa_input.json
```

3. Run the QA module with the converted input:
```bash
# Using Python API
from agent_tools.dualipa.qa.processor import process_extraction_json
qa_response = await process_extraction_json(input_data="/path/to/qa_input.json", output_file="/path/to/qa_output.json")

# Using CLI tool
python -m agent_tools.dualipa.qa /path/to/qa_input.json --output /path/to/qa_output.json
```

## Notes on Format Conversion

When converting from the extraction format to the QA-compatible format:

### DeepSeek Format Conversion
- `uuid` is preserved directly
- `section_hierarchy_depth` is mapped to `breadcrumb`
- `title` is preserved directly
- `content` is preserved directly
- `extraction_focus` is set to "technical details"
- `summary_instructions` is generated from the title
- `type` is set to "documentation"

### Standard Format Conversion
- Required fields are added if missing
- Nested content (tables, code, images) is preserved but structured differently

## Troubleshooting

Common issues and their solutions:

1. **Missing Fields**: If the validation fails due to missing fields, check that your extraction output follows the expected format.

2. **Incorrect Format**: If you get "Output is neither a dict nor a list" error, check that your JSON is properly formatted.

3. **QA Module Import Error**: If you're unable to import the QA module, ensure the `agent_tools.dualipa.qa` package is installed and accessible in your Python path.

4. **Empty Output**: If the extraction produces an empty output, check that your repository contains the expected files and that the extraction configuration is correct.