# DuaLipa Documentation Extraction Validation Framework

This document describes the validation framework used to ensure that our documentation extraction pipeline is working correctly. The framework uses a Test-Driven Development (TDD) approach, requiring expected format definitions before tests are run.

## Overview

The validation framework provides a comprehensive approach to validating both the structure and content of extracted documentation. It ensures:

1. **Structural Integrity**: Hierarchical relationships between blocks are maintained
2. **Content Accuracy**: Extracted content contains the expected semantic information
3. **Format Consistency**: HTML and markdown extraction outputs maintain consistency
4. **Basic Format Validation**: Outputs are compatible with the QA module

## Key Components

### 1. Expected Format Definition

Each test requires an expected format definition (JSON file) that specifies:

- Required block types
- Hierarchical relationships
- Content validation criteria
- Validation thresholds

Example: `length_expected_format.json`

### 2. Validation Functions

The core validation functions in `validation.py` include:

- `validate_qa_output`: Basic format validation
- `validate_structure`: Hierarchical structure validation
- `validate_content_against_expected`: Semantic content validation
- `validate_markdown_and_html_structure`: Format consistency validation
- `validate_extraction_result`: Comprehensive validation combining all others

### 3. Validation Tools

Standalone scripts for running validation:

- `test_validation_framework.py`: Validate a single extraction against an expected format
- `validate_all_tests.py`: Batch validation of multiple tests
- `convert_for_validation.py`: Convert raw extraction outputs to validation-compatible format
- `detect_doc_type.py`: Automatically detect document type and select appropriate template

### 4. Standardized Templates

The framework includes specialized templates for different documentation sources:

- `expected_format_template.json`: Generic template for all documentation
- `arangodb_expected_format.json`: Specialized for ArangoDB API documentation
- `html_docs_expected_format.json`: Specialized for HTML-based documentation
- `markdown_docs_expected_format.json`: Specialized for Markdown-based documentation

## Using the Framework

### Creating Expected Format Files

1. Start with the appropriate template file for your documentation source
2. Define the structure requirements in the `expected_structure` section
3. Define content validation criteria in the `expected_content_validation` section
4. Define structure consistency checks in the `structure_consistency` section

### Writing Tests with Expected Formats

```python
# Example test file
import sys
from pathlib import Path
from validation import validate_extraction_result, load_expected_format

# Extract documentation
extraction_result = extract_documentation(url)

# Load expected format
expected_format = load_expected_format("length_expected_format.json")

# Validate extraction result
validation_result = validate_extraction_result(extraction_result, expected_format)

# Check if validation passed
assert validation_result["valid"], f"Validation failed with score {validation_result['overall_score']}%"
```

### Running the Validation Framework

To validate a single extraction:

```bash
python test_validation_framework.py --extraction length_function_extraction.json --expected length_expected_format.json
```

To validate all tests:

```bash
python validate_all_tests.py --output-dir validation_results
```

#### Advanced Options

The framework now supports advanced validation options:

1. **Automatic Template Detection**

```bash
python validate_all_tests.py --auto-detect --output-dir validation_results
```

2. **Format Conversion**

```bash
python validate_all_tests.py --convert --output-dir validation_results
```

3. **Processing a Directory of Extractions**

```bash
python validate_all_tests.py --input-dir extracted_docs --auto-detect --convert --output-dir validation_results
```

4. **Testing a Specific Extraction**

```bash
python validate_all_tests.py --specific-test length_function --auto-detect --convert
```

## Validation Scoring

The framework uses a scoring mechanism with configurable thresholds:

- **Structure validation**: Default threshold 75%
- **Content validation**: Default threshold 85% 
- **Structure consistency**: Default threshold 75%
- **Overall validation**: Average of all validation scores

A test passes if all validation scores exceed their respective thresholds.

## Format Conversion

The `convert_for_validation.py` script converts raw extraction outputs to the expected format:

```bash
python convert_for_validation.py --input extraction_output.json --output converted_output.json
```

The conversion process:
1. Identifies document sections, pages, and blocks
2. Maintains hierarchical relationships
3. Extracts embedded content (tables, code blocks)
4. Formats everything according to the deepseek format expected by the validation framework

## Expected Format Structure

```json
{
  "description": "Expected format for documentation extraction validation",
  "version": "1.0",
  "expected_structure": {
    "required_block_types": ["documentation", "doc_section"],
    "hierarchy": [
      {
        "parent_type": "documentation",
        "child_types": ["doc_page"]
      }
    ],
    "metadata_checks": [
      {
        "field": "uuid",
        "requirement": "uuid_format"
      }
    ],
    "validation_threshold": 75
  },
  "expected_content_validation": {
    "function_name": "LENGTH",
    "function_purpose": ["returns the length of a string"],
    "parameters": [
      {
        "name": "str",
        "type": "string",
        "description": ["input string"]
      }
    ],
    "return_type": "number",
    "examples": [
      {
        "code": "LENGTH(\"foobar\")",
        "output": "6"
      }
    ],
    "required_keywords": ["LENGTH", "string"],
    "validation_threshold": 85
  },
  "structure_consistency": {
    "required_root_blocks": ["documentation"],
    "hierarchical_types": [
      {
        "parent": "documentation",
        "children": ["doc_page"]
      }
    ],
    "validation_threshold": 75
  }
}
```

## Automatic Document Type Detection

The framework can automatically detect the type of documentation and select the appropriate template:

```bash
python detect_doc_type.py --input extraction_output.json
```

Currently supported document types:
- ArangoDB API documentation
- HTML-based documentation
- Markdown-based documentation
- Generic documentation (fallback)

## Troubleshooting

If validation fails, check:

1. Extraction output structure
2. Expected format definition
3. Content patterns in the expected content validation
4. Hierarchical relationships in extraction outputs
5. Conversion to validation-compatible format

### Common Issues

1. **Wrong Template**: Use `--auto-detect` to let the framework choose the appropriate template
2. **Raw Format Not Compatible**: Use `--convert` to convert the extraction to the validation format
3. **Missing Hierarchical Relationships**: Check parent-child relationships in the extraction
4. **Content Not Matching**: Review the content validation criteria in the expected format

## Additional Resources

- See `length_expected_format.json` and `array_intersection_expected_format.json` for examples
- Check documentation source format to ensure appropriate validation criteria
- Use the document type detection to identify the best template for your documentation
- For complex extractions, review the conversion logs to ensure proper transformation