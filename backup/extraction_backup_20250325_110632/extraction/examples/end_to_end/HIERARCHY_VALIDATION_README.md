# Hierarchy Validation Framework Guide

This document provides a quick reference guide for using the hierarchy validation tools in the DuaLipa extraction module.

## Overview

The Hierarchy Validation Framework verifies that parent-child relationships in extraction outputs are properly maintained. It checks for:

1. Valid bidirectional references between parent and child blocks
2. Absence of circular references in hierarchical relationships
3. Proper structure as defined in expected formats
4. Content validation against expected values

## Validation Tools

The framework provides several tools for validating extraction outputs:

### 1. `validate_hierarchy.py`

Validates a single extraction file and generates an HTML visualization.

```bash
python validate_hierarchy.py --input /path/to/extraction.json --output-dir ./hierarchy_validation --extraction-name arangodb
```

Parameters:
- `--input`: Path to the extraction JSON file
- `--output-dir`: Directory to save validation results
- `--extraction-name`: Name of the extraction for report titles

### 2. `validate_all_hierarchies.py`

Batch validates all extraction outputs in the project and generates a summary page.

```bash
python validate_all_hierarchies.py --output-dir ./hierarchy_validation
```

Parameters:
- `--output-dir`: Directory to save validation results

### 3. `generate_expected_formats.py`

Generates expected format templates for validation.

```bash
python generate_expected_formats.py --update-tests
```

Parameters:
- `--update-tests`: Updates test files to use the generated formats

## Expected Format Files

The validation framework uses expected format files to define validation criteria. These files are stored in the `expected_formats` directory and include:

- `arangodb_expected_format.json`: For ArangoDB documentation
- `readthedocs_expected_format.json`: For ReadTheDocs documentation
- `length_string_expected_format.json`: For LENGTH string function
- `array_intersection_array_expected_format.json`: For ARRAY_INTERSECTION function
- `deepseek_expected_format.json`: For deepseek format output

## Format-Specific Tests

For deepseek format validation (used by the QA system), ensure the test file uses the appropriate expected format:

```bash
python test_validation_framework.py --extraction extraction.json --expected expected_formats/deepseek_expected_format.json
```

## HTML Visualization

The validation tools generate interactive HTML visualizations:

1. **Hierarchy Visualization**: Shows the parent-child relationships between blocks
2. **Summary Page**: Shows validation statistics and errors
3. **Index Page**: For batch validation, shows an overview of all validations

## Integration with Extraction Pipeline

To validate extraction outputs as part of your pipeline:

```python
from validation import validate_extraction_result, load_expected_format

# Load the extraction result
extraction_result = load_extraction_result('output.json')

# Load the expected format
expected_format = load_expected_format('expected_formats/arangodb_expected_format.json')

# Validate the extraction result
validation_results = validate_extraction_result(extraction_result, expected_format)

# Check if validation passed
if validation_results['valid']:
    print('Validation passed!')
else:
    print('Validation failed!')
    for error in validation_results.get('structure_validation', {}).get('errors', []):
        print(f'Error: {error}')
```

## Common Issues

1. **Missing References**: Check that all blocks have proper parent_uuid and child_uuids fields
2. **Circular References**: Fix loops in parent-child relationships
3. **Orphaned Blocks**: Ensure all blocks have a parent or are intended to be root blocks
4. **Format Issues**: For deepseek format, ensure all required fields are present (uuid, title, content, section_hierarchy_depth)

## Contributing

When adding new documentation types or functions, create a new expected format file in the `expected_formats` directory using `generate_expected_formats.py` with appropriate validation criteria.