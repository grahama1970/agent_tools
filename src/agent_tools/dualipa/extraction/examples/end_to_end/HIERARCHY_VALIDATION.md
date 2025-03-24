# Hierarchy Validation Framework

This document provides comprehensive documentation for the hierarchy validation framework in the DuaLipa extraction module. The framework ensures extraction outputs maintain proper parent-child relationships, which are crucial for LLM processing.

## 1. Importance of Hierarchical Structure

Documentation extractions must maintain hierarchical structure to be useful:

1. **Contextual Understanding**: LLMs need hierarchy to understand the context of code examples
2. **Navigation**: Applications need hierarchy to provide proper navigation between sections
3. **Relationships**: Proper child → parent and parent → child relationships are essential
4. **Semantic Structure**: Documentation follows natural hierarchies (doc → page → section → subsection → code blocks)

## 2. Validation Framework Architecture

The framework consists of several interconnected components:

### 2.1 Core Validation Logic (`validation.py`)

- `validate_qa_output()`: Validates basic QA compatibility requirements
- `validate_structure()`: Validates hierarchical structure against expected format
- `validate_content_against_expected()`: Validates semantic content against expected values
- `validate_extraction_result()`: Comprehensive validation combining all checks

### 2.2 Hierarchy-Specific Validation (`hierarchy_validator.py`)

- `validate_parent_child_relationships()`: Verifies bidirectional references between blocks
- `find_circular_references()`: Detects circular references in the hierarchy
- `visualize_hierarchy()`: Generates HTML visualization of the hierarchy

### 2.3 Command-Line Interface Tools

- `validate_hierarchy.py`: Validates a single extraction
- `validate_all_hierarchies.py`: Batch validates all extractions
- `generate_expected_formats.py`: Generates expected format templates

### 2.4 Expected Formats

JSON files that define validation criteria for different document types, including:
- Basic structure requirements
- Hierarchical relationship rules
- Content validation requirements
- Metadata verification rules

## 3. Validation Criteria

The framework validates extractions against several types of criteria:

### 3.1 Structure Validation

- **Block Types**: Checks for the presence of required block types
- **Hierarchy**: Validates that parent-child relationships follow expected patterns
- **Bidirectional References**: Ensures both parent → child and child → parent references exist
- **Type Consistency**: Verifies that child blocks have allowed types

### 3.2 Content Validation

- **Required Keywords**: Checks for the presence of domain-specific terminology
- **Function Details**: For function documentation, validates parameters, return types, etc.
- **Examples**: Validates that code examples are present

### 3.3 Format Consistency

- **Root Blocks**: Ensures appropriate root-level blocks exist
- **Hierarchy Depth**: Validates the nesting level of document sections
- **Metadata**: Checks required metadata fields

## 4. Expected Format Files

Expected format files define validation criteria and are structured as follows:

```json
{
  "description": "Expected format for ArangoDB documentation extraction",
  "version": "1.0",
  "expected_structure": {
    "required_block_types": ["documentation", "doc_page", "doc_section"],
    "hierarchy": [
      {
        "parent_type": "documentation",
        "child_types": ["doc_page"]
      },
      {
        "parent_type": "doc_page",
        "child_types": ["doc_section"]
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
    "required_keywords": ["ArangoDB", "function"],
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

### 4.1 Available Expected Formats

- `arangodb_expected_format.json`: For ArangoDB documentation
- `readthedocs_expected_format.json`: For ReadTheDocs documentation
- `length_string_expected_format.json`: For LENGTH string function
- `array_intersection_array_expected_format.json`: For ARRAY_INTERSECTION function
- `deepseek_expected_format.json`: For deepseek format output
- Many more domain-specific formats

## 5. HTML Visualization

The framework generates interactive HTML visualizations to make validation results easy to understand:

### 5.1 Hierarchy Visualization

- Visual representation of parent-child relationships
- Color-coded by block type
- Expandable/collapsible sections
- Error highlighting
- Metadata display

### 5.2 Summary Page

- Overall validation results
- Statistics (total blocks, root blocks, child blocks, etc.)
- Error and warning lists
- Links to detailed reports

### 5.3 Index Page (Batch Validation)

- Overview of all validation results
- Status indicators for each extraction
- Statistics for all extractions
- Links to individual reports

## 6. Integration Guide

### 6.1 Adding Validation to Extraction Pipeline

```python
from validation import validate_extraction_result, load_expected_format

def extract_and_validate(source_url, expected_format_path):
    # Extract content
    extraction_result = extract_content(source_url)
    
    # Load expected format
    expected_format = load_expected_format(expected_format_path)
    
    # Validate extraction
    validation_results = validate_extraction_result(extraction_result, expected_format)
    
    # Check validation results
    if validation_results["valid"]:
        return extraction_result
    else:
        log_validation_errors(validation_results)
        raise ValidationError("Extraction failed validation")
```

### 6.2 Creating Custom Expected Formats

To create a custom expected format for a new document type:

1. Use `generate_expected_formats.py` as a starting point
2. Define required block types, hierarchy, and content requirements
3. Set appropriate validation thresholds
4. Save the format in the `expected_formats` directory

### 6.3 Converting Between Formats

For formats that don't match the expected structure, use the conversion tools:

```python
from convert_for_validation import convert_to_validation_format

# Convert extraction format
converted_data = convert_to_validation_format(extraction_data)

# Validate converted data
validation_results = validate_extraction_result(converted_data, expected_format)
```

## 7. Troubleshooting

### 7.1 Common Validation Errors

| Error | Possible Cause | Solution |
|-------|---------------|----------|
| Missing parent reference | Block has no parent_uuid | Add parent_uuid field |
| Reference to non-existent child | UUID in child_uuids doesn't exist | Remove invalid child reference |
| Child has incorrect type | Child block has disallowed type | Change child type or relationship |
| Circular reference detected | Hierarchy contains a loop | Break the circular reference |
| Missing required block type | Required block type not present | Add missing block type |

### 7.2 Format-Specific Issues

- **ArangoDB**: Ensure code blocks have language="aql"
- **ReadTheDocs**: Check section nesting levels
- **Deepseek Format**: Ensure all required fields are present

### 7.3 Using Validation Results for Debugging

The validation results object contains detailed information:

```json
{
  "valid": false,
  "overall_score": 78.5,
  "structure_validation": {
    "valid": true,
    "score": 85.0,
    "errors": [],
    "successes": ["..."],
    "total_checks": 20,
    "passed_checks": 17
  },
  "content_validation": {
    "valid": false,
    "score": 72.0,
    "errors": ["Missing required keyword: function"],
    "successes": ["..."],
    "total_checks": 10,
    "passed_checks": 7
  },
  "format_validation": {
    "valid": true
  }
}
```

## 8. Extending the Framework

To extend the validation framework for new requirements:

1. Add new validation functions in `validation.py`
2. Create new expected format templates
3. Update visualization code for new block types
4. Add new tests for the validation functions

## 9. Performance Considerations

The validation framework is designed to be efficient, but for large extractions:

1. Use batch validation for multiple files
2. Adjust validation thresholds as needed
3. Focus on structure validation first, then content

## 10. Future Improvements

Planned enhancements to the validation framework:

1. Schema-based validation using JSON Schema
2. Integration with continuous integration (CI) pipelines
3. Automatic correction of common hierarchy issues
4. Visual diff tool for comparing extraction hierarchies
5. Performance optimizations for large extractions