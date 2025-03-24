# ArangoDB Documentation Extraction Tests

This directory contains specialized tests for extracting and validating ArangoDB documentation content. These tests ensure the quality and completeness of documentation extraction, particularly focusing on complex elements like code examples, tables, and section hierarchies.

## Test Descriptions

### 1. LENGTH Function Test (`length_function_test.py`)

Tests the extraction of the ArangoDB LENGTH function documentation from https://docs.arangodb.com/3.12/aql/functions/string/#length. 

**Purpose**: Validates extraction of function documentation with:
- Proper content identification
- Code examples for both string and array LENGTH usage
- Tables for function syntax, parameters, and return values
- Section hierarchies

**Expected Results**: 
- Documentation blocks containing LENGTH function content
- Code blocks showing LENGTH usage with strings and arrays
- Tables describing the function parameters and return values

### 2. Array INTERSECTION Test (`array_intersection_test.py`)

Tests the extraction of the ArangoDB INTERSECTION function documentation from https://docs.arangodb.com/3.12/aql/functions/array/#intersection.

**Purpose**: Validates extraction of array function documentation with:
- Complex content including multiple code examples
- Tables for function signatures
- Proper parent-child relationships between blocks

**Expected Results**:
- Documentation blocks with INTERSECTION function content
- Code blocks showing various INTERSECTION usage patterns
- Tables describing the function syntax and parameters

### 3. AQL Main Page Test (`arangodb_aql_test.py`)

Tests the extraction of the main AQL page from https://docs.arangodb.com/stable/aql/.

**Purpose**: Validates extraction of overview documentation with:
- High-level AQL language descriptions
- Multiple sections and subsections
- Links to other documentation pages

**Expected Results**:
- Documentation blocks with AQL overview
- Proper section hierarchy extraction
- Relationships between different documentation elements

### 4. General ArangoDB Documentation Test (`arangodb_validator.py`)

Tests multiple ArangoDB documentation pages including:
- https://docs.arangodb.com/stable/aql/operations/return/
- https://docs.arangodb.com/stable/indexing/
- https://docs.arangodb.com/stable/aql/fundamentals/
- https://docs.arangodb.com/stable/aql/

**Purpose**: Validates the general extraction capabilities across different types of documentation pages.

**Expected Results**:
- Multiple documentation blocks from different pages
- Proper section, code, and table extraction
- Consistent block relationships and hierarchies

## Running the Tests

Use the `run_specific_tests.py` script to execute tests either individually or all at once:

```bash
# Run all tests
python run_specific_tests.py --all

# Run individual tests
python run_specific_tests.py --length-function
python run_specific_tests.py --array-intersection
python run_specific_tests.py --aql-main
python run_specific_tests.py --general-docs
```

## Test Output

Each test generates detailed output files for analysis:

- `*_extraction.json`: Raw extraction output for manual review
- `*_summary.json`: Test summary including success indicators and error messages
- Console output: Detailed logging of the extraction and validation process

## Integrating with Blind Tests

These tests are designed to be integrated with the `blind_test.py` framework, which provides comprehensive validation across multiple documentation sources.

## Fallback Mechanism

If direct extraction fails, these tests implement a fallback mechanism that generates synthetic examples from markdown content. This ensures tests can still validate the basic extraction infrastructure even when specific elements aren't perfectly extracted.

## Schema Validation

Each test includes schema validation against expected formats defined in corresponding `*_expected_format.json` files. The validation is flexible to accommodate different extraction capabilities.