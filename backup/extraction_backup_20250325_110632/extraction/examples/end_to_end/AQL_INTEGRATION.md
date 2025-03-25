# ArangoDB AQL Documentation Integration

This document explains the specific integration with ArangoDB's main AQL documentation page (https://docs.arangodb.com/stable/aql/) within the DuaLipa extraction project.

## Overview

The ArangoDB Query Language (AQL) is a central feature of ArangoDB, and its documentation requires special attention in our extraction system. This integration ensures that we properly extract and structure the content from the main AQL documentation page, making it available for downstream processing and the QA system.

## Implementation Details

### Test Components

1. **Specialized Test Function**: 
   - `test_arangodb_aql_main_page()` in `blind_test.py` specifically tests the extraction of the main AQL page
   - `validate_arangodb_aql_main_page()` validates the structure and content of extracted AQL blocks

2. **Expected Structure**:
   - Defined in `arangodb_expected_format.json` through the `main_aql_doc` block
   - Requires proper document hierarchy (documentation → pages → sections → code blocks/tables)
   - Specifically validates AQL-related code blocks and tables

3. **Blind Test Repository**:
   - Creates a temporary repository with markdown files containing explicit links to the AQL documentation
   - Includes sample AQL code blocks and tables to ensure proper format even when external documentation is unavailable
   - Mimics the structure of real repositories linking to ArangoDB documentation

### Validation Process

The validation process focuses on:

1. **URL Verification**:
   - Ensures that the extracted documentation comes from the correct URL (https://docs.arangodb.com/stable/aql/)
   - Validates that source_url metadata is preserved

2. **Content Validation**:
   - Checks for "AQL" in the documentation title
   - Looks for essential AQL sections such as operations, functions, and syntax
   - Validates that query examples are properly extracted as code blocks

3. **Structure Validation**:
   - Confirms proper parent-child relationships between documentation elements
   - Ensures sections have been properly parsed from HTML headers
   - Validates that code blocks and tables are properly associated with their parent sections

4. **Type Validation**:
   - Verifies that AQL code blocks have the correct language specified (javascript or aql)
   - Ensures tables related to AQL operations are properly structured

### Fallback Mechanisms

To handle cases where documentation cannot be downloaded:

1. **Markdown Substitutes**:
   - Uses code blocks and tables from markdown files as substitutes when needed
   - Ensures validation doesn't fail due to external service unavailability

2. **Format Normalization**:
   - Creates consistent block structure regardless of the source (HTML or markdown)
   - Standardizes the representation of code blocks and tables

## Results Analysis

The test generates a summary file (`arangodb_aql_main_summary.json`) containing:

1. **Block Counts**:
   - Number of documentation blocks, pages, sections, code blocks, and tables
   - Origin URL information

2. **Validation Status**:
   - Whether the extraction met all validation criteria
   - Any errors or warnings encountered

## Running the Test

To specifically test the AQL main page extraction:

```bash
python blind_test.py --aql-main-page-only
```

## Integration with QA System

The AQL documentation extraction is specifically formatted to enable:

1. **Answering AQL syntax questions**:
   - Understanding AQL keyword usage and syntax
   - Explaining query components

2. **Providing AQL example code**:
   - Offering sample queries for common operations
   - Demonstrating AQL best practices

3. **Explaining AQL operations**:
   - Clarifying how AQL operations like FOR, FILTER, SORT work
   - Showing operations in table format with descriptions