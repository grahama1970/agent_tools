#!/usr/bin/env python3
"""
Generate Expected Format Files for DuaLipa Extraction Tests.

This script creates structurally and semantically accurate expected format templates
for all tests within the extraction module. These templates are used to validate 
that extraction outputs maintain proper structure and relationships.
"""

import os
import sys
import json
import uuid
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_expected_formats")

# Define the base path
BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def create_documentation_expected_format(doc_type: str) -> Dict[str, Any]:
    """
    Create an expected format template for generic documentation extraction.
    
    Args:
        doc_type: The type of documentation (e.g., "arangodb", "readthedocs")
        
    Returns:
        Dictionary containing expected format
    """
    # Define type-specific settings
    if doc_type == "arangodb":
        sample_url = "https://docs.arangodb.com/stable/aql/"
        description = "Expected format for ArangoDB documentation extraction"
        required_keywords = ["ArangoDB", "AQL", "query", "function"]
    elif doc_type == "readthedocs":
        sample_url = "https://python.readthedocs.io/en/latest/"
        description = "Expected format for Read the Docs documentation extraction"
        required_keywords = ["Python", "documentation", "reference", "module"]
    else:  # Generic format
        sample_url = "https://example.com/docs/"
        description = "Expected format for generic documentation extraction"
        required_keywords = ["documentation", "reference", "guide", "tutorial"]

    # Create the expected format structure
    expected_format = {
        "description": description,
        "version": "1.0",
        "expected_structure": {
            "required_block_types": [
                "documentation",
                "doc_page",
                "doc_section",
                "code_block",
                "table"
            ],
            "hierarchy": [
                {
                    "parent_type": "documentation",
                    "child_types": ["doc_page"]
                },
                {
                    "parent_type": "doc_page",
                    "child_types": ["doc_section"]
                },
                {
                    "parent_type": "doc_section",
                    "child_types": ["doc_section", "code_block", "table"]
                }
            ],
            "metadata_checks": [
                {
                    "field": "uuid",
                    "requirement": "uuid_format"
                },
                {
                    "field": "metadata.language",
                    "requirement": "not_empty"
                },
                {
                    "field": "metadata.source_url",
                    "requirement": "not_empty"
                }
            ],
            "validation_threshold": 75
        },
        "expected_content_validation": {
            "required_keywords": required_keywords,
            "validation_threshold": 85
        },
        "structure_consistency": {
            "required_root_blocks": ["documentation"],
            "hierarchical_types": [
                {
                    "parent": "documentation",
                    "children": ["doc_page"]
                },
                {
                    "parent": "doc_page",
                    "children": ["doc_section"]
                },
                {
                    "parent": "doc_section",
                    "children": ["code_block", "table"]
                }
            ],
            "skip_metadata_checks": False,
            "skip_root_checks": False,
            "validation_threshold": 75
        },
        "sample_blocks": {
            "documentation": {
                "uuid": str(uuid.uuid4()),
                "id": f"docs_{doc_type}",
                "name": f"Documentation: {doc_type}",
                "type": "documentation",
                "language": "html",
                "content": f"Documentation site: {sample_url}",
                "source_url": sample_url,
                "child_uuids": ["doc-page-uuid-placeholder"],
                "metadata": {
                    "language": "html",
                    "source_url": sample_url,
                    "doc_type": doc_type
                }
            },
            "doc_page": {
                "uuid": "doc-page-uuid-placeholder",
                "id": f"docs_{doc_type}_index.html",
                "name": "index.html",
                "type": "doc_page",
                "language": "html",
                "content": f"Documentation page from {sample_url}",
                "file_path": f"/path/to/downloaded/{doc_type}/index.html",
                "parent_uuid": "doc-uuid-placeholder",
                "child_uuids": ["section-uuid-placeholder"],
                "metadata": {
                    "language": "html",
                    "source_url": sample_url,
                    "relative_path": "index.html",
                    "doc_type": doc_type
                }
            },
            "doc_section": {
                "uuid": "section-uuid-placeholder",
                "id": f"{doc_type}_section_0",
                "name": "Main Section",
                "type": "doc_section",
                "language": "html",
                "content": "Main section content",
                "file_path": f"/path/to/downloaded/{doc_type}/index.html",
                "parent_uuid": "doc-page-uuid-placeholder",
                "child_uuids": ["code-block-uuid-placeholder", "table-uuid-placeholder"],
                "metadata": {
                    "language": "html",
                    "level": 1,
                    "has_code": True,
                    "has_tables": True,
                    "section_hierarchy": ["Main Section"]
                }
            },
            "code_block": {
                "uuid": "code-block-uuid-placeholder",
                "id": f"{doc_type}_code_0",
                "name": "Example Code",
                "type": "code_block",
                "language": "python" if doc_type == "readthedocs" else "javascript",
                "content": "print('Hello, world!')" if doc_type == "readthedocs" else "console.log('Hello, world!');",
                "file_path": f"/path/to/downloaded/{doc_type}/index.html",
                "parent_uuid": "section-uuid-placeholder",
                "metadata": {
                    "language": "python" if doc_type == "readthedocs" else "javascript"
                }
            },
            "table": {
                "uuid": "table-uuid-placeholder",
                "id": f"{doc_type}_table_0",
                "name": "Example Table",
                "type": "table",
                "language": "html",
                "content": "{'headers': ['Column 1', 'Column 2'], 'rows': [['Value 1', 'Value 2'], ['Value 3', 'Value 4']]}",
                "file_path": f"/path/to/downloaded/{doc_type}/index.html",
                "parent_uuid": "section-uuid-placeholder",
                "metadata": {
                    "headers": ["Column 1", "Column 2"],
                    "rows": [["Value 1", "Value 2"], ["Value 3", "Value 4"]]
                }
            }
        }
    }
    
    # Fix placeholder UUIDs
    doc_uuid = expected_format["sample_blocks"]["documentation"]["uuid"]
    expected_format["sample_blocks"]["doc_page"]["parent_uuid"] = doc_uuid
    
    return expected_format


def create_function_expected_format(function_name: str, function_type: str) -> Dict[str, Any]:
    """
    Create an expected format template for specific function documentation.
    
    Args:
        function_name: Name of the function (e.g., "LENGTH", "ARRAY_INTERSECTION")
        function_type: Type of function (e.g., "string", "array")
        
    Returns:
        Dictionary containing expected format
    """
    # Define function-specific settings
    if function_name == "LENGTH":
        if function_type == "string":
            description = "Expected format for LENGTH string function documentation extraction"
            function_purpose = ["returns the length of a string", "calculates string length"]
            parameter_type = "string"
            example_code = 'RETURN LENGTH("hello world")'
            example_output = "11"
        else:  # array
            description = "Expected format for LENGTH array function documentation extraction"
            function_purpose = ["returns the number of elements in an array", "counts array elements"]
            parameter_type = "array"
            example_code = "RETURN LENGTH([1, 2, 3, 4, 5])"
            example_output = "5"
    elif function_name == "ARRAY_INTERSECTION":
        description = "Expected format for ARRAY_INTERSECTION function documentation extraction"
        function_purpose = ["returns the intersection of arrays", "finds common elements"]
        parameter_type = "array, array"
        example_code = "RETURN ARRAY_INTERSECTION([1, 2, 3], [2, 3, 4])"
        example_output = "[2, 3]"
    else:  # Generic function
        description = f"Expected format for {function_name} function documentation extraction"
        function_purpose = ["performs a function", "processes data"]
        parameter_type = "any"
        example_code = f"RETURN {function_name}(value)"
        example_output = "result"

    # Create the expected format structure
    expected_format = {
        "description": description,
        "version": "1.0",
        "expected_structure": {
            "required_block_types": [
                "documentation",
                "doc_page",
                "doc_section",
                "code_block",
                "table"
            ],
            "hierarchy": [
                {
                    "parent_type": "documentation",
                    "child_types": ["doc_page"]
                },
                {
                    "parent_type": "doc_page",
                    "child_types": ["doc_section"]
                },
                {
                    "parent_type": "doc_section",
                    "child_types": ["code_block", "table"]
                }
            ],
            "metadata_checks": [
                {
                    "field": "uuid",
                    "requirement": "uuid_format"
                },
                {
                    "field": "metadata.language",
                    "requirement": "not_empty"
                }
            ],
            "validation_threshold": 75
        },
        "expected_content_validation": {
            "function_name": function_name,
            "function_purpose": function_purpose,
            "parameters": [
                {
                    "name": "value",
                    "type": parameter_type,
                    "description": [
                        f"A {parameter_type} value",
                        f"input {parameter_type}"
                    ]
                }
            ],
            "return_type": "result",
            "examples": [
                {
                    "code": example_code,
                    "output": example_output
                }
            ],
            "required_keywords": [
                function_name,
                "function",
                "return",
                parameter_type
            ],
            "validation_threshold": 85
        },
        "structure_consistency": {
            "required_root_blocks": ["documentation"],
            "hierarchical_types": [
                {
                    "parent": "documentation",
                    "children": ["doc_page"]
                },
                {
                    "parent": "doc_page",
                    "children": ["doc_section"]
                },
                {
                    "parent": "doc_section",
                    "children": ["code_block", "table"]
                }
            ],
            "validation_threshold": 75
        },
        "sample_blocks": {
            "documentation": {
                "uuid": str(uuid.uuid4()),
                "id": f"docs_arangodb_{function_name.lower()}",
                "name": f"Documentation: arangodb_{function_name.lower()}",
                "type": "documentation",
                "language": "html",
                "content": f"Documentation site: https://docs.arangodb.com/stable/aql/functions/{function_type}/#{function_name.lower()}",
                "source_url": f"https://docs.arangodb.com/stable/aql/functions/{function_type}/#{function_name.lower()}",
                "child_uuids": ["doc-page-uuid-placeholder"],
                "metadata": {
                    "language": "html",
                    "source_url": f"https://docs.arangodb.com/stable/aql/functions/{function_type}/#{function_name.lower()}",
                    "doc_type": "arangodb"
                }
            },
            "doc_page": {
                "uuid": "doc-page-uuid-placeholder",
                "id": f"docs_arangodb_{function_name.lower()}_index.html",
                "name": "index.html",
                "type": "doc_page",
                "language": "html",
                "content": f"Documentation page for {function_name} function",
                "file_path": f"/path/to/downloaded/arangodb/{function_type}/{function_name.lower()}.html",
                "parent_uuid": "doc-uuid-placeholder",
                "child_uuids": ["section-uuid-placeholder"],
                "metadata": {
                    "language": "html",
                    "source_url": f"https://docs.arangodb.com/stable/aql/functions/{function_type}/#{function_name.lower()}",
                    "relative_path": f"aql/functions/{function_type}/{function_name.lower()}.html",
                    "doc_type": "arangodb"
                }
            },
            "function_section": {
                "uuid": "section-uuid-placeholder",
                "id": f"functions_arangodb_{function_name.lower()}_function",
                "name": function_name,
                "type": "doc_section",
                "language": "markdown",
                "content": f"# {function_name}\n\n{function_purpose[0]}.",
                "file_path": f"/path/to/downloaded/arangodb/{function_type}/{function_name.lower()}.html",
                "parent_uuid": "doc-page-uuid-placeholder",
                "child_uuids": ["code-block-uuid-placeholder", "table-uuid-placeholder"],
                "metadata": {
                    "language": "markdown",
                    "level": 1,
                    "has_code": True,
                    "has_tables": True,
                    "section_hierarchy": [function_name]
                }
            },
            "syntax_table": {
                "uuid": "table-uuid-placeholder",
                "id": f"{function_name.lower()}_syntax_table",
                "name": f"{function_name} Function Syntax",
                "type": "table",
                "language": "html",
                "content": f"{{'headers': ['Syntax', 'Description'], 'rows': [['{function_name}(value)', '{function_purpose[0]}']]}}",
                "file_path": f"/path/to/downloaded/arangodb/{function_type}/{function_name.lower()}.html",
                "parent_uuid": "section-uuid-placeholder",
                "metadata": {
                    "source": "markdown",
                    "headers": ["Syntax", "Description"],
                    "rows": [[f"{function_name}(value)", function_purpose[0]]]
                }
            },
            "example_code": {
                "uuid": "code-block-uuid-placeholder",
                "id": f"{function_name.lower()}_example",
                "name": f"{function_name} example",
                "type": "code_block",
                "language": "aql",
                "content": example_code + " // returns " + example_output,
                "file_path": f"/path/to/downloaded/arangodb/{function_type}/{function_name.lower()}.html",
                "parent_uuid": "section-uuid-placeholder",
                "metadata": {
                    "language": "aql",
                    "source": "html"
                }
            }
        }
    }
    
    # Fix placeholder UUIDs
    doc_uuid = expected_format["sample_blocks"]["documentation"]["uuid"]
    expected_format["sample_blocks"]["doc_page"]["parent_uuid"] = doc_uuid
    
    return expected_format


def generate_expected_formats() -> None:
    """Generate all expected format files."""
    # Create the expected formats directory if it doesn't exist
    expected_formats_dir = BASE_PATH / "expected_formats"
    expected_formats_dir.mkdir(exist_ok=True)
    
    # Generate documentation formats
    doc_types = ["arangodb", "readthedocs", "html", "markdown"]
    for doc_type in doc_types:
        format_data = create_documentation_expected_format(doc_type)
        file_path = expected_formats_dir / f"{doc_type}_expected_format.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(format_data, f, indent=2)
        logger.info(f"Generated {doc_type} expected format: {file_path}")
    
    # Generate function formats
    functions = [
        ("LENGTH", "string"),
        ("LENGTH", "array"),
        ("ARRAY_INTERSECTION", "array"),
        ("DOCUMENT", "document"),
        ("ATTRIBUTES", "document")
    ]
    
    for function_name, function_type in functions:
        format_data = create_function_expected_format(function_name, function_type)
        file_path = expected_formats_dir / f"{function_name.lower()}_{function_type}_expected_format.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(format_data, f, indent=2)
        logger.info(f"Generated {function_name} {function_type} expected format: {file_path}")
    
    # Copy existing formats if they already exist
    existing_formats = {
        "arangodb_expected_format.json": "arangodb_expected_format.json",
        "length_expected_format.json": "length_string_expected_format.json",
        "array_intersection_expected_format.json": "array_intersection_array_expected_format.json"
    }
    
    for src_name, dst_name in existing_formats.items():
        src_path = BASE_PATH / src_name
        dst_path = expected_formats_dir / dst_name
        if src_path.exists() and not dst_path.exists():
            with open(src_path, 'r', encoding='utf-8') as src_file:
                data = json.load(src_file)
                with open(dst_path, 'w', encoding='utf-8') as dst_file:
                    json.dump(data, dst_file, indent=2)
            logger.info(f"Copied existing format from {src_path} to {dst_path}")
    
    logger.info(f"Generated all expected format files in {expected_formats_dir}")


def update_test_files() -> None:
    """Update test files to use the correct expected formats."""
    # Find all test files
    test_files = list(BASE_PATH.glob("*test*.py"))
    test_files.extend(BASE_PATH.glob("test_*.py"))
    
    # Create the expected formats directory path
    expected_formats_dir = BASE_PATH / "expected_formats"
    
    for test_file in test_files:
        logger.info(f"Updating test file: {test_file}")
        
        # Read the file contents
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip files that don't need updating
        if "expected_format" not in content:
            logger.info(f"  No expected_format references found, skipping")
            continue
        
        # Determine the appropriate expected format file
        format_file = None
        if "arangodb" in test_file.name.lower():
            format_file = "arangodb_expected_format.json"
        elif "length" in test_file.name.lower():
            format_file = "length_string_expected_format.json"
        elif "array_intersection" in test_file.name.lower():
            format_file = "array_intersection_array_expected_format.json"
        elif "readthedocs" in test_file.name.lower():
            format_file = "readthedocs_expected_format.json"
        elif "html" in test_file.name.lower():
            format_file = "html_expected_format.json"
        elif "markdown" in test_file.name.lower():
            format_file = "markdown_expected_format.json"
        
        if format_file:
            # Check if the expected format file exists in the expected_formats directory
            if (expected_formats_dir / format_file).exists():
                format_path = str(expected_formats_dir / format_file)
                
                # Update references to expected format files
                new_content = content
                
                # Update hardcoded paths
                if "expected_format_path" in new_content:
                    new_content = new_content.replace(
                        'expected_format_path = os.path.join(current_dir, "expected_format.json")',
                        f'expected_format_path = os.path.join(current_dir, "expected_formats/{format_file}")'
                    )
                    new_content = new_content.replace(
                        'expected_format_path = os.path.join(current_dir, "length_expected_format.json")',
                        f'expected_format_path = os.path.join(current_dir, "expected_formats/{format_file}")'
                    )
                    new_content = new_content.replace(
                        'expected_format_path = os.path.join(current_dir, "array_intersection_expected_format.json")',
                        f'expected_format_path = os.path.join(current_dir, "expected_formats/{format_file}")'
                    )
                    new_content = new_content.replace(
                        'expected_format_path = os.path.join(current_dir, "arangodb_expected_format.json")',
                        f'expected_format_path = os.path.join(current_dir, "expected_formats/{format_file}")'
                    )
                
                # Update argparse defaults
                if "--expected" in new_content:
                    new_content = new_content.replace(
                        'default="/home/grahama/workspace/experiments/agent_tools/test_repos/samples/deepseek_markdown_extraction_example.json"',
                        f'default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_formats/{format_file}")'
                    )
                
                # Write updated content if changed
                if new_content != content:
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    logger.info(f"  Updated expected format path to {format_file}")
                else:
                    logger.info(f"  No updates needed")
            else:
                logger.warning(f"  Expected format file {format_file} not found in {expected_formats_dir}")
        else:
            logger.warning(f"  Could not determine appropriate expected format for {test_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate expected formats for tests")
    parser.add_argument("--update-tests", action="store_true", help="Update test files to use the generated formats")
    args = parser.parse_args()
    
    # Generate expected formats
    generate_expected_formats()
    
    # Update test files if requested
    if args.update_tests:
        update_test_files()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()