#!/usr/bin/env python3
"""
test_end_to_end.py

Official Documentation:
- pytest: https://docs.pytest.org/
- json (Python standard library): https://docs.python.org/3/library/json.html
- pathlib: https://docs.python.org/3/library/pathlib.html

This module contains blind, end-to-end tests for the fetch_docs pipeline. It verifies that processing
sample documentation pages (for example, ArangoDB AQL documentation) produces an ordered JSON output that
matches the verified expected result. These tests use real, verified data (not mocks) to help the agent reason
about how functions are related and why they might fail.
"""

import json
from pathlib import Path
import pytest

# Import the processing function from our main module.
from agent_tools.fetch_docs.main import process_directory

@pytest.fixture(scope="module")
def expected_arangodb_aql():
    """
    Load the verified expected JSON output for the ArangoDB AQL documentation page.
    This file should exist in tests/expected_results/arangodb_aql.json.
    """
    expected_file = Path("tests/expected_results/arangodb_aql.json")
    assert expected_file.exists(), f"Expected results file not found: {expected_file}"
    with expected_file.open("r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def sample_arangodb_aql_dir():
    """
    Provide the path to the sample input directory for the ArangoDB AQL documentation page.
    The sample HTML files should be in tests/sample_data/arangodb_aql/.
    """
    sample_dir = Path("tests/sample_data/arangodb_aql")
    assert sample_dir.exists(), f"Sample data directory not found: {sample_dir}"
    return sample_dir

def sort_recursive(item):
    """
    Recursively sort lists within a JSON-like structure.
    This helps make blind comparisons robust against nonessential ordering differences.
    """
    if isinstance(item, dict):
        return {k: sort_recursive(v) for k, v in item.items()}
    elif isinstance(item, list):
        # Sort each element (by its JSON string) and return a sorted list.
        return sorted([sort_recursive(x) for x in item], key=lambda x: json.dumps(x, sort_keys=True))
    else:
        return item

def test_arangodb_aql_end_to_end(expected_arangodb_aql, sample_arangodb_aql_dir):
    """
    End-to-end blind test for processing the ArangoDB AQL documentation page.
    
    This test processes the sample HTML files in the sample_data directory and compares the resulting JSON output
    with the verified expected output stored in expected_results. The comparison is performed on sorted
    (recursively normalized) data to account for nonessential ordering differences.
    """
    # Run the processing function on the sample input directory.
    actual_data = process_directory(sample_arangodb_aql_dir)
    
    # Recursively sort both actual and expected data.
    sorted_actual = sort_recursive(actual_data)
    sorted_expected = sort_recursive(expected_arangodb_aql)
    
    assert sorted_actual == sorted_expected, (
        "The actual processed JSON does not match the expected blind result. "
        "Please verify that the sample data and expected output are correct."
    )

def test_json_format():
    """
    Smoke test to ensure that the process_directory function produces well-formed JSON output.
    
    This test processes any sample HTML files available in tests/sample_data and tries to serialize the output.
    If the output cannot be serialized, the test fails.
    """
    sample_dir = Path("tests/sample_data")
    html_files = list(sample_dir.rglob("*.html"))
    if not html_files:
        pytest.skip("No sample HTML files available in tests/sample_data")
    
    actual_data = process_directory(sample_dir)
    
    try:
        json_output = json.dumps(actual_data, indent=2)
    except Exception as e:
        pytest.fail(f"Output is not well-formed JSON: {e}")
