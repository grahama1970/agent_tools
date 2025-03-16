#!/usr/bin/env python3
"""
test_pipeline.py

Official Documentation:
- pytest: https://docs.pytest.org/
- pathlib: https://docs.python.org/3/library/pathlib.html

This module contains tests that run against real HTML files to ensure the pipeline extracts content as expected.
"""

import json
from pathlib import Path
import pytest
from main import process_file

# Ensure that a sample HTML file exists in the tests/sample_data directory.
SAMPLE_FILE = Path("tests/sample_data/sample.html")

@pytest.fixture(scope="module")
def sample_extraction():
    assert SAMPLE_FILE.exists(), f"Sample file {SAMPLE_FILE} does not exist."
    return process_file(SAMPLE_FILE)

def test_file_hierarchy(sample_extraction):
    # Blind check: the file path should match the sample file location.
    assert "file" in sample_extraction
    assert SAMPLE_FILE.as_posix() in sample_extraction["file"]

def test_sections_extracted(sample_extraction):
    # Check that at least one section was extracted.
    sections = sample_extraction.get("sections", [])
    assert isinstance(sections, list)
    assert len(sections) > 0, "No sections extracted from the sample file."

def test_token_count(sample_extraction):
    # Check that each section has a non-zero token count.
    for sec in sample_extraction["sections"]:
        assert sec.get("token_count", 0) > 0, f"Section '{sec.get('header', '')}' has zero tokens."

def test_output_structure(sample_extraction):
    # Blind-check that the JSON structure includes expected keys.
    expected_keys = {"file", "sections"}
    assert expected_keys.issubset(sample_extraction.keys())
    
    # Write the blind-check output to a file for manual verification if needed.
    output_path = Path("tests/sample_data/sample_extraction_output.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sample_extraction, f, indent=2)

if __name__ == "__main__":
    pytest.main()
