"""Test fixtures and configuration for QA generation tests.

This module provides common fixtures for QA generation tests, including:
- Sample document sections (markdown and code)
- Sample extraction JSON with metadata
- File fixtures for testing I/O operations

Behavioral Summary:
1. The fixtures provide realistic mock data for testing QA generation
2. This includes multisection content with UUID identification
3. Code and documentation sections are handled separately
4. The fixtures prepare both in-memory data structures and temp files
5. All fixtures support async test operations via pytest-asyncio

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- json: https://docs.python.org/3/library/json.html
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

import os
import json
import pytest
import asyncio
from typing import Dict, Any, List
from pathlib import Path

# Sample data for tests
SAMPLE_SECTION = {
    "uuid": "93e7517c-78f7-44ce-90bc-4a32c25ad877",
    "type": "documentation",
    "content": "## Feature Overview\nThis module provides QA generation capabilities.",
    "extraction_focus": "technical details",
    "summary_instructions": "Generate 3 QA pairs focusing on API usage"
}

SAMPLE_CODE_SECTION = {
    "uuid": "a4e8629d-12b3-40de-89bc-1a45d62a9c8e",
    "type": "code",
    "content": """
def calculate_average(numbers):
    # Calculate the average of a list of numbers
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
""",
    "extraction_focus": "implementation details",
    "summary_instructions": "Generate QA pairs focusing on code functionality"
}

SAMPLE_EXTRACTION_JSON = {
    "sections": [SAMPLE_SECTION, SAMPLE_CODE_SECTION],
    "extraction_metadata": {
        "model_used": "gpt-4-turbo",
        "timestamp": "2025-03-19T14:49:00Z"
    }
}


@pytest.fixture
def sample_section():
    """Return a sample documentation section."""
    return SAMPLE_SECTION


@pytest.fixture
def sample_code_section():
    """Return a sample code section."""
    return SAMPLE_CODE_SECTION


@pytest.fixture
def sample_extraction_json():
    """Return a sample extraction JSON."""
    return SAMPLE_EXTRACTION_JSON


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file with sample extraction data."""
    json_file = tmp_path / "extraction.json"
    with open(json_file, 'w') as f:
        json.dump(SAMPLE_EXTRACTION_JSON, f)
    return json_file