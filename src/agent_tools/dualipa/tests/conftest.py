"""
Common test fixtures for dualipa tests.

Official Documentation References:
- pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- tempfile: https://docs.python.org/3/library/tempfile.html
- os: https://docs.python.org/3/library/os.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- shutil: https://docs.python.org/3/library/shutil.html
"""

import pytest
import os
import json
import tempfile
import shutil
import asyncio
from pathlib import Path


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest-asyncio to use the 'strict' mode."""
    config.option.asyncio_mode = "strict"


@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data that persists for the session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def temp_directory():
    """Create a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir) 