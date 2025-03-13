"""
Global test fixtures and configuration for cursor_rules tests.

This file contains fixtures and setup that should be available to all tests.
Import patterns defined here should be followed across all test files.
"""

# Environment setup must be first
from agent_tools.cursor_rules.utils.helpers.file_utils import load_env_file
load_env_file()

# Standard library imports
import asyncio
import os
import tempfile
import shutil
from pathlib import Path

# Third-party imports
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch

# Import custom exception for database connection issues
from agent_tools.cursor_rules.utils.exceptions import DatabaseConnectionError

# Define test directories for easier reference
TEST_ROOT = Path(__file__).parent
FIXTURES_DIR = TEST_ROOT / "models" / "fixtures"

# Use the recommended approach for setting the default event loop scope
def pytest_configure(config):
    """Configure pytest-asyncio to use module-scoped event loops by default."""
    config.option.asyncio_mode = "auto"
    
    # Set the default loop scope to module to match our database fixtures
    pytest.asyncio_loop_scope = "module"

# Example of a shared fixture for database testing
@pytest_asyncio.fixture(scope="function")
async def test_db():
    """Fixture to create a test database connection.
    
    Uses asyncio.to_thread to prevent blocking the event loop when
    connecting to the ArangoDB database.
    """
    from agent_tools.cursor_rules.utils.db import setup_database
    
    # Use a unique database name for testing to avoid conflicts
    db_name = f"cursor_rules_test_{os.getpid()}"
    
    try:
        # Create database asynchronously
        db = await asyncio.to_thread(
            setup_database,
            db_name,
            "http://localhost:8529",
            "root",
            "openSesame"
        )
        yield db
        
        # Clean up: drop the test database after the test completes
        if await asyncio.to_thread(db.has_database, db_name):
            await asyncio.to_thread(db.drop)
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {e}") 