#!/usr/bin/env python3
"""
Test file for the rule_search_cli module.
Tests the actual behavior of CLI commands with real database operations.

Documentation References:
- Click: https://click.palletsprojects.com/
- pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- python-arango: https://python-arango.readthedocs.io/
"""

import pytest
import asyncio
from click.testing import CliRunner
from datetime import datetime

from agent_tools.cursor_rules.cli.rule_search_cli import cli, _user_query_command, _reasoning_task_command
from agent_tools.cursor_rules.core.cursor_rules import setup_cursor_rules_db

@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()

@pytest.fixture(scope="module")
async def db():
    """Setup a test database connection.
    Using module scope to match the event_loop fixture from conftest.py"""
    config = {'host': 'http://localhost:8529', 'username': 'root', 'password': 'openSesame'}
    db = await asyncio.to_thread(setup_cursor_rules_db, config, 'test_db')
    yield db
    # Cleanup will happen automatically when the database connection closes

@pytest.mark.asyncio
async def test_user_query_command_async(db):
    """Test the user-query command with actual database operations."""
    query = 'How should I handle database operations?'
    config = {'host': 'http://localhost:8529', 'username': 'root', 'password': 'openSesame'}
    
    # Call the command and capture output
    output = []
    def mock_echo(message):
        output.append(message)
    
    import click
    original_echo = click.echo
    click.echo = mock_echo
    try:
        await _user_query_command(config, 'test_db', query, 5)
    finally:
        click.echo = original_echo
    
    # Verify the output contains relevant information
    result = '\n'.join(output)
    assert 'Rules related to' in result
    assert 'database operations' in result.lower()

@pytest.mark.asyncio
async def test_reasoning_task_command_async(db):
    """Test the reasoning-task command with actual database operations."""
    task = 'I need to implement error handling'
    config = {'host': 'http://localhost:8529', 'username': 'root', 'password': 'openSesame'}
    
    # Call the command and capture output
    output = []
    def mock_echo(message):
        output.append(message)
    
    import click
    original_echo = click.echo
    click.echo = mock_echo
    try:
        await _reasoning_task_command(config, 'test_db', task, 5)
    finally:
        click.echo = original_echo
    
    # Verify the output contains relevant information
    result = '\n'.join(output)
    assert 'Rules related to' in result
    assert 'error handling' in result.lower()

# Non-async tests don't need the asyncio mark
def test_cli_user_query(runner):
    """Test the CLI interface for user-query."""
    result = runner.invoke(cli, ['user-query', 'How should I handle database operations?'])
    assert result.exit_code == 0
    assert 'Rules related to' in result.output

def test_cli_reasoning_task(runner):
    """Test the CLI interface for reasoning-task."""
    result = runner.invoke(cli, ['reasoning-task', 'I need to implement error handling'])
    assert result.exit_code == 0
    assert 'Rules related to' in result.output

def test_invalid_command(runner):
    """Test behavior with invalid command."""
    result = runner.invoke(cli, ['invalid-command'])
    assert result.exit_code != 0

def test_empty_query(runner):
    """Test behavior with empty query."""
    result = runner.invoke(cli, ['user-query', ''])
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_search_test_failures_command(runner, db):
    """Test the search-test-failures command."""
    # First ensure we have some test data
    test_failures = db.collection('test_failures')
    test_data = {
        "test_name": "test_async_operation",
        "error_message": "RuntimeError: Event loop is closed",
        "analysis": "The async operation failed due to event loop closure",
        "timestamp": datetime.now().isoformat()
    }
    test_failures.insert(test_data)
    
    # Test table format (default)
    result = runner.invoke(cli, ['search-test-failures', 'event loop closed'])
    assert result.exit_code == 0
    assert 'Test Name' in result.output
    assert 'Error Message' in result.output
    assert 'Analysis' in result.output
    assert 'Relevance' in result.output
    assert 'test_async_operation' in result.output
    assert 'Event loop is closed' in result.output
    
    # Test JSON format
    result = runner.invoke(cli, ['search-test-failures', 'event loop closed', '--format', 'json'])
    assert result.exit_code == 0
    assert '"test_name": "test_async_operation"' in result.output
    assert '"error_message": "RuntimeError: Event loop is closed"' in result.output
    assert '"score":' in result.output  # Verify BM25 score is included
    
    # Test with no matches
    result = runner.invoke(cli, ['search-test-failures', 'nonexistent xyz'])
    assert result.exit_code == 0
    assert 'No matching test failures found.' in result.output 