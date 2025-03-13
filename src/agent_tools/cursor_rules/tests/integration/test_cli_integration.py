#!/usr/bin/env python3
"""
Comprehensive integration test for all CLI commands with a real ArangoDB instance.

This script tests all CLI commands in the enhanced_cli.py file with a real ArangoDB instance,
ensuring that they work correctly in a non-mocked environment.

Documentation references:
- ArangoDB Python Driver: https://python-driver.arangodb.com/
- Click CLI: https://click.palletsprojects.com/en/8.1.x/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Any, Tuple

import click
from loguru import logger
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the required setup function
from agent_tools.cursor_rules.core.enhanced_db import setup_enhanced_cursor_rules_db
from agent_tools.cursor_rules.core.ai_knowledge_db import setup_ai_knowledge_db


async def run_cli_command_async(command: str) -> Tuple[int, str, str]:
    """
    Run a CLI command asynchronously and return the exit code, stdout, and stderr.
    
    Args:
        command: The command to run
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    logger.info(f"Running command: {command}")
    
    # Use asyncio.subprocess to avoid blocking
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True
    )
    
    stdout, stderr = await process.communicate()
    exit_code = process.returncode
    
    return exit_code, stdout.decode(), stderr.decode()


def run_cli_command(command: str) -> Tuple[int, str, str]:
    """
    Synchronous wrapper for run_cli_command_async.
    
    Args:
        command: The command to run
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    return asyncio.run(run_cli_command_async(command))


async def check_command_success_async(command: str) -> bool:
    """
    Check if a CLI command runs successfully.
    
    Args:
        command: The command to run
        
    Returns:
        True if the command ran successfully, False otherwise
    """
    exit_code, stdout, stderr = await run_cli_command_async(command)
    
    if exit_code == 0:
        logger.info(f"Command succeeded: {command}")
        logger.debug(f"Output: {stdout}")
        return True
    else:
        logger.error(f"Command failed: {command}")
        logger.error(f"Exit code: {exit_code}")
        logger.error(f"stderr: {stderr}")
        return False


def check_command_success(command: str) -> bool:
    """
    Synchronous wrapper for check_command_success_async.
    
    Args:
        command: The command to run
        
    Returns:
        True if the command ran successfully, False otherwise
    """
    return asyncio.run(check_command_success_async(command))


async def direct_db_check_async(host: str, username: str, password: str, db_name: str) -> bool:
    """
    Directly check if the database can be accessed correctly using the async pattern
    from LESSONS_LEARNED.md.
    
    Args:
        host: ArangoDB host
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name
        
    Returns:
        True if the database can be accessed, False otherwise
    """
    try:
        # Following the pattern from LESSONS_LEARNED.md
        config = {
            "arango_config": {
                "hosts": [host],
                "username": username,
                "password": password
            }
        }
        
        # ALWAYS use this exact pattern from LESSONS_LEARNED.md
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name=db_name
        )
        
        # Check if we can access the collections
        collections = await asyncio.to_thread(db.collections)
        collections_list = await asyncio.to_thread(list, collections)
        logger.info(f"Successfully connected to database. Found {len(collections_list)} collections.")
        
        # Try to get the rules collection
        if await asyncio.to_thread(db.has_collection, "rules"):
            rules_collection = await asyncio.to_thread(db.collection, "rules")
            all_rules = await asyncio.to_thread(rules_collection.all)
            rules_count = await asyncio.to_thread(list, all_rules)
            logger.info(f"Rules collection exists with {len(rules_count)} documents.")
        else:
            logger.warning("Rules collection does not exist.")
        
        return True
    except Exception as e:
        logger.error(f"Error accessing database directly: {e}")
        return False


def direct_db_check(host: str, username: str, password: str, db_name: str) -> bool:
    """
    Synchronous wrapper for direct_db_check_async.
    
    Args:
        host: ArangoDB host
        username: ArangoDB username
        password: ArangoDB password
        db_name: Database name
        
    Returns:
        True if the database can be accessed, False otherwise
    """
    return asyncio.run(direct_db_check_async(host, username, password, db_name))


@pytest.mark.asyncio
async def test_setup_command_async() -> bool:
    """Test the setup command"""
    logger.info("Testing setup command")
    
    # Reset the database to ensure a clean slate
    cmd = "python -m src.agent_tools.cursor_rules.enhanced_cli setup --reset"
    return await check_command_success_async(cmd)


def test_setup_command() -> bool:
    """Synchronous wrapper for test_setup_command_async."""
    return asyncio.run(test_setup_command_async())


@pytest.mark.asyncio
async def test_search_command_async() -> bool:
    """Test the search command"""
    logger.info("Testing search command")
    
    # Search for a general term that should return results
    cmd = "python -m src.agent_tools.cursor_rules.enhanced_cli search 'code'"
    return await check_command_success_async(cmd)


def test_search_command() -> bool:
    """Synchronous wrapper for test_search_command_async."""
    return asyncio.run(test_search_command_async())


@pytest.mark.asyncio
async def test_list_command_async() -> bool:
    """Test the list command"""
    logger.info("Testing list command")
    
    cmd = "python -m src.agent_tools.cursor_rules.enhanced_cli list"
    return await check_command_success_async(cmd)


def test_list_command() -> bool:
    """Synchronous wrapper for test_list_command_async."""
    return asyncio.run(test_list_command_async())


async def run_all_tests_async(host: str, username: str, password: str, db_name: str) -> Dict[str, bool]:
    """
    Run all integration tests and return the results.
    
    Returns:
        Dictionary mapping test names to boolean results
    """
    results = {}
    
    # First do a direct database check
    results["direct_db_check"] = await direct_db_check_async(host, username, password, db_name)
    
    # Test setup first as other commands depend on it
    results["setup"] = await test_setup_command_async()
    
    # Only proceed with other tests if setup succeeded
    if results["setup"]:
        # Give the database a moment to initialize
        await asyncio.sleep(1)
        
        # Run tests for other commands
        results["list"] = await test_list_command_async()
        results["search"] = await test_search_command_async()
    else:
        logger.error("Setup failed, skipping other tests")
        results.update({
            "list": False,
            "search": False
        })
    
    return results


def run_all_tests(host: str, username: str, password: str, db_name: str) -> Dict[str, bool]:
    """Synchronous wrapper for run_all_tests_async."""
    return asyncio.run(run_all_tests_async(host, username, password, db_name))


def print_results(results: Dict[str, bool]) -> None:
    """
    Print the test results in a formatted way.
    
    Args:
        results: Dictionary mapping test names to boolean results
    """
    click.echo("\n=== CLI Integration Test Results ===")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        color = "green" if passed else "red"
        click.echo(f"  {name}: {click.style(status, fg=color)}")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        click.echo(click.style("\nAll tests passed!", fg="green", bold=True))
    else:
        click.echo(click.style("\nSome tests failed!", fg="red", bold=True))


@click.command()
@click.option("--host", default="http://localhost:8529", help="ArangoDB host")
@click.option("--username", default="root", help="ArangoDB username")
@click.option("--password", default="openSesame", help="ArangoDB password")
@click.option("--db-name", default="cursor_rules_enhanced", help="Database name")
@click.option("--verbose", "-v", is_flag=True, help="Display verbose output")
def main(host, username, password, db_name, verbose):
    """Run integration tests for all CLI commands with a real ArangoDB instance."""
    # Set environment variables for the CLI commands
    os.environ["ARANGO_HOST"] = host
    os.environ["ARANGO_USERNAME"] = username
    os.environ["ARANGO_PASSWORD"] = password
    os.environ["ARANGO_DB_NAME"] = db_name
    
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    click.echo(f"Testing CLI commands with ArangoDB at {host}")
    click.echo(f"Database: {db_name}, Username: {username}")
    
    # Run all tests
    results = run_all_tests(host, username, password, db_name)
    
    # Print results
    print_results(results)
    
    # Exit with appropriate status code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main() 