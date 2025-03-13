#!/usr/bin/env python3
"""
CLI command to demonstrate the rule search functionality.

This module provides a CLI command to search for rules related to user queries
or reasoning tasks.

Documentation References:
- Click: https://click.palletsprojects.com/
- pytest: https://docs.pytest.org/
- python-arango: https://python-arango.readthedocs.io/
- tabulate: https://github.com/astanin/python-tabulate
"""

import asyncio
import click
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from tabulate import tabulate
from ..core.cursor_rules import setup_cursor_rules_db
from ..utils.rule_search import (
    search_for_user_query,
    search_for_reasoning_task,
    get_related_rules,
    format_rules_for_agent
)
from ..utils.test_state import (
    store_test_state,
    get_test_state,
    get_all_test_states,
    ensure_test_collections,
    _search_test_failures
)

def run_async(coro):
    """Run an async function in the event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

@click.group()
@click.option('--host', default="http://localhost:8529", help="ArangoDB host")
@click.option('--username', default="root", help="ArangoDB username")
@click.option('--password', default="openSesame", help="ArangoDB password")
@click.option('--db-name', default="cursor_rules", help="Database name")
@click.pass_context
def cli(ctx, host, username, password, db_name):
    """CLI command to search for rules related to user queries or reasoning tasks."""
    # Store connection config in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = {
        'host': host,
        'username': username, 
        'password': password
    }
    ctx.obj['db_name'] = db_name

@cli.command()
@click.argument('query')
@click.option('--limit', type=int, default=5, help="Maximum number of results")
@click.pass_context
def user_query(ctx, query, limit):
    """Search for rules related to a user query."""
    if not query.strip():
        raise click.UsageError("Query cannot be empty")
    
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Run the async command
    asyncio.run(_user_query_command(config, db_name, query, limit))

async def _user_query_command(config, db_name, query, limit):
    """Async implementation of the user_query command."""
    try:
        # Validate query
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Connect to the database
        db = await asyncio.to_thread(setup_cursor_rules_db, config, db_name=db_name)
        
        # Search for rules related to the user query
        formatted = await search_for_user_query(db, query, limit=limit)
        
        # Print the formatted results
        click.echo(formatted)
        
        return formatted
        
    except Exception as e:
        click.echo(f"Error searching for rules: {e}")
        raise

@cli.command()
@click.argument('task')
@click.option('--limit', type=int, default=5, help="Maximum number of results")
@click.pass_context
def reasoning_task(ctx, task, limit):
    """Search for rules related to a reasoning task."""
    if not task.strip():
        raise click.UsageError("Task description cannot be empty")
        
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Run the async command
    asyncio.run(_reasoning_task_command(config, db_name, task, limit))

async def _reasoning_task_command(config, db_name, task, limit):
    """Async implementation of the reasoning_task command."""
    try:
        # Validate task
        if not task.strip():
            raise ValueError("Task description cannot be empty")
            
        # Connect to the database
        db = await asyncio.to_thread(setup_cursor_rules_db, config, db_name=db_name)
        
        # Search for rules related to the reasoning task
        formatted = await search_for_reasoning_task(db, task, limit=limit)
        
        # Print the formatted results
        click.echo(formatted)
        
        return formatted
        
    except Exception as e:
        click.echo(f"Error searching for rules: {e}")
        raise

@cli.command()
@click.argument('tag_name')
@click.option('--total', type=int, required=True, help="Total number of tests")
@click.option('--passed', type=int, required=True, help="Number of passed tests")
@click.option('--failed', type=int, required=True, help="Number of failed tests")
@click.option('--failing-tests', required=True, help="Comma-separated list of failing test names")
@click.option('--cli-implementation', default="rule_search_cli.py", help="CLI implementation used")
@click.option('--notes', help="Additional notes about this test state")
@click.pass_context
def store_test_state_cmd(ctx, tag_name, total, passed, failed, failing_tests, cli_implementation, notes):
    """Store a test state in the database.
    
    This command allows the AI agent to record test state information for future reference.
    """
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Convert comma-separated failing tests to list
    failing_tests_list = [test.strip() for test in failing_tests.split(',') if test.strip()]
    
    # Prepare test results
    test_results = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "failing_tests": failing_tests_list,
        "cli_implementation": cli_implementation
    }
    
    # Run the async command
    asyncio.run(_store_test_state_command(config, db_name, tag_name, test_results, notes))

async def _store_test_state_command(config, db_name, tag_name, test_results, notes):
    """Async implementation of the store_test_state command."""
    try:
        # Connect to the database
        db = await asyncio.to_thread(setup_cursor_rules_db, config, db_name=db_name)
        
        # Store the test state
        result = await store_test_state(db, tag_name, test_results, notes)
        
        # Print confirmation
        click.echo(f"Test state for tag '{tag_name}' stored successfully with ID: {result['_id']}")
        timestamp = datetime.now().isoformat()
        click.echo(f"Timestamp: {timestamp}")
        
        # Display test results in a table
        headers = ["Status", "Count"]
        table_data = [
            ["Total Tests", test_results['total']],
            ["Passed", test_results['passed']],
            ["Failed", test_results['failed']]
        ]
        click.echo("\nTest Results:")
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Display failing tests if any
        if test_results['failing_tests']:
            click.echo("\nFailing Tests:")
            failing_table = [[i+1, test] for i, test in enumerate(test_results['failing_tests'])]
            click.echo(tabulate(failing_table, headers=["#", "Test Name"], tablefmt="grid"))
        
    except Exception as e:
        click.echo(f"Error storing test state: {e}")
        raise

@cli.command()
@click.option('--tag-name', help="Filter by git tag name")
@click.option('--limit', type=int, default=1, help="Maximum number of test states to retrieve")
@click.option('--all', is_flag=True, help="Retrieve all test states")
@click.option('--format', 'output_format', type=click.Choice(['table', 'text', 'json']), default='table', help="Output format")
@click.pass_context
def get_test_state_cmd(ctx, tag_name, limit, all, output_format):
    """Retrieve test states from the database.
    
    This command allows the AI agent to retrieve previously stored test state information.
    """
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Run the async command
    asyncio.run(_get_test_state_command(config, db_name, tag_name, limit, all, output_format))

async def _get_test_state_command(config, db_name, tag_name, limit, all, output_format):
    """Async implementation of the get_test_state command."""
    try:
        # Connect to the database
        db = await asyncio.to_thread(setup_cursor_rules_db, config, db_name=db_name)
        
        # Get test states
        if all:
            test_states = await get_all_test_states(db)
        else:
            test_states = await get_test_state(db, tag_name, limit)
        
        # Print results
        if not test_states:
            click.echo("No test states found.")
            return
        
        if output_format == 'json':
            # JSON format
            click.echo(json.dumps(test_states, indent=2))
        
        elif output_format == 'table':
            # Table format (default)
            click.echo("\n=== Test States Summary ===")
            
            # Summary table
            headers = ["#", "Tag", "Timestamp", "Total", "Passed", "Failed", "Pass Rate"]
            table_data = []
            
            for i, state in enumerate(test_states):
                timestamp = state.get('timestamp', 'N/A')
                total = state.get('tests_total', 0)
                passed = state.get('tests_passed', 0)
                failed = state.get('tests_failed', 0)
                
                # Calculate pass rate
                pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"
                
                # Add row to table
                table_data.append([
                    i+1,
                    state.get('tag_name', 'N/A'),
                    timestamp,
                    total,
                    passed,
                    failed,
                    pass_rate
                ])
            
            click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Detailed information for each test state
            for i, state in enumerate(test_states):
                click.echo(f"\n=== Test State #{i+1} Details ===")
                
                # Basic information
                basic_info = [
                    ["Tag", state.get('tag_name', 'N/A')],
                    ["Timestamp", state.get('timestamp', 'N/A')],
                    ["CLI Implementation", state.get('cli_implementation', 'N/A')],
                ]
                click.echo(tabulate(basic_info, tablefmt="simple"))
                
                # Test results
                total = state.get('tests_total', 0)
                passed = state.get('tests_passed', 0)
                failed = state.get('tests_failed', 0)
                
                test_results = [
                    ["Total Tests", total],
                    ["Passed", passed],
                    ["Failed", failed],
                    ["Pass Rate", f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"]
                ]
                click.echo("\nTest Results:")
                click.echo(tabulate(test_results, tablefmt="grid"))
                
                # Failing tests
                if state.get('failing_tests'):
                    click.echo("\nFailing Tests:")
                    failing_table = [[i+1, test] for i, test in enumerate(state['failing_tests'])]
                    click.echo(tabulate(failing_table, headers=["#", "Test Name"], tablefmt="grid"))
                
                # Notes
                if state.get('notes'):
                    click.echo("\nNotes:")
                    click.echo(state['notes'])
                
                click.echo("\n" + "=" * 50)
        
        else:
            # Simple text format
            for i, state in enumerate(test_states):
                click.echo(f"\nTest State {i+1}:")
                click.echo("=" * 50)
                click.echo(f"Tag: {state.get('tag_name', 'N/A')}")
                click.echo(f"Timestamp: {state.get('timestamp', 'N/A')}")
                click.echo(f"Tests total: {state.get('tests_total', 0)}, passed: {state.get('tests_passed', 0)}, failed: {state.get('tests_failed', 0)}")
                
                if state.get('failing_tests'):
                    click.echo("\nFailing tests:")
                    for test in state['failing_tests']:
                        click.echo(f"  - {test}")
                
                if state.get('notes'):
                    click.echo(f"\nNotes: {state['notes']}")
                click.echo("=" * 50)
        
    except Exception as e:
        click.echo(f"Error retrieving test states: {e}")
        raise

@cli.command(name='search-test-failures')
@click.argument('query')
@click.option('--limit', type=int, default=5, help="Maximum number of results")
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help="Output format")
@click.pass_context
def search_test_failures_command(ctx, query, limit, format):
    """Search for test failures using hybrid search.
    
    Args:
        QUERY: Search query to find relevant test failures
    """
    run_async(_search_test_failures_command(ctx, query, limit, format))

async def _search_test_failures_command(ctx, query, limit, format):
    """Async implementation of search_test_failures command."""
    # Get database connection from context
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Connect to database
    db = await asyncio.to_thread(setup_cursor_rules_db, config, db_name)
    
    # Ensure collections exist and view is updated with the right analyzers
    await ensure_test_collections(db)
    
    # Use the ArangoSearch view with proper text analyzers
    view_name = "test_states_view"
    
    # Create AQL query
    aql_query = """
    FOR doc IN test_states_view 
    SEARCH ANALYZER(doc.test_name IN TOKENS(@query, "text_en"), "text_en")
    OR ANALYZER(doc.error_message IN TOKENS(@query, "text_en"), "text_en")
    OR ANALYZER(doc.analysis IN TOKENS(@query, "text_en"), "text_en")
    FILTER IS_DOCUMENT(doc) AND STARTS_WITH(doc._id, 'test_failures/')
    SORT BM25(doc) DESC 
    LIMIT 10
    RETURN { 
        "doc": doc,
        "score": BM25(doc) 
    }
    """
    
    bind_vars = {"query": query}
    
    # Execute query using asyncio.to_thread
    cursor = await asyncio.to_thread(db.aql.execute, aql_query, bind_vars=bind_vars)
    search_results = await asyncio.to_thread(list, cursor)
    
    if not search_results:
        click.echo("No matching test failures found.")
        return
    
    if format == 'json':
        click.echo(json.dumps(search_results, indent=2))
        return
    
    # Format output using tabulate for table format
    table_data = []
    headers = ["Test Name", "Timestamp", "Error Message", "Analysis", "Relevance"]
    
    for result in search_results:
        doc = result["doc"]
        score = result["score"]
        
        table_data.append([
            doc.get("test_name", "Unknown"),
            doc.get("timestamp", "Unknown"),
            doc.get("error_message", "Unknown")[:50] + "..." if len(doc.get("error_message", "")) > 50 else doc.get("error_message", "Unknown"),
            doc.get("analysis", "Unknown")[:50] + "..." if len(doc.get("analysis", "")) > 50 else doc.get("analysis", "Unknown"),
            f"{score:.2f}"
        ])
    
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == '__main__':
    cli() 