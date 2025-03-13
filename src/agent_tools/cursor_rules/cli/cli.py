#!/usr/bin/env python3
"""
Enhanced CLI for Cursor Rules

This module provides a command-line interface for interacting with the enhanced
graph-based knowledge system for cursor rules.
"""

import os
import sys
import json
import click
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from tabulate import tabulate

# Import from enhanced_db module
from agent_tools.cursor_rules.core.enhanced_db import (
    setup_enhanced_cursor_rules_db,
    multi_hop_rule_discovery,
    knowledge_path_between_resources,
    rule_complete_context,
    hybrid_cross_collection_search,
    contextual_recommendation
)
from agent_tools.cursor_rules.core.ai_knowledge_db import (
    setup_ai_knowledge_db,
    load_schema,
    create_document_collections,
    create_edge_collections,
    create_named_graphs,
    create_views,
    create_analyzers,
    store_schema_doc,
    get_schema_doc
)

from agent_tools.cursor_rules.scenarios.common_queries import (
    store_scenario,
    get_scenario_by_title,
    list_all_scenarios,
    search_scenarios,
    import_scenarios_from_file
)

def print_rule(rule: Dict[str, Any], verbose: bool = False) -> None:
    """Print a rule to the console."""
    # Print rule number and title
    click.echo(f"\nRule {rule.get('rule_number', 'Unknown')}: {rule.get('title', 'Untitled')}")
    
    # Print description if available
    if rule.get('description'):
        click.echo(f"Description: {rule['description']}")
    
    # Print content if available
    if rule.get('content'):
        click.echo("\nContent:")
        click.echo(rule['content'])
    
    # Print additional fields if verbose
    if verbose:
        for key, value in rule.items():
            if key not in ['rule_number', 'title', 'description', 'content']:
                click.echo(f"{key}: {value}")

def print_resource(resource: Dict[str, Any], resource_type: str, verbose: bool = False) -> None:
    """Print a resource to the console."""
    click.echo("\n" + "=" * 80)
    click.echo(f"{resource_type}: {resource.get('title', 'Untitled')}")
    click.echo("-" * 80)
    
    # Print specific fields based on resource type
    if resource_type == "RULES":
        click.echo(f"Rule #: {resource.get('rule_number', 'N/A')}")
        click.echo(f"Type: {resource.get('rule_type', 'N/A')}")
        click.echo(f"Applies to: {resource.get('glob_pattern', 'N/A')}")
    
    click.echo(f"Description: {resource.get('description', 'No description')}")
    
    if verbose:
        if "content" in resource:
            click.echo("\nContent:")
            click.echo("-" * 80)
            click.echo(resource["content"])
            click.echo("-" * 80)

        if "steps" in resource:
            click.echo("\nSteps:")
            click.echo("-" * 80)
            for i, step in enumerate(resource["steps"], 1):
                click.echo(f"{i}. {step}")
            click.echo("-" * 80)
        
        if "examples" in resource:
            click.echo("\nExamples:")
            click.echo("-" * 80)
            for i, example in enumerate(resource["examples"], 1):
                click.echo(f"Example {i}:")
                click.echo(f"  Description: {example.get('description', 'No description')}")
                if "code" in example:
                    click.echo(f"  Code:\n{example['code']}")
            click.echo("-" * 80)

def print_path(path: Dict[str, Any]) -> None:
    """Print a path between two resources."""
    if not path or "vertices" not in path or "edges" not in path:
        click.echo("No path found.")
        return
    
    vertices = path["vertices"]
    edges = path["edges"]
    
    if not vertices:
        click.echo("No path found.")
        return
    
    click.echo("\nPath:")
    click.echo("-" * 80)
    
    for i, vertex in enumerate(vertices):
        # Print vertex information
        v_type = vertex.get("_id", "").split("/")[0]
        click.echo(f"{v_type}: {vertex.get('title', 'Untitled')}")
        
        # If there's an edge after this vertex, print it
        if i < len(edges):
            edge = edges[i]
            e_type = edge.get("_id", "").split("/")[0]
            click.echo(f"  ↓ {e_type}: {edge.get('label', '')}")
    
    click.echo("-" * 80)

@click.group()
@click.option('--host', default="http://localhost:8529", help="ArangoDB host")
@click.option('--username', default="root", help="ArangoDB username")
@click.option('--password', default="openSesame", help="ArangoDB password")
@click.option('--db-name', default="cursor_rules_enhanced", help="Database name")
@click.option('--verbose', '-v', is_flag=True, help="Show verbose output")
@click.pass_context
def cli(ctx, host, username, password, db_name, verbose):
    """Enhanced CLI for Cursor Rules knowledge system.
    
    This tool provides access to the cursor rules knowledge system, including
    rules, examples, troubleshooting guides, lessons learned, and prompt templates.
    
    Features include graph traversal, hybrid search, and contextual recommendations.
    """
    # Initialize the context object with configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = {
        "arango_config": {
            "hosts": [host],
            "username": username,
            "password": password
        }
    }
    ctx.obj['db_name'] = db_name
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('rule_key')
@click.option('--depth', type=int, default=2, help="Maximum traversal depth")
@click.pass_context
def related(ctx, rule_key, depth):
    """Find related resources for a rule.
    
    This command discovers resources connected to a rule through relationships,
    including related rules, examples, troubleshooting guides, and more.
    
    Example: enhanced-cli related rule-001 --depth 3
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function using the event loop
    asyncio.run(_related_command(config, db_name, verbose, rule_key, depth))

async def _related_command(config, db_name, verbose, rule_key, depth):
    """Async implementation of related command following LESSONS_LEARNED.md patterns."""
    # Ensure rule_key has the proper format
    if not rule_key.startswith('rule-'):
        rule_key = f'rule-{rule_key}'
    
    click.echo(f"Finding resources related to rule #{rule_key} (depth: {depth})")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name
        )
        
        # Function to get related resources
        def get_related_resources():
            # First verify the rule exists
            rules_collection = db.collection('rules')
            rule = rules_collection.get(rule_key)
            if not rule:
                return {"start_rule": None, "related_rules": []}
            
            return multi_hop_rule_discovery(db, rule_key, max_depth=depth)
        
        # Get related resources using asyncio.to_thread
        result = await asyncio.to_thread(get_related_resources)
        
        start_rule = result.get("start_rule")
        related_rules = result.get("related_rules", [])
        
        if not start_rule:
            click.echo(f"Rule #{rule_key} not found")
            return
        
        # Print the starting rule
        click.echo("\nSTARTING RULE:")
        print_resource(start_rule, "rules", verbose)
        
        if not related_rules:
            click.echo("\nNo related resources found")
            return
        
        # Group related resources by collection
        by_collection = {}
        for resource in related_rules:
            collection = resource.get("collection", "unknown")
            if collection not in by_collection:
                by_collection[collection] = []
            by_collection[collection].append(resource)
        
        # Print related resources by collection
        for collection, resources in by_collection.items():
            click.echo(f"\n{collection.upper()} ({len(resources)}):")
            click.echo("-" * 80)
            
            for i, resource in enumerate(resources, 1):
                depth = resource.get("depth", "?")
                rule = resource.get("rule", {})
                title = rule.get("title", "Untitled")
                relationship = resource.get("relationship_type", "related")
                strength = resource.get("strength", 0)
                
                click.echo(f"{i}. {title} (depth: {depth}, relation: {relationship}, strength: {strength:.2f})")
                
                if verbose:
                    print_resource(rule, collection, False)
    except Exception as e:
        click.echo(f"Error finding related resources: {e}")

@cli.command()
@click.argument('from_id')
@click.argument('to_id')
@click.pass_context
def path(ctx, from_id, to_id):
    """Find the path between two resources.
    
    This command finds the shortest path between two resources in the knowledge graph,
    showing how different resources are connected.
    
    Example: enhanced-cli path 001 004
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function using the event loop
    asyncio.run(_path_command(config, db_name, verbose, from_id, to_id))

async def _path_command(config, db_name, verbose, from_id, to_id):
    """Async implementation of path command following LESSONS_LEARNED.md patterns."""
    click.echo(f"Finding path from {from_id} to {to_id}")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name
        )
        
        # Function to find the path
        def find_path():
            return knowledge_path_between_resources(db, from_id, to_id)
        
        # Get path result using asyncio.to_thread
        result = await asyncio.to_thread(find_path)
        
        if not result or "path" not in result:
            click.echo("No path found")
            return
        
        # Print the path
        print_path(result)
    except Exception as e:
        click.echo(f"Error finding path: {e}")

@cli.command()
@click.argument('rule_key')
@click.pass_context
def context(ctx, rule_key):
    """Get complete context for a rule.
    
    This command retrieves the full context for a rule, including examples,
    related rules, troubleshooting guides, lessons learned, and prompts.
    
    Example: enhanced-cli context 001-code-advice-rules
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function using the event loop
    asyncio.run(_context_command(config, db_name, verbose, rule_key))

async def _context_command(config, db_name, verbose, rule_key):
    """Async implementation of context command following LESSONS_LEARNED.md patterns."""
    click.echo(f"Getting complete context for rule {rule_key}")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name
        )
        
        # Function to get rule context
        def get_rule_context():
            return rule_complete_context(db, rule_key)
        
        # Get the rule context using asyncio.to_thread
        result = await asyncio.to_thread(get_rule_context)
        
        rule = result.get("rule")
        if not rule:
            click.echo(f"Rule {rule_key} not found")
            return
        
        # Print the rule
        click.echo("\nRULE:")
        print_resource(rule, "rules", verbose)
        
        # Print examples
        examples = result.get("examples", [])
        if examples:
            click.echo(f"\nEXAMPLES ({len(examples)}):")
            for i, example in enumerate(examples, 1):
                click.echo(f"{i}. {example.get('title', 'Untitled Example')}")
                if verbose:
                    print_resource(example, "rule_examples", True)
        
        # Print related rules
        related_rules = result.get("related_rules", [])
        if related_rules:
            click.echo(f"\nRELATED RULES ({len(related_rules)}):")
            for i, related in enumerate(related_rules, 1):
                rule_data = related.get("rule", {})
                relationship = related.get("relationship_type", "related")
                strength = related.get("strength", 0)
                
                click.echo(f"{i}. {rule_data.get('title', 'Untitled')} ({relationship}, strength: {strength:.2f})")
                if verbose:
                    print_resource(rule_data, "rules", False)
        
        # Print troubleshooting guides
        guides = result.get("troubleshooting", [])
        if guides:
            click.echo(f"\nTROUBLESHOOTING GUIDES ({len(guides)}):")
            for i, guide_data in enumerate(guides, 1):
                guide = guide_data.get("guide", {})
                resolution = guide_data.get("resolution_type", "helps with")
                
                click.echo(f"{i}. {guide.get('title', 'Untitled')} ({resolution})")
                if verbose:
                    print_resource(guide, "troubleshooting_guides", False)
        
        # Print lessons
        lessons = result.get("lessons", [])
        if lessons:
            click.echo(f"\nLESSONS LEARNED ({len(lessons)}):")
            for i, lesson_data in enumerate(lessons, 1):
                lesson = lesson_data.get("lesson", {})
                reference = lesson_data.get("reference_type", "references")
                
                click.echo(f"{i}. {lesson.get('title', 'Untitled')} ({reference})")
                if verbose:
                    print_resource(lesson, "lessons_learned", False)
        
        # Print prompts
        prompts = result.get("prompts", [])
        if prompts:
            click.echo(f"\nPROMPT TEMPLATES ({len(prompts)}):")
            for i, prompt_data in enumerate(prompts, 1):
                prompt = prompt_data.get("prompt", {})
                usage = prompt_data.get("usage_type", "uses")
                
                click.echo(f"{i}. {prompt.get('title', 'Untitled')} ({usage})")
                if verbose:
                    print_resource(prompt, "prompt_templates", False)
    except Exception as e:
        click.echo(f"Error getting rule context: {e}")

@cli.command()
@click.argument('query')
@click.option('--limit', type=int, default=10, help="Maximum number of results")
@click.option('--semantic/--no-semantic', default=True, help="Enable/disable semantic search component")
@click.option('--bm25/--no-bm25', default=True, help="Enable/disable BM25 text search component")
@click.option('--glossary/--no-glossary', default=True, help="Enable/disable glossary matching component")
@click.option('--threshold', type=float, default=0.4, help="Minimum hybrid score threshold (0-1)")
@click.option('--boost-recency', is_flag=True, default=True, help="Boost recently accessed items")
@click.pass_context
def search(ctx, query, limit, semantic, bm25, glossary, threshold, boost_recency):
    """Search across all collections using configurable hybrid search.
    
    This command performs a hybrid search across all collections in the knowledge system.
    You can enable/disable different search components:
    - Semantic search (vector similarity)
    - BM25 text search (keyword matching)
    - Glossary matching (term-based matching)
    
    Example: enhanced-cli search "python async patterns" --no-glossary --threshold 0.3
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function using the event loop
    asyncio.run(_search_command(config, db_name, verbose, query, limit, semantic, bm25, glossary, threshold, boost_recency))

async def _search_command(config, db_name, verbose, query, limit, semantic, bm25, glossary, threshold, boost_recency):
    """Async implementation of search command following LESSONS_LEARNED.md patterns."""
    components = []
    if semantic:
        components.append("semantic")
    if bm25:
        components.append("bm25")
    if glossary:
        components.append("glossary")
    
    click.echo(f"Searching for: {query}")
    click.echo(f"Components enabled: {', '.join(components)}")
    click.echo(f"Threshold: {threshold}, Boost recency: {boost_recency}")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name
        )
        
        # Function to perform the search
        def perform_search():
            return hybrid_cross_collection_search(
                db, 
                query, 
                limit=limit,
                semantic=semantic,
                bm25=bm25,
                glossary=glossary,
                threshold=threshold,
                boost_recency=boost_recency,
                verbose=verbose
            )
        
        # Perform search using asyncio.to_thread
        results = await asyncio.to_thread(perform_search)
        
        if not results:
            click.echo("No results found")
            return
        
        # Group results by collection and sort by score
        by_collection = {}
        for result in results:
            collection = result.get("collection", "unknown")
            if collection not in by_collection:
                by_collection[collection] = []
            by_collection[collection].append(result)
        
        # Sort results within each collection by score
        for collection in by_collection:
            by_collection[collection].sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Print results by collection
        total_results = sum(len(items) for items in by_collection.values())
        click.echo(f"\nFound {total_results} results across {len(by_collection)} collections:")
        
        for collection, items in by_collection.items():
            click.echo(f"\n{collection.upper()} ({len(items)} results)")
            click.echo("-" * (len(collection) + 15))
            
            for i, item in enumerate(items, 1):
                doc = item.get("resource", {})
                score = item.get("relevance", 0)
                components = item.get("components", {})
                
                # Extract title and description with fallbacks
                title = doc.get("title", doc.get("name", "Untitled"))
                description = doc.get("description", "No description")
                
                # Print result with ID, score, and component scores
                click.echo(f"{i}. {title}")
                click.echo(f"   ID: {doc.get('_key', 'unknown')}")
                click.echo(f"   Total Score: {score:.3f}")
                
                # Print component scores if available
                if components and verbose:
                    click.echo("   Component Scores:")
                    if semantic and "semantic_score" in components:
                        click.echo(f"     - Semantic: {components['semantic_score']:.3f}")
                    if bm25 and "bm25_score" in components:
                        click.echo(f"     - BM25: {components['bm25_score']:.3f}")
                    if glossary and "glossary_score" in components:
                        click.echo(f"     - Glossary: {components['glossary_score']:.3f}")
                    if "importance_boost" in components:
                        click.echo(f"     - Importance: {components['importance_boost']:.3f}")
                    if "recency_boost" in components:
                        click.echo(f"     - Recency: {components['recency_boost']:.3f}")
                
                if verbose:
                    # Print more details in verbose mode
                    click.echo(f"   Description: {description}")
                    if "content" in doc:
                        content_preview = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                        click.echo(f"   Content Preview: {content_preview}")
                    if "glossary_terms" in doc and glossary:
                        click.echo(f"   Glossary Terms: {', '.join(doc['glossary_terms'])}")
                    click.echo("   ---")
    except Exception as e:
        click.echo(f"Error searching: {e}")

@cli.command()
@click.argument('file_pattern')
@click.option('--language', help="Programming language")
@click.option('--limit', type=int, default=5, help="Maximum number of results per category")
@click.pass_context
def recommend(ctx, file_pattern, language, limit):
    """Get contextual recommendations for the current file.
    
    This command provides recommendations based on the current file context,
    suggesting relevant rules, examples, troubleshooting guides, and lessons.
    
    Example: enhanced-cli recommend "*.py" --language python --limit 3
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function using the event loop
    asyncio.run(_recommend_command(config, db_name, verbose, file_pattern, language, limit))

async def _recommend_command(config, db_name, verbose, file_pattern, language, limit):
    """Async implementation of recommend command following LESSONS_LEARNED.md patterns."""
    click.echo(f"Getting recommendations for {file_pattern}" + 
              (f" ({language})" if language else ""))
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name
        )
        
        # Function to get recommendations
        def get_recommendations():
            return contextual_recommendation(db, file_pattern, language, limit)
        
        # Get recommendations using asyncio.to_thread
        results = await asyncio.to_thread(get_recommendations)
        
        # Print rules
        rules = results.get("rules", [])
        if rules:
            click.echo(f"\nRELEVANT RULES ({len(rules)}):")
            click.echo("-" * 80)
            
            for i, rule in enumerate(rules, 1):
                title = rule.get("title", "Untitled")
                click.echo(f"{i}. {rule.get('rule_number', 'N/A')}: {title}")
                
                if verbose:
                    print_resource(rule, "rules", False)
        else:
            click.echo("\nNo relevant rules found")
        
        # Print examples
        examples = results.get("examples", [])
        if examples:
            click.echo(f"\nRELEVANT EXAMPLES ({len(examples)}):")
            click.echo("-" * 80)
            
            for i, example in enumerate(examples, 1):
                title = example.get("title", "Untitled Example")
                click.echo(f"{i}. {title}")
                
                if verbose:
                    print_resource(example, "rule_examples", True)
        
        # Print troubleshooting guides
        guides = results.get("troubleshooting", [])
        if guides:
            click.echo(f"\nRELEVANT TROUBLESHOOTING ({len(guides)}):")
            click.echo("-" * 80)
            
            for i, guide in enumerate(guides, 1):
                title = guide.get("title", "Untitled")
                click.echo(f"{i}. {title}")
                
                if verbose:
                    print_resource(guide, "troubleshooting_guides", verbose)
        
        # Print lessons
        lessons = results.get("lessons", [])
        if lessons:
            click.echo(f"\nRELEVANT LESSONS ({len(lessons)}):")
            click.echo("-" * 80)
            
            for i, lesson in enumerate(lessons, 1):
                title = lesson.get("title", "Untitled")
                click.echo(f"{i}. {title}")
            
            if verbose:
                    print_resource(lesson, "lessons_learned", verbose)
    except Exception as e:
        click.echo(f"Error getting recommendations: {e}")

@cli.command()
@click.option('--reset', is_flag=True, help="Reset the database")
@click.pass_context
def setup(ctx, reset):
    """Initialize or reset the enhanced database.
    
    This command sets up the enhanced database structure, creating all collections,
    edge collections, views, and graphs needed for the knowledge system.
    
    Example: enhanced-cli setup --reset
    """
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    # Run the async function using the event loop
    asyncio.run(_setup_command(config, db_name, reset))

async def _setup_command(config, db_name, reset):
    """Async implementation of setup command following LESSONS_LEARNED.md patterns."""
    click.echo(f"Setting up enhanced database '{db_name}'" + 
              (" (with reset)" if reset else ""))
    
    try:
        # Set up the database using asyncio.to_thread
        db = await asyncio.to_thread(
            setup_enhanced_cursor_rules_db, config, db_name, reset
        )
        
        # Just print a simple summary - don't try to get detailed collection info
        # since that's causing issues with Click contexts in threads
        click.echo("\nDatabase setup complete!")
        
    except Exception as e:
        click.echo(f"Error setting up database: {e}")

@cli.command()
@click.option("--host", default="http://localhost:8529", help="ArangoDB host")
@click.option("--username", default="root", help="ArangoDB username")
@click.option("--password", default="openSesame", help="ArangoDB password") 
@click.option("--db-name", default=f"test_cursor_rules_{os.getpid()}", help="Test database name")
@click.option("--cleanup/--no-cleanup", default=True, help="Clean up test database after the test")
@click.option("--verbose", "-v", is_flag=True, help="Display detailed output")
def test_integration(host, username, password, db_name, cleanup, verbose):
    """
    Test ArangoDB integration to verify database functionality.
    
    This command tests the core ArangoDB operations:
    1. Database connection and setup
    2. Collection creation (document and edge collections)  
    3. Graph creation with edge definitions
    4. View creation with links
    5. Analyzer creation
    6. Schema document storage and retrieval
    
    Returns a success message if all tests pass, or an error message with details if any test fails.
    """
    click.echo(f"Testing ArangoDB integration with database: {db_name}")
    click.echo(f"Host: {host}, Username: {username}")
    
    # Create async event loop for running tests
    loop = asyncio.get_event_loop()
    
    try:
        # Run the tests
        result = loop.run_until_complete(_run_integration_tests(
            host=host,
            username=username, 
            password=password,
            db_name=db_name,
            verbose=verbose
        ))
        
        # If we get here, tests were successful
        click.secho("\n✅ All integration tests passed successfully!", fg="green", bold=True)
        
        for test_name, (success, message) in result.items():
            if verbose or not success:
                click.secho(f"  {test_name}: {'✓' if success else '✗'}", fg="green" if success else "red")
                if message and (verbose or not success):
                    click.echo(f"    {message}")
        
        # Clean up the test database if requested
        if cleanup:
            try:
                click.echo(f"\nCleaning up test database: {db_name}")
                from arango import ArangoClient
                client = ArangoClient(hosts=host)
                sys_db = client.db("_system", username=username, password=password)
                if sys_db.has_database(db_name):
                    sys_db.delete_database(db_name)
                    click.echo(f"Test database {db_name} deleted successfully.")
                else:
                    click.echo(f"Test database {db_name} not found, nothing to clean up.")
            except Exception as e:
                click.secho(f"Warning: Could not clean up test database: {e}", fg="yellow")
        
    except Exception as e:
        click.secho(f"\n❌ Integration tests failed: {e}", fg="red", bold=True)
        sys.exit(1)

async def _run_integration_tests(host, username, password, db_name, verbose):
    """
    Run integration tests for ArangoDB and return results.
    """
    results = {}
    
    try:
        if verbose:
            click.echo("Testing database connection and setup...")
        
        # Use minimal test schema
        test_schema = {
            "description": "Test AI Knowledge Schema",
            "database_name": db_name,
            "collections": {
                "test_methods": {
                    "description": "Test methods collection",
                    "fields": {
                        "name": {"type": "string", "description": "Method name"},
                        "language": {"type": "string", "description": "Programming language"}
                    }
                },
                "test_patterns": {
                    "description": "Test patterns collection",
                    "fields": {
                        "name": {"type": "string", "description": "Pattern name"},
                        "language": {"type": "string", "description": "Programming language"}
                    }
                }
            },
            "edge_collections": {
                "test_requires": {
                    "description": "Test requires relationship",
                    "fields": {
                        "requirement_type": {"type": "string", "description": "Type of requirement"}
                    }
                }
            },
            "named_graphs": {
                "test_knowledge_graph": {
                    "description": "Test knowledge graph",
                    "edge_definitions": [
                        {
                            "edge_collection": "test_requires",
                            "from_collections": ["test_methods"],
                            "to_collections": ["test_patterns"]
                        }
                    ]
                }
            },
            "views": {
                "test_search_view": {
                    "description": "Test search view",
                    "properties": {
                        "primarySort": [],
                        "cleanupIntervalStep": 2,
                        "commitIntervalMsec": 1000
                    },
                    "links": {
                        "test_methods": {
                            "fields": {
                                "name": {"analyzers": ["text_en"]}
                            }
                        },
                        "test_patterns": {
                            "fields": {
                                "name": {"analyzers": ["text_en"]}
                            }
                        }
                    }
                }
            },
            "analyzers": {
                "text_en": {
                    "type": "text",
                    "properties": {
                        "locale": "en",
                        "case": "lower",
                        "stopwords": [],
                        "accent": False,
                        "stemming": True
                    },
                    "features": ["frequency", "norm", "position"]
                }
            }
        }
        
        # Set up test database
        db = await setup_ai_knowledge_db(
            host=host,
            username=username,
            password=password,
            db_name=db_name,
            reset=True  # Always start fresh
        )
        
        results["Database connection"] = (True, f"Successfully connected to {db_name}")
        
        # Test 2: Document collection creation
        try:
            collections = await asyncio.to_thread(create_document_collections, db, test_schema)
            if len(collections) >= 2 and "test_methods" in collections and "test_patterns" in collections:
                results["Document collections"] = (True, f"Created {len(collections)} document collections")
        except Exception as e:
            results["Document collections"] = (False, f"Error creating document collections: {e}")
        
        # Test 3: Edge collection creation
        try:
            edge_collections = await asyncio.to_thread(create_edge_collections, db, test_schema)
            if len(edge_collections) >= 1 and "test_requires" in edge_collections:
                results["Edge collections"] = (True, f"Created {len(edge_collections)} edge collections")
            else:
                results["Edge collections"] = (False, f"Failed to create all edge collections: {edge_collections}")
        except Exception as e:
            results["Edge collections"] = (False, f"Error creating edge collections: {e}")
        
        # Test 4: Named graph creation
        try:
            graphs = await asyncio.to_thread(create_named_graphs, db, test_schema)
            if len(graphs) >= 1 and "test_knowledge_graph" in graphs:
                results["Graphs"] = (True, f"Created {len(graphs)} graphs")
            else:
                results["Graphs"] = (False, f"Failed to create all graphs: {graphs}")
                
            # Verify graphs were created in database
            def get_graph_names():
                return [g["name"] for g in db.graphs()]
            
            graph_names = await asyncio.to_thread(get_graph_names)
            if "test_knowledge_graph" in graph_names:
                results["Graph verification"] = (True, "Graph exists in database")
            else:
                results["Graph verification"] = (False, f"Graph not found in database. Graphs: {graph_names}")
        except Exception as e:
            results["Graphs"] = (False, f"Error creating graphs: {e}")
        
        # Test 5: Analyzer creation
        try:
            analyzers = await asyncio.to_thread(create_analyzers, db, test_schema)
            if len(analyzers) >= 1 and "text_en" in analyzers:
                results["Analyzers"] = (True, f"Created/verified {len(analyzers)} analyzers")
            else:
                results["Analyzers"] = (False, f"Failed to create all analyzers: {analyzers}")
        except Exception as e:
            results["Analyzers"] = (False, f"Error creating analyzers: {e}")
        
        # Test 6: View creation
        try:
            views = await asyncio.to_thread(create_views, db, test_schema)
            if len(views) >= 1 and "test_search_view" in views:
                results["Views"] = (True, f"Created {len(views)} views")
            else:
                results["Views"] = (False, f"Failed to create all views: {views}")
                
            # Verify views were created in database
            def get_view_names():
                return [v["name"] for v in db.views()]
            
            view_names = await asyncio.to_thread(get_view_names)
            if "test_search_view" in view_names:
                results["View verification"] = (True, "View exists in database")
            else:
                results["View verification"] = (False, f"View not found in database. Views: {view_names}")
        except Exception as e:
            results["Views"] = (False, f"Error creating views: {e}")
        
        # Test 7: Schema document storage
        try:
            schema_key = await asyncio.to_thread(store_schema_doc, db, test_schema)
            if schema_key:
                results["Schema storage"] = (True, f"Stored schema with key {schema_key}")
            else:
                results["Schema storage"] = (False, "Failed to store schema document")
        except Exception as e:
            results["Schema storage"] = (False, f"Error storing schema: {e}")
        
        # Test 8: Schema document retrieval (get latest)
        try:
            schema_doc = await get_schema_doc(db, None)
            if schema_doc and "description" in schema_doc.get("schema", {}):
                results["Schema retrieval"] = (True, f"Retrieved latest schema document")
            else:
                results["Schema retrieval"] = (False, f"Failed to retrieve schema document")
        except Exception as e:
            results["Schema retrieval"] = (False, f"Error retrieving schema: {e}")
            
    except Exception as e:
        # Handle any unexpected errors
        results["Database setup"] = (False, f"Failed to set up database: {e}")
    
    return results

@cli.command()
@click.pass_context
def list(ctx):
    """List all rules in the database."""
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function in the event loop
    asyncio.run(_list_command(config, db_name, verbose))

async def _list_command(config, db_name, verbose):
    """Async implementation of list command following LESSONS_LEARNED.md patterns."""
    click.echo("Listing all rules in the database")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(setup_enhanced_cursor_rules_db, config, db_name)
        
        # Function to get rules in a thread-safe manner
        def get_rules():
            # Check if rules collection exists
            if not db.has_collection("rules"):
                click.echo("Rules collection does not exist")
                return []
                
            click.echo("Getting rules from database...")
            rules_collection = db.collection("rules")
            click.echo(f"Rules collection: {rules_collection.name}")
            
            # Log collection count
            try:
                count_cursor = db.aql.execute("RETURN LENGTH(FOR doc IN rules RETURN doc)")
                rule_count = next(count_cursor)
                click.echo(f"Rules count from AQL: {rule_count}")
            except Exception as e:
                click.echo(f"Error counting rules: {e}")
            
            # Try direct collection access
            try:
                rules = []
                for doc in rules_collection.all():
                    rules.append(dict(doc))
                click.echo(f"Found {len(rules)} rules using collection.all()")
                return rules
            except Exception as e:
                click.echo(f"Error retrieving rules via collection.all(): {e}")
            
            # Try AQL as fallback
            try:
                cursor = db.aql.execute("FOR doc IN rules RETURN doc")
                rules = []
                for doc in cursor:
                    rules.append(dict(doc))
                click.echo(f"Found {len(rules)} rules using AQL")
                return rules
            except Exception as e:
                click.echo(f"Error retrieving rules via AQL: {e}")
                return []
        
        # Get rules using asyncio.to_thread
        rules = await asyncio.to_thread(get_rules)
        
        if not rules:
            click.echo("No rules found in the database")
            return
        
        # Create a table with the rules
        rows = []
        headers = ["Number", "Title", "Type", "Pattern", "Tags"]
        
        for rule in rules:
            # Extract the rule number from the _key (format: rule-001)
            rule_key = rule.get("_key", "")
            rule_number = rule_key.replace("rule-", "") if rule_key.startswith("rule-") else ""
            
            rows.append([
                rule_number,
                rule.get("title", ""),
                rule.get("type", ""),
                rule.get("pattern", ""),
                ", ".join(rule.get("tags", []))
            ])
        
        # Sort by rule number
        rows.sort(key=lambda x: x[0])
        
        # Print table
        click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
            
        if verbose:
            click.echo(f"\nTotal rules: {len(rules)}")
    
    except Exception as e:
        click.echo(f"Error listing rules: {e}")

@cli.command()
@click.option('--count', default=5, help="Number of sample rules to create")
@click.option('--reset', is_flag=True, help="Reset the rules collection before creating samples")
@click.pass_context
def load_samples(ctx, count, reset):
    """Load sample rules into the database."""
    # Create configuration for database connection
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    verbose = ctx.obj['verbose']
    
    # Run the async function in the event loop
    asyncio.run(_load_samples_command(config, db_name, count, reset))

async def _load_samples_command(config, db_name, count, reset=False):
    """Async implementation of load-samples command following LESSONS_LEARNED.md patterns."""
    click.echo(f"Loading {count} sample rules into the database")
    
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(setup_enhanced_cursor_rules_db, config, db_name)
        
        # Function to create sample rules
        def create_sample_rules():
            created_rules = []
            
            # Create rules collection if it doesn't exist
            try:
                if not db.has_collection("rules"):
                    click.echo("Creating 'rules' collection")
                    db.create_collection("rules")
                    click.echo("'rules' collection created successfully")
                else:
                    click.echo("'rules' collection already exists")
                    
                    # Reset rules if requested
                    if reset:
                        click.echo("Resetting 'rules' collection")
                        rules_collection = db.collection("rules")
                        rules_collection.truncate()
                        click.echo("'rules' collection truncated")
                
                rules_collection = db.collection("rules")
                click.echo(f"Rules collection: {rules_collection.name}")
                
                # Ensure other required collections exist
                for collection_name in ["rule_examples", "rule_references_rule"]:
                    if not db.has_collection(collection_name):
                        click.echo(f"Creating '{collection_name}' collection")
                        if collection_name.startswith("rule_references"):
                            db.create_collection(collection_name, edge=True)
                        else:
                            db.create_collection(collection_name)
                        click.echo(f"'{collection_name}' collection created successfully")
                    else:
                        click.echo(f"'{collection_name}' collection already exists")
                
                click.echo("Collections in database:")
                for collection in db.collections():
                    click.echo(f" - {collection['name']}")
                
                created_rules = []
                for i in range(1, count + 1):
                    rule_number = f"{i:03d}"
                    rule_key = f"rule-{rule_number}"
                    
                    # Sample rule data
                    rule_data = {
                        "_key": rule_key,
                        "title": f"Sample Rule {rule_number}",
                        "description": f"This is sample rule {rule_number}",
                        "type": "guideline",
                        "pattern": "*.py",
                        "tags": ["sample", f"rule-{rule_number}"]
                    }
                    
                    try:
                        rules_collection.insert(rule_data, overwrite=True)
                        click.echo(f"Created rule {rule_key}")
                        created_rules.append(rule_data)
                    except Exception as e:
                        click.echo(f"Error creating rule {rule_key}: {e}")
                        
                # Count rules using AQL
                try:
                    result = db.aql.execute("RETURN LENGTH(FOR r IN rules RETURN 1)")
                    rule_count = next(result)
                    click.echo(f"Database has {rule_count} rules")
                except Exception as e:
                    click.echo(f"Error counting rules: {e}")
                        
                # Create example rule references
                if len(created_rules) >= 2:
                    from_rule = created_rules[0]
                    to_rule = created_rules[1]
                    
                    edge = {
                        "_from": f"rules/{from_rule['_key']}",
                        "_to": f"rules/{to_rule['_key']}",
                        "relationship": "references",
                        "strength": 0.75
                    }
                    
                    try:
                        edges_collection = db.collection("rule_references_rule")
                        edges_collection.insert(edge, overwrite=True)
                    except Exception as e:
                        click.echo(f"Error creating relationship between {from_rule['_key']} and {to_rule['_key']}: {e}")
                
                return len(created_rules)
            except Exception as e:
                click.echo(f"Error setting up collections: {e}")
        return 0
        
        # Create sample rules using asyncio.to_thread
        num_created = await asyncio.to_thread(create_sample_rules)
        
        click.echo(f"Successfully created {num_created} sample rules in the database")
        click.echo("You can now use the 'list', 'search', and other commands to interact with them")
    except Exception as e:
        click.echo(f"Error loading sample rules: {e}")

@cli.command(name="scenario")
@click.argument('query', required=False)
@click.option("--title", help="Scenario title")
@click.option("--description", help="Scenario description")
@click.option("--query-example", help="Example query for this scenario")
@click.option("--expected-format", help="Expected format of the results")
@click.option("--category", help="Category for the scenario")
@click.option("--priority", type=int, default=2, help="Priority (1=high, 3=low)")
@click.option("--import-file", help="Import scenarios from a JSON file")
@click.option("--search", help="Search for scenarios")
@click.option("--search-type", type=click.Choice(["keyword", "semantic", "hybrid"]), default="hybrid", help="Type of search")
@click.option("--list", is_flag=True, help="List all scenarios")
@click.option("--get", help="Get scenario by title")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def scenario_command(
    ctx, query, title, description, query_example, expected_format, category, 
    priority, import_file, search, search_type, list, get, verbose
):
    """Manage query scenarios for AI agent retrieval.
    
    This command helps manage common query scenarios that AI agents
    frequently need to handle. These scenarios guide the implementation
    of the knowledge retrieval system.
    
    Examples:
        # Search for scenarios by query
        enhanced_cli.py scenario "How to handle errors"
        
        # Add a new scenario
        enhanced_cli.py scenario \\
            --title "Find method usage" \\
            --description "How to use a specific method" \\
            --query-example "How do I use requests.get?" \\
            --expected-format "Code example with explanation" \\
            --category "method_usage"
            
        # Import scenarios from a file
        enhanced_cli.py scenario --import-file scenarios.json
            
        # Search for scenarios
        enhanced_cli.py scenario --search "error message" --search-type hybrid
            
        # List all scenarios
        enhanced_cli.py scenario --list
        
        # Get a specific scenario
        enhanced_cli.py scenario --get "Find method usage"
    """
    # Create configuration for database connection
    config = ctx.obj["config"]
    db_name = ctx.obj.get("db_name", "cursor_rules_enhanced")
    
    if verbose:
        click.echo(f"Using database: {db_name}")
    
    # If a query is provided directly, treat it as a search
    if query:
        search = query
    
    # Run the async function using the event loop
    asyncio.run(_scenario_command_async(
        config=config,
        db_name=db_name,
        title=title,
        description=description,
        query_example=query_example,
        expected_format=expected_format,
        category=category,
        priority=priority,
        import_file=import_file,
        search=search,
        search_type=search_type,
        list=list,
        get=get,
        verbose=verbose
    ))

async def _scenario_command_async(
    config, db_name, title, description, query_example, expected_format, category,
    priority, import_file, search, search_type, list, get, verbose
):
    """Async implementation of scenario command."""
    try:
        # Connect to the database using asyncio.to_thread
        db = await asyncio.to_thread(setup_enhanced_cursor_rules_db, config, db_name)
        
        # Import scenarios from file
        if import_file:
            try:
                imported = await asyncio.to_thread(import_scenarios_from_file, db, import_file)
                click.echo(f"Imported {len(imported)} scenarios from {import_file}")
                if verbose:
                    for scenario in imported:
                        click.echo(f"  - {scenario.get('title')}")
                return
            except Exception as e:
                click.echo(f"Error importing scenarios: {e}", err=True)
                return
        
        # List all scenarios
        if list:
            scenarios = await asyncio.to_thread(list_all_scenarios, db)
            click.echo(f"Found {len(scenarios)} scenarios:")
            
            # Create a table for output
            table = []
            headers = ["#", "Title", "Category", "Priority"]
            
            for i, scenario in enumerate(scenarios, 1):
                table.append([
                    i,
                    scenario.get("title", "N/A"),
                    scenario.get("category", "N/A"),
                    scenario.get("priority", "N/A")
                ])
            
            if table:
                click.echo(tabulate(table, headers=headers, tablefmt="pretty"))
            return
        
        # Get scenario by title
        if get:
            scenario = await asyncio.to_thread(get_scenario_by_title, db, get)
            if scenario:
                click.echo(f"Scenario: {scenario.get('title')}")
                click.echo(f"Description: {scenario.get('description')}")
                click.echo(f"Query example: {scenario.get('query_example')}")
                click.echo(f"Expected format: {scenario.get('expected_result_format')}")
                click.echo(f"Category: {scenario.get('category')}")
                click.echo(f"Priority: {scenario.get('priority', 'N/A')}")
            else:
                click.echo(f"Scenario not found: {get}", err=True)
            return
        
        # Search for scenarios
        if search:
            results = await asyncio.to_thread(search_scenarios, db, search, search_type)
        if not results:
            click.echo("No scenarios found matching the search criteria.")
            return
        
        click.echo(f"Found {len(results)} matching scenarios:")
        
        # Create a table for output
        table = []
        headers = ["#", "Title", "Category", "Query Example"]
        
        for i, scenario in enumerate(results, 1):
            table.append([
                i,
                scenario.get("title", "N/A"),
                scenario.get("category", "N/A"),
                scenario.get("query_example", "N/A")
            ])
        
        if table:
            click.echo(tabulate(table, headers=headers, tablefmt="pretty"))
            return
        
        # Store a new scenario
        if title and description and query_example and expected_format and category:
            scenario = {
                "title": title,
                "description": description,
                "query_example": query_example,
                "expected_result_format": expected_format,
                "category": category,
                "priority": priority
            }
            
            try:
                stored = await asyncio.to_thread(store_scenario, db, scenario)
                click.echo(f"Stored scenario: {stored.get('title')}")
                if verbose:
                    click.echo(json.dumps(stored, indent=2))
            except Exception as e:
                click.echo(f"Error storing scenario: {e}", err=True)
            return
        
        # Show help if no action specified
        click.echo(ctx.get_help())
        
    except Exception as e:
        click.echo(f"Error in scenario command: {e}", err=True)

if __name__ == "__main__":
    cli(obj={}) 