#!/usr/bin/env python
"""
Command Line Interface for Cursor Rules Database

This script provides a CLI for interacting with the cursor rules database.
"""
import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

import click

from agent_tools.cursor_rules.cursor_rules import (
    setup_cursor_rules_db,
    get_all_rules,
    get_examples_for_rule,
    semantic_search,
    hybrid_search,
    bm25_keyword_search,
    EMBEDDING_AVAILABLE
)

def print_rule(rule: Dict[str, Any], verbose: bool = False) -> None:
    """Print a rule in a readable format."""
    click.echo(f"\n[Rule {rule['rule_number']}] {rule['title']}")
    click.echo(f"Type: {rule['rule_type']}")
    click.echo(f"Applies to: {rule['glob_pattern']}")
    click.echo(f"Description: {rule['description']}")
    
    if verbose:
        click.echo("\nContent:")
        click.echo("-" * 80)
        click.echo(rule['content'])
        click.echo("-" * 80)

def print_example(example: Dict[str, Any], verbose: bool = False) -> None:
    """Print an example in a readable format."""
    click.echo(f"\n[Example] {example['title']}")
    click.echo(f"Description: {example['description']}")
    
    if verbose or True:  # Always show code examples
        click.echo("\nGood Example:")
        click.echo("-" * 80)
        click.echo(example['good_example'])
        click.echo("-" * 80)
        
        click.echo("\nBad Example:")
        click.echo("-" * 80)
        click.echo(example['bad_example'])
        click.echo("-" * 80)

# Create the Click command group
@click.group()
@click.option('--host', default="http://localhost:8529", help="ArangoDB host")
@click.option('--username', default="root", help="ArangoDB username")
@click.option('--password', default="openSesame", help="ArangoDB password")
@click.option('--db-name', default="cursor_rules_test", help="Database name")
@click.option('--verbose', '-v', is_flag=True, help="Show verbose output")
@click.pass_context
def cli(ctx, host, username, password, db_name, verbose):
    """Cursor Rules Database CLI tool."""
    # Ensure we have a dict to store our context
    ctx.ensure_object(dict)
    
    # Store options in context
    ctx.obj['host'] = host
    ctx.obj['username'] = username
    ctx.obj['password'] = password
    ctx.obj['db_name'] = db_name
    ctx.obj['verbose'] = verbose
    
    # Create config for ArangoDB
    config = {
        "arango_config": {
            "hosts": [host],
            "username": username,
            "password": password
        }
    }
    ctx.obj['config'] = config
    
    # Don't connect for init command
    if ctx.invoked_subcommand != 'init':
        db = setup_cursor_rules_db(config, db_name=db_name)
        if not db:
            click.echo("Failed to connect to database", err=True)
            ctx.exit(1)
        ctx.obj['db'] = db

@cli.command()
@click.option('--rules-dir', default=".cursor/rules", help="Path to rules directory")
@click.pass_context
def init(ctx, rules_dir):
    """Initialize or update database from rules directory."""
    click.echo(f"Initializing database from rules directory: {rules_dir}")
    
    config = ctx.obj['config']
    db_name = ctx.obj['db_name']
    
    db = setup_cursor_rules_db(config, rules_dir, db_name=db_name)
    if not db:
        click.echo("Failed to initialize database", err=True)
        return 1
    
    click.echo("Database initialized successfully")
    return 0

@cli.command()
@click.option('--examples', is_flag=True, help="Show examples for each rule")
@click.pass_context
def list(ctx, examples):
    """List all rules in the database."""
    db = ctx.obj['db']
    verbose = ctx.obj['verbose']
    
    # Get all rules
    rules = get_all_rules(db)
    if not rules:
        click.echo("No rules found in the database.")
        return 1
    
    # Display rules
    click.echo(f"Found {len(rules)} rules:")
    for rule in rules:
        click.echo(f"{rule['rule_number']}: {rule['title']}")
        
        if verbose:
            print_rule(rule, verbose=verbose)
            
            if examples:
                rule_examples = get_examples_for_rule(db, rule["_key"])
                if rule_examples:
                    for example in rule_examples:
                        print_example(example, verbose=verbose)
                else:
                    click.echo(f"No examples found for rule {rule['rule_number']}")
    
    return 0

@cli.command()
@click.argument('rule_number')
@click.option('--examples', is_flag=True, help="Show examples for the rule")
@click.pass_context
def show(ctx, rule_number, examples):
    """Show details of a specific rule."""
    db = ctx.obj['db']
    
    # Get all rules
    rules = get_all_rules(db)
    
    # Find the rule with the specified number
    matching_rules = [r for r in rules if r['rule_number'] == rule_number]
    if not matching_rules:
        click.echo(f"Rule {rule_number} not found.", err=True)
        return 1
    
    # Display the rule
    rule = matching_rules[0]
    print_rule(rule, verbose=True)
    
    # Show examples if requested
    if examples:
        rule_examples = get_examples_for_rule(db, rule["_key"])
        if rule_examples:
            for example in rule_examples:
                print_example(example, verbose=True)
        else:
            click.echo(f"No examples found for rule {rule['rule_number']}")
    
    return 0

@cli.command()
@click.argument('query')
@click.option('--semantic', is_flag=True, help="Use semantic search with embeddings")
@click.option('--hybrid', is_flag=True, help="Use hybrid search (BM25 + vector similarity)")
@click.option('--keyword', is_flag=True, help="Use BM25 keyword search with stemming")
@click.option('--limit', type=int, default=5, help="Maximum number of results")
@click.pass_context
def search(ctx, query, semantic, hybrid, keyword, limit):
    """Search for rules using various search methods."""
    db = ctx.obj['db']
    verbose = ctx.obj['verbose']
    
    # Validate search options
    if hybrid and not EMBEDDING_AVAILABLE:
        click.echo("Hybrid search is not available. Missing embedding utilities.", err=True)
        return 1
    
    if semantic and not EMBEDDING_AVAILABLE:
        click.echo("Semantic search is not available. Missing embedding utilities.", err=True)
        return 1
    
    click.echo(f"Searching for: {query}")
    
    # BM25 keyword search with stemming
    if keyword:
        click.echo("Using BM25 keyword search with stemming")
        results = bm25_keyword_search(db, query, limit=limit)
        
        if results:
            click.echo(f"Found {len(results)} rules matching keywords:")
            for i, result in enumerate(results, 1):
                rule = result["rule"]
                score = result.get("score", 0)
                matched_terms = result.get("matched_terms", [])
                
                click.echo(f"{i}. {rule.get('rule_number', '')}: {rule.get('title', 'Untitled')}")
                click.echo(f"   Score: {score:.4f}")
                if matched_terms:
                    click.echo(f"   Matched terms: {', '.join(matched_terms)}")
                
                if verbose:
                    print_rule(rule, verbose=verbose)
        else:
            click.echo("No relevant rules found.")
        return 0
    
    # Hybrid search (BM25 + vector similarity)
    if hybrid:
        click.echo("Using hybrid search (BM25 + vector similarity)")
        results = hybrid_search(db, query, limit=limit)
        
        if not results:
            click.echo("No results found.")
            return 0
        
        click.echo(f"Found {len(results)} relevant rules:")
        for i, result in enumerate(results, 1):
            rule = result["rule"]
            hybrid_score = result.get("hybrid_score", 0)
            vector_score = result.get("vector_score", 0)
            text_score = result.get("text_score", 0)
            
            click.echo(f"{i}. {rule['rule_number']}: {rule['title']}")
            click.echo(f"   Hybrid Score: {hybrid_score:.4f} (Text: {text_score:.4f}, Vector: {vector_score:.4f})")
            
            if verbose:
                print_rule(rule, verbose=verbose)
                
    # Semantic search
    elif semantic:
        click.echo("Using semantic search")
        results = semantic_search(db, query, limit=limit)
        
        if not results:
            click.echo("No results found.")
            return 0
        
        click.echo(f"Found {len(results)} relevant rules:")
        for i, result in enumerate(results, 1):
            rule = result["rule"]
            similarity = result["similarity"]
            click.echo(f"{i}. {rule['rule_number']}: {rule['title']} (Similarity: {similarity:.4f})")
            
            if verbose:
                print_rule(rule, verbose=verbose)
                
    # Default simple keyword search
    else:
        # Simple keyword search in rules
        rules = get_all_rules(db)
        query_lower = query.lower()
        
        # Search in title, description, and content
        matching_rules = []
        for rule in rules:
            if (query_lower in rule['title'].lower() or 
                query_lower in rule['description'].lower() or 
                query_lower in rule['content'].lower()):
                matching_rules.append(rule)
        
        if not matching_rules:
            click.echo("No results found.")
            return 0
        
        click.echo(f"Found {len(matching_rules)} matching rules:")
        for i, rule in enumerate(matching_rules, 1):
            click.echo(f"{i}. {rule['rule_number']}: {rule['title']}")
            
            if verbose:
                print_rule(rule, verbose=verbose)
    
    return 0

if __name__ == "__main__":
    # Use obj to pass context between commands
    cli(obj={}) 