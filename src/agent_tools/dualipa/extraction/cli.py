#!/usr/bin/env python3
"""
TestStateManager CLI

A comprehensive CLI for interacting with the TestStateManager, providing
robust memory management for the extraction process and addressing limitations
in AI context maintenance.
"""

import os
import sys
import argparse
import json
import datetime
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .test_state_manager import get_state_manager, TestStateManager


def format_output(content, width=80):
    """Format output for CLI display."""
    if isinstance(content, str):
        return textwrap.fill(content, width=width)
    elif isinstance(content, dict):
        return json.dumps(content, indent=2)
    elif isinstance(content, list):
        return json.dumps(content, indent=2)
    else:
        return str(content)


def initialize_database(args):
    """Initialize the state database."""
    db_path = args.db_path
    state_manager = get_state_manager(db_path)
    print(f"Database initialized at: {db_path if db_path else 'in-memory'}")
    
    # Create initial checkpoint
    state_manager.set_checkpoint("initialization", "CLI initialization")
    
    # Set creation metadata
    state_manager.set_metadata("cli_creation_time", datetime.datetime.now().isoformat())
    state_manager.set_metadata("created_by", "TestStateManager CLI")
    
    print("State database initialized successfully.")


def set_command(args):
    """Set a key-value pair in the state store."""
    state_manager = get_state_manager(args.db_path)
    
    # Handle value types
    value = args.value
    if args.type == 'int':
        value = int(value)
    elif args.type == 'float':
        value = float(value)
    elif args.type == 'bool':
        value = value.lower() in ('true', 'yes', '1', 'y')
    elif args.type == 'json':
        value = json.loads(value)
    
    state_manager.set(args.key, value)
    print(f"Set {args.key} = {value}")


def get_command(args):
    """Get a value from the state store."""
    state_manager = get_state_manager(args.db_path)
    value = state_manager.get(args.key, args.default)
    
    if args.quiet:
        print(value)
    else:
        print(f"{args.key}: {format_output(value)}")


def list_command(args):
    """List all keys in the state store."""
    state_manager = get_state_manager(args.db_path)
    
    # This requires a new method in TestStateManager
    state_manager.cursor.execute("SELECT key, value_type FROM state ORDER BY key")
    keys = state_manager.cursor.fetchall()
    
    if not keys:
        print("No keys found in state store.")
        return
    
    print(f"State store contains {len(keys)} keys:")
    for row in keys:
        key = row['key']
        value_type = row['value_type']
        print(f"- {key} ({value_type})")


def delete_command(args):
    """Delete a key from the state store."""
    state_manager = get_state_manager(args.db_path)
    
    # This requires a new method in TestStateManager
    with state_manager.transaction():
        state_manager.cursor.execute("DELETE FROM state WHERE key = ?", (args.key,))
    
    print(f"Deleted key: {args.key}")


def checkpoint_command(args):
    """Create a checkpoint."""
    state_manager = get_state_manager(args.db_path)
    state_manager.set_checkpoint(args.name, args.description)
    print(f"Checkpoint created: {args.name}")
    print(f"Description: {args.description}")


def list_checkpoints_command(args):
    """List all checkpoints."""
    state_manager = get_state_manager(args.db_path)
    checkpoints = state_manager.get_checkpoint_history()
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for i, cp in enumerate(checkpoints, 1):
        print(f"{i}. {cp['name']} ({cp['timestamp']})")
        if cp['description']:
            print(f"   {cp['description']}")


def context_command(args):
    """Update the current context."""
    state_manager = get_state_manager(args.db_path)
    
    # Create a dict of provided values
    context_data = {}
    if args.task:
        context_data['task'] = args.task
    if args.goal:
        context_data['goal'] = args.goal
    if args.progress:
        context_data['progress'] = args.progress
    if args.assumptions:
        context_data['assumptions'] = args.assumptions
    if args.problems:
        context_data['problems'] = args.problems
    if args.next_steps:
        context_data['next_steps'] = args.next_steps
    if args.notes:
        context_data['notes'] = args.notes
    
    # Update context
    state_manager.update_context(args.key, **context_data)
    print(f"Context updated: {args.key}")


def show_context_command(args):
    """Show current context."""
    state_manager = get_state_manager(args.db_path)
    
    if args.key:
        context = state_manager.get_context(args.key)
    else:
        context = state_manager.get_context()
    
    if not context:
        print("No context found.")
        return
    
    print("\n=== CONTEXT ===")
    print(f"Key: {context.get('context_key', 'N/A')}")
    print(f"Task: {context.get('task', 'N/A')}")
    print(f"Goal: {context.get('goal', 'N/A')}")
    print(f"Progress: {context.get('progress', 'N/A')}")
    
    if args.verbose:
        print(f"Assumptions: {context.get('assumptions', 'N/A')}")
        print(f"Problems: {context.get('problems', 'N/A')}")
    
    print(f"Next Steps: {context.get('next_steps', 'N/A')}")
    
    if args.verbose and context.get('notes'):
        print(f"Notes: {context.get('notes', 'N/A')}")
    
    print(f"Last Updated: {context.get('updated_at', 'N/A')}")
    print("==============")


def list_contexts_command(args):
    """List all contexts."""
    state_manager = get_state_manager(args.db_path)
    
    # This requires a new method in TestStateManager
    state_manager.cursor.execute("""
        SELECT context_key, task, updated_at 
        FROM context 
        ORDER BY updated_at DESC
    """)
    contexts = state_manager.cursor.fetchall()
    
    if not contexts:
        print("No contexts found.")
        return
    
    print(f"Found {len(contexts)} contexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"{i}. {ctx['context_key']} - {ctx['task']} ({ctx['updated_at']})")


def add_doc_command(args):
    """Add documentation."""
    state_manager = get_state_manager(args.db_path)
    
    # Handle file input
    content = args.content
    if args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    
    # Add documentation
    state_manager.add_documentation(
        args.topic,
        content,
        summary=args.summary,
        source=args.source,
        examples=args.examples,
        related_topics=args.related_topics,
        importance=args.importance
    )
    
    print(f"Documentation added: {args.topic}")


def update_doc_command(args):
    """Update existing documentation."""
    state_manager = get_state_manager(args.db_path)
    
    # Get existing doc
    doc = state_manager.get_documentation(topic=args.topic)
    if not doc:
        print(f"No documentation found for topic: {args.topic}")
        return
    
    # Prepare updates
    content = args.content or doc.get('content', '')
    if args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    
    summary = args.summary or doc.get('summary', '')
    source = args.source or doc.get('source', '')
    examples = args.examples or doc.get('examples', '')
    related_topics = args.related_topics or doc.get('related_topics', '')
    importance = args.importance or doc.get('importance', 5)
    
    # Update documentation
    state_manager.add_documentation(
        args.topic,
        content,
        summary=summary,
        source=source,
        examples=examples,
        related_topics=related_topics,
        importance=importance
    )
    
    print(f"Documentation updated: {args.topic}")


def show_doc_command(args):
    """Show documentation."""
    state_manager = get_state_manager(args.db_path)
    
    if args.topic:
        doc = state_manager.get_documentation(topic=args.topic)
        if not doc:
            print(f"No documentation found for topic: {args.topic}")
            return
        
        print(f"\n=== DOCUMENTATION: {doc.get('topic')} ===")
        if args.summary_only:
            print(f"Summary: {doc.get('summary', 'N/A')}")
        else:
            print(doc.get('content', 'N/A'))
            
            if doc.get('examples') and not args.no_examples:
                print("\nExamples:")
                print(doc.get('examples', 'N/A'))
                
            if args.verbose:
                print(f"\nSource: {doc.get('source', 'N/A')}")
                print(f"Related Topics: {doc.get('related_topics', 'N/A')}")
                print(f"Importance: {doc.get('importance', 'N/A')}")
                print(f"Created: {doc.get('created_at', 'N/A')}")
                print(f"Updated: {doc.get('updated_at', 'N/A')}")
                
        print("===========================")
    
    elif args.search:
        results = state_manager.get_documentation(search_term=args.search)
        if not results:
            print(f"No documentation found matching '{args.search}'")
            return
        
        print(f"\nFound {len(results)} documentation topics matching '{args.search}':")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.get('topic')} - {doc.get('summary', 'N/A')}")
    
    else:
        topics = state_manager.get_documentation()
        if not topics:
            print("No documentation available.")
            return
        
        print(f"\nAvailable documentation topics ({len(topics)}):")
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic.get('topic')} - {topic.get('summary', 'N/A')}")


def delete_doc_command(args):
    """Delete documentation."""
    state_manager = get_state_manager(args.db_path)
    
    # This requires a new method in TestStateManager
    with state_manager.transaction():
        state_manager.cursor.execute("DELETE FROM documentation WHERE topic = ?", (args.topic,))
    
    print(f"Documentation deleted: {args.topic}")


def load_project_docs_command(args):
    """Load documentation from project docs directory."""
    state_manager = get_state_manager(args.db_path)
    docs_path = args.path or "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/docs"
    
    docs_path = Path(docs_path)
    if not docs_path.exists():
        print(f"Documentation path does not exist: {docs_path}")
        return
    
    loaded_count = 0
    for doc_file in docs_path.glob("**/*.md"):
        # Skip if already loaded and not forcing reload
        if not args.force:
            existing_doc = state_manager.get_documentation(topic=doc_file.stem)
            if existing_doc:
                continue
                
        with open(doc_file, 'r') as f:
            content = f.read()
            
            # Extract summary (first paragraph or first line)
            lines = content.split('\n')
            summary = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    summary = line[:100] + ('...' if len(line) > 100 else '')
                    break
            
            # Get file modification time
            mod_time = doc_file.stat().st_mtime
            
            # Add documentation
            state_manager.add_documentation(
                doc_file.stem,
                content,
                summary=summary,
                source=str(doc_file),
                importance=8,  # High importance for project docs
            )
            loaded_count += 1
    
    print(f"Loaded {loaded_count} documentation files from {docs_path}")


def report_command(args):
    """Generate report."""
    state_manager = get_state_manager(args.db_path)
    
    if args.output:
        state_manager.generate_report(args.output)
        print(f"Report generated: {args.output}")
    else:
        state_manager.generate_report()


def export_command(args):
    """Export database to JSON."""
    state_manager = get_state_manager(args.db_path)
    
    # Build export data
    export_data = {
        'state': {},
        'context': state_manager.get_context(),
        'documentation': [],
        'checkpoints': state_manager.get_checkpoint_history(),
        'verification': state_manager.get_verification_history(),
        'metadata': {},
        'extraction_stats': state_manager.get_extraction_stats(),
        'repo_stats': state_manager.get_repo_stats(),
        'export_timestamp': datetime.datetime.now().isoformat()
    }
    
    # Get all state
    state_manager.cursor.execute("SELECT key, value, value_type FROM state")
    for row in state_manager.cursor.fetchall():
        export_data['state'][row['key']] = json.loads(row['value'])
    
    # Get all documentation
    for doc in state_manager.get_documentation():
        full_doc = state_manager.get_documentation(topic=doc['topic'])
        if full_doc:
            export_data['documentation'].append(full_doc)
    
    # Get all metadata
    state_manager.cursor.execute("SELECT key, value FROM metadata")
    for row in state_manager.cursor.fetchall():
        export_data['metadata'][row['key']] = json.loads(row['value'])
    
    # Export to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Database exported to: {args.output}")
    else:
        print(json.dumps(export_data, indent=2))


def import_command(args):
    """Import database from JSON."""
    state_manager = get_state_manager(args.db_path)
    
    # Read import file
    with open(args.input, 'r') as f:
        import_data = json.load(f)
    
    # Import state
    for key, value in import_data.get('state', {}).items():
        state_manager.set(key, value)
    
    # Import documentation
    for doc in import_data.get('documentation', []):
        state_manager.add_documentation(
            doc['topic'],
            doc.get('content', ''),
            summary=doc.get('summary', ''),
            source=doc.get('source', ''),
            examples=doc.get('examples', ''),
            related_topics=doc.get('related_topics', ''),
            importance=doc.get('importance', 5)
        )
    
    # Import metadata
    for key, value in import_data.get('metadata', {}).items():
        state_manager.set_metadata(key, value)
    
    # Set checkpoint for import
    state_manager.set_checkpoint(
        "import",
        f"Imported from {args.input} at {datetime.datetime.now().isoformat()}"
    )
    
    print(f"Database imported from: {args.input}")


def reminder_command(args):
    """Show a reminder of current context."""
    state_manager = get_state_manager(args.db_path)
    state_manager.remind_me()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TestStateManager CLI for robust state management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        TestStateManager CLI
        
        This CLI provides comprehensive management capabilities for the TestStateManager,
        a system designed to maintain state between test steps and address limitations
        in context maintenance.
        
        Example usage:
            # Initialize a new database
            python -m dualipa.extraction.cli init --db-path extraction_state.db
            
            # Set a new context 
            python -m dualipa.extraction.cli context current-task --task "Extracting Python files" --goal "Complete repository extraction"
            
            # Remind me what I'm doing
            python -m dualipa.extraction.cli remind
            
            # Load project documentation
            python -m dualipa.extraction.cli load-docs
            
            # Add documentation
            python -m dualipa.extraction.cli add-doc "extraction-process" "Detailed steps..." --summary "How extraction works"
            
            # Search documentation
            python -m dualipa.extraction.cli show-doc --search "extraction"
        """)
    )
    
    parser.add_argument(
        "--db-path",
        help="Path to the state database (defaults to in-memory)",
        default=None
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize database
    init_parser = subparsers.add_parser("init", help="Initialize a new state database")
    
    # Key-value store commands
    set_parser = subparsers.add_parser("set", help="Set a value in the state store")
    set_parser.add_argument("key", help="Key to set")
    set_parser.add_argument("value", help="Value to set")
    set_parser.add_argument(
        "--type",
        choices=["str", "int", "float", "bool", "json"],
        default="str",
        help="Value type"
    )
    
    get_parser = subparsers.add_parser("get", help="Get a value from the state store")
    get_parser.add_argument("key", help="Key to get")
    get_parser.add_argument("--default", help="Default value if key doesn't exist")
    get_parser.add_argument("--quiet", "-q", action="store_true", help="Only output the value")
    
    list_parser = subparsers.add_parser("list", help="List all keys in the state store")
    
    delete_parser = subparsers.add_parser("delete", help="Delete a key from the state store")
    delete_parser.add_argument("key", help="Key to delete")
    
    # Checkpoint commands
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Create a checkpoint")
    checkpoint_parser.add_argument("name", help="Checkpoint name")
    checkpoint_parser.add_argument("--description", "-d", help="Checkpoint description")
    
    list_checkpoints_parser = subparsers.add_parser("list-checkpoints", help="List all checkpoints")
    
    # Context commands
    context_parser = subparsers.add_parser("context", help="Update the current context")
    context_parser.add_argument("key", help="Context key")
    context_parser.add_argument("--task", "-t", help="What I'm currently doing")
    context_parser.add_argument("--goal", "-g", help="Why I'm doing it")
    context_parser.add_argument("--progress", "-p", help="Current step in the process")
    context_parser.add_argument("--assumptions", "-a", help="Key assumptions I'm making")
    context_parser.add_argument("--problems", help="Issues I've encountered")
    context_parser.add_argument("--next-steps", "-n", help="What to do next")
    context_parser.add_argument("--notes", help="Additional context")
    
    show_context_parser = subparsers.add_parser("show-context", help="Show current context")
    show_context_parser.add_argument("--key", "-k", help="Specific context key to show")
    show_context_parser.add_argument("--verbose", "-v", action="store_true", help="Show all context fields")
    
    list_contexts_parser = subparsers.add_parser("list-contexts", help="List all contexts")
    
    # Documentation commands
    add_doc_parser = subparsers.add_parser("add-doc", help="Add documentation")
    add_doc_parser.add_argument("topic", help="Documentation topic")
    add_doc_parser.add_argument("content", nargs="?", help="Documentation content")
    add_doc_parser.add_argument("--file", "-f", help="Read content from file")
    add_doc_parser.add_argument("--summary", "-s", help="Short summary")
    add_doc_parser.add_argument("--source", help="Source of documentation")
    add_doc_parser.add_argument("--examples", "-e", help="Usage examples")
    add_doc_parser.add_argument("--related-topics", "-r", help="Related topics")
    add_doc_parser.add_argument(
        "--importance",
        "-i",
        type=int,
        choices=range(1, 11),
        default=5,
        help="Importance level (1-10)"
    )
    
    update_doc_parser = subparsers.add_parser("update-doc", help="Update existing documentation")
    update_doc_parser.add_argument("topic", help="Documentation topic")
    update_doc_parser.add_argument("content", nargs="?", help="Documentation content")
    update_doc_parser.add_argument("--file", "-f", help="Read content from file")
    update_doc_parser.add_argument("--summary", "-s", help="Short summary")
    update_doc_parser.add_argument("--source", help="Source of documentation")
    update_doc_parser.add_argument("--examples", "-e", help="Usage examples")
    update_doc_parser.add_argument("--related-topics", "-r", help="Related topics")
    update_doc_parser.add_argument(
        "--importance",
        "-i",
        type=int,
        choices=range(1, 11),
        help="Importance level (1-10)"
    )
    
    show_doc_parser = subparsers.add_parser("show-doc", help="Show documentation")
    show_doc_parser.add_argument("--topic", "-t", help="Specific topic to show")
    show_doc_parser.add_argument("--search", "-s", help="Search term in documentation")
    show_doc_parser.add_argument("--summary-only", action="store_true", help="Show only the summary")
    show_doc_parser.add_argument("--no-examples", action="store_true", help="Don't show examples")
    show_doc_parser.add_argument("--verbose", "-v", action="store_true", help="Show all metadata")
    
    delete_doc_parser = subparsers.add_parser("delete-doc", help="Delete documentation")
    delete_doc_parser.add_argument("topic", help="Documentation topic to delete")
    
    load_project_docs_parser = subparsers.add_parser("load-docs", help="Load documentation from project docs directory")
    load_project_docs_parser.add_argument("--path", "-p", help="Path to docs directory")
    load_project_docs_parser.add_argument("--force", "-f", action="store_true", help="Force reload of existing docs")
    
    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--output", "-o", help="Output file path")
    
    # Import/Export commands
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("--output", "-o", help="Output file path")
    
    import_parser = subparsers.add_parser("import", help="Import database from JSON")
    import_parser.add_argument("input", help="Input file path")
    
    # Reminder command
    reminder_parser = subparsers.add_parser("remind", help="Show a reminder of what I'm currently doing")
    
    # Parse args and dispatch
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch to command handlers
    if args.command == "init":
        initialize_database(args)
    elif args.command == "set":
        set_command(args)
    elif args.command == "get":
        get_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "delete":
        delete_command(args)
    elif args.command == "checkpoint":
        checkpoint_command(args)
    elif args.command == "list-checkpoints":
        list_checkpoints_command(args)
    elif args.command == "context":
        context_command(args)
    elif args.command == "show-context":
        show_context_command(args)
    elif args.command == "list-contexts":
        list_contexts_command(args)
    elif args.command == "add-doc":
        add_doc_command(args)
    elif args.command == "update-doc":
        update_doc_command(args)
    elif args.command == "show-doc":
        show_doc_command(args)
    elif args.command == "delete-doc":
        delete_doc_command(args)
    elif args.command == "load-docs":
        load_project_docs_command(args)
    elif args.command == "report":
        report_command(args)
    elif args.command == "export":
        export_command(args)
    elif args.command == "import":
        import_command(args)
    elif args.command == "remind":
        reminder_command(args)


if __name__ == "__main__":
    main()