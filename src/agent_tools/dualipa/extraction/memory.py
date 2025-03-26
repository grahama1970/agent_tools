#!/usr/bin/env python3
"""
Memory Management Helper Script

Provides simplified access to the TestStateManager for AI use cases,
focusing on reliable memory persistence and context awareness.
"""

import os
import sys
import argparse
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .test_state_manager import (
    get_state_manager, 
    what_am_i_doing, 
    remember_context, 
    add_docs, 
    get_docs
)


def remember(task, goal, progress, next_steps, notes=None):
    """Remember my current context with simple arguments."""
    manager = get_state_manager()
    context_key = f"memory-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    kwargs = {
        'task': task,
        'goal': goal,
        'progress': progress,
        'next_steps': next_steps
    }
    
    if notes:
        kwargs['notes'] = notes
        
    manager.update_context(context_key, **kwargs)
    return f"Context saved as {context_key}"


def recall(search_term=None, last=False):
    """Recall context information with search."""
    manager = get_state_manager()
    
    if last:
        context = manager.get_context()
        if not context:
            return "No context found."
            
        return {
            'task': context.get('task'),
            'goal': context.get('goal'),
            'progress': context.get('progress'),
            'next_steps': context.get('next_steps')
        }
    
    if search_term:
        # This requires a new method in TestStateManager
        manager.cursor.execute("""
            SELECT context_key, task, goal, progress, next_steps, updated_at
            FROM context
            WHERE 
                task LIKE ? OR
                goal LIKE ? OR
                progress LIKE ? OR
                next_steps LIKE ? OR
                notes LIKE ?
            ORDER BY updated_at DESC
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", 
              f"%{search_term}%", f"%{search_term}%"))
        
        results = []
        for row in manager.cursor.fetchall():
            results.append({
                'key': row['context_key'],
                'task': row['task'],
                'updated_at': row['updated_at']
            })
        
        if not results:
            return f"No contexts found matching '{search_term}'."
            
        return results
    
    # Default to returning most recent context
    return recall(last=True)


def save_docs(topic, content, summary=None, importance=7):
    """Save documentation for future reference."""
    add_docs(topic, content, summary=summary or "", importance=importance)
    return f"Documentation saved: {topic}"


def find_docs(topic=None, search=None):
    """Find documentation by topic or search term."""
    manager = get_state_manager()
    
    if topic:
        doc = manager.get_documentation(topic=topic)
        if not doc:
            return f"No documentation found for topic: {topic}"
        return doc
    
    if search:
        results = manager.get_documentation(search_term=search)
        if not results:
            return f"No documentation found matching '{search}'"
        return results
    
    # Get all topics
    topics = manager.get_documentation()
    if not topics:
        return "No documentation available."
    return topics


def load_project_docs(force=False):
    """Load documentation from the project docs directory."""
    manager = get_state_manager()
    docs_path = Path("/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/docs")
    
    if not docs_path.exists():
        return f"Documentation path does not exist: {docs_path}"
    
    loaded_count = 0
    for doc_file in docs_path.glob("**/*.md"):
        # Skip if already loaded and not forcing reload
        if not force:
            existing_doc = manager.get_documentation(topic=doc_file.stem)
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
            
            # Add documentation
            manager.add_documentation(
                doc_file.stem,
                content,
                summary=summary,
                source=str(doc_file),
                importance=8,  # High importance for project docs
            )
            loaded_count += 1
    
    return f"Loaded {loaded_count} documentation files from {docs_path}"


def get_verification_summary():
    """Get a summary of verification results."""
    manager = get_state_manager()
    
    verifications = manager.get_verification_history()
    total = len(verifications)
    passed = sum(1 for v in verifications if v['passed'])
    failed = total - passed
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': (passed / total * 100) if total > 0 else 0,
        'recent': verifications[-5:] if verifications else []
    }


def log_error(error_type, details, recovery_action=None):
    """Log an error pattern for future prevention."""
    manager = get_state_manager()
    
    # Store in metadata
    errors = manager.get_metadata('error_patterns', [])
    errors.append({
        'error_type': error_type,
        'details': details,
        'recovery_action': recovery_action,
        'timestamp': datetime.datetime.now().isoformat()
    })
    
    manager.set_metadata('error_patterns', errors)
    return f"Error logged: {error_type}"


def suggest_recovery(error_type=None):
    """Suggest recovery based on past similar errors."""
    manager = get_state_manager()
    
    errors = manager.get_metadata('error_patterns', [])
    
    if error_type:
        matching = [e for e in errors if e['error_type'] == error_type]
        if not matching:
            return f"No recovery suggestions found for error type: {error_type}"
        
        # Return most recent matching error
        return matching[-1].get('recovery_action', 'No recovery action specified')
    
    # Return all error types
    error_types = set(e['error_type'] for e in errors)
    return list(error_types)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Memory management helper for AI operations"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Remember command
    remember_parser = subparsers.add_parser("remember", help="Remember current context")
    remember_parser.add_argument("task", help="What I'm currently doing")
    remember_parser.add_argument("goal", help="Why I'm doing it")
    remember_parser.add_argument("progress", help="Current step in the process")
    remember_parser.add_argument("next_steps", help="What to do next")
    remember_parser.add_argument("--notes", "-n", help="Additional notes")
    
    # Recall command
    recall_parser = subparsers.add_parser("recall", help="Recall context information")
    recall_parser.add_argument("--search", "-s", help="Search term in contexts")
    recall_parser.add_argument("--last", "-l", action="store_true", help="Get most recent context")
    
    # Save docs command
    save_docs_parser = subparsers.add_parser("save-docs", help="Save documentation")
    save_docs_parser.add_argument("topic", help="Documentation topic")
    save_docs_parser.add_argument("content", help="Documentation content")
    save_docs_parser.add_argument("--summary", "-s", help="Documentation summary")
    save_docs_parser.add_argument(
        "--importance",
        "-i",
        type=int,
        choices=range(1, 11),
        default=7,
        help="Importance level (1-10)"
    )
    
    # Find docs command
    find_docs_parser = subparsers.add_parser("find-docs", help="Find documentation")
    find_docs_parser.add_argument("--topic", "-t", help="Specific topic to find")
    find_docs_parser.add_argument("--search", "-s", help="Search term in documentation")
    
    # Load project docs command
    load_docs_parser = subparsers.add_parser("load-docs", help="Load project documentation")
    load_docs_parser.add_argument("--force", "-f", action="store_true", help="Force reload existing docs")
    
    # Verification summary command
    verify_parser = subparsers.add_parser("verify-summary", help="Get verification summary")
    
    # Error logging command
    error_parser = subparsers.add_parser("log-error", help="Log an error pattern")
    error_parser.add_argument("error_type", help="Type of error")
    error_parser.add_argument("details", help="Error details")
    error_parser.add_argument("--recovery", "-r", help="Recovery action")
    
    # Suggest recovery command
    recovery_parser = subparsers.add_parser("suggest-recovery", help="Suggest error recovery")
    recovery_parser.add_argument("--error-type", "-e", help="Type of error")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return
    
    # Dispatch to command handlers
    result = None
    
    if args.command == "remember":
        result = remember(args.task, args.goal, args.progress, args.next_steps, args.notes)
    elif args.command == "recall":
        result = recall(args.search, args.last)
    elif args.command == "save-docs":
        result = save_docs(args.topic, args.content, args.summary, args.importance)
    elif args.command == "find-docs":
        result = find_docs(args.topic, args.search)
    elif args.command == "load-docs":
        result = load_project_docs(args.force)
    elif args.command == "verify-summary":
        result = get_verification_summary()
    elif args.command == "log-error":
        result = log_error(args.error_type, args.details, args.recovery)
    elif args.command == "suggest-recovery":
        result = suggest_recovery(args.error_type)
    
    # Print result
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2))
    else:
        print(result)


# Simple aliases for better AI context management
def think(thought):
    """Record a thought for future reference."""
    return save_docs("thought-" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), thought)


def remind_me():
    """Get a reminder of what I'm currently doing."""
    what_am_i_doing()


def note(topic, content):
    """Take a quick note about something."""
    return save_docs(topic, content)


def recall_thought(search=None):
    """Recall a previous thought."""
    if search:
        return find_docs(search=search)
    else:
        # Try to find thoughts
        return find_docs(search="thought-")


if __name__ == "__main__":
    main()