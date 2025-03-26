#!/usr/bin/env python3
"""
Agent Memory CLI

Command-line interface for the agent memory system.
This tool helps the agent maintain state between interactions.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the repository to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_tools.agent_memory.memory_store import get_memory_store

def format_value(value: Any) -> str:
    """Format a value for display in the CLI."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value)

def handle_remember(args):
    """Handle the remember command."""
    memory = get_memory_store(args.db_path)
    
    # Determine the value type and parse accordingly
    if args.value.startswith('{') and args.value.endswith('}'):
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
    elif args.value.startswith('[') and args.value.endswith(']'):
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
    elif args.value.lower() == 'true':
        value = True
    elif args.value.lower() == 'false':
        value = False
    elif args.value.isdigit():
        value = int(args.value)
    elif args.value.replace('.', '', 1).isdigit():
        value = float(args.value)
    else:
        value = args.value
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print(f"Error: Metadata must be valid JSON. Got: {args.metadata}")
            return 1
    
    success = memory.remember(args.key, value, metadata)
    if success:
        print(f"Successfully stored memory: {args.key}")
        return 0
    else:
        print(f"Failed to store memory: {args.key}")
        return 1

def handle_recall(args):
    """Handle the recall command."""
    memory = get_memory_store(args.db_path)
    
    if args.with_metadata:
        value, metadata = memory.recall_with_metadata(args.key)
        if value is None:
            print(f"Memory not found: {args.key}")
            return 1
        
        print(f"Key: {args.key}")
        print(f"Value: {format_value(value)}")
        print(f"Metadata: {format_value(metadata)}")
    else:
        value = memory.recall(args.key)
        if value is None:
            print(f"Memory not found: {args.key}")
            return 1
        
        print(format_value(value))
    
    return 0

def handle_forget(args):
    """Handle the forget command."""
    memory = get_memory_store(args.db_path)
    
    success = memory.forget(args.key)
    if success:
        print(f"Successfully removed memory: {args.key}")
        return 0
    else:
        print(f"Failed to remove memory: {args.key}")
        return 1

def handle_list(args):
    """Handle the list command."""
    memory = get_memory_store(args.db_path)
    
    keys = memory.list_keys(args.pattern)
    if not keys:
        print("No memories found")
        return 0
    
    if args.verbose:
        for key in keys:
            value, metadata = memory.recall_with_metadata(key)
            print(f"Key: {key}")
            print(f"Value: {format_value(value)}")
            print(f"Metadata: {format_value(metadata)}")
            print("-" * 40)
    else:
        for key in keys:
            print(key)
    
    return 0

def handle_add_conversation(args):
    """Handle the add-conversation command."""
    memory = get_memory_store(args.db_path)
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print(f"Error: Metadata must be valid JSON. Got: {args.metadata}")
            return 1
    
    success = memory.add_conversation_entry(args.role, args.content, metadata)
    if success:
        print(f"Successfully added conversation entry")
        return 0
    else:
        print(f"Failed to add conversation entry")
        return 1

def handle_get_conversation(args):
    """Handle the get-conversation command."""
    memory = get_memory_store(args.db_path)
    
    history = memory.get_conversation_history(args.limit)
    if not history:
        print("No conversation history found")
        return 0
    
    if args.json:
        print(json.dumps(history, indent=2))
    else:
        for entry in history:
            timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {entry['role']}: {entry['content']}")
    
    return 0

def handle_add_task(args):
    """Handle the add-task command."""
    memory = get_memory_store(args.db_path)
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print(f"Error: Metadata must be valid JSON. Got: {args.metadata}")
            return 1
    
    task_id = memory.add_task(args.description, args.status, metadata)
    if task_id != -1:
        print(f"Successfully added task with ID: {task_id}")
        return 0
    else:
        print(f"Failed to add task")
        return 1

def handle_update_task(args):
    """Handle the update-task command."""
    memory = get_memory_store(args.db_path)
    
    success = memory.update_task_status(args.id, args.status)
    if success:
        print(f"Successfully updated task {args.id} to {args.status}")
        return 0
    else:
        print(f"Failed to update task {args.id}")
        return 1

def handle_get_tasks(args):
    """Handle the get-tasks command."""
    memory = get_memory_store(args.db_path)
    
    tasks = memory.get_tasks(args.status)
    if not tasks:
        print("No tasks found")
        return 0
    
    if args.json:
        print(json.dumps(tasks, indent=2))
    else:
        for task in tasks:
            created = datetime.fromtimestamp(task["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
            updated = datetime.fromtimestamp(task["updated_at"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"ID: {task['id']}")
            print(f"Description: {task['description']}")
            print(f"Status: {task['status']}")
            print(f"Created: {created}")
            print(f"Updated: {updated}")
            print("-" * 40)
    
    return 0

def handle_query(args):
    """Handle the query command."""
    memory = get_memory_store(args.db_path)
    
    # Parse params if provided
    params = None
    if args.params:
        try:
            params = tuple(json.loads(args.params))
        except json.JSONDecodeError:
            print(f"Error: Params must be valid JSON array. Got: {args.params}")
            return 1
    
    results = memory.execute_query(args.sql, params)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results:
            print(row)
    
    return 0

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Agent Memory CLI")
    parser.add_argument("--db-path", default="agent_memory.db", help="Path to the SQLite database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # remember command
    remember_parser = subparsers.add_parser("remember", help="Store a memory")
    remember_parser.add_argument("key", help="Key to store the memory under")
    remember_parser.add_argument("value", help="Value to store")
    remember_parser.add_argument("--metadata", help="JSON metadata to store with the memory")
    remember_parser.set_defaults(func=handle_remember)
    
    # recall command
    recall_parser = subparsers.add_parser("recall", help="Retrieve a memory")
    recall_parser.add_argument("key", help="Key to retrieve")
    recall_parser.add_argument("--with-metadata", action="store_true", help="Include metadata in the output")
    recall_parser.set_defaults(func=handle_recall)
    
    # forget command
    forget_parser = subparsers.add_parser("forget", help="Delete a memory")
    forget_parser.add_argument("key", help="Key to delete")
    forget_parser.set_defaults(func=handle_forget)
    
    # list command
    list_parser = subparsers.add_parser("list", help="List memory keys")
    list_parser.add_argument("--pattern", help="SQL LIKE pattern to filter keys")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show values and metadata")
    list_parser.set_defaults(func=handle_list)
    
    # add-conversation command
    add_conversation_parser = subparsers.add_parser("add-conversation", help="Add a conversation entry")
    add_conversation_parser.add_argument("role", choices=["user", "assistant"], help="Role of the speaker")
    add_conversation_parser.add_argument("content", help="Content of the message")
    add_conversation_parser.add_argument("--metadata", help="JSON metadata to store with the entry")
    add_conversation_parser.set_defaults(func=handle_add_conversation)
    
    # get-conversation command
    get_conversation_parser = subparsers.add_parser("get-conversation", help="Get conversation history")
    get_conversation_parser.add_argument("--limit", type=int, help="Maximum number of entries to return")
    get_conversation_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_conversation_parser.set_defaults(func=handle_get_conversation)
    
    # add-task command
    add_task_parser = subparsers.add_parser("add-task", help="Add a task")
    add_task_parser.add_argument("description", help="Description of the task")
    add_task_parser.add_argument("--status", default="pending", help="Initial status of the task")
    add_task_parser.add_argument("--metadata", help="JSON metadata to store with the task")
    add_task_parser.set_defaults(func=handle_add_task)
    
    # update-task command
    update_task_parser = subparsers.add_parser("update-task", help="Update a task's status")
    update_task_parser.add_argument("id", type=int, help="ID of the task to update")
    update_task_parser.add_argument("status", help="New status for the task")
    update_task_parser.set_defaults(func=handle_update_task)
    
    # get-tasks command
    get_tasks_parser = subparsers.add_parser("get-tasks", help="Get task list")
    get_tasks_parser.add_argument("--status", help="Filter tasks by status")
    get_tasks_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_tasks_parser.set_defaults(func=handle_get_tasks)
    
    # query command
    query_parser = subparsers.add_parser("query", help="Execute a SQL query")
    query_parser.add_argument("sql", help="SQL query to execute")
    query_parser.add_argument("--params", help="JSON array of parameters for the query")
    query_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    query_parser.set_defaults(func=handle_query)
    
    args = parser.parse_args()
    
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())