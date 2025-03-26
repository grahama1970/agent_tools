#!/usr/bin/env python3
"""
Initialize AI Memory System

This script initializes the AI memory system, loads documentation,
and sets up the vector database for semantic search.

Usage:
  python initialize_memory.py [--db-path DB_PATH] [--skip-docs] [--force-reload]
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("memory_init.log")
    ]
)
logger = logging.getLogger("memory_init")

# Add repository root to path
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

try:
    # Import memory system
    from src.agent_tools.dualipa.extraction.ai_memory_system import (
        initialize_memory_system,
        load_project_docs,
        get_system_status,
        save_docs,
        find_docs,
        recall,
        log_error,
        suggest_recovery,
        batch_operation,
        vector_store
    )
    from src.agent_tools.dualipa.extraction.test_state_manager import get_state_manager
    
    # Flag for importing documentation
    DOCS_IMPORTED = True
    
except ImportError as e:
    logger.error(f"Error importing memory system: {e}")
    DOCS_IMPORTED = False
    print(f"Failed to import memory system: {e}")
    sys.exit(1)


def create_initial_documentation(manager, force=False):
    """Create initial documentation about the memory system itself."""
    # Check if already created
    if not force and manager.get_documentation(topic="memory-system"):
        logger.info("Initial documentation already exists. Skipping.")
        return False
        
    # Create documentation about the memory system
    save_docs(
        "memory-system",
        """# AI Memory System
        
The AI Memory System provides persistent memory capabilities for AI assistants, 
overcoming context limitations and enabling sophisticated knowledge management.

## Key Features

1. **Context Management**: Track the current task and progress
2. **Documentation Storage**: Save and retrieve documentation with semantic search
3. **Error Pattern Learning**: Log errors and suggest recovery actions
4. **Vector-based Search**: Find semantically similar information using embeddings
5. **Relationship Discovery**: Automatically find connections between related knowledge

## Basic Usage

Remember what you're doing:
```
remember "task description" "goal" "current progress" "next steps"
```

Recall your current context:
```
recall --last
```

Search documentation:
```
find-docs --search "query" --semantic
```

Log an error pattern:
```
log-error "error type" "details" --recovery "how to fix it"
```

Get system status:
```
status
```

## Important Concepts

- **Vector Search**: Information is converted to vector embeddings for semantic similarity search
- **Bidirectional Relationships**: Documents can automatically discover relationships
- **Context Staleness**: System tracks when context is outdated based on priority levels
- **Batch Operations**: Multiple memory operations can be performed in a single request
        """,
        summary="Overview of the AI Memory System for persistent AI memory",
        importance=10,
        tags=["system", "memory", "documentation"]
    )
    
    # Create documentation about embeddings
    save_docs(
        "embedding-capabilities",
        """# Embedding Capabilities
        
The AI Memory System uses vector embeddings to enable semantic search and relationship discovery.

## Available Embedding Methods

1. **Transformer Embeddings**: When available, uses proper transformer embeddings from the fetch_docs module
2. **TF-IDF Fallback**: If transformer embeddings are not available, falls back to TF-IDF vectorization

## Semantic Search

Semantic search allows finding relevant information even when exact keyword matches are not present.
The system uses cosine similarity between vector embeddings to identify the most semantically similar content.

## Automatic Relationship Discovery

When proper embeddings are available, the system can automatically discover relationships between documents
based on semantic similarity. This helps build a knowledge graph without explicit manual linking.

## Usage Tips

- Enable semantic search with the `--semantic` flag in search commands
- Higher similarity scores (closer to 1.0) indicate stronger relationships
- The system will improve over time as more documents are added and related
        """,
        summary="Overview of the embedding and semantic search capabilities",
        importance=8,
        tags=["system", "embeddings", "search"]
    )
    
    # Add documentation about CLI usage
    save_docs(
        "memory-cli",
        """# Memory System CLI
        
The memory system provides a command-line interface for interacting with the persistent memory.

## Global Arguments

- `--db-path PATH`: Path to the database file (defaults to in-memory)

## Available Commands

### init
Initialize the memory system
```
initialize_memory.py init [--skip-docs]
```

### remember
Remember current context
```
initialize_memory.py remember "task" "goal" "progress" "next steps" [--notes "notes"] [--priority 1-10] [--tags tag1 tag2]
```

### recall
Recall context information
```
initialize_memory.py recall [--search "term"] [--last] [--tag "tag"] [--days N] [--priority-min N] [--semantic]
```

### save-docs
Save documentation
```
initialize_memory.py save-docs "topic" "content" [--summary "summary"] [--importance 1-10] [--source "source"] [--tags tag1 tag2] [--related topic1 topic2]
```

### find-docs
Find documentation
```
initialize_memory.py find-docs [--topic "topic"] [--search "term"] [--tag "tag"] [--importance-min N] [--no-semantic] [--max-results N]
```

### log-error
Log an error pattern
```
initialize_memory.py log-error "error_type" "details" [--recovery "action"] [--severity 1-10] [--tags tag1 tag2]
```

### suggest-recovery
Suggest recovery for an error
```
initialize_memory.py suggest-recovery [--error-type "type"] [--details "details"] [--no-semantic]
```

### batch
Execute multiple operations in a batch
```
initialize_memory.py batch operations.json
```

### status
Get memory system status
```
initialize_memory.py status
```
        """,
        summary="Guide to using the memory system command-line interface",
        importance=9,
        tags=["system", "cli", "usage"]
    )
    
    return True


def main():
    """Main function to initialize the memory system."""
    parser = argparse.ArgumentParser(description="Initialize AI Memory System")
    
    parser.add_argument(
        "--db-path",
        help="Path to the state database (defaults to memory.db in current directory)",
        default="memory.db"
    )
    
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip loading project documentation"
    )
    
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload existing documentation"
    )
    
    # Allow command to be passed directly
    parser.add_argument(
        "command",
        nargs="?",
        default="init",
        help="Command to execute (init, status, find-docs, etc.)"
    )
    
    # Pass remaining arguments to the command
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the command"
    )
    
    args = parser.parse_args()
    
    db_path = args.db_path
    
    try:
        # Execute the command
        if args.command == "init":
            logger.info(f"Initializing memory system with database: {db_path}")
            
            # Initialize memory system
            result = initialize_memory_system(db_path, not args.skip_docs)
            print(f"Memory system initialization: {result}")
            
            # Get state manager
            manager = get_state_manager(db_path)
            
            # Create initial documentation about the memory system
            if create_initial_documentation(manager, args.force_reload):
                print("Created initial memory system documentation")
            
            # Display status
            status = get_system_status()
            
            print("\nMemory System Status:")
            print(f"  Database: {db_path}")
            print(f"  Documents: {status['database']['documentation_entries']}")
            print(f"  Contexts: {status['database']['context_entries']}")
            
            # Vector store status
            if status['vector_store']['available']:
                print("\nVector Store:")
                print(f"  Method: {status['vector_store']['embedding_method']}")
                print(f"  Model: {status['vector_store']['embedding_model']}")
                print(f"  Documents: {status['vector_store']['document_count']}")
                print(f"  Proper embeddings: {'Yes' if status['vector_store']['proper_embeddings'] else 'No'}")
            else:
                print("\nVector Store: Not available")
            
            print("\nInitialization complete!")
            
        elif args.command == "status":
            # Initialize with existing database
            initialize_memory_system(db_path, False)
            
            # Display status
            status = get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.command == "find-docs":
            # Get the search term from args
            search_term = None
            topic = None
            tag = None
            importance_min = 0
            no_semantic = False
            max_results = 10
            
            # Parse the remaining arguments manually
            i = 0
            while i < len(args.args):
                arg = args.args[i]
                if arg in ["--search", "-s"] and i + 1 < len(args.args):
                    search_term = args.args[i + 1]
                    i += 2
                elif arg in ["--topic", "-t"] and i + 1 < len(args.args):
                    topic = args.args[i + 1]
                    i += 2
                elif arg == "--tag" and i + 1 < len(args.args):
                    tag = args.args[i + 1]
                    i += 2
                elif arg in ["--importance-min", "-i"] and i + 1 < len(args.args):
                    importance_min = int(args.args[i + 1])
                    i += 2
                elif arg == "--no-semantic":
                    no_semantic = True
                    i += 1
                elif arg in ["--max-results", "-m"] and i + 1 < len(args.args):
                    max_results = int(args.args[i + 1])
                    i += 2
                elif arg in ["--db-path"] and i + 1 < len(args.args):
                    db_path = args.args[i + 1]
                    i += 2
                else:
                    # Skip unknown arguments
                    i += 1
            
            # Initialize with existing database
            initialize_memory_system(db_path, False)
            
            # Find documentation
            results = find_docs(
                topic=topic,
                search=search_term,
                tag=tag,
                importance_min=importance_min,
                semantic=not no_semantic,
                max_results=max_results
            )
            
            # Print results
            if isinstance(results, (dict, list)):
                print(json.dumps(results, indent=2))
            else:
                print(results)
                
        elif args.command == "recall":
            # Get the arguments
            search_term = None
            last = False
            tag = None
            days = None
            priority_min = None
            semantic = False
            
            # Parse the remaining arguments manually
            i = 0
            while i < len(args.args):
                arg = args.args[i]
                if arg in ["--search", "-s"] and i + 1 < len(args.args):
                    search_term = args.args[i + 1]
                    i += 2
                elif arg in ["--last", "-l"]:
                    last = True
                    i += 1
                elif arg in ["--tag", "-t"] and i + 1 < len(args.args):
                    tag = args.args[i + 1]
                    i += 2
                elif arg in ["--days", "-d"] and i + 1 < len(args.args):
                    days = int(args.args[i + 1])
                    i += 2
                elif arg in ["--priority-min", "-p"] and i + 1 < len(args.args):
                    priority_min = int(args.args[i + 1])
                    i += 2
                elif arg in ["--semantic", "-v"]:
                    semantic = True
                    i += 1
                elif arg in ["--db-path"] and i + 1 < len(args.args):
                    db_path = args.args[i + 1]
                    i += 2
                else:
                    # Skip unknown arguments
                    i += 1
                        
            # Initialize with existing database
            initialize_memory_system(db_path, False)
            
            # Recall context
            results = recall(
                search_term=search_term,
                last=last,
                tag=tag,
                days=days,
                priority_min=priority_min,
                semantic=semantic
            )
            
            # Print results
            if isinstance(results, (dict, list)):
                print(json.dumps(results, indent=2))
            else:
                print(results)
        
        else:
            print(f"Unknown command: {args.command}")
            print("Available commands: init, status, find-docs, recall")
            return 1
                
    except Exception as e:
        logger.error(f"Error initializing memory system: {e}")
        print(f"Failed to initialize memory system: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())