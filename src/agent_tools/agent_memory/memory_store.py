#!/usr/bin/env python3
"""
Memory store for agent state management.

This module provides a simple SQLite-based memory store for the agent
to maintain state between interactions with the user.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_memory.log")
    ]
)
logger = logging.getLogger("agent_memory")

class MemoryStore:
    """SQLite-based memory store for agent state management."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        """
        Initialize the memory store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            metadata TEXT,
            created_at REAL,
            updated_at REAL
        )
        ''')
        
        # Create conversation table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            role TEXT,
            content TEXT,
            metadata TEXT
        )
        ''')
        
        # Create tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            status TEXT,
            created_at REAL,
            updated_at REAL,
            metadata TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Memory store initialized with database at {self.db_path}")
        
    def remember(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a memory by key.
        
        Args:
            key: Unique identifier for the memory
            value: Value to store
            metadata: Optional metadata about the memory
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Convert value and metadata to JSON strings
            value_json = json.dumps(value)
            metadata_json = json.dumps(metadata or {})
            
            # Check if key already exists
            cursor.execute("SELECT id FROM memories WHERE key = ?", (key,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing memory
                cursor.execute(
                    "UPDATE memories SET value = ?, metadata = ?, updated_at = ? WHERE key = ?",
                    (value_json, metadata_json, now, key)
                )
            else:
                # Create new memory
                cursor.execute(
                    "INSERT INTO memories (key, value, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (key, value_json, metadata_json, now, now)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored memory: {key}")
            return True
            
        except Exception as e:
            conn.close()
            logger.error(f"Error storing memory: {e}")
            return False
            
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory by key.
        
        Args:
            key: Unique identifier for the memory
            
        Returns:
            The stored value, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return json.loads(result[0])
            else:
                logger.info(f"Memory not found: {key}")
                return None
                
        except Exception as e:
            conn.close()
            logger.error(f"Error recalling memory: {e}")
            return None
    
    def recall_with_metadata(self, key: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Retrieve a memory with its metadata.
        
        Args:
            key: Unique identifier for the memory
            
        Returns:
            Tuple of (value, metadata), or (None, None) if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT value, metadata FROM memories WHERE key = ?", (key,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                value = json.loads(result[0])
                metadata = json.loads(result[1])
                return (value, metadata)
            else:
                logger.info(f"Memory not found: {key}")
                return (None, None)
                
        except Exception as e:
            conn.close()
            logger.error(f"Error recalling memory with metadata: {e}")
            return (None, None)
    
    def forget(self, key: str) -> bool:
        """
        Delete a memory by key.
        
        Args:
            key: Unique identifier for the memory
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM memories WHERE key = ?", (key,))
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted memory: {key}")
            return True
            
        except Exception as e:
            conn.close()
            logger.error(f"Error deleting memory: {e}")
            return False
    
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all memory keys, optionally filtered by pattern.
        
        Args:
            pattern: Optional SQL LIKE pattern to filter keys
            
        Returns:
            List of memory keys
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if pattern:
                cursor.execute("SELECT key FROM memories WHERE key LIKE ?", (pattern,))
            else:
                cursor.execute("SELECT key FROM memories")
                
            results = cursor.fetchall()
            conn.close()
            
            return [result[0] for result in results]
            
        except Exception as e:
            conn.close()
            logger.error(f"Error listing memory keys: {e}")
            return []
    
    def add_conversation_entry(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an entry to the conversation history.
        
        Args:
            role: Role of the speaker (user, assistant)
            content: Content of the message
            metadata: Optional metadata about the message
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            metadata_json = json.dumps(metadata or {})
            cursor.execute(
                "INSERT INTO conversation (timestamp, role, content, metadata) VALUES (?, ?, ?, ?)",
                (now, role, content, metadata_json)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added conversation entry: {role}")
            return True
            
        except Exception as e:
            conn.close()
            logger.error(f"Error adding conversation entry: {e}")
            return False
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            limit: Optional limit on the number of entries to return
            
        Returns:
            List of conversation entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if limit:
                cursor.execute(
                    "SELECT timestamp, role, content, metadata FROM conversation ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            else:
                cursor.execute("SELECT timestamp, role, content, metadata FROM conversation ORDER BY timestamp DESC")
                
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for result in results:
                timestamp, role, content, metadata_json = result
                history.append({
                    "timestamp": timestamp,
                    "role": role,
                    "content": content,
                    "metadata": json.loads(metadata_json)
                })
            
            return history
            
        except Exception as e:
            conn.close()
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def add_task(self, description: str, status: str = "pending", metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a task to the task list.
        
        Args:
            description: Description of the task
            status: Initial status of the task
            metadata: Optional metadata about the task
            
        Returns:
            Task ID if successful, -1 otherwise
        """
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            metadata_json = json.dumps(metadata or {})
            cursor.execute(
                "INSERT INTO tasks (description, status, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (description, status, now, now, metadata_json)
            )
            
            task_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Added task: {description} (ID: {task_id})")
            return task_id
            
        except Exception as e:
            conn.close()
            logger.error(f"Error adding task: {e}")
            return -1
    
    def update_task_status(self, task_id: int, status: str) -> bool:
        """
        Update a task's status.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.now().timestamp()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, task_id)
            )
            
            if cursor.rowcount == 0:
                conn.close()
                logger.warning(f"Task not found: {task_id}")
                return False
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated task {task_id} status to {status}")
            return True
            
        except Exception as e:
            conn.close()
            logger.error(f"Error updating task status: {e}")
            return False
    
    def get_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get task list, optionally filtered by status.
        
        Args:
            status: Optional status to filter by
            
        Returns:
            List of tasks
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if status:
                cursor.execute(
                    "SELECT id, description, status, created_at, updated_at, metadata FROM tasks WHERE status = ? ORDER BY updated_at DESC",
                    (status,)
                )
            else:
                cursor.execute(
                    "SELECT id, description, status, created_at, updated_at, metadata FROM tasks ORDER BY updated_at DESC"
                )
                
            results = cursor.fetchall()
            conn.close()
            
            tasks = []
            for result in results:
                task_id, description, status, created_at, updated_at, metadata_json = result
                tasks.append({
                    "id": task_id,
                    "description": description,
                    "status": status,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "metadata": json.loads(metadata_json)
                })
            
            return tasks
            
        except Exception as e:
            conn.close()
            logger.error(f"Error getting tasks: {e}")
            return []
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        Execute a raw SQL query on the database.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            List of query results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            # If this was a modifying query, commit changes
            if query.strip().lower().startswith(("insert", "update", "delete", "create", "drop")):
                conn.commit()
                
            conn.close()
            return results
            
        except Exception as e:
            conn.close()
            logger.error(f"Error executing query: {e}")
            return []


# Singleton instance
_instance = None

def get_memory_store(db_path: str = "agent_memory.db") -> MemoryStore:
    """
    Get the memory store instance.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        MemoryStore instance
    """
    global _instance
    if _instance is None:
        _instance = MemoryStore(db_path)
    return _instance


if __name__ == "__main__":
    # Example usage
    memory = get_memory_store()
    
    # Store some memories
    memory.remember("greeting", "Hello, world!")
    memory.remember("counter", 1)
    memory.remember("config", {"mode": "verbose", "timeout": 30})
    
    # Retrieve memories
    print(f"Greeting: {memory.recall('greeting')}")
    print(f"Counter: {memory.recall('counter')}")
    print(f"Config: {memory.recall('config')}")
    
    # Update a memory
    memory.remember("counter", memory.recall("counter") + 1)
    print(f"Updated counter: {memory.recall('counter')}")
    
    # List keys
    print(f"All memory keys: {memory.list_keys()}")
    
    # Add conversation entries
    memory.add_conversation_entry("user", "Hello!")
    memory.add_conversation_entry("assistant", "Hi there! How can I help you?")
    
    # Get conversation history
    print(f"Conversation history: {memory.get_conversation_history()}")
    
    # Add tasks
    task_id = memory.add_task("Implement memory store")
    memory.add_task("Test the system")
    
    # Update task status
    memory.update_task_status(task_id, "completed")
    
    # Get tasks
    print(f"All tasks: {memory.get_tasks()}")
    print(f"Completed tasks: {memory.get_tasks('completed')}")
    print(f"Pending tasks: {memory.get_tasks('pending')}")