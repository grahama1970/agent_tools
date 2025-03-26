#!/usr/bin/env python3
"""
Test State Manager

A reliable state management system using SQLite in-memory database
to maintain state between test steps for the DuaLipa extraction module.

This addresses limitations in context maintenance by providing persistent,
queryable state that can be explicitly verified at each step.
"""

import os
import sys
import json
import sqlite3
import datetime
import threading
import traceback
from typing import Any, Dict, List, Union, Optional
from contextlib import contextmanager

class TestStateManager:
    """
    State manager for extraction tests using SQLite in-memory database.
    
    This class provides reliable state management that doesn't depend on
    context tracking. It stores key-value data, verification logs, and
    extraction statistics to ensure tests can reliably track and verify state.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, persistent_path=None):
        """Get singleton instance with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = TestStateManager(persistent_path)
            return cls._instance
    
    def __init__(self, persistent_path=None):
        """
        Initialize the state manager with a SQLite database.
        
        Args:
            persistent_path: Optional path to persist database to disk
        """
        # Use persistent database if path provided, otherwise in-memory
        if persistent_path:
            self.db_path = persistent_path
            self.persistent_path = None  # No need for backup if already using file
        else:
            self.db_path = ":memory:"
            self.persistent_path = persistent_path
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Initialize database tables
        self._setup_database()
        
        # Checkpoint identifier
        self.current_checkpoint = "initial"
        
        # Track database state
        self.transaction_count = 0
        self.is_initializing = True
        
        # Initialize with creation timestamp
        self.set_metadata("creation_timestamp", datetime.datetime.now().isoformat())
        self.is_initializing = False
        
        if self.db_path == ":memory:":
            print(f"TestStateManager initialized with in-memory database")
        else:
            print(f"TestStateManager initialized with database at {self.db_path}")
    
    def _setup_database(self):
        """Set up the database tables."""
        # State table for key-value storage
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT,
            value_type TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Verification log for tracking test assertions
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint TEXT,
            step TEXT,
            expected TEXT,
            actual TEXT,
            passed BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # File tracking for extraction tests
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_tracking (
            file_path TEXT PRIMARY KEY,
            file_type TEXT,
            extracted BOOLEAN DEFAULT FALSE,
            extraction_time TIMESTAMP,
            extracted_uuid TEXT,
            size INTEGER,
            included_in_output BOOLEAN DEFAULT FALSE
        )
        ''')
        
        # Repository statistics
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS repo_stats (
            extension TEXT PRIMARY KEY,
            count INTEGER,
            extracted_count INTEGER DEFAULT 0,
            percentage REAL
        )
        ''')
        
        # Checkpoint tracking
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkpoints (
            name TEXT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        ''')
        
        # Metadata for general information
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Context tracking for assistant
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context_key TEXT UNIQUE,
            task TEXT,                 -- What I'm currently doing
            goal TEXT,                 -- Why I'm doing it
            progress TEXT,             -- Current step in the process
            assumptions TEXT,          -- Key assumptions I'm making
            problems TEXT,             -- Issues I've encountered
            next_steps TEXT,           -- What to do next
            notes TEXT,                -- Additional context
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Documentation store for self-reference
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT UNIQUE,
            content TEXT,              -- Main documentation content
            summary TEXT,              -- Short summary of documentation
            source TEXT,               -- Where this came from
            examples TEXT,             -- Usage examples
            related_topics TEXT,       -- Related topics for cross-reference
            importance INTEGER,        -- 1-10 scale of importance
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        self.transaction_count += 1
        try:
            yield
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Transaction failed: {e}")
            traceback.print_exc()
            raise
        finally:
            self.transaction_count -= 1
            # If we have a persistent path and no active transactions, backup
            if self.persistent_path and self.transaction_count == 0 and not self.is_initializing:
                self._backup_to_disk()
    
    def _backup_to_disk(self):
        """Backup in-memory database to disk for debugging."""
        if not self.persistent_path:
            return
            
        try:
            # Create backup connection
            disk_conn = sqlite3.connect(self.persistent_path)
            
            # Backup in-memory to disk
            self.conn.backup(disk_conn)
            
            # Close disk connection
            disk_conn.close()
        except Exception as e:
            print(f"Failed to backup database to {self.persistent_path}: {e}")
    
    def set(self, key: str, value: Any):
        """
        Set a value in the state store.
        
        Args:
            key: State key
            value: Value to store (will be JSON serialized)
        """
        # Determine value type for proper retrieval
        value_type = type(value).__name__
        
        # JSON serialize value
        serialized = json.dumps(value)
        
        with self.transaction():
            self.cursor.execute(
                "INSERT OR REPLACE INTO state (key, value, value_type, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (key, serialized, value_type)
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the state store.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            The stored value, properly deserialized
        """
        self.cursor.execute("SELECT value, value_type FROM state WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        
        if not result:
            return default
            
        value, value_type = result
        
        # Deserialize based on stored type
        deserialized = json.loads(value)
        
        return deserialized
    
    def set_checkpoint(self, name: str, description: str = ""):
        """
        Set a named checkpoint in the test process.
        
        Args:
            name: Checkpoint name
            description: Optional description
        """
        self.current_checkpoint = name
        
        with self.transaction():
            self.cursor.execute(
                "INSERT OR REPLACE INTO checkpoints (name, timestamp, description) VALUES (?, CURRENT_TIMESTAMP, ?)",
                (name, description)
            )
    
    def log_verification(self, step: str, expected: Any, actual: Any, passed: bool):
        """
        Log a verification step result.
        
        Args:
            step: Verification step name
            expected: Expected value
            actual: Actual value
            passed: Whether verification passed
        """
        # Serialize values for storage
        expected_str = json.dumps(expected)
        actual_str = json.dumps(actual)
        
        with self.transaction():
            self.cursor.execute(
                "INSERT INTO verification_log (checkpoint, step, expected, actual, passed, timestamp) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (self.current_checkpoint, step, expected_str, actual_str, passed)
            )
    
    def verify(self, step: str, expected: Any, actual: Any) -> bool:
        """
        Verify a value and log the result.
        
        Args:
            step: Verification step name
            expected: Expected value
            actual: Actual value
            
        Returns:
            Whether verification passed
        """
        # Handle different types of comparisons
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            passed = sorted(expected) == sorted(actual)
        else:
            passed = expected == actual
        
        # Log verification
        self.log_verification(step, expected, actual, passed)
        
        # Print result for clarity
        result_str = "PASSED ✓" if passed else "FAILED ❌"
        print(f"Verification '{step}': {result_str}")
        
        if not passed:
            print(f"  Expected: {expected}")
            print(f"  Actual: {actual}")
        
        return passed
    
    def assert_verify(self, step: str, expected: Any, actual: Any, message: str = None):
        """
        Assert that a verification passes and log the result.
        
        Args:
            step: Verification step name
            expected: Expected value
            actual: Actual value
            message: Optional assertion message
            
        Raises:
            AssertionError: If verification fails
        """
        passed = self.verify(step, expected, actual)
        
        if not passed:
            default_message = f"Verification '{step}' failed: expected {expected}, got {actual}"
            raise AssertionError(message or default_message)
    
    def track_file(self, file_path: str, file_type: str, size: int = 0):
        """
        Track a file for extraction.
        
        Args:
            file_path: Path to the file
            file_type: File type (extension)
            size: File size in bytes
        """
        with self.transaction():
            self.cursor.execute(
                "INSERT OR REPLACE INTO file_tracking (file_path, file_type, size) VALUES (?, ?, ?)",
                (file_path, file_type, size)
            )
    
    def mark_file_extracted(self, file_path: str, extracted_uuid: str):
        """
        Mark a file as extracted.
        
        Args:
            file_path: Path to the file
            extracted_uuid: UUID of the extracted block
        """
        with self.transaction():
            self.cursor.execute(
                "UPDATE file_tracking SET extracted = TRUE, extraction_time = CURRENT_TIMESTAMP, extracted_uuid = ? WHERE file_path = ?",
                (extracted_uuid, file_path)
            )
    
    def mark_file_included_in_output(self, file_path: str):
        """
        Mark a file as included in the final output.
        
        Args:
            file_path: Path to the file
        """
        with self.transaction():
            self.cursor.execute(
                "UPDATE file_tracking SET included_in_output = TRUE WHERE file_path = ?",
                (file_path,)
            )
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        # Get total tracked files
        self.cursor.execute("SELECT COUNT(*) FROM file_tracking")
        total_files = self.cursor.fetchone()[0]
        
        # Get extracted files
        self.cursor.execute("SELECT COUNT(*) FROM file_tracking WHERE extracted = TRUE")
        extracted_files = self.cursor.fetchone()[0]
        
        # Get included files
        self.cursor.execute("SELECT COUNT(*) FROM file_tracking WHERE included_in_output = TRUE")
        included_files = self.cursor.fetchone()[0]
        
        # Get stats by file type
        self.cursor.execute("""
        SELECT file_type, 
               COUNT(*) as total, 
               SUM(CASE WHEN extracted THEN 1 ELSE 0 END) as extracted,
               SUM(CASE WHEN included_in_output THEN 1 ELSE 0 END) as included
        FROM file_tracking
        GROUP BY file_type
        """)
        
        by_type = {}
        for row in self.cursor.fetchall():
            by_type[row['file_type']] = {
                'total': row['total'],
                'extracted': row['extracted'],
                'included': row['included'],
                'extraction_rate': (row['extracted'] / row['total']) * 100 if row['total'] > 0 else 0
            }
        
        return {
            'total_files': total_files,
            'extracted_files': extracted_files,
            'included_files': included_files,
            'extraction_rate': (extracted_files / total_files) * 100 if total_files > 0 else 0,
            'inclusion_rate': (included_files / extracted_files) * 100 if extracted_files > 0 else 0,
            'by_type': by_type
        }
    
    def set_repo_stats(self, extension: str, count: int, percentage: float):
        """
        Set repository statistics for a file extension.
        
        Args:
            extension: File extension
            count: Number of files with extension
            percentage: Percentage of all files
        """
        with self.transaction():
            self.cursor.execute(
                "INSERT OR REPLACE INTO repo_stats (extension, count, percentage) VALUES (?, ?, ?)",
                (extension, count, percentage)
            )
    
    def update_extracted_count(self, extension: str, count: int):
        """
        Update the count of extracted files for an extension.
        
        Args:
            extension: File extension
            count: Number of extracted files
        """
        with self.transaction():
            self.cursor.execute(
                "UPDATE repo_stats SET extracted_count = ? WHERE extension = ?",
                (count, extension)
            )
    
    def get_repo_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with repository statistics by extension
        """
        self.cursor.execute("SELECT extension, count, extracted_count, percentage FROM repo_stats")
        
        stats = {}
        for row in self.cursor.fetchall():
            stats[row['extension']] = {
                'count': row['count'],
                'extracted': row['extracted_count'],
                'percentage': row['percentage'],
                'extraction_rate': (row['extracted_count'] / row['count']) * 100 if row['count'] > 0 else 0
            }
        
        return stats
    
    def get_verification_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of verifications.
        
        Returns:
            List of verification records
        """
        self.cursor.execute("""
        SELECT checkpoint, step, expected, actual, passed, timestamp
        FROM verification_log
        ORDER BY timestamp
        """)
        
        history = []
        for row in self.cursor.fetchall():
            history.append({
                'checkpoint': row['checkpoint'],
                'step': row['step'],
                'expected': json.loads(row['expected']),
                'actual': json.loads(row['actual']),
                'passed': row['passed'],
                'timestamp': row['timestamp']
            })
        
        return history
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of checkpoints.
        
        Returns:
            List of checkpoint records
        """
        self.cursor.execute("SELECT name, timestamp, description FROM checkpoints ORDER BY timestamp")
        
        history = []
        for row in self.cursor.fetchall():
            history.append({
                'name': row['name'],
                'timestamp': row['timestamp'],
                'description': row['description']
            })
        
        return history
    
    def set_metadata(self, key: str, value: Any):
        """
        Set metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        # JSON serialize value
        serialized = json.dumps(value)
        
        with self.transaction():
            self.cursor.execute(
                "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (key, serialized)
            )
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata.
        
        Args:
            key: Metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The stored metadata value
        """
        self.cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        
        if not result:
            return default
            
        # Deserialize value
        return json.loads(result[0])
    
    def generate_report(self, report_path: str = None):
        """
        Generate a comprehensive test report.
        
        Args:
            report_path: Path to save report (defaults to stdout)
        """
        # Collect report data
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'extraction_stats': self.get_extraction_stats(),
            'repo_stats': self.get_repo_stats(),
            'verifications': self.get_verification_history(),
            'checkpoints': self.get_checkpoint_history(),
            'metadata': {}
        }
        
        # Get all metadata
        self.cursor.execute("SELECT key, value FROM metadata")
        for row in self.cursor.fetchall():
            report['metadata'][row['key']] = json.loads(row['value'])
        
        # Format report
        formatted_report = json.dumps(report, indent=2)
        
        if report_path:
            with open(report_path, 'w') as f:
                f.write(formatted_report)
            print(f"Report saved to {report_path}")
        else:
            print(formatted_report)
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def update_context(self, context_key: str, **kwargs):
        """
        Update context tracking for maintaining state of what I'm doing.
        
        Args:
            context_key: Unique identifier for this context
            **kwargs: Context fields to update (task, goal, progress, etc.)
        """
        # Get existing context or create default values
        self.cursor.execute("SELECT * FROM context WHERE context_key = ?", (context_key,))
        existing = self.cursor.fetchone()
        
        if existing:
            # Update existing context
            fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['task', 'goal', 'progress', 'assumptions', 'problems', 'next_steps', 'notes']:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            if fields:
                values.append(context_key)
                with self.transaction():
                    self.cursor.execute(
                        f"UPDATE context SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE context_key = ?",
                        tuple(values)
                    )
        else:
            # Create new context
            fields = ['context_key']
            values = [context_key]
            placeholders = ['?']
            
            for key, value in kwargs.items():
                if key in ['task', 'goal', 'progress', 'assumptions', 'problems', 'next_steps', 'notes']:
                    fields.append(key)
                    values.append(value)
                    placeholders.append('?')
            
            with self.transaction():
                self.cursor.execute(
                    f"INSERT INTO context ({', '.join(fields)}) VALUES ({', '.join(placeholders)})",
                    tuple(values)
                )
        
        # Print context update for visibility
        print(f"Context updated: {context_key}")
        for key, value in kwargs.items():
            if key in ['task', 'goal', 'progress', 'next_steps']:  # Only print key fields
                print(f"- {key}: {value}")
    
    def get_context(self, context_key: str = None) -> Dict[str, Any]:
        """
        Get context for maintaining state of what I'm doing.
        
        Args:
            context_key: Optional specific context to retrieve
            
        Returns:
            Dictionary of context information
        """
        if context_key:
            # Get specific context
            self.cursor.execute("SELECT * FROM context WHERE context_key = ?", (context_key,))
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'context_key': row['context_key'],
                    'task': row['task'],
                    'goal': row['goal'],
                    'progress': row['progress'],
                    'assumptions': row['assumptions'],
                    'problems': row['problems'],
                    'next_steps': row['next_steps'],
                    'notes': row['notes'],
                    'updated_at': row['updated_at']
                }
            else:
                return {}
        else:
            # Get most recent context
            self.cursor.execute("SELECT * FROM context ORDER BY updated_at DESC LIMIT 1")
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'context_key': row['context_key'],
                    'task': row['task'],
                    'goal': row['goal'],
                    'progress': row['progress'],
                    'assumptions': row['assumptions'],
                    'problems': row['problems'],
                    'next_steps': row['next_steps'],
                    'notes': row['notes'],
                    'updated_at': row['updated_at']
                }
            else:
                return {}
    
    def add_documentation(self, topic: str, content: str, summary: str = "", source: str = "",
                          examples: str = "", related_topics: str = "", importance: int = 5):
        """
        Add documentation that I can refer back to when confused.
        
        Args:
            topic: Documentation topic (unique identifier)
            content: Main documentation content
            summary: Short summary of documentation
            source: Source of documentation
            examples: Usage examples
            related_topics: Related topics for cross-reference
            importance: Importance level (1-10)
        """
        with self.transaction():
            self.cursor.execute(
                """
                INSERT OR REPLACE INTO documentation
                (topic, content, summary, source, examples, related_topics, importance, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (topic, content, summary, source, examples, related_topics, importance)
            )
        
        print(f"Documentation added: {topic}")
    
    def get_documentation(self, topic: str = None, search_term: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get documentation for reference.
        
        Args:
            topic: Specific topic to retrieve
            search_term: Term to search for in documentation
            
        Returns:
            Documentation information
        """
        if topic:
            # Get specific topic
            self.cursor.execute("SELECT * FROM documentation WHERE topic = ?", (topic,))
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'topic': row['topic'],
                    'content': row['content'],
                    'summary': row['summary'],
                    'source': row['source'],
                    'examples': row['examples'],
                    'related_topics': row['related_topics'],
                    'importance': row['importance'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
            else:
                return {}
        
        elif search_term:
            # Search in topics and content
            self.cursor.execute(
                """
                SELECT * FROM documentation 
                WHERE topic LIKE ? OR content LIKE ? OR summary LIKE ?
                ORDER BY importance DESC, updated_at DESC
                """,
                (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%")
            )
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'topic': row['topic'],
                    'summary': row['summary'],
                    'importance': row['importance'],
                    'updated_at': row['updated_at']
                })
            
            return results
        
        else:
            # Get all topics sorted by importance
            self.cursor.execute("SELECT topic, summary, importance FROM documentation ORDER BY importance DESC, updated_at DESC")
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'topic': row['topic'],
                    'summary': row['summary'],
                    'importance': row['importance']
                })
            
            return results
    
    def remember_my_place(self, what_im_doing: str, why_im_doing_it: str, what_step: str, what_next: str):
        """
        Convenience method to remember where I am in a test process.
        
        Args:
            what_im_doing: Current task
            why_im_doing_it: Goal of task
            what_step: Current step
            what_next: Next steps
        """
        self.update_context(
            self.current_checkpoint,
            task=what_im_doing,
            goal=why_im_doing_it,
            progress=what_step,
            next_steps=what_next
        )
    
    def remind_me(self):
        """Print a reminder of what I'm currently doing."""
        context = self.get_context()
        
        if context:
            print("\n=== CURRENT CONTEXT ===")
            print(f"Task: {context.get('task', 'No task defined')}")
            print(f"Goal: {context.get('goal', 'No goal defined')}")
            print(f"Progress: {context.get('progress', 'No progress reported')}")
            print(f"Next Steps: {context.get('next_steps', 'No next steps defined')}")
            print("=======================\n")
        else:
            print("No context defined yet.")
    
    def __del__(self):
        """Ensure the connection is closed on deletion."""
        self.close()


# Helper functions to use the state manager
def get_state_manager(persistent_path=None):
    """Get the state manager instance."""
    return TestStateManager.get_instance(persistent_path)


# CLI-style helpers for context management
def what_am_i_doing():
    """Get a reminder of what I'm currently doing."""
    state_manager = get_state_manager()
    state_manager.remind_me()


def remember_context(what_im_doing, why_im_doing_it, what_step, what_next):
    """Remember my current context."""
    state_manager = get_state_manager()
    state_manager.remember_my_place(what_im_doing, why_im_doing_it, what_step, what_next)


def add_docs(topic, content, summary="", importance=5):
    """Add documentation for later reference."""
    state_manager = get_state_manager()
    state_manager.add_documentation(topic, content, summary=summary, importance=importance)


def get_docs(topic=None, search=None):
    """Get documentation for reference."""
    state_manager = get_state_manager()
    if topic:
        doc = state_manager.get_documentation(topic=topic)
        if doc:
            print(f"\n=== DOCUMENTATION: {doc.get('topic')} ===")
            print(doc.get('content'))
            if doc.get('examples'):
                print("\nExamples:")
                print(doc.get('examples'))
            print("===========================\n")
        else:
            print(f"No documentation found for topic: {topic}")
    elif search:
        results = state_manager.get_documentation(search_term=search)
        if results:
            print(f"\nFound {len(results)} documentation topics matching '{search}':")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.get('topic')} - {doc.get('summary')}")
        else:
            print(f"No documentation found matching '{search}'")
    else:
        topics = state_manager.get_documentation()
        if topics:
            print("\nAvailable documentation topics:")
            for i, topic in enumerate(topics, 1):
                print(f"{i}. {topic.get('topic')} - {topic.get('summary')}")
        else:
            print("No documentation available.")

def verify_extraction_completeness(repo_stats, extraction_results, state_manager=None):
    """
    Verify extraction completeness against repository statistics.
    
    Args:
        repo_stats: Repository statistics
        extraction_results: Extraction results
        state_manager: Optional state manager instance
        
    Returns:
        True if extraction is complete, False otherwise
    """
    if state_manager is None:
        state_manager = get_state_manager()
    
    # Set checkpoint
    state_manager.set_checkpoint("verify_extraction", "Verifying extraction completeness")
    
    # Get file counts
    expected_files = {}
    for ext, stats in repo_stats.items():
        if ext.startswith('.'):
            expected_files[ext] = stats['count']
    
    # Get extracted file counts
    extracted_files = {}
    for file_path in extraction_results:
        ext = os.path.splitext(file_path)[1]
        extracted_files[ext] = extracted_files.get(ext, 0) + 1
    
    # Verify Python files
    python_expected = expected_files.get('.py', 0)
    python_extracted = extracted_files.get('.py', 0)
    
    # Log verification
    python_complete = state_manager.verify(
        "python_files_extracted",
        python_expected,
        python_extracted
    )
    
    # Update stats in state manager
    for ext, count in extracted_files.items():
        state_manager.update_extracted_count(ext, count)
    
    # Overall verification
    all_passed = True
    
    # Check for critical files
    critical_files = [
        "tests/js/common/test-data/search/docs/generate_ii_sa_dataset.py",
        "utils/gantt.py"
    ]
    
    for critical_file in critical_files:
        critical_found = any(critical_file in file_path for file_path in extraction_results)
        all_passed = all_passed and state_manager.verify(
            f"critical_file_{os.path.basename(critical_file)}",
            True,
            critical_found
        )
    
    # Generate report
    state_manager.set_metadata("extraction_verification_result", all_passed)
    
    return all_passed


if __name__ == "__main__":
    # Example usage
    state_manager = get_state_manager("test_state.db")
    
    # Set initial repository stats
    state_manager.set_checkpoint("initialization", "Setting up test")
    
    # Track some example files
    state_manager.track_file("example.py", ".py", 1024)
    state_manager.track_file("main.js", ".js", 2048)
    
    # Mark a file as extracted
    state_manager.mark_file_extracted("example.py", "123e4567-e89b-12d3-a456-426614174000")
    
    # Set repository stats
    state_manager.set_repo_stats(".py", 10, 25.0)
    state_manager.set_repo_stats(".js", 20, 50.0)
    
    # Update extracted count
    state_manager.update_extracted_count(".py", 5)
    
    # Verify something
    state_manager.set_checkpoint("verification", "Verifying extraction")
    state_manager.verify("python_file_count", 10, 10)
    
    # Generate report
    state_manager.generate_report("state_report.json")
    
    print("Test state manager demonstration complete")