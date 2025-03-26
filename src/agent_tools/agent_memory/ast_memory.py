#!/usr/bin/env python3
"""
AST Memory Integration

This module provides specialized memory functions for AST extraction
that can be used directly by the agent.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the src directory to the Python path
from .memory_store import get_memory_store

class AstMemory:
    """Memory system for AST extraction specifically for agent use."""
    
    def __init__(self, db_path: str = "ast_agent_memory.db"):
        """
        Initialize the AST memory system.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.memory = get_memory_store(db_path)
        self.namespace = "ast_extraction"
        
        # Initialize if needed
        self._init_memory()
        
    def _init_memory(self):
        """Initialize memory with default values if needed."""
        if not self.memory.recall(f"{self.namespace}:initialized"):
            self.memory.remember(f"{self.namespace}:initialized", True)
            self.memory.remember(f"{self.namespace}:file_count", 0)
            self.memory.remember(f"{self.namespace}:extraction_count", 0)
            self.memory.remember(f"{self.namespace}:error_count", 0)
            self.memory.remember(f"{self.namespace}:files_processed", [])
            self.memory.remember(f"{self.namespace}:patterns", {})
            self.memory.remember(f"{self.namespace}:errors", [])
            
    def record_file_processed(self, file_path: str, language: str, result: Dict[str, Any]) -> None:
        """
        Record a processed file in memory.
        
        Args:
            file_path: Path to the file
            language: Programming language
            result: Extraction result
        """
        # Update file count
        file_count = self.memory.recall(f"{self.namespace}:file_count")
        self.memory.remember(f"{self.namespace}:file_count", file_count + 1)
        
        # Update processed files list
        files_processed = self.memory.recall(f"{self.namespace}:files_processed")
        files_processed.append({
            "file_path": file_path,
            "language": language,
            "timestamp": time.time()
        })
        self.memory.remember(f"{self.namespace}:files_processed", files_processed)
        
        # Store the file result
        file_key = f"{self.namespace}:file:{Path(file_path).name}"
        self.memory.remember(file_key, result)
        
        # Record language patterns
        self._record_language_patterns(language, result)
        
    def _record_language_patterns(self, language: str, result: Dict[str, Any]) -> None:
        """
        Record language-specific patterns from extraction result.
        
        Args:
            language: Programming language
            result: Extraction result
        """
        patterns = self.memory.recall(f"{self.namespace}:patterns")
        
        # Initialize language patterns if needed
        if language not in patterns:
            patterns[language] = {
                "classes": {"count": 0, "examples": []},
                "functions": {"count": 0, "examples": []},
                "imports": {"count": 0, "examples": []},
                "nested_classes": {"count": 0, "examples": []},
                "inheritance": {"count": 0, "examples": []}
            }
        
        lang_patterns = patterns[language]
        
        # Update class patterns
        if "classes" in result:
            lang_patterns["classes"]["count"] += len(result["classes"])
            
            # Store examples (limit to 5)
            for cls in result["classes"][:2]:
                if len(lang_patterns["classes"]["examples"]) < 5:
                    lang_patterns["classes"]["examples"].append({
                        "name": cls.get("name", "Unknown"),
                        "file": result.get("file_path", "Unknown")
                    })
            
            # Check for nested classes
            for cls in result.get("classes", []):
                if cls.get("inner_classes", []):
                    lang_patterns["nested_classes"]["count"] += 1
                    
                    if len(lang_patterns["nested_classes"]["examples"]) < 5:
                        lang_patterns["nested_classes"]["examples"].append({
                            "parent": cls.get("name", "Unknown"),
                            "children": [ic.get("name", "Unknown") for ic in cls.get("inner_classes", [])],
                            "file": result.get("file_path", "Unknown")
                        })
                        
                # Check for inheritance
                if cls.get("inherits_from", []):
                    lang_patterns["inheritance"]["count"] += 1
                    
                    if len(lang_patterns["inheritance"]["examples"]) < 5:
                        lang_patterns["inheritance"]["examples"].append({
                            "class": cls.get("name", "Unknown"),
                            "inherits_from": cls.get("inherits_from", []),
                            "file": result.get("file_path", "Unknown")
                        })
        
        # Update function patterns
        if "functions" in result:
            lang_patterns["functions"]["count"] += len(result["functions"])
            
            # Store examples (limit to 5)
            for func in result["functions"][:2]:
                if len(lang_patterns["functions"]["examples"]) < 5:
                    lang_patterns["functions"]["examples"].append({
                        "name": func.get("name", "Unknown"),
                        "file": result.get("file_path", "Unknown")
                    })
        
        # Update import patterns
        if "imports" in result:
            lang_patterns["imports"]["count"] += len(result["imports"])
        
        # Save updated patterns
        self.memory.remember(f"{self.namespace}:patterns", patterns)
        
    def record_extraction_error(self, file_path: str, error_type: str, error_message: str) -> None:
        """
        Record an extraction error in memory.
        
        Args:
            file_path: Path to the file
            error_type: Type of error
            error_message: Error message
        """
        # Update error count
        error_count = self.memory.recall(f"{self.namespace}:error_count")
        self.memory.remember(f"{self.namespace}:error_count", error_count + 1)
        
        # Update errors list
        errors = self.memory.recall(f"{self.namespace}:errors")
        errors.append({
            "file_path": file_path,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time()
        })
        self.memory.remember(f"{self.namespace}:errors", errors)
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extractions from memory.
        
        Returns:
            Dictionary with statistics
        """
        file_count = self.memory.recall(f"{self.namespace}:file_count")
        extraction_count = self.memory.recall(f"{self.namespace}:extraction_count")
        error_count = self.memory.recall(f"{self.namespace}:error_count")
        patterns = self.memory.recall(f"{self.namespace}:patterns")
        
        # Calculate language statistics
        languages = {}
        for language, lang_patterns in patterns.items():
            languages[language] = {
                "class_count": lang_patterns["classes"]["count"],
                "function_count": lang_patterns["functions"]["count"],
                "import_count": lang_patterns["imports"]["count"],
                "nested_class_count": lang_patterns["nested_classes"]["count"],
                "inheritance_count": lang_patterns["inheritance"]["count"]
            }
        
        return {
            "file_count": file_count,
            "extraction_count": extraction_count,
            "error_count": error_count,
            "success_rate": ((file_count - error_count) / file_count * 100) if file_count > 0 else 0,
            "languages": languages
        }
    
    def get_language_patterns(self, language: str) -> Dict[str, Any]:
        """
        Get patterns for a specific language.
        
        Args:
            language: Programming language
            
        Returns:
            Dictionary with language patterns
        """
        patterns = self.memory.recall(f"{self.namespace}:patterns")
        return patterns.get(language, {})
    
    def get_recent_files(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recently processed files.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of recently processed files
        """
        files_processed = self.memory.recall(f"{self.namespace}:files_processed")
        return sorted(files_processed, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent extraction errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent extraction errors
        """
        errors = self.memory.recall(f"{self.namespace}:errors")
        return sorted(errors, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_file_result(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Get extraction result for a file.
        
        Args:
            file_name: Name of the file
            
        Returns:
            Extraction result or None if not found
        """
        file_key = f"{self.namespace}:file:{file_name}"
        return self.memory.recall(file_key)
    
    def clear_memory(self) -> None:
        """Clear all AST extraction memory."""
        keys = self.memory.list_keys(f"{self.namespace}:*")
        for key in keys:
            self.memory.forget(key)
        
        # Reinitialize
        self._init_memory()


# Singleton instance
_instance = None

def get_ast_memory(db_path: str = "ast_agent_memory.db") -> AstMemory:
    """
    Get the AST memory instance.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        AstMemory instance
    """
    global _instance
    if _instance is None:
        _instance = AstMemory(db_path)
    return _instance


if __name__ == "__main__":
    # Example usage
    ast_memory = get_ast_memory()
    
    # Record a file processed
    ast_memory.record_file_processed(
        "/path/to/example.py",
        "python",
        {
            "file_path": "/path/to/example.py",
            "language": "python",
            "classes": [
                {
                    "name": "ExampleClass",
                    "inherits_from": ["BaseClass"],
                    "inner_classes": [
                        {"name": "InnerClass"}
                    ]
                }
            ],
            "functions": [
                {"name": "example_function"}
            ],
            "imports": [
                {"text": "import os"}
            ]
        }
    )
    
    # Record an error
    ast_memory.record_extraction_error(
        "/path/to/error.py",
        "parse_error",
        "Failed to parse file"
    )
    
    # Get statistics
    stats = ast_memory.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    # Get Python patterns
    python_patterns = ast_memory.get_language_patterns("python")
    print(f"Python patterns: {json.dumps(python_patterns, indent=2)}")
    
    # Get recent files
    recent_files = ast_memory.get_recent_files()
    print(f"Recent files: {json.dumps(recent_files, indent=2)}")
    
    # Get errors
    errors = ast_memory.get_errors()
    print(f"Errors: {json.dumps(errors, indent=2)}")