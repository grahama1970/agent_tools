#!/usr/bin/env python3
"""
AST Agent CLI

Command-line interface for the AST memory system.
This tool helps the agent track AST extraction patterns and results.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the repository to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_tools.agent_memory.ast_memory import get_ast_memory

def handle_record_file(args):
    """Handle the record-file command."""
    ast_memory = get_ast_memory(args.db_path)
    
    # Load extraction result from file
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            result = json.load(f)
    except Exception as e:
        print(f"Error loading extraction result: {e}")
        return 1
    
    # Record the file
    try:
        file_path = args.path or result.get("file_path", "unknown")
        language = args.language or result.get("language", "unknown")
        
        # If result is wrapped in "results", unwrap it
        if "results" in result and isinstance(result["results"], list) and len(result["results"]) == 1:
            result = result["results"][0]
        
        ast_memory.record_file_processed(file_path, language, result)
        print(f"Successfully recorded file: {file_path}")
        return 0
    except Exception as e:
        print(f"Error recording file: {e}")
        return 1

def handle_record_error(args):
    """Handle the record-error command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        ast_memory.record_extraction_error(args.file_path, args.error_type, args.message)
        print(f"Successfully recorded error for file: {args.file_path}")
        return 0
    except Exception as e:
        print(f"Error recording error: {e}")
        return 1

def handle_get_stats(args):
    """Handle the get-stats command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        stats = ast_memory.get_statistics()
        
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("AST Extraction Statistics:")
            print(f"Files processed: {stats['file_count']}")
            print(f"Extraction errors: {stats['error_count']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            
            if "languages" in stats:
                print("\nLanguage Statistics:")
                for language, lang_stats in stats["languages"].items():
                    print(f"\n{language.upper()}:")
                    print(f"  Classes: {lang_stats['class_count']}")
                    print(f"  Functions: {lang_stats['function_count']}")
                    print(f"  Imports: {lang_stats['import_count']}")
                    print(f"  Nested classes: {lang_stats['nested_class_count']}")
                    print(f"  Inheritance: {lang_stats['inheritance_count']}")
        
        return 0
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return 1

def handle_get_patterns(args):
    """Handle the get-patterns command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        if args.language:
            patterns = ast_memory.get_language_patterns(args.language)
            if not patterns:
                print(f"No patterns found for language: {args.language}")
                return 0
                
            if args.json:
                print(json.dumps(patterns, indent=2))
            else:
                print(f"Patterns for {args.language.upper()}:")
                for pattern_type, pattern_data in patterns.items():
                    print(f"\n{pattern_type.capitalize()}:")
                    print(f"  Count: {pattern_data['count']}")
                    
                    if pattern_data.get("examples"):
                        print("  Examples:")
                        for example in pattern_data["examples"]:
                            if pattern_type == "nested_classes":
                                print(f"    {example['parent']} with inner classes: {', '.join(example['children'])}")
                            elif pattern_type == "inheritance":
                                print(f"    {example['class']} inherits from: {', '.join(example['inherits_from'])}")
                            else:
                                print(f"    {example.get('name', 'Unknown')} in {example.get('file', 'Unknown')}")
        else:
            # Get all language patterns from statistics
            stats = ast_memory.get_statistics()
            languages = stats.get("languages", {})
            
            if args.json:
                print(json.dumps(languages, indent=2))
            else:
                print("Language Patterns Summary:")
                for language, lang_stats in languages.items():
                    print(f"\n{language.upper()}:")
                    for key, value in lang_stats.items():
                        print(f"  {key}: {value}")
        
        return 0
    except Exception as e:
        print(f"Error getting patterns: {e}")
        return 1

def handle_get_recent_files(args):
    """Handle the get-recent-files command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        files = ast_memory.get_recent_files(args.limit)
        
        if not files:
            print("No files found")
            return 0
            
        if args.json:
            print(json.dumps(files, indent=2))
        else:
            print("Recent Files:")
            for file in files:
                timestamp = datetime.fromtimestamp(file["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {file['file_path']} ({file['language']}) - {timestamp}")
        
        return 0
    except Exception as e:
        print(f"Error getting recent files: {e}")
        return 1

def handle_get_errors(args):
    """Handle the get-errors command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        errors = ast_memory.get_errors(args.limit)
        
        if not errors:
            print("No errors found")
            return 0
            
        if args.json:
            print(json.dumps(errors, indent=2))
        else:
            print("Recent Errors:")
            for error in errors:
                timestamp = datetime.fromtimestamp(error["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {error['file_path']} - {error['error_type']}")
                print(f"    {error['error_message']}")
                print(f"    {timestamp}")
                print()
        
        return 0
    except Exception as e:
        print(f"Error getting errors: {e}")
        return 1

def handle_get_file_result(args):
    """Handle the get-file-result command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        result = ast_memory.get_file_result(args.file_name)
        
        if not result:
            print(f"No result found for file: {args.file_name}")
            return 0
            
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Result for {args.file_name}:")
            print(f"  Language: {result.get('language', 'unknown')}")
            print(f"  Classes: {len(result.get('classes', []))}")
            print(f"  Functions: {len(result.get('functions', []))}")
            print(f"  Imports: {len(result.get('imports', []))}")
            
            if args.verbose:
                if "classes" in result and result["classes"]:
                    print("\nClasses:")
                    for cls in result["classes"]:
                        print(f"  {cls.get('name', 'Unknown')}")
                        if cls.get("inherits_from"):
                            print(f"    Inherits from: {', '.join(cls['inherits_from'])}")
                        if cls.get("inner_classes"):
                            inner_names = [ic.get("name", "Unknown") for ic in cls["inner_classes"]]
                            print(f"    Inner classes: {', '.join(inner_names)}")
                
                if "functions" in result and result["functions"]:
                    print("\nFunctions:")
                    for func in result["functions"]:
                        print(f"  {func.get('name', 'Unknown')}")
        
        return 0
    except Exception as e:
        print(f"Error getting file result: {e}")
        return 1

def handle_clear_memory(args):
    """Handle the clear-memory command."""
    ast_memory = get_ast_memory(args.db_path)
    
    try:
        ast_memory.clear_memory()
        print("Successfully cleared AST memory")
        return 0
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return 1

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AST Agent CLI")
    parser.add_argument("--db-path", default="ast_agent_memory.db", help="Path to the SQLite database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # record-file command
    record_file_parser = subparsers.add_parser("record-file", help="Record a processed file")
    record_file_parser.add_argument("file", help="Path to the extraction result JSON file")
    record_file_parser.add_argument("--path", help="Override file path")
    record_file_parser.add_argument("--language", help="Override language")
    record_file_parser.set_defaults(func=handle_record_file)
    
    # record-error command
    record_error_parser = subparsers.add_parser("record-error", help="Record an extraction error")
    record_error_parser.add_argument("file_path", help="Path to the file with error")
    record_error_parser.add_argument("error_type", help="Type of error")
    record_error_parser.add_argument("message", help="Error message")
    record_error_parser.set_defaults(func=handle_record_error)
    
    # get-stats command
    get_stats_parser = subparsers.add_parser("get-stats", help="Get AST extraction statistics")
    get_stats_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_stats_parser.set_defaults(func=handle_get_stats)
    
    # get-patterns command
    get_patterns_parser = subparsers.add_parser("get-patterns", help="Get language-specific patterns")
    get_patterns_parser.add_argument("--language", help="Language to get patterns for")
    get_patterns_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_patterns_parser.set_defaults(func=handle_get_patterns)
    
    # get-recent-files command
    get_recent_files_parser = subparsers.add_parser("get-recent-files", help="Get recently processed files")
    get_recent_files_parser.add_argument("--limit", type=int, default=5, help="Maximum number of files to return")
    get_recent_files_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_recent_files_parser.set_defaults(func=handle_get_recent_files)
    
    # get-errors command
    get_errors_parser = subparsers.add_parser("get-errors", help="Get recent extraction errors")
    get_errors_parser.add_argument("--limit", type=int, default=5, help="Maximum number of errors to return")
    get_errors_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_errors_parser.set_defaults(func=handle_get_errors)
    
    # get-file-result command
    get_file_result_parser = subparsers.add_parser("get-file-result", help="Get extraction result for a file")
    get_file_result_parser.add_argument("file_name", help="Name of the file")
    get_file_result_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    get_file_result_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    get_file_result_parser.set_defaults(func=handle_get_file_result)
    
    # clear-memory command
    clear_memory_parser = subparsers.add_parser("clear-memory", help="Clear all AST extraction memory")
    clear_memory_parser.set_defaults(func=handle_clear_memory)
    
    args = parser.parse_args()
    
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())