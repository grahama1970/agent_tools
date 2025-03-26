#!/usr/bin/env python3
"""
Single file test for AST extraction with memory inspection.

This script runs AST extraction on a single file and provides
detailed output about the extraction results and memory state.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add repository root to path
sys.path.append(str(Path(__file__).parent))

# Import AST extractor and memory components
from src.agent_tools.dualipa.extraction.extractors.code.ast_extractor import AstExtractor
from src.agent_tools.dualipa.extraction.test_state_manager import get_state_manager
from src.agent_tools.dualipa.extraction.extraction_memory import (
    init_extraction_memory,
    track_extraction_start,
    track_extraction_completion,
    get_extraction_context,
    find_extraction_knowledge,
    save_extraction_knowledge
)


def test_file(file_path: str, db_path: str = "extraction_memory.db", 
              verbose: bool = True, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Test AST extraction on a single file with memory inspection.
    
    Args:
        file_path: Path to the file to test
        db_path: Path to the memory database
        verbose: Whether to print verbose output
        output_path: Optional path to save the output
        
    Returns:
        Dictionary with test results
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return {"error": "File not found"}
    
    # Initialize memory
    init_extraction_memory(db_path)
    
    # Get initial context
    initial_context = get_extraction_context()
    
    # Initialize extractor
    extractor = AstExtractor(memory_db_path=db_path)
    
    # Track extraction start
    file_name = os.path.basename(file_path)
    track_extraction_start(
        file_name,
        "single_test",
        {"file_path": file_path, "timestamp": os.path.getmtime(file_path)}
    )
    
    # Perform extraction
    result = extractor.extract_file(file_path)
    
    # Track completion
    track_extraction_completion(
        file_name,
        f"Completed test extraction of {file_name}",
        extractor.get_statistics()
    )
    
    # Get final context
    final_context = get_extraction_context()
    
    # Search for related knowledge
    related_knowledge = find_extraction_knowledge(file_name)
    
    # Save test result knowledge
    save_extraction_knowledge(
        f"test_result_{file_name}",
        f"# Test Results for {file_name}\n\n"
        f"File: {file_path}\n\n"
        f"Extraction result contains:\n"
        f"- Classes: {len(result.get('classes', []))}\n"
        f"- Functions: {len(result.get('functions', []))}\n"
        f"- Imports: {len(result.get('imports', []))}\n\n"
        f"Memory integration successful.",
        summary=f"Test results for {file_name} extraction with memory integration",
        tags=["test", "ast", "memory", os.path.splitext(file_path)[1].lstrip('.')]
    )
    
    # Prepare report
    report = {
        "file_path": file_path,
        "extraction_result": result,
        "initial_context": initial_context,
        "final_context": final_context,
        "related_knowledge": related_knowledge,
        "statistics": extractor.get_statistics()
    }
    
    # Print verbose output
    if verbose:
        print("\n" + "="*80)
        print(f"AST EXTRACTION TEST: {file_path}")
        print("="*80)
        
        print("\nINITIAL CONTEXT:")
        print(f"Task: {initial_context.get('task', 'N/A')}")
        print(f"Progress: {initial_context.get('progress', 'N/A')}")
        print(f"Next steps: {initial_context.get('next_steps', 'N/A')}")
        
        print("\nEXTRACTION RESULT:")
        print(f"Classes: {len(result.get('classes', []))}")
        for cls in result.get('classes', []):
            print(f"  - {cls['name']}")
            if cls.get('inner_classes'):
                print(f"    Inner classes: {[ic['name'] for ic in cls['inner_classes']]}")
            if cls.get('inherits_from'):
                print(f"    Inherits from: {cls['inherits_from']}")
        
        print(f"Functions: {len(result.get('functions', []))}")
        for func in result.get('functions', []):
            print(f"  - {func['name']}")
            
        print(f"Imports: {len(result.get('imports', []))}")
        
        print("\nFINAL CONTEXT:")
        print(f"Task: {final_context.get('task', 'N/A')}")
        print(f"Progress: {final_context.get('progress', 'N/A')}")
        print(f"Next steps: {final_context.get('next_steps', 'N/A')}")
        
        print("\nMEMORY STATISTICS:")
        stats = extractor.get_statistics()
        print(f"Files processed: {stats.get('files_processed', 0)}")
        print(f"Files extracted: {stats.get('files_extracted', 0)}")
        print(f"Extraction errors: {stats.get('extraction_errors', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.1f}%")
        
        print("="*80 + "\n")
    
    # Save output if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        if verbose:
            print(f"Report saved to {output_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test AST extraction on a single file with memory inspection")
    parser.add_argument("file_path", help="Path to the file to test")
    parser.add_argument("--db-path", default="extraction_memory.db", help="Path to the memory database")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output", help="Path to save the output")
    
    args = parser.parse_args()
    
    test_file(args.file_path, args.db_path, not args.quiet, args.output)


if __name__ == "__main__":
    main()