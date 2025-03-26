#!/usr/bin/env python3
"""
Comprehensive test script for AST extraction with memory integration.

This script allows testing various complex code structures and uses the
AI memory system to track extraction patterns, learn from errors, and
improve extraction over time.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add repository root to path
sys.path.append(str(Path(__file__).parent))

# Import AST extractor and memory components
from src.agent_tools.dualipa.extraction.extractors.code.ast_extractor import AstExtractor
from src.agent_tools.dualipa.extraction.test_state_manager import (
    get_state_manager,
    what_am_i_doing,
    remember_context
)
from src.agent_tools.dualipa.extraction.extraction_memory import (
    init_extraction_memory,
    track_extraction_start,
    track_extraction_progress,
    track_extraction_completion,
    record_extraction_error,
    find_similar_errors,
    save_extraction_knowledge,
    find_extraction_knowledge,
    get_extraction_context
)

# Optional import for memory visualization
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class AstTester:
    """Test harness for AST extraction with memory integration."""
    
    def __init__(self, memory_db_path: str = "extraction_test.db", verbose: bool = False):
        """
        Initialize the AST tester.
        
        Args:
            memory_db_path: Path to the memory database
            verbose: Whether to print verbose output
        """
        self.memory_db_path = memory_db_path
        self.verbose = verbose
        self.results = {}
        
        # Initialize memory system
        self.state_manager = get_state_manager(memory_db_path)
        init_extraction_memory(memory_db_path)
        
        # Initialize extractor with memory integration
        self.extractor = AstExtractor(memory_db_path=memory_db_path)
        
        # Set up test session
        self._setup_test_session()
    
    def _setup_test_session(self):
        """Set up a new test session with context tracking."""
        session_id = f"ast-test-{int(time.time())}"
        self.session_id = session_id
        
        # Record context in memory
        remember_context(
            "AST Extraction Testing",
            "Test the AST extractor's ability to parse complex code structures",
            "Initializing test session",
            "Prepare to run extraction tests on complex code samples"
        )
        
        if self.verbose:
            print(f"Test session {session_id} initialized with memory at {self.memory_db_path}")
            print("-" * 70)
    
    def test_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Test AST extraction on a single file with memory tracking.
        
        Args:
            file_path: Path to the file to test
            language: Optional language override
            
        Returns:
            Dictionary with test results
        """
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            record_extraction_error("file_not_found", error_msg, file_path)
            return {"error": error_msg}
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1]
        
        if self.verbose:
            print(f"\nTESTING: {file_path}")
            print("-" * 70)
            
            # Check what's in memory before test
            print("MEMORY STATE BEFORE TEST:")
            context = get_extraction_context()
            print(f"Current context: {context}")
            print("-" * 70)
        
        # Track test start in memory
        track_extraction_start(
            file_name,
            "ast_test",
            {
                "file_path": file_path,
                "file_extension": file_ext,
                "language": language,
                "timestamp": time.time()
            }
        )
        
        # Update context
        remember_context(
            "AST Extraction Testing",
            "Test the AST extractor's ability to parse complex code structures",
            f"Extracting from {file_name}",
            "Process file and analyze results"
        )
        
        # Perform extraction with timing
        start_time = time.time()
        try:
            result = self.extractor.extract_file(file_path, language)
            success = True
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            record_extraction_error("extraction_failure", error_msg, file_path)
            result = {"error": error_msg, "file_path": file_path}
            success = False
        
        duration = time.time() - start_time
        
        # Store results
        test_result = {
            "file_path": file_path,
            "file_name": file_name,
            "duration": duration,
            "success": success,
            "result": result,
            "timestamp": time.time()
        }
        
        self.results[file_path] = test_result
        
        # Track completion in memory
        if success:
            structures_found = {
                "classes": len(result.get("classes", [])),
                "functions": len(result.get("functions", [])),
                "imports": len(result.get("imports", []))
            }
            
            # Add language-specific counts
            if result.get("language") == "typescript" or result.get("language") == "javascript":
                structures_found["interfaces"] = len(result.get("interfaces", []))
                structures_found["exports"] = len(result.get("exports", []))
            elif result.get("language") == "go":
                structures_found["structs"] = len(result.get("structs", []))
            elif result.get("language") == "rust":
                structures_found["traits"] = len(result.get("traits", []))
                structures_found["modules"] = len(result.get("modules", []))
            
            track_extraction_completion(
                file_name,
                f"Extracted {file_name} successfully in {duration:.4f}s",
                {
                    "duration": duration,
                    "structures_found": structures_found,
                    "memory_usage": self.extractor.get_statistics()
                }
            )
            
            # Save knowledge about extraction patterns
            if result.get("classes") and any(cls.get("inner_classes") for cls in result.get("classes", [])):
                save_extraction_knowledge(
                    f"nested_classes_{result.get('language')}",
                    f"Successfully extracted nested classes in {file_name}",
                    summary=f"Nested class extraction pattern for {result.get('language')}",
                    tags=[result.get('language'), "nested_classes", "success_pattern"]
                )
                
            if result.get("language") == "typescript" and result.get("interfaces"):
                save_extraction_knowledge(
                    f"typescript_interfaces_{file_name}",
                    f"Successfully extracted TypeScript interfaces in {file_name}",
                    summary=f"TypeScript interface extraction pattern",
                    tags=["typescript", "interfaces", "success_pattern"]
                )
        else:
            record_extraction_error(
                "extraction_error",
                f"Failed to extract {file_name}: {result.get('error', 'Unknown error')}",
                file_path,
                severity=7
            )
            
            # Find similar errors for suggestion
            similar_errors = find_similar_errors(str(result.get('error', '')))
            if similar_errors and isinstance(similar_errors, list):
                test_result["similar_errors"] = similar_errors
                
                if self.verbose:
                    print("\nSIMILAR ERRORS FOUND:")
                    for err in similar_errors:
                        print(f"- {err.get('error_type')}: {err.get('recovery_action', 'No recovery action')}")
        
        if self.verbose:
            print("\nEXTRACTION RESULTS:")
            print(f"Duration: {duration:.4f} seconds")
            
            if success:
                if "classes" in result:
                    print(f"Classes found: {len(result['classes'])}")
                    for cls in result["classes"]:
                        print(f"  - {cls['name']}")
                        
                        # Show inheritance
                        if "inherits_from" in cls and cls["inherits_from"]:
                            print(f"    inherits from: {cls['inherits_from']}")
                            
                        # Show nested classes
                        if "inner_classes" in cls and cls["inner_classes"]:
                            print(f"    inner classes: {[ic['name'] for ic in cls['inner_classes']]}")
                
                if "interfaces" in result:
                    print(f"Interfaces found: {len(result['interfaces'])}")
                    for interface in result["interfaces"]:
                        print(f"  - {interface['name']}")
                
                if "functions" in result:
                    print(f"Functions found: {len(result['functions'])}")
                
                if "imports" in result:
                    print(f"Imports found: {len(result['imports'])}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("-" * 70)
            
            # Check memory after test
            print("MEMORY STATE AFTER TEST:")
            context = get_extraction_context()
            print(f"Updated context: {context}")
            print("-" * 70)
        
        # Save results to file
        output_file = f"{file_name}_extraction_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        if self.verbose:
            print(f"Results saved to {output_file}")
        
        return test_result
    
    def test_directory(self, dir_path: str, pattern: str = "**/*.*", languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test AST extraction on all matching files in a directory.
        
        Args:
            dir_path: Path to the directory to test
            pattern: Glob pattern for files to test
            languages: Optional list of languages to test
            
        Returns:
            Dictionary with test results
        """
        if not os.path.isdir(dir_path):
            return {"error": f"Directory not found: {dir_path}"}
        
        # Find matching files
        matched_files = list(Path(dir_path).glob(pattern))
        
        if not matched_files:
            return {"error": f"No files matching pattern '{pattern}' found in {dir_path}"}
        
        # Initialize results
        directory_results = {
            "directory": dir_path,
            "pattern": pattern,
            "file_count": len(matched_files),
            "languages": languages,
            "results": {},
            "stats": {
                "success_count": 0,
                "error_count": 0,
                "total_duration": 0,
                "by_language": {}
            }
        }
        
        # Start tracking in memory
        track_extraction_start(
            os.path.basename(dir_path),
            "directory_test",
            {
                "directory": dir_path,
                "pattern": pattern,
                "file_count": len(matched_files),
                "languages": languages,
                "timestamp": time.time()
            }
        )
        
        # Track progress for updates
        total_files = len(matched_files)
        progress_interval = max(1, total_files // 10)  # Update at 10% intervals
        
        # Process each file
        for i, file_path in enumerate(matched_files):
            file_path_str = str(file_path)
            
            # Check if file should be processed based on languages filter
            file_ext = os.path.splitext(file_path_str)[1].lstrip('.')
            if languages and file_ext not in languages:
                continue
                
            # Test file
            result = self.test_file(file_path_str)
            directory_results["results"][file_path_str] = result
            
            # Update stats
            directory_results["stats"]["total_duration"] += result["duration"]
            
            if result["success"]:
                directory_results["stats"]["success_count"] += 1
            else:
                directory_results["stats"]["error_count"] += 1
            
            # Update language stats
            language = result.get("result", {}).get("language")
            if language:
                if language not in directory_results["stats"]["by_language"]:
                    directory_results["stats"]["by_language"][language] = {
                        "count": 0,
                        "success_count": 0,
                        "error_count": 0,
                        "total_duration": 0
                    }
                
                lang_stats = directory_results["stats"]["by_language"][language]
                lang_stats["count"] += 1
                lang_stats["total_duration"] += result["duration"]
                
                if result["success"]:
                    lang_stats["success_count"] += 1
                else:
                    lang_stats["error_count"] += 1
            
            # Report progress at intervals
            if (i + 1) % progress_interval == 0 or (i + 1) == total_files:
                progress_pct = ((i + 1) / total_files) * 100
                
                track_extraction_progress(
                    os.path.basename(dir_path),
                    "directory_extraction",
                    f"Processed {i + 1}/{total_files} files ({progress_pct:.1f}%)",
                    "Continue processing files",
                    {
                        "processed": i + 1,
                        "total": total_files,
                        "success_count": directory_results["stats"]["success_count"],
                        "error_count": directory_results["stats"]["error_count"]
                    }
                )
                
                if self.verbose:
                    print(f"Progress: {i + 1}/{total_files} files ({progress_pct:.1f}%)")
        
        # Complete tracking
        track_extraction_completion(
            os.path.basename(dir_path),
            f"Completed directory test of {total_files} files",
            directory_results["stats"]
        )
        
        # Save overall results
        output_file = f"{os.path.basename(dir_path)}_directory_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(directory_results, f, indent=2)
            
        if self.verbose:
            print(f"\nDirectory test complete. Results saved to {output_file}")
            print(f"Success rate: {directory_results['stats']['success_count']}/{total_files} files ({directory_results['stats']['success_count']/total_files*100:.1f}%)")
        
        return directory_results
    
    def compare_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two extraction results to identify differences.
        
        Args:
            result1: First extraction result
            result2: Second extraction result
            
        Returns:
            Dictionary with comparison results
        """
        def count_elements(result):
            counts = {}
            for key in ["classes", "functions", "imports", "interfaces", "exports", "structs", "traits", "modules"]:
                if key in result:
                    counts[key] = len(result[key])
            return counts
        
        counts1 = count_elements(result1)
        counts2 = count_elements(result2)
        
        # Find differences
        differences = {
            "element_counts": {
                "result1": counts1,
                "result2": counts2
            },
            "missing_in_result2": {},
            "missing_in_result1": {},
            "differences": {}
        }
        
        # Compare element counts
        for key in set(counts1.keys()) | set(counts2.keys()):
            if key in counts1 and key in counts2:
                if counts1[key] != counts2[key]:
                    differences["differences"][key] = {
                        "result1_count": counts1[key],
                        "result2_count": counts2[key]
                    }
            elif key in counts1:
                differences["missing_in_result2"][key] = counts1[key]
            else:
                differences["missing_in_result1"][key] = counts2[key]
        
        # Compare classes in detail
        if "classes" in result1 and "classes" in result2:
            differences["class_details"] = self._compare_classes(result1["classes"], result2["classes"])
        
        # Compare functions in detail
        if "functions" in result1 and "functions" in result2:
            differences["function_details"] = self._compare_functions(result1["functions"], result2["functions"])
        
        return differences
    
    def _compare_classes(self, classes1, classes2):
        """Compare classes from two extraction results."""
        classes1_dict = {cls["name"]: cls for cls in classes1}
        classes2_dict = {cls["name"]: cls for cls in classes2}
        
        result = {
            "only_in_first": [],
            "only_in_second": [],
            "in_both_with_differences": []
        }
        
        # Find classes only in first result
        for name in classes1_dict:
            if name not in classes2_dict:
                result["only_in_first"].append(name)
        
        # Find classes only in second result
        for name in classes2_dict:
            if name not in classes1_dict:
                result["only_in_second"].append(name)
        
        # Compare classes that exist in both
        for name in set(classes1_dict.keys()) & set(classes2_dict.keys()):
            class1 = classes1_dict[name]
            class2 = classes2_dict[name]
            
            differences = {}
            
            # Compare inheritance
            if class1.get("inherits_from") != class2.get("inherits_from"):
                differences["inherits_from"] = {
                    "first": class1.get("inherits_from"),
                    "second": class2.get("inherits_from")
                }
            
            # Compare nested classes
            inner1 = {inner.get("name"): inner for inner in class1.get("inner_classes", [])}
            inner2 = {inner.get("name"): inner for inner in class2.get("inner_classes", [])}
            
            if set(inner1.keys()) != set(inner2.keys()):
                differences["inner_classes"] = {
                    "only_in_first": list(set(inner1.keys()) - set(inner2.keys())),
                    "only_in_second": list(set(inner2.keys()) - set(inner1.keys()))
                }
            
            # Compare methods
            methods1 = {method.get("name"): method for method in class1.get("methods", [])}
            methods2 = {method.get("name"): method for method in class2.get("methods", [])}
            
            if set(methods1.keys()) != set(methods2.keys()):
                differences["methods"] = {
                    "only_in_first": list(set(methods1.keys()) - set(methods2.keys())),
                    "only_in_second": list(set(methods2.keys()) - set(methods1.keys()))
                }
            
            if differences:
                result["in_both_with_differences"].append({
                    "name": name,
                    "differences": differences
                })
        
        return result
    
    def _compare_functions(self, functions1, functions2):
        """Compare functions from two extraction results."""
        functions1_dict = {func["name"]: func for func in functions1}
        functions2_dict = {func["name"]: func for func in functions2}
        
        result = {
            "only_in_first": [],
            "only_in_second": [],
            "in_both_with_differences": []
        }
        
        # Find functions only in first result
        for name in functions1_dict:
            if name not in functions2_dict:
                result["only_in_first"].append(name)
        
        # Find functions only in second result
        for name in functions2_dict:
            if name not in functions1_dict:
                result["only_in_second"].append(name)
        
        return result
    
    def visualize_results(self, output_path: Optional[str] = None):
        """
        Visualize test results if matplotlib is available.
        
        Args:
            output_path: Optional path to save visualization
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available. Install matplotlib for visualization support.")
            return
        
        if not self.results:
            print("No results to visualize.")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Gather data
        languages = {}
        durations = []
        success_counts = 0
        error_counts = 0
        
        for result in self.results.values():
            durations.append(result["duration"])
            
            if result["success"]:
                success_counts += 1
                language = result.get("result", {}).get("language")
                if language:
                    languages[language] = languages.get(language, 0) + 1
            else:
                error_counts += 1
        
        # Plot 1: Success vs Failure
        labels = ['Success', 'Error']
        sizes = [success_counts, error_counts]
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Extraction Success Rate')
        
        # Plot 2: Language distribution
        if languages:
            lang_labels = list(languages.keys())
            lang_sizes = list(languages.values())
            ax2.bar(lang_labels, lang_sizes)
            ax2.set_title('Languages Processed')
            ax2.set_ylabel('Number of Files')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test AST extraction with memory integration")
    
    parser.add_argument(
        "--db-path",
        default="extraction_test.db",
        help="Path to the memory database"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Test file command
    file_parser = subparsers.add_parser("file", help="Test a single file")
    file_parser.add_argument("file_path", help="Path to the file to test")
    file_parser.add_argument("--language", help="Override language detection")
    
    # Test directory command
    dir_parser = subparsers.add_parser("dir", help="Test a directory of files")
    dir_parser.add_argument("dir_path", help="Path to the directory to test")
    dir_parser.add_argument("--pattern", default="**/*.*", help="Glob pattern for files to test")
    dir_parser.add_argument("--languages", nargs="+", help="List of language extensions to test (e.g., py js ts)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two extraction results")
    compare_parser.add_argument("file1", help="Path to first result JSON file")
    compare_parser.add_argument("file2", help="Path to second result JSON file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize tester
    tester = AstTester(args.db_path, args.verbose)
    
    # Execute command
    if args.command == "file":
        tester.test_file(args.file_path, args.language)
    elif args.command == "dir":
        tester.test_directory(args.dir_path, args.pattern, args.languages)
    elif args.command == "compare":
        # Load results
        try:
            with open(args.file1, 'r') as f:
                result1 = json.load(f)
                
            with open(args.file2, 'r') as f:
                result2 = json.load(f)
                
            differences = tester.compare_results(result1, result2)
            
            # Save comparison results
            output_file = "comparison_results.json"
            with open(output_file, "w") as f:
                json.dump(differences, f, indent=2)
                
            print(f"Comparison results saved to {output_file}")
            
        except Exception as e:
            print(f"Error comparing results: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()