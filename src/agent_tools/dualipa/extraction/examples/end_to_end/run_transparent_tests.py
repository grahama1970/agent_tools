#!/usr/bin/env python3
"""
run_transparent_tests.py

This script runs all the transparent validation tests for documentation extraction.
It makes it easy to run multiple tests in sequence and collect all results.

Key features:
- Runs both ArangoDB and ReadTheDocs extraction tests
- Saves all results in a single parent directory
- Creates a summary report of all test results

Example usage:
    python run_transparent_tests.py --output-dir test_results
"""

import os
import sys
import json
import argparse
import logging
import importlib
from pathlib import Path
import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_transparent_tests")

# Constants
TEST_RESULTS_DIR = Path("test_results")

def create_results_directory(output_dir: Optional[Path] = None) -> Path:
    """
    Create and return the path to a results directory.
    
    Args:
        output_dir: Optional directory to use, defaults to test_results/all_docs_yyyy-mm-dd_hhmmss
        
    Returns:
        Path to the results directory
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if output_dir is None:
        output_dir = TEST_RESULTS_DIR / f"all_docs_{timestamp}"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created results directory: {output_dir}")
    return output_dir


def run_test_modules(output_dir: Path) -> Dict[str, Any]:
    """
    Run all transparent validation test modules.
    
    Args:
        output_dir: Directory to save the results
        
    Returns:
        Dictionary of test results
    """
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tests": {},
        "success": True,
    }
    
    # Import the test modules
    try:
        # Only import these modules when needed
        from test_arangodb_extraction_transparent import run_test as run_arangodb_test
        from test_readthedocs_extraction_transparent import run_test as run_readthedocs_test
        
        # Create subdirectories for each test
        arangodb_dir = output_dir / "arangodb"
        readthedocs_dir = output_dir / "readthedocs"
        
        # Run the ArangoDB test
        logger.info("\n===== Running ArangoDB Documentation Test =====")
        arangodb_results = run_arangodb_test(arangodb_dir)
        results["tests"]["arangodb"] = arangodb_results
        
        # Update overall success
        if not arangodb_results.get("success", False):
            results["success"] = False
            
        # Run the ReadTheDocs test
        logger.info("\n===== Running ReadTheDocs Documentation Test =====")
        readthedocs_results = run_readthedocs_test(readthedocs_dir)
        results["tests"]["readthedocs"] = readthedocs_results
        
        # Update overall success
        if not readthedocs_results.get("success", False):
            results["success"] = False
            
    except ImportError as e:
        logger.error(f"Error importing test modules: {e}")
        results["success"] = False
        results["error"] = f"Error importing test modules: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        results["success"] = False
        results["error"] = f"Unexpected error: {str(e)}"
    
    # Create a summary HTML report
    create_summary_report(results, output_dir)
    
    # Save the results to a JSON file
    results_file = output_dir / "all_tests_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved all test results to {results_file}")
    return results


def create_summary_report(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create an HTML summary report of all test results.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save the report
    """
    # Create a summary HTML file
    summary_path = output_dir / "summary.html"
    
    # Collect statistics
    arangodb_stats = results.get("tests", {}).get("arangodb", {}).get("statistics", {})
    readthedocs_stats = results.get("tests", {}).get("readthedocs", {}).get("statistics", {})
    
    # Get output files
    arangodb_files = results.get("tests", {}).get("arangodb", {}).get("output_files", {})
    readthedocs_files = results.get("tests", {}).get("readthedocs", {}).get("output_files", {})
    
    # Create status strings
    arangodb_status = "✅ Passed" if results.get("tests", {}).get("arangodb", {}).get("success", False) else "❌ Failed"
    readthedocs_status = "✅ Passed" if results.get("tests", {}).get("readthedocs", {}).get("success", False) else "❌ Failed"
    
    # Create HTML content
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Documentation Extraction Test Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
        }}
        .test-report {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .success {{
            color: green;
            font-weight: bold;
        }}
        .failure {{
            color: red;
            font-weight: bold;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Documentation Extraction Test Summary</h1>
            <p>Test run: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Overall status: <span class="{'success' if results.get('success', False) else 'failure'}">
                {'✅ All tests passed' if results.get('success', False) else '❌ Some tests failed'}
            </span></p>
        </div>

        <div class="section">
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Total Blocks</th>
                    <th>Summary</th>
                </tr>
                <tr>
                    <td>ArangoDB Documentation</td>
                    <td class="{'success' if 'Passed' in arangodb_status else 'failure'}">{arangodb_status}</td>
                    <td>{arangodb_stats.get("total_blocks", "N/A")}</td>
                    <td>
                        <a href="arangodb/extraction_summary.html">View Details</a>
                    </td>
                </tr>
                <tr>
                    <td>ReadTheDocs Documentation</td>
                    <td class="{'success' if 'Passed' in readthedocs_status else 'failure'}">{readthedocs_status}</td>
                    <td>{readthedocs_stats.get("total_blocks", "N/A")}</td>
                    <td>
                        <a href="readthedocs/extraction_summary.html">View Details</a>
                    </td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>ArangoDB Documentation</h2>
            <div class="test-report">
                <h3>Extraction Statistics</h3>
                <div class="stats">
                    <div class="stat-card">
                        <p><strong>Total Blocks:</strong> {arangodb_stats.get("total_blocks", "N/A")}</p>
                        <p><strong>Documentation Blocks:</strong> {arangodb_stats.get("doc_blocks", "N/A")}</p>
                        <p><strong>Page Blocks:</strong> {arangodb_stats.get("page_blocks", "N/A")}</p>
                    </div>
                    <div class="stat-card">
                        <p><strong>Section Blocks:</strong> {arangodb_stats.get("section_blocks", "N/A")}</p>
                        <p><strong>Code Blocks:</strong> {arangodb_stats.get("code_blocks", "N/A")}</p>
                        <p><strong>Table Blocks:</strong> {arangodb_stats.get("table_blocks", "N/A")}</p>
                    </div>
                </div>
                
                <h3>Output Files</h3>
                <ul>
                    <li><a href="arangodb/{arangodb_files.get('summary', '').split('/')[-1] if 'summary' in arangodb_files else 'extraction_summary.html'}">Summary Report</a></li>
                    <li><a href="arangodb/{arangodb_files.get('html', '').split('/')[-1] if 'html' in arangodb_files else 'arangodb_aql.html'}">Original HTML</a></li>
                    <li><a href="arangodb/{arangodb_files.get('blocks', '').split('/')[-1] if 'blocks' in arangodb_files else 'arangodb_blocks.json'}">Extracted Blocks (JSON)</a></li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>ReadTheDocs Documentation</h2>
            <div class="test-report">
                <h3>Extraction Statistics</h3>
                <div class="stats">
                    <div class="stat-card">
                        <p><strong>Total Blocks:</strong> {readthedocs_stats.get("total_blocks", "N/A")}</p>
                        <p><strong>Documentation Blocks:</strong> {readthedocs_stats.get("doc_blocks", "N/A")}</p>
                        <p><strong>Page Blocks:</strong> {readthedocs_stats.get("page_blocks", "N/A")}</p>
                    </div>
                    <div class="stat-card">
                        <p><strong>Section Blocks:</strong> {readthedocs_stats.get("section_blocks", "N/A")}</p>
                        <p><strong>Code Blocks:</strong> {readthedocs_stats.get("code_blocks", "N/A")}</p>
                        <p><strong>Table Blocks:</strong> {readthedocs_stats.get("table_blocks", "N/A")}</p>
                    </div>
                </div>
                
                <h3>Output Files</h3>
                <ul>
                    <li><a href="readthedocs/{readthedocs_files.get('summary', '').split('/')[-1] if 'summary' in readthedocs_files else 'extraction_summary.html'}">Summary Report</a></li>
                    <li><a href="readthedocs/{readthedocs_files.get('html', '').split('/')[-1] if 'html' in readthedocs_files else 'readthedocs.html'}">Original HTML</a></li>
                    <li><a href="readthedocs/{readthedocs_files.get('blocks', '').split('/')[-1] if 'blocks' in readthedocs_files else 'readthedocs_blocks.json'}">Extracted Blocks (JSON)</a></li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>""")
    
    logger.info(f"Created summary report at {summary_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all transparent validation tests.")
    parser.add_argument("--output-dir", type=str, help="Directory to save test results.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Create results directory
    results_dir = create_results_directory(output_dir)
    
    print(f"Starting all documentation extraction tests...")
    print(f"Results will be saved to: {results_dir}\n")
    
    # Run all tests
    results = run_test_modules(results_dir)
    
    # Print overall result
    if results.get("success"):
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        if "error" in results:
            print(f"Error: {results['error']}")
    
    print(f"\nResults directory: {results_dir}")
    print(f"Summary report: {results_dir}/summary.html")
    
    # Try to open the summary report in browser
    try:
        import webbrowser
        webbrowser.open(str(results_dir / "summary.html"))
        print("Opened summary report in browser")
    except:
        print("Please open the summary report manually to view detailed results")


if __name__ == "__main__":
    main()