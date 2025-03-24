#!/usr/bin/env python3
"""
run_transparent_tests.py

This script runs all the transparent validation tests for documentation extraction.
It makes it easy to run multiple tests in sequence and collect all results.

Key features:
- Runs both ArangoDB and ReadTheDocs extraction tests
- Saves all results in a single parent directory
- Creates a summary report of all test results
- Provides a modern dashboard with Tailwind CSS and Lucide icons

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
import time
import threading
import http.server
import socketserver
from typing import Dict, List, Any, Optional, Tuple

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
    Create an HTML summary report of all test results with Tailwind CSS and Lucide icons.
    
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
    
    # Create relative paths for links
    arangodb_summary = "arangodb/extraction_summary.html"
    readthedocs_summary = "readthedocs/extraction_summary.html"
    arangodb_html = "arangodb/arangodb_aql.html"
    readthedocs_html = "readthedocs/readthedocs.html"
    arangodb_blocks = "arangodb/arangodb_blocks.json" 
    readthedocs_blocks = "readthedocs/readthedocs_blocks.json"
    
    # Create HTML content with Tailwind CSS and Lucide icons
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Extraction Test Dashboard</title>
    <!-- Include Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Include Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
        }}
        .icon {{
            stroke: currentColor;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
            fill: none;
        }}
        .pulse {{
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: .5;
            }}
        }}
    </style>
</head>
<body class="bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <div class="container max-w-7xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="bg-white dark:bg-gray-800 shadow-md rounded-lg p-6 mb-8">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-2xl font-bold text-indigo-600 dark:text-indigo-400">Documentation Extraction Dashboard</h1>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Test run: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                <div class="flex items-center">
                    <span class="flex items-center px-4 py-2 rounded-full {results.get('success', False) and 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' or 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}">
                        <i data-lucide="{results.get('success', False) and 'check-circle' or 'alert-circle'}" class="w-5 h-5 mr-2"></i>
                        <span class="font-medium">{results.get('success', False) and 'All Tests Passed' or 'Some Tests Failed'}</span>
                    </span>
                </div>
            </div>
        </header>

        <!-- Test Summary -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- ArangoDB Test Card -->
            <div class="bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden">
                <div class="px-6 py-4 bg-indigo-50 dark:bg-indigo-900 border-b border-indigo-100 dark:border-indigo-800">
                    <div class="flex justify-between items-center">
                        <h2 class="text-lg font-semibold text-indigo-700 dark:text-indigo-300">ArangoDB Documentation</h2>
                        <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium {arangodb_status.startswith('✅') and 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' or 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}">
                            <i data-lucide="{arangodb_status.startswith('✅') and 'check' or 'x'}" class="w-4 h-4 mr-1"></i>
                            {arangodb_status.replace('✅', '').replace('❌', '').strip()}
                        </span>
                    </div>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-3 gap-4 mb-6">
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-indigo-600 dark:text-indigo-400">{arangodb_stats.get("total_blocks", "N/A")}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Total Blocks</span>
                        </div>
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-indigo-600 dark:text-indigo-400">{arangodb_stats.get("section_blocks", "N/A")}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Sections</span>
                        </div>
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-indigo-600 dark:text-indigo-400">{arangodb_stats.get("code_blocks", 0) + arangodb_stats.get("table_blocks", 0)}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Code & Tables</span>
                        </div>
                    </div>
                    <div class="flex flex-col space-y-2">
                        <a href="{arangodb_summary}" class="flex items-center px-4 py-2 bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-900 dark:hover:bg-indigo-800 rounded-md text-indigo-700 dark:text-indigo-300 transition">
                            <i data-lucide="file-text" class="w-5 h-5 mr-2"></i>
                            <span>View Report Summary</span>
                        </a>
                        <a href="{arangodb_html}" class="flex items-center px-4 py-2 bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-md text-gray-700 dark:text-gray-300 transition">
                            <i data-lucide="code" class="w-5 h-5 mr-2"></i>
                            <span>Original HTML</span>
                        </a>
                        <a href="{arangodb_blocks}" class="flex items-center px-4 py-2 bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-md text-gray-700 dark:text-gray-300 transition">
                            <i data-lucide="database" class="w-5 h-5 mr-2"></i>
                            <span>Extracted Blocks (JSON)</span>
                        </a>
                    </div>
                </div>
            </div>

            <!-- ReadTheDocs Test Card -->
            <div class="bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden">
                <div class="px-6 py-4 bg-purple-50 dark:bg-purple-900 border-b border-purple-100 dark:border-purple-800">
                    <div class="flex justify-between items-center">
                        <h2 class="text-lg font-semibold text-purple-700 dark:text-purple-300">ReadTheDocs Documentation</h2>
                        <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium {readthedocs_status.startswith('✅') and 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' or 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}">
                            <i data-lucide="{readthedocs_status.startswith('✅') and 'check' or 'x'}" class="w-4 h-4 mr-1"></i>
                            {readthedocs_status.replace('✅', '').replace('❌', '').strip()}
                        </span>
                    </div>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-3 gap-4 mb-6">
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-purple-600 dark:text-purple-400">{readthedocs_stats.get("total_blocks", "N/A")}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Total Blocks</span>
                        </div>
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-purple-600 dark:text-purple-400">{readthedocs_stats.get("section_blocks", "N/A")}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Sections</span>
                        </div>
                        <div class="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg text-center">
                            <span class="block text-2xl font-bold text-purple-600 dark:text-purple-400">{readthedocs_stats.get("code_blocks", 0) + readthedocs_stats.get("table_blocks", 0)}</span>
                            <span class="text-sm text-gray-500 dark:text-gray-400">Code & Tables</span>
                        </div>
                    </div>
                    <div class="flex flex-col space-y-2">
                        <a href="{readthedocs_summary}" class="flex items-center px-4 py-2 bg-purple-50 hover:bg-purple-100 dark:bg-purple-900 dark:hover:bg-purple-800 rounded-md text-purple-700 dark:text-purple-300 transition">
                            <i data-lucide="file-text" class="w-5 h-5 mr-2"></i>
                            <span>View Report Summary</span>
                        </a>
                        <a href="{readthedocs_html}" class="flex items-center px-4 py-2 bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-md text-gray-700 dark:text-gray-300 transition">
                            <i data-lucide="code" class="w-5 h-5 mr-2"></i>
                            <span>Original HTML</span>
                        </a>
                        <a href="{readthedocs_blocks}" class="flex items-center px-4 py-2 bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-md text-gray-700 dark:text-gray-300 transition">
                            <i data-lucide="database" class="w-5 h-5 mr-2"></i>
                            <span>Extracted Blocks (JSON)</span>
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <!-- Block Type Analysis -->
        <div class="grid grid-cols-1 gap-6 mb-8">
            <div class="bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden">
                <div class="px-6 py-4 bg-green-50 dark:bg-green-900 border-b border-green-100 dark:border-green-800">
                    <h2 class="text-lg font-semibold text-green-700 dark:text-green-300">Block Type Analysis</h2>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <!-- ArangoDB Block Types -->
                        <div>
                            <h3 class="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">ArangoDB Blocks</h3>
                            <div class="space-y-2">
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="layout" class="w-5 h-5 mr-2 text-indigo-600 dark:text-indigo-400"></i>
                                        <span>Documentation</span>
                                    </div>
                                    <span class="font-semibold">{arangodb_stats.get("doc_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="file" class="w-5 h-5 mr-2 text-indigo-600 dark:text-indigo-400"></i>
                                        <span>Pages</span>
                                    </div>
                                    <span class="font-semibold">{arangodb_stats.get("page_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="book-open" class="w-5 h-5 mr-2 text-indigo-600 dark:text-indigo-400"></i>
                                        <span>Sections</span>
                                    </div>
                                    <span class="font-semibold">{arangodb_stats.get("section_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="code" class="w-5 h-5 mr-2 text-indigo-600 dark:text-indigo-400"></i>
                                        <span>Code Blocks</span>
                                    </div>
                                    <span class="font-semibold">{arangodb_stats.get("code_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="table" class="w-5 h-5 mr-2 text-indigo-600 dark:text-indigo-400"></i>
                                        <span>Tables</span>
                                    </div>
                                    <span class="font-semibold">{arangodb_stats.get("table_blocks", 0)}</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- ReadTheDocs Block Types -->
                        <div>
                            <h3 class="text-md font-medium text-gray-700 dark:text-gray-300 mb-3">ReadTheDocs Blocks</h3>
                            <div class="space-y-2">
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="layout" class="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400"></i>
                                        <span>Documentation</span>
                                    </div>
                                    <span class="font-semibold">{readthedocs_stats.get("doc_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="file" class="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400"></i>
                                        <span>Pages</span>
                                    </div>
                                    <span class="font-semibold">{readthedocs_stats.get("page_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="book-open" class="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400"></i>
                                        <span>Sections</span>
                                    </div>
                                    <span class="font-semibold">{readthedocs_stats.get("section_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="code" class="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400"></i>
                                        <span>Code Blocks</span>
                                    </div>
                                    <span class="font-semibold">{readthedocs_stats.get("code_blocks", 0)}</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                                    <div class="flex items-center">
                                        <i data-lucide="table" class="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400"></i>
                                        <span>Tables</span>
                                    </div>
                                    <span class="font-semibold">{readthedocs_stats.get("table_blocks", 0)}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Commands -->
        <div class="bg-white dark:bg-gray-800 shadow-md rounded-lg overflow-hidden mb-8">
            <div class="px-6 py-4 bg-yellow-50 dark:bg-yellow-900 border-b border-yellow-100 dark:border-yellow-800">
                <h2 class="text-lg font-semibold text-yellow-700 dark:text-yellow-300">Quick Commands</h2>
            </div>
            <div class="p-6">
                <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">Here are some useful commands to analyze the extraction results:</p>
                <div class="space-y-3">
                    <div class="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                        <p class="text-sm font-mono"># Count blocks by type</p>
                        <code class="block mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm overflow-x-auto">
                            cat {arangodb_blocks} | grep "type" | sort | uniq -c
                        </code>
                    </div>
                    <div class="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                        <p class="text-sm font-mono"># Examine documentation hierarchy</p>
                        <code class="block mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm overflow-x-auto">
                            cat {arangodb_blocks} | grep -E "uuid|type|parent_uuid|child_uuids"
                        </code>
                    </div>
                    <div class="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                        <p class="text-sm font-mono"># Check for orphaned blocks (no parent)</p>
                        <code class="block mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm overflow-x-auto">
                            cat {arangodb_blocks} | jq '.[] | select(.type != "documentation" and (.parent_uuid == null or .parent_uuid == ""))'
                        </code>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-gray-500 dark:text-gray-400 text-sm">
            <p>Test results created at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p class="mt-1">DuaLiPA Documentation Extraction Testing Framework</p>
        </footer>
    </div>

    <!-- Initialize Lucide Icons -->
    <script>
        lucide.createIcons();
    </script>
</body>
</html>
""")
    
    logger.info(f"Created summary report at {summary_path}")


import socket
import threading
import http.server
import socketserver

def get_ip_address():
    """
    Get the primary IP address of the machine.
    This will work in both regular Linux and WSL2 environments.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't actually establish a connection, but gets the IP
        # that would be used for an outbound connection
        s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback if that fails
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except:
            ip = '127.0.0.1'  # Last resort fallback
    finally:
        s.close()
    return ip

def get_wsl_ip():
    """
    More reliable WSL2 IP detection.
    
    This approach looks at the nameserver in resolv.conf which
    is typically the Windows host in WSL2.
    """
    try:
        if not is_running_in_wsl():
            return None
            
        # First try using socket to get IP connected to DNS
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"⚠️ WSL IP detection method 1 failed: {str(e)}", flush=True)
        
        # Second try using resolv.conf
        try:
            with open('/etc/resolv.conf') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        # Get interface IP using socket
                        nameserver = line.split()[1]
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect((nameserver, 53))
                        ip = s.getsockname()[0]
                        s.close()
                        return ip
        except Exception as e:
            print(f"⚠️ WSL IP detection method 2 failed: {str(e)}", flush=True)
            
        # Last resort
        return '127.0.0.1'

def get_wsl_windows_ip():
    """
    Attempt to get the Windows host IP from WSL.
    This is useful for accessing the server from outside WSL.
    """
    try:
        if not is_running_in_wsl():
            return None
            
        # First method: The default gateway in WSL2 is usually the Windows host
        import subprocess
        output = subprocess.check_output(
            "ip route | grep default | awk '{print $3}'",
            shell=True, text=True
        ).strip()
        if output:
            print(f"✅ Found Windows host IP (method 1): {output}", flush=True)
            return output
            
        # Second method: Check resolv.conf
        try:
            with open('/etc/resolv.conf') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[1]
                        print(f"✅ Found Windows host IP (method 2): {ip}", flush=True)
                        return ip
        except Exception as e:
            print(f"⚠️ Windows host IP detection method 2 failed: {str(e)}", flush=True)
            
        return None
    except Exception as e:
        print(f"⚠️ Windows host IP detection method 1 failed: {str(e)}", flush=True)
        return None

def get_tailscale_ip():
    """Get the Tailscale IP address if available."""
    try:
        # Run the tailscale status command and parse output
        import subprocess
        output = subprocess.check_output(["tailscale", "status", "--self"], 
                                          stderr=subprocess.STDOUT, 
                                          text=True)
        
        # Parse the output to find the IP address
        lines = output.strip().split('\n')
        for line in lines:
            if line.startswith("100."):  # Tailscale IPs start with 100.
                return line.split()[0]
        
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        # Either tailscale command failed or isn't installed
        return None

def is_running_in_wsl():
    """Check if we're running in a WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def create_terminal_link(file_path, text=None):
    """
    Creates a clickable terminal link that works in most modern terminals.
    
    Args:
        file_path: Path to the file
        text: Text to display for the link (defaults to file_path)
        
    Returns:
        Formatted string with terminal hyperlink
    """
    if text is None:
        text = file_path
    
    # Convert to absolute path
    if not str(file_path).startswith('/'):
        file_path = os.path.abspath(file_path)
    
    # Format as a terminal hyperlink
    # This works in many modern terminals like iTerm2, GNOME Terminal, etc.
    return f"\033]8;;file://{file_path}\033\\{text}\033]8;;\033\\"

def wait_for_server_startup(port, timeout=5):
    """
    Wait until server is actually listening.
    
    Args:
        port: The port to check
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if server is ready, False otherwise
    """
    print(f"🔍 Checking if server is ready on port {port}...", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                print(f"✅ Server is ready and listening on port {port}", flush=True)
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.5)
    
    print(f"⚠️ Server readiness check timed out after {timeout} seconds", flush=True)
    return False


def print_server_urls(results_dir, port):
    """
    Print all server access URLs with improved visibility.
    
    Args:
        results_dir: Directory where test results are stored
        port: Port the server is listening on
    """
    # Get machine IP addresses
    local_ip = get_ip_address()
    tailscale_ip = get_tailscale_ip()
    
    # WSL2-specific handling
    is_wsl = is_running_in_wsl()
    windows_host_ip = get_wsl_windows_ip() if is_wsl else None
    wsl_ip = get_wsl_ip() if is_wsl else None
    
    # Create paths to important files
    summary_path = os.path.join(results_dir, "summary.html")
    
    # Create horizontal separator
    separator = "=" * 80
    
    # Print header
    print(f"\n{separator}", flush=True)
    print(f"🌐 WEB SERVER READY - PORT {port}", flush=True)
    print(f"📊 ACCESS THE TEST REPORTS USING THESE URLS:", flush=True)
    print(f"{separator}", flush=True)
    
    # Show localhost links
    print(f"\n🏠 LOCAL ACCESS (SAME MACHINE):", flush=True)
    print(f"  • Main Summary: http://localhost:{port}/summary.html", flush=True)
    print(f"  • ArangoDB Test: http://localhost:{port}/arangodb/extraction_summary.html", flush=True)
    print(f"  • ReadTheDocs Test: http://localhost:{port}/readthedocs/extraction_summary.html", flush=True)
    
    # Show WSL2-specific links if applicable
    if is_wsl:
        print(f"\n🪟 WINDOWS HOST ACCESS (FROM WSL2):", flush=True)
        if wsl_ip:
            print(f"  • Main Summary: http://{wsl_ip}:{port}/summary.html", flush=True)
            print(f"  • ArangoDB Test: http://{wsl_ip}:{port}/arangodb/extraction_summary.html", flush=True)
            print(f"  • ReadTheDocs Test: http://{wsl_ip}:{port}/readthedocs/extraction_summary.html", flush=True)
        else:
            print(f"  • Could not detect WSL IP address", flush=True)
        
        if windows_host_ip:
            print(f"\n🖥️ ACCESS FROM OTHER COMPUTERS ON YOUR NETWORK:", flush=True)
            print(f"  • Main Summary: http://{windows_host_ip}:{port}/summary.html", flush=True)
            print(f"  • ArangoDB Test: http://{windows_host_ip}:{port}/arangodb/extraction_summary.html", flush=True)
            print(f"  • ReadTheDocs Test: http://{windows_host_ip}:{port}/readthedocs/extraction_summary.html", flush=True)
            
        print(f"\n💡 WSL2 NETWORK CONFIGURATION:", flush=True)
        print(f"  If URLs don't work, run these Windows commands (PowerShell Admin):", flush=True)
        print(f"  ```", flush=True)
        print(f"  # Port forwarding", flush=True)
        print(f"  netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={wsl_ip or local_ip}", flush=True)
        print(f"  # Firewall rule", flush=True)
        print(f"  New-NetFirewallRule -DisplayName \"WSL2 Port {port}\" -Direction Inbound -LocalPort {port} -Action Allow -Protocol TCP", flush=True)
        print(f"  ```", flush=True)
    else:
        # Show regular local network links for non-WSL
        print(f"\n📊 LOCAL NETWORK LINKS:", flush=True)
        print(f"  • Main Summary: http://{local_ip}:{port}/summary.html", flush=True)
        print(f"  • ArangoDB Test: http://{local_ip}:{port}/arangodb/extraction_summary.html", flush=True)
        print(f"  • ReadTheDocs Test: http://{local_ip}:{port}/readthedocs/extraction_summary.html", flush=True)
    
    # Show Tailscale links if available
    if tailscale_ip:
        print(f"\n🔒 TAILSCALE SECURE LINKS:", flush=True)
        print(f"  • Main Summary: http://{tailscale_ip}:{port}/summary.html", flush=True)
        print(f"  • ArangoDB Test: http://{tailscale_ip}:{port}/arangodb/extraction_summary.html", flush=True)
        print(f"  • ReadTheDocs Test: http://{tailscale_ip}:{port}/readthedocs/extraction_summary.html", flush=True)
    
    print(f"\n{separator}", flush=True)
    print(f"SERVER WILL RUN UNTIL YOU PRESS CTRL+C", flush=True)
    print(f"{separator}\n", flush=True)
    
    # Output again to a file for reference
    url_file = os.path.join(results_dir, "server_urls.txt")
    try:
        with open(url_file, 'w') as f:
            f.write(f"WEB SERVER URLS - PORT {port}\n\n")
            f.write(f"Local access: http://localhost:{port}/summary.html\n")
            if is_wsl and wsl_ip:
                f.write(f"Windows access: http://{wsl_ip}:{port}/summary.html\n")
            if windows_host_ip:
                f.write(f"Network access: http://{windows_host_ip}:{port}/summary.html\n")
            if tailscale_ip:
                f.write(f"Tailscale access: http://{tailscale_ip}:{port}/summary.html\n")
        print(f"URLs also saved to: {url_file}", flush=True)
    except Exception as e:
        print(f"Could not save URLs to file: {e}", flush=True)

def start_http_server(directory, port=0):
    """
    Start a simple HTTP server in a separate thread.
    Works in both regular Linux and WSL2 environments.
    
    Args:
        directory: The directory to serve files from
        port: Port to use (0 means auto-select)
        
    Returns:
        tuple: (server_thread, server_port)
    """
    # Important: First change to the directory to serve files from
    original_dir = os.getcwd()
    os.chdir(str(directory))
    
    # Log for debugging
    print(f"🌐 Starting HTTP server in directory: {os.getcwd()}", flush=True)
    
    # Set ports to try - using less common ports to avoid conflicts
    if port == 0:
        # Try ports in the less common range
        ports_to_try = [8765, 9876, 7654, 6543, 5432]
    else:
        # Try the specified port and some alternatives
        ports_to_try = [port, port+1, port+2, port+1000, port+2000]
    
    # In WSL2, we need to bind to all interfaces (0.0.0.0) to be accessible from Windows
    host = "0.0.0.0" if is_running_in_wsl() else ""
    
    # Create a custom handler that suppresses console output
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress the default logging to keep console clean
            pass
    
    # Try each port in sequence
    for attempt_port in ports_to_try:
        try:
            print(f"🔄 Trying port: {attempt_port} on host: {host or 'all interfaces'}", flush=True)
            
            # Bind to the port
            httpd = socketserver.TCPServer((host, attempt_port), QuietHTTPRequestHandler)
            port = attempt_port
            
            print(f"✅ Successfully bound to port {port}", flush=True)
            
            # Create a daemon thread that will be killed when the main process exits
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait to confirm thread is running
            import time
            time.sleep(1)
            
            if server_thread.is_alive():
                print(f"✅ Server thread started successfully", flush=True)
                
                # Verify server is actually listening
                if wait_for_server_startup(port):
                    return server_thread, port
                else:
                    print(f"⚠️ Server thread is alive but not responding", flush=True)
                    continue
            else:
                print(f"❌ Server thread failed to start", flush=True)
                continue
        
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠️ Port {attempt_port} is already in use, trying another port", flush=True)
                continue
            else:
                print(f"❌ OSError: {e}", flush=True)
                raise
        
        except Exception as e:
            print(f"❌ Error starting HTTP server: {e}", flush=True)
            os.chdir(original_dir)
            raise
    
    # If we get here, all ports failed
    print("❌ Could not find an available port", flush=True)
    os.chdir(original_dir)
    raise RuntimeError("All ports are in use")

def serve_with_docker(results_dir: Path, port: int) -> bool:
    """
    Serve the results directory using a Docker container.
    
    Args:
        results_dir: Directory to serve
        port: Port to use
        
    Returns:
        bool: True if started successfully, False otherwise
    """
    try:
        # Import the report_server module
        from agent_tools.report_server import serve_with_docker as docker_serve
        from agent_tools.report_server import create_terminal_link
        
        print(f"🐳 Starting Docker container to serve reports...", flush=True)
        success, url = docker_serve(results_dir, port)
        
        if success and url:
            # Create terminal links to key files
            summary_url = f"{url.rstrip('/')}/summary.html"
            arangodb_url = f"{url.rstrip('/')}/arangodb/extraction_summary.html"
            readthedocs_url = f"{url.rstrip('/')}/readthedocs/extraction_summary.html"
            
            # Print clickable links (using terminal hyperlinks)
            print(f"\n📊 CLICKABLE REPORT LINKS (works in modern terminals):", flush=True)
            print(f"  • Main Summary: \033]8;;{summary_url}\033\\Click to Open Summary\033]8;;\033\\", flush=True)
            print(f"  • ArangoDB Test: \033]8;;{arangodb_url}\033\\Click to Open ArangoDB\033]8;;\033\\", flush=True)
            print(f"  • ReadTheDocs Test: \033]8;;{readthedocs_url}\033\\Click to Open ReadTheDocs\033]8;;\033\\", flush=True)
        
        return success
    except ImportError as e:
        print(f"⚠️ Could not import report_server module: {e}", flush=True)
        print(f"⚠️ Falling back to local Docker implementation", flush=True)
        
        # Fallback to local docker-serve.sh script
        import subprocess
        import os
        
        # Get absolute path to docker-serve.sh
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        docker_serve_path = script_dir / "docker-serve.sh"
        
        # Make sure the script is executable
        try:
            os.chmod(docker_serve_path, 0o755)
        except Exception as e:
            print(f"⚠️ Could not make docker-serve.sh executable: {e}", flush=True)
            
        # Run the docker-serve.sh script
        try:
            subprocess.run(
                [str(docker_serve_path), "--directory", str(results_dir), "--port", str(port)],
                check=True,
                text=True
            )
            return True
        except Exception as e:
            print(f"❌ Error starting Docker container: {e}", flush=True)
            return False
    except Exception as e:
        print(f"❌ Unexpected error starting Docker container: {e}", flush=True)
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all transparent validation tests.")
    parser.add_argument("--output-dir", type=str, help="Directory to save test results.")
    parser.add_argument("--serve", action="store_true", help="Start a web server to view results")
    parser.add_argument("--docker-serve", action="store_true", help="Start a Docker container to serve results (preferred method)")
    parser.add_argument("--port", type=int, default=12345, help="Port to use for web server (default: 12345)")
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
    
    # Create paths for important files
    summary_path = results_dir / "summary.html"
    arangodb_path = results_dir / "arangodb" / "extraction_summary.html"
    readthedocs_path = results_dir / "readthedocs" / "extraction_summary.html"
    
    # Show local file links (for local use)
    print("\n📄 Local file access:", flush=True)
    print(f"  - Main Summary: {os.path.abspath(summary_path)}", flush=True)
    print(f"  - ArangoDB Test: {os.path.abspath(arangodb_path)}", flush=True)
    print(f"  - ReadTheDocs Test: {os.path.abspath(readthedocs_path)}", flush=True)
    
    # Start HTTP server if requested
    if args.docker_serve:
        # Use Docker-based approach (preferred)
        serve_with_docker(results_dir, args.port)
    elif args.serve:
        try:
            # First start HTTP server using built-in Python server
            print(f"\n🚀 Starting web server...", flush=True)
            server_thread, port = start_http_server(results_dir, args.port)
            
            # Print URLs immediately after server is confirmed running
            print_server_urls(results_dir, port)
            
            # Server maintenance loop with heartbeat to ensure output is flushed
            print(f"💓 Server heartbeat started (will update every 60 seconds)", flush=True)
            heartbeat_count = 0
            try:
                while True:
                    heartbeat_count += 1
                    time.sleep(60)
                    print(f"💓 Server heartbeat #{heartbeat_count} - still running on port {port}", flush=True)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down server...", flush=True)
                
        except Exception as e:
            print(f"\n❌ Error starting Python web server: {e}", flush=True)
            
            # Try Docker server as fallback
            if not serve_with_docker(results_dir, args.port):
                print("Falling back to local file access", flush=True)
    else:
        # Traditional file links for local use
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(summary_path)}")
            print("\nOpened summary report in browser", flush=True)
        except Exception as e:
            print(f"\nCouldn't open browser automatically: {e}", flush=True)
            print("Please use the file paths above to view the reports manually", flush=True)


if __name__ == "__main__":
    main()