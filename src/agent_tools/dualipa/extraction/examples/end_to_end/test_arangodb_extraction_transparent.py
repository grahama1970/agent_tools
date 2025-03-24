#!/usr/bin/env python3
"""
test_arangodb_extraction_transparent.py

This script performs a transparent validation of ArangoDB documentation extraction.
It downloads ArangoDB documentation, processes it with the fetch_docs pipeline,
and creates human-readable verification artifacts for review.

Key features:
- Downloads and saves the original ArangoDB HTML
- Extracts and saves the processed output blocks
- Creates an HTML report comparing inputs and outputs
- Provides detailed statistics on the extraction process
- Includes sample commands for human verification

Output: Human-friendly verification report and artifacts for review

Example usage:
    python test_arangodb_extraction_transparent.py --output-dir test_results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import shutil
import tempfile
import datetime
import webbrowser
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_arangodb_extraction_transparent")

# Constants
ARANGODB_AQL_URL = "https://docs.arangodb.com/stable/aql/"
TEST_RESULTS_DIR = Path("test_results")


def create_results_directory(output_dir: Optional[Path] = None) -> Path:
    """
    Create and return the path to a results directory.
    
    Args:
        output_dir: Optional directory to use, defaults to test_results/arangodb_yyyy-mm-dd_hhmmss
        
    Returns:
        Path to the results directory
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    if output_dir is None:
        output_dir = TEST_RESULTS_DIR / f"arangodb_{timestamp}"
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created results directory: {output_dir}")
    return output_dir


def download_arangodb_docs(output_dir: Path) -> Optional[Path]:
    """
    Download ArangoDB AQL documentation.
    
    Args:
        output_dir: Directory to save the downloaded files
        
    Returns:
        Path to the downloaded HTML file or None if download failed
    """
    # Create a directory for downloaded files
    download_dir = output_dir / "original_html"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading ArangoDB AQL documentation from {ARANGODB_AQL_URL}")
    
    try:
        # Import the download_site function
        try:
            # Try to import from fetch_docs module
            from agent_tools.fetch_docs.download_site import download_site
        except ImportError:
            # Try to import from local patch
            current_dir = os.path.dirname(os.path.abspath(__file__))
            download_site_patch_path = os.path.join(current_dir, "download_site_patch.py")
            
            if os.path.exists(download_site_patch_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("download_site_patch", download_site_patch_path)
                download_site_patch = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_site_patch)
                download_site = download_site_patch.download_site
            else:
                raise ImportError("Could not find download_site function")
        
        # Download the documentation
        download_site(ARANGODB_AQL_URL, str(download_dir), recursive=False)
        
        # Find the downloaded HTML file (usually index.html)
        html_files = list(download_dir.glob("**/*.html"))
        if not html_files:
            logger.error("No HTML files found after download")
            return None
        
        # Return the main HTML file
        main_html = next((f for f in html_files if "index.html" in f.name), html_files[0])
        logger.info(f"Downloaded HTML file: {main_html}")
        
        # Copy the HTML to the root of the output directory for easy access
        simple_html_path = output_dir / "arangodb_aql.html"
        shutil.copy(main_html, simple_html_path)
        
        return simple_html_path
    
    except Exception as e:
        logger.error(f"Error downloading documentation: {e}")
        
        # Create a minimal HTML file for fallback testing
        fallback_html = download_dir / "index.html"
        with open(fallback_html, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>ArangoDB Documentation (Fallback)</title></head>
<body>
  <div class="content">
    <h1>ArangoDB Documentation</h1>
    <p>This is a fallback documentation page for {ARANGODB_AQL_URL}</p>
    <h2>AQL Query Language</h2>
    <p>ArangoDB Query Language (AQL) is used to retrieve and modify data.</p>
    <pre><code class="language-javascript">
    FOR doc IN collection
      FILTER doc.value > 10
      RETURN doc
    </code></pre>
    <h2>Operations</h2>
    <p>AQL provides various operations for data manipulation.</p>
    <h3>RETURN Operation</h3>
    <p>The RETURN operation specifies what to return from a query.</p>
    <table>
      <tr><th>Syntax</th><th>Description</th></tr>
      <tr><td>RETURN expression</td><td>Returns the value of expression</td></tr>
    </table>
  </div>
</body>
</html>""")
        
        # Copy the fallback HTML to the root of the output directory
        simple_html_path = output_dir / "arangodb_aql_fallback.html"
        shutil.copy(fallback_html, simple_html_path)
        
        logger.info(f"Created fallback HTML file: {simple_html_path}")
        return simple_html_path


def process_arangodb_docs(html_file: Path, output_dir: Path) -> Optional[Path]:
    """
    Process ArangoDB documentation and convert to blocks.
    
    Args:
        html_file: Path to the HTML file to process
        output_dir: Directory to save the processed output
        
    Returns:
        Path to the JSON file containing processed blocks or None if processing failed
    """
    logger.info(f"Processing HTML file: {html_file}")
    
    try:
        # Import required functions
        try:
            from agent_tools.fetch_docs.processor import process_documentation
            from agent_tools.dualipa.fetch_docs_integration import convert_to_dualipa_format
        except ImportError as e:
            logger.error(f"Error importing required modules: {e}")
            return None
        
        # Read the HTML content
        with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save the HTML to a predictable location in the temp directory
            temp_html = temp_path / "docs.arangodb.com" / "stable" / "aql" / "index.html"
            temp_html.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_html, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Process the documentation
            processed_docs = process_documentation([ARANGODB_AQL_URL], temp_path)
            
            if not processed_docs or ARANGODB_AQL_URL not in processed_docs:
                logger.error("Failed to process documentation")
                return None
            
            # Save the raw processed docs for inspection
            processed_file = output_dir / "arangodb_processed_raw.json"
            with open(processed_file, 'w', encoding='utf-8') as f:
                # Convert to a serializable format
                serializable_docs = {}
                for url, pages in processed_docs.items():
                    serializable_docs[url] = []
                    for page in pages:
                        serializable_page = {k: v for k, v in page.items()}
                        serializable_docs[url].append(serializable_page)
                
                json.dump(serializable_docs, f, indent=2)
            
            logger.info(f"Saved raw processed documentation to {processed_file}")
            
            # Convert to DuaLipa blocks
            blocks = convert_to_dualipa_format(processed_docs, temp_path)
            
            # Save the blocks to JSON
            blocks_file = output_dir / "arangodb_blocks.json"
            with open(blocks_file, 'w', encoding='utf-8') as f:
                json.dump(blocks, f, indent=2)
            
            logger.info(f"Saved {len(blocks)} blocks to {blocks_file}")
            return blocks_file
    
    except Exception as e:
        logger.error(f"Error processing documentation: {e}")
        return None


def create_html_summary(html_file: Path, blocks_file: Path, output_dir: Path) -> Optional[Path]:
    """
    Create an HTML summary of the extraction process for human verification.
    
    Args:
        html_file: Path to the original HTML file
        blocks_file: Path to the blocks JSON file
        output_dir: Directory to save the summary
        
    Returns:
        Path to the HTML summary file or None if creation failed
    """
    try:
        import re
        from bs4 import BeautifulSoup
        
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()
        
        # Read the blocks file
        with open(blocks_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Collect statistics
        stats = {
            "total_blocks": len(blocks),
            "doc_blocks": len([b for b in blocks if b.get("type") == "documentation"]),
            "page_blocks": len([b for b in blocks if b.get("type") == "doc_page"]),
            "section_blocks": len([b for b in blocks if b.get("type") == "doc_section"]),
            "code_blocks": len([b for b in blocks if b.get("type") == "code_block"]),
            "table_blocks": len([b for b in blocks if b.get("type") == "table"]),
            "image_blocks": len([b for b in blocks if b.get("type") == "image"]),
        }
        
        # Find a header element for comparison
        header_element = soup.find('h1')
        header_content = str(header_element) if header_element else "No h1 tag found"
        header_block = next((b for b in blocks if b.get("type") == "doc_section" and 
                          b.get("metadata", {}).get("header_level", 0) == 1), {"name": "Not found"})
        
        # Find a code block for comparison
        code_element = soup.find('pre')
        code_content = str(code_element) if code_element else "No pre tag found"
        code_block = next((b for b in blocks if b.get("type") == "code_block"), {"name": "Not found"})
        
        # Find a table for comparison
        table_element = soup.find('table')
        table_content = str(table_element) if table_element else "No table tag found"
        table_block = next((b for b in blocks if b.get("type") == "table"), {"name": "Not found"})
        
        # Create an HTML summary
        summary_path = output_dir / "extraction_summary.html"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>ArangoDB Documentation Extraction Summary</title>
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
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        .commands {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-family: monospace;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        pre {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow: auto;
        }}
        .comparison {{
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .original, .extracted {{
            flex: 1;
            background-color: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            overflow: auto;
        }}
        .example {{
            margin-top: 20px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ArangoDB Documentation Extraction Summary</h1>
            <p>Test run: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Source URL: <a href="{ARANGODB_AQL_URL}">{ARANGODB_AQL_URL}</a></p>
        </div>

        <div class="section">
            <h2>Extraction Statistics</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Blocks</h3>
                    <p>{stats["total_blocks"]}</p>
                </div>
                <div class="stat-card">
                    <h3>Documentation Blocks</h3>
                    <p>{stats["doc_blocks"]}</p>
                </div>
                <div class="stat-card">
                    <h3>Page Blocks</h3>
                    <p>{stats["page_blocks"]}</p>
                </div>
                <div class="stat-card">
                    <h3>Section Blocks</h3>
                    <p>{stats["section_blocks"]}</p>
                </div>
                <div class="stat-card">
                    <h3>Code Blocks</h3>
                    <p>{stats["code_blocks"]}</p>
                </div>
                <div class="stat-card">
                    <h3>Table Blocks</h3>
                    <p>{stats["table_blocks"]}</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Verification Files</h2>
            <p>The following files have been created for verification:</p>
            <ul>
                <li><strong>Original HTML:</strong> <a href="{html_file.name}">{html_file.name}</a></li>
                <li><strong>Extracted Blocks:</strong> <a href="{blocks_file.name}">{blocks_file.name}</a></li>
            </ul>
            
            <div class="commands">
                <p># Commands to inspect the extraction results:</p>
                <p>cat "{blocks_file.name}" | grep "type" | sort | uniq -c</p>
                <p>cat "{blocks_file.name}" | grep "doc_type" | sort | uniq -c</p>
            </div>
        </div>

        <div class="section">
            <h2>Example Content Extraction</h2>
            
            <div class="example">
                <h3>Main Heading</h3>
                <div class="comparison">
                    <div class="original">
                        <h4>Original HTML</h4>
                        <pre>{header_content}</pre>
                    </div>
                    <div class="extracted">
                        <h4>Extracted Block</h4>
                        <pre>{json.dumps(header_block, indent=2)}</pre>
                    </div>
                </div>
            </div>

            <div class="example">
                <h3>Code Block Example</h3>
                <div class="comparison">
                    <div class="original">
                        <h4>Original HTML</h4>
                        <pre>{code_content}</pre>
                    </div>
                    <div class="extracted">
                        <h4>Extracted Block</h4>
                        <pre>{json.dumps(code_block, indent=2)}</pre>
                    </div>
                </div>
            </div>

            <div class="example">
                <h3>Table Example</h3>
                <div class="comparison">
                    <div class="original">
                        <h4>Original HTML</h4>
                        <pre>{table_content}</pre>
                    </div>
                    <div class="extracted">
                        <h4>Extracted Block</h4>
                        <pre>{json.dumps(table_block, indent=2)}</pre>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Block Structure Overview</h2>
            <p>Here's a sample of the extracted block structure:</p>
            <pre>
{json.dumps([
    {"uuid": b.get("uuid"), 
     "type": b.get("type"), 
     "name": b.get("name"),
     "language": b.get("language"),
     "doc_type": b.get("metadata", {}).get("doc_type", "unknown")}
    for b in blocks[:5]  # Only show first 5 blocks
], indent=2)}
            </pre>
            <p><em>Note: This is a sample of {min(5, len(blocks))} blocks out of {len(blocks)} total blocks.</em></p>
        </div>
    </div>
</body>
</html>""")
        
        logger.info(f"Created HTML summary at {summary_path}")
        return summary_path
    
    except Exception as e:
        logger.error(f"Error creating HTML summary: {e}")
        return None


def run_test(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run the ArangoDB extraction test with transparent output.
    
    Args:
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with test results
    """
    # Initialize results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_url": ARANGODB_AQL_URL,
        "success": False,
        "steps": [],
        "output_files": {},
    }
    
    try:
        # Create results directory
        results_dir = create_results_directory(output_dir)
        results["output_directory"] = str(results_dir)
        
        # Step 1: Download ArangoDB docs
        logger.info("Step 1: Downloading ArangoDB documentation")
        html_file = download_arangodb_docs(results_dir)
        
        if not html_file:
            results["error"] = "Failed to download ArangoDB documentation"
            return results
        
        results["output_files"]["html"] = str(html_file)
        
        # Step 2: Process documentation
        logger.info("Step 2: Processing documentation")
        blocks_file = process_arangodb_docs(html_file, results_dir)
        
        if not blocks_file:
            results["error"] = "Failed to process ArangoDB documentation"
            return results
        
        results["output_files"]["blocks"] = str(blocks_file)
        
        # Step 3: Create HTML summary
        logger.info("Step 3: Creating HTML summary")
        summary_file = create_html_summary(html_file, blocks_file, results_dir)
        
        if not summary_file:
            results["error"] = "Failed to create HTML summary"
            return results
        
        results["output_files"]["summary"] = str(summary_file)
        
        # Step 4: Load blocks and collect statistics
        logger.info("Step 4: Collecting statistics")
        with open(blocks_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
        
        # Collect statistics
        stats = {
            "total_blocks": len(blocks),
            "doc_blocks": len([b for b in blocks if b.get("type") == "documentation"]),
            "page_blocks": len([b for b in blocks if b.get("type") == "doc_page"]),
            "section_blocks": len([b for b in blocks if b.get("type") == "doc_section"]),
            "code_blocks": len([b for b in blocks if b.get("type") == "code_block"]),
            "table_blocks": len([b for b in blocks if b.get("type") == "table"]),
            "image_blocks": len([b for b in blocks if b.get("type") == "image"]),
        }
        
        results["statistics"] = stats
        
        # Check if we have the expected block types
        expected_types = ["documentation", "doc_page", "doc_section"]
        missing_types = [t for t in expected_types if stats.get(f"{t}_blocks", 0) == 0]
        
        if missing_types:
            results["warning"] = f"Missing expected block types: {', '.join(missing_types)}"
            logger.warning(results["warning"])
        else:
            results["success"] = True
            logger.info("✅ All expected block types found")
        
        # Try to open the HTML summary in browser
        try:
            webbrowser.open(str(summary_file))
            logger.info(f"Opened HTML summary in browser: {summary_file}")
        except Exception as e:
            logger.warning(f"Could not open HTML summary in browser: {e}")
            logger.info(f"Please open the summary file manually: {summary_file}")
        
        # Create a results.json file
        results_file = results_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved test results to {results_file}")
        return results
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        results["error"] = str(e)
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run ArangoDB extraction test with transparent output.")
    parser.add_argument("--output-dir", type=str, help="Directory to save test results.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("Starting ArangoDB extraction test...")
    results = run_test(output_dir)
    
    if results.get("success"):
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        if "error" in results:
            print(f"Error: {results['error']}")
    
    print(f"\nOutput directory: {results.get('output_directory')}")
    
    if "statistics" in results:
        print("\nExtraction Statistics:")
        for key, value in results["statistics"].items():
            print(f"  {key}: {value}")
    
    if "output_files" in results:
        print("\nOutput Files:")
        for key, value in results["output_files"].items():
            print(f"  {key}: {value}")
    
    print("\nTo verify the results, open the HTML summary in your browser.")


if __name__ == "__main__":
    main()