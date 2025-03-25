#!/usr/bin/env python3
"""
test_playwright_fetch.py

Test script to verify Playwright-based website fetching functionality.
This tests the download_site_with_playwright function from the fetch_docs module.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_playwright_fetch")

def test_playwright_fetch(url: str, output_dir: str = "test_playwright_output"):
    """
    Test the Playwright-based website fetching functionality.
    
    Args:
        url: URL to fetch
        output_dir: Directory to save the results
    """
    try:
        # Try to import the download_site function
        try:
            from agent_tools.fetch_docs.download_site import download_site_with_playwright
            logger.info("Successfully imported download_site_with_playwright from fetch_docs module")
        except ImportError:
            # Try importing from direct path
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
            from agent_tools.fetch_docs.download_site import download_site_with_playwright
            logger.info("Successfully imported download_site_with_playwright after path adjustment")
            
        # Make sure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Start timing
        start_time = time.time()
        
        # Test the Playwright function
        logger.info(f"Fetching {url} with Playwright...")
        stats = download_site_with_playwright(url, output_path, wait_time=5)
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        # Check results
        if stats["success"]:
            logger.info(f"✅ Successfully downloaded {url} with Playwright")
            logger.info(f"⏱️ Download took {duration:.2f} seconds")
            logger.info(f"📊 Downloaded {stats['pages_downloaded']} pages")
            logger.info(f"🗂️ Output directory: {output_path}")
            
            # Output downloaded files
            logger.info("📄 Downloaded files:")
            for file_path in output_path.glob("**/*.html"):
                logger.info(f"  - {file_path}")
                
            # Save stats to output directory
            stats_file = output_path / "playwright_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"📊 Saved stats to {stats_file}")
            
            # Return success
            return True
        else:
            logger.error(f"❌ Failed to download {url}: {stats.get('error_message', 'Unknown error')}")
            # Save error stats
            error_file = output_path / "playwright_error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"📊 Saved error info to {error_file}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Playwright-based website fetching")
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("--output-dir", default="test_playwright_output", help="Directory to save results")
    args = parser.parse_args()
    
    print(f"Testing Playwright fetch functionality for {args.url}...")
    success = test_playwright_fetch(args.url, args.output_dir)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
    
if __name__ == "__main__":
    main()