"""
Example demonstrating integration with fetch_docs module.

This example shows how to download documentation using fetch_docs and
convert it to the extraction format.
"""

import os
from pathlib import Path

# Import integration components
from agent_tools.dualipa.extraction.integration import DocumentationDownloader, HTMLProcessor
from agent_tools.dualipa.extraction.docs_integration import convert_to_dualipa_format

def download_and_extract_docs(url: str, output_dir: str, use_playwright: bool = False):
    """
    Download and extract documentation from a URL.
    
    Args:
        url: URL to download
        output_dir: Directory to save output
        use_playwright: Whether to use Playwright for JavaScript rendering
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Download documentation
    print(f"Downloading documentation from {url}...")
    downloader = DocumentationDownloader(output_dir, use_playwright=use_playwright)
    success = downloader.download(url, recursive=True)
    
    if not success:
        print("Download failed.")
        return None
    
    print("Download completed successfully.")
    
    # Process HTML content
    print("Processing HTML content...")
    processor = HTMLProcessor(output_dir)
    processed_docs = processor.process_directory()
    
    # Convert to extraction format
    print("Converting to extraction format...")
    blocks = convert_to_dualipa_format(processed_docs, output_dir)
    
    # Save extraction output
    extraction_output_file = os.path.join(output_dir, "extraction_output.json")
    import json
    with open(extraction_output_file, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)
    
    print(f"Extraction completed. Output saved to {extraction_output_file}")
    print(f"Extracted {len(blocks)} blocks.")
    
    return blocks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and extract documentation")
    parser.add_argument("url", help="URL to download")
    parser.add_argument("--output-dir", default="docs_output", help="Output directory")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for JavaScript rendering")
    
    args = parser.parse_args()
    
    download_and_extract_docs(args.url, args.output_dir, args.playwright)
