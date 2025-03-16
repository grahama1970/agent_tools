#!/usr/bin/env python3
"""
download_pages.py

Official Documentation:
- wget: https://www.gnu.org/software/wget/manual/wget.html
- subprocess (Python standard library): https://docs.python.org/3/library/subprocess.html

This module uses wget (invoked via subprocess) to download a documentation page.
It can operate in two modes:
1. Recursive mode (default): Downloads the entire site, preserving the directory structure.
2. Single-page mode: Downloads only the specified page (non-recursive).

The downloaded files will be stored in the specified output directory.
"""

import subprocess
from pathlib import Path
import sys

def download_site(root_url: str, output_dir: str, recursive: bool = True) -> None:
    """
    Download the site starting from root_url using wget.
    
    Args:
        root_url (str): The URL of the site/page to download.
        output_dir (str): The directory where the site/page will be stored.
        recursive (bool): If True (default), downloads recursively; if False, downloads only the single page.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wget_command = [
        "wget",
        "--no-clobber",         # Do not overwrite existing files
        "--page-requisites",    # Download all assets needed to display HTML
        "--html-extension",     # Save files with a .html extension
        "--convert-links",      # Convert links for local viewing
        "--restrict-file-names=windows",
        "--domains", root_url.split("/")[2],
    ]
    
    if recursive:
        wget_command.append("--recursive")
        wget_command.append("--no-parent")
    
    wget_command.extend(["--directory-prefix", str(output_path), root_url])
    
    print(f"Running command: {' '.join(wget_command)}")
    try:
        subprocess.run(wget_command, check=True)
        print("Site downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading site: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python download_pages.py <root_url> <output_directory> [--single-page]")
        sys.exit(1)
    
    url = sys.argv[1]
    out_dir = sys.argv[2]
    # Check for an optional flag indicating single-page download
    recursive = True
    if len(sys.argv) == 4 and sys.argv[3] == "--single-page":
        recursive = False
    download_site(url, out_dir, recursive)
