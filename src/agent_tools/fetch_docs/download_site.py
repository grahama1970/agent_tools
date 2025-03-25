#!/usr/bin/env python3
"""
download_site.py

Official Documentation:
- wget: https://www.gnu.org/software/wget/manual/wget.html
- subprocess (Python standard library): https://docs.python.org/3/library/subprocess.html
- Playwright: https://playwright.dev/python/docs/intro

This module downloads documentation pages using either:
1. wget (for static sites)
2. Playwright (for JavaScript-rendered sites)

It can operate in two modes:
1. Recursive mode (default): Downloads the entire site, preserving the directory structure.
2. Single-page mode: Downloads only the specified page (non-recursive).

The downloaded files will be stored in the specified output directory.
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_site")

# Check if Playwright is available
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logger.warning("Playwright not available. Will use wget only.")
    PLAYWRIGHT_AVAILABLE = False


def check_playwright_installed() -> bool:
    """
    Check if Playwright is installed and browsers are available.
    
    Returns:
        bool: True if Playwright is available, False otherwise
    """
    if not PLAYWRIGHT_AVAILABLE:
        return False
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception as e:
        logger.error(f"Error checking Playwright: {e}")
        return False


def download_site_with_wget(root_url: str, output_dir: str, recursive: bool = True) -> bool:
    """
    Download the site starting from root_url using wget.
    
    Args:
        root_url (str): The URL of the site/page to download.
        output_dir (str): The directory where the site/page will be stored.
        recursive (bool): If True (default), downloads recursively; if False, downloads only the single page.
        
    Returns:
        bool: True if successful, False otherwise
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
    
    logger.info(f"Running wget command: {' '.join(wget_command)}")
    try:
        subprocess.run(wget_command, check=True)
        logger.info("Site downloaded successfully with wget.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading site with wget: {e}")
        return False


def download_site_with_playwright(
    url: str, 
    output_dir: Union[str, Path], 
    wait_time: int = 5,
    recursive: bool = False,
    max_depth: int = 2,
    timeout: int = 30000
) -> Dict[str, Any]:
    """
    Download a website using Playwright, which supports JavaScript rendering.
    
    Args:
        url: URL to download
        output_dir: Directory to save the downloaded files
        wait_time: Time in seconds to wait for JavaScript rendering
        recursive: Whether to download linked pages
        max_depth: Maximum depth for recursive downloading
        timeout: Maximum time in ms to wait for page load
        
    Returns:
        Dict containing download statistics and metadata
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright is not available. Please install it first.")
        return {"success": False, "error": "Playwright not available"}
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract domain from URL
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    base_path = parsed_url.path
    
    # Statistics
    stats = {
        "success": False,
        "pages_downloaded": 0,
        "errors": 0,
        "pages": {}
    }
    
    try:
        # Set up Playwright
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            # Download the page
            page = context.new_page()
            logger.info(f"Navigating to {url}")
            
            try:
                page.goto(url, timeout=timeout, wait_until="networkidle")
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout when loading {url}, continuing with what we have")
            
            # Wait for JavaScript to render
            logger.info(f"Waiting {wait_time} seconds for JavaScript rendering")
            time.sleep(wait_time)
            
            # Get the rendered HTML
            html_content = page.content()
            
            # Create directory structure similar to wget
            site_dir = output_path / domain
            if base_path and base_path != "/":
                for part in base_path.strip("/").split("/"):
                    if part:
                        site_dir = site_dir / part
            
            site_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the HTML
            html_file = site_dir / "index.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            stats["pages_downloaded"] += 1
            stats["pages"][url] = {
                "path": str(html_file),
                "size": len(html_content)
            }
            
            logger.info(f"Saved {url} to {html_file}")
            
            # If recursive, get links and download them
            visited_urls = {url}
            if recursive and max_depth > 0:
                links = page.eval_on_selector_all('a[href]', '''
                    (elements) => elements.map(el => {
                        const href = el.getAttribute('href');
                        return {
                            href: href,
                            text: el.textContent.trim()
                        };
                    })
                ''')
                
                # Filter links
                same_domain_links = []
                for link in links:
                    href = link.get('href', '')
                    if not href or href.startswith('#') or href.startswith('javascript:'):
                        continue
                    
                    # Handle relative links
                    if href.startswith('/'):
                        link_url = f"{parsed_url.scheme}://{domain}{href}"
                    elif not href.startswith('http'):
                        # Relative to current path
                        current_path = '/'.join(url.split('/')[:-1]) + '/'
                        link_url = f"{current_path}{href}"
                    else:
                        link_url = href
                        
                        # Check if same domain
                        link_domain = urlparse(link_url).netloc
                        if link_domain != domain:
                            continue
                    
                    if link_url not in visited_urls:
                        same_domain_links.append(link_url)
                        visited_urls.add(link_url)
                
                # Download linked pages recursively
                for link_url in same_domain_links[:10]:  # Limit to 10 for testing
                    try:
                        sub_stats = download_site_with_playwright(
                            link_url, 
                            output_dir,
                            wait_time=wait_time, 
                            recursive=True, 
                            max_depth=max_depth - 1,
                            timeout=timeout
                        )
                        
                        # Update stats
                        stats["pages_downloaded"] += sub_stats.get("pages_downloaded", 0)
                        stats["errors"] += sub_stats.get("errors", 0)
                        stats["pages"].update(sub_stats.get("pages", {}))
                        
                    except Exception as e:
                        logger.error(f"Error downloading linked page {link_url}: {e}")
                        stats["errors"] += 1
            
            # Also download CSS and JavaScript resources
            resources = {}
            
            # Get CSS files
            css_links = page.eval_on_selector_all('link[rel="stylesheet"]', '''
                (elements) => elements.map(el => el.getAttribute('href'))
            ''')
            
            for css_link in css_links:
                if not css_link:
                    continue
                    
                # Handle relative paths
                if css_link.startswith('/'):
                    css_url = f"{parsed_url.scheme}://{domain}{css_link}"
                elif not css_link.startswith('http'):
                    current_path = '/'.join(url.split('/')[:-1]) + '/'
                    css_url = f"{current_path}{css_link}"
                else:
                    css_url = css_link
                
                resources[css_link] = css_url
            
            # Get JavaScript files
            js_links = page.eval_on_selector_all('script[src]', '''
                (elements) => elements.map(el => el.getAttribute('src'))
            ''')
            
            for js_link in js_links:
                if not js_link:
                    continue
                    
                # Handle relative paths
                if js_link.startswith('/'):
                    js_url = f"{parsed_url.scheme}://{domain}{js_link}"
                elif not js_link.startswith('http'):
                    current_path = '/'.join(url.split('/')[:-1]) + '/'
                    js_url = f"{current_path}{js_link}"
                else:
                    js_url = js_link
                
                resources[js_link] = js_url
            
            # Download resources (limited to avoid excessive downloads)
            for resource_url in list(resources.values())[:10]:
                try:
                    resource_page = context.new_page()
                    resource_page.goto(resource_url, timeout=timeout)
                    
                    # Get filename from URL
                    resource_filename = resource_url.split('/')[-1]
                    if not resource_filename or '?' in resource_filename:
                        # Generate a name if there isn't a clear one
                        extension = '.css' if resource_url.endswith('.css') else '.js'
                        resource_filename = f"resource_{len(resources)}{extension}"
                    
                    # Remove query parameters
                    if '?' in resource_filename:
                        resource_filename = resource_filename.split('?')[0]
                    
                    # Save resource
                    resource_path = site_dir / "resources"
                    resource_path.mkdir(exist_ok=True)
                    resource_file = resource_path / resource_filename
                    
                    with open(resource_file, 'w', encoding='utf-8') as f:
                        f.write(resource_page.content())
                    
                    logger.info(f"Saved resource {resource_url} to {resource_file}")
                    
                except Exception as e:
                    logger.error(f"Error downloading resource {resource_url}: {e}")
                    stats["errors"] += 1
            
            # Close browser
            browser.close()
            
            # Update stats
            stats["success"] = True
            
            # Save stats
            stats_file = output_path / "download_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            return stats
    
    except Exception as e:
        logger.error(f"Error downloading site with Playwright: {e}")
        stats["errors"] += 1
        stats["error_message"] = str(e)
        return stats


def download_site(root_url: str, output_dir: str, recursive: bool = True, use_playwright: bool = False) -> bool:
    """
    Download a website using either wget or Playwright.
    
    Args:
        root_url: The URL of the site/page to download
        output_dir: The directory where the site/page will be stored
        recursive: If True (default), downloads recursively
        use_playwright: If True, use Playwright instead of wget
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First try with wget (unless Playwright is explicitly requested)
    if not use_playwright:
        success = download_site_with_wget(root_url, output_dir, recursive)
        
        # If successful or Playwright is not available, return the result
        if success or not PLAYWRIGHT_AVAILABLE:
            return success
        
        # If wget failed and Playwright is available, try with Playwright
        logger.info("wget failed. Attempting download with Playwright...")
    
    # Check if Playwright is properly installed
    if not check_playwright_installed():
        logger.error(
            "Playwright is not available or browser installation not found. "
            "Please install with: pip install playwright && playwright install"
        )
        logger.error("Continuing with wget-based download only.")
        return download_site_with_wget(root_url, output_dir, recursive)
    
    # Use Playwright to download the site
    logger.info(f"Downloading {root_url} with Playwright")
    result = download_site_with_playwright(
        root_url,
        output_dir,
        recursive=recursive,
        max_depth=2 if recursive else 0
    )
    
    return result.get("success", False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download documentation websites")
    parser.add_argument("url", help="URL to download")
    parser.add_argument("output_dir", help="Directory to save downloaded files")
    parser.add_argument("--single-page", action="store_true", help="Download single page only (non-recursive)")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for JavaScript-rendered sites")
    
    args = parser.parse_args()
    
    recursive = not args.single_page
    success = download_site(args.url, args.output_dir, recursive, args.playwright)
    
    if not success:
        sys.exit(1)