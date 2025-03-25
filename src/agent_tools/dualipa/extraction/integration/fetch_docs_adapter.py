"""
Integration adapter for the fetch_docs module.

This module provides interfaces for downloading and processing HTML documentation
using the fetch_docs module.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import from fetch_docs module
try:
    from agent_tools.fetch_docs.download_site import download_site, download_site_with_playwright
    from agent_tools.fetch_docs.extract_sections import extract_sections_from_html
    FETCH_DOCS_AVAILABLE = True
except ImportError:
    FETCH_DOCS_AVAILABLE = False
    print("Warning: fetch_docs module not available.")

class DocumentationDownloader:
    """
    Interface for downloading documentation using fetch_docs.
    
    This class provides a simplified interface for downloading documentation
    using either wget or Playwright.
    """
    
    def __init__(self, output_dir: str, use_playwright: bool = False):
        """
        Initialize the DocumentationDownloader.
        
        Args:
            output_dir: Directory to save downloaded files
            use_playwright: Whether to use Playwright for JavaScript rendering
        """
        self.output_dir = output_dir
        self.use_playwright = use_playwright
        
        if not FETCH_DOCS_AVAILABLE:
            raise ImportError("fetch_docs module is required for DocumentationDownloader")
    
    def download(self, url: str, recursive: bool = True) -> bool:
        """
        Download documentation from the specified URL.
        
        Args:
            url: URL to download
            recursive: Whether to download recursively
            
        Returns:
            bool: True if successful, False otherwise
        """
        return download_site(url, self.output_dir, recursive, self.use_playwright)
    
    def download_with_playwright(self, url: str, recursive: bool = True, 
                                max_depth: int = 2) -> Dict[str, Any]:
        """
        Download documentation using Playwright for JavaScript rendering.
        
        Args:
            url: URL to download
            recursive: Whether to download recursively
            max_depth: Maximum recursion depth
            
        Returns:
            Dict containing download statistics
        """
        return download_site_with_playwright(
            url, self.output_dir, recursive=recursive, max_depth=max_depth
        )

class HTMLProcessor:
    """
    Interface for processing HTML content from downloaded documentation.
    
    This class provides methods for extracting structured content from HTML files.
    """
    
    def __init__(self, content_dir: str):
        """
        Initialize the HTMLProcessor.
        
        Args:
            content_dir: Directory containing downloaded HTML files
        """
        self.content_dir = content_dir
        
        if not FETCH_DOCS_AVAILABLE:
            raise ImportError("fetch_docs module is required for HTMLProcessor")
    
    def extract_sections(self, html_file: str) -> List[Dict[str, Any]]:
        """
        Extract sections from an HTML file.
        
        Args:
            html_file: Path to the HTML file
            
        Returns:
            List of extracted sections with hierarchical structure
        """
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return extract_sections_from_html(html_content)
    
    def process_directory(self, doc_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all HTML files in the content directory.
        
        Args:
            doc_type: Optional documentation type (e.g., "readthedocs", "arangodb")
            
        Returns:
            Dictionary mapping URLs to lists of processed documents
        """
        processed_docs = {}
        
        # Get all HTML files
        for root, _, files in os.walk(self.content_dir):
            for file in files:
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.content_dir)
                    
                    # Extract URL
                    url_parts = relative_path.split(os.sep)
                    if len(url_parts) >= 1:
                        domain = url_parts[0]
                        url = f"https://{domain}"
                        
                        # Extract sections
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                html_content = f.read()
                            
                            sections = extract_sections_from_html(html_content)
                            
                            if url not in processed_docs:
                                processed_docs[url] = []
                            
                            processed_docs[url].append({
                                "file": file_path,
                                "relative_path": relative_path,
                                "sections": sections,
                                "doc_type": doc_type or self._detect_doc_type(domain)
                            })
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        
        return processed_docs
    
    def _detect_doc_type(self, domain: str) -> str:
        """
        Detect documentation type from domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Detected documentation type
        """
        if "readthedocs.io" in domain or "readthedocs.org" in domain:
            return "readthedocs"
        elif "arangodb.com" in domain:
            return "arangodb"
        else:
            return "unknown"
