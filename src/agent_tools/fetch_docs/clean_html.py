#!/usr/bin/env python3
"""
clean_html.py

Official Documentation:
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Bleach: https://bleach.readthedocs.io/
- markdownify: https://github.com/matthewwithanm/python-markdownify

This module provides functions for minimal HTML cleanup. It removes unwanted tags (scripts, styles, etc.)
while preserving the content structure so that later conversion to markdown maintains header information.
"""

from bs4 import BeautifulSoup, Tag
from loguru import logger  # Documentation: https://loguru.readthedocs.io/
import bleach  # For final cleanup, if needed

def remove_unwanted_elements(soup: BeautifulSoup) -> None:
    """
    Remove elements that are clearly extraneous.
    """
    # Remove tags that are not needed
    for tag in soup.find_all(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

def clean_html(html: str) -> str:
    """
    Clean the HTML by removing unwanted elements.
    
    Args:
        html (str): Raw HTML input.
        
    Returns:
        str: Cleaned HTML.
    """
    logger.debug(f"Starting HTML cleaning. Original length: {len(html)}")
    soup = BeautifulSoup(html, "lxml")
    remove_unwanted_elements(soup)
    
    # Optionally, you could use bleach here before converting to markdown
    # For example: cleaned = bleach.clean(str(soup), tags=['p','h1','h2','h3','img','table'], strip=True)
    cleaned = str(soup)
    
    logger.debug(f"Finished HTML cleaning. Cleaned length: {len(cleaned)}")
    return cleaned

def convert_to_markdown(cleaned_html: str) -> str:
    """
    Convert cleaned HTML to markdown using markdownify.
    
    Args:
        cleaned_html (str): Cleaned HTML.
        
    Returns:
        str: Markdown text.
    """
    from markdownify import markdownify as md  # Documentation: https://github.com/matthewwithanm/python-markdownify
    markdown_text = md(cleaned_html, heading_style="ATX")
    return markdown_text

if __name__ == "__main__":
    # Simple usage demo
    sample_html = "<html><head><title>Test</title></head><body><nav>Navigation</nav><h1>Header</h1><p>Paragraph text.</p></body></html>"
    cleaned = clean_html(sample_html)
    md_text = convert_to_markdown(cleaned)
    print("Converted Markdown:\n", md_text)
