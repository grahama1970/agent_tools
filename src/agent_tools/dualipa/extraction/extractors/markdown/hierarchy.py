"""
Markdown hierarchy extraction for DuaLipa.

This module handles extraction of hierarchical structure from markdown content,
focusing on sections, subsections, and their relationships.

Key Features:
1. Section hierarchy extraction
2. Breadcrumb path generation
3. Section metadata handling
4. Content organization

Dependencies:
- markdown-it-py: For markdown parsing
- loguru: For logging

Related Files:
- parser.py: Provides parsing functionality
- extractor.py: Uses hierarchy for content extraction
"""

import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from loguru import logger

from .parser import parse_markdown, extract_content_blocks

def extract_hierarchical_sections(content: str) -> List[Dict[str, Any]]:
    """
    Extract sections and their hierarchy from markdown content.
    
    Args:
        content: Markdown content to process
        
    Returns:
        List of section dictionaries with hierarchy information
    """
    # Parse the markdown
    parsed = parse_markdown(content)
    tokens = parsed["tokens"]
    
    # Find all headings and their positions
    sections = []
    current_section = None
    
    for token in tokens:
        if token.get("type") == "heading_open":
            # Get heading level from h1, h2, etc.
            level = int(token.get("tag", "h1")[1])
            
            # Get heading text from next token
            title = ""
            if "children" in tokens[tokens.index(token) + 1]:
                title = tokens[tokens.index(token) + 1]["content"]
            
            # Create new section
            current_section = {
                "title": title,
                "level": level,
                "content": [],
                "subsections": [],
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "heading_level": level,
                    "content_blocks": []
                }
            }
            sections.append(current_section)
        
        # Add content to current section
        elif current_section is not None:
            current_section["content"].append(token)
    
    # Build hierarchy
    root_sections = []
    section_stack = []
    
    for section in sections:
        while section_stack and section_stack[-1]["level"] >= section["level"]:
            section_stack.pop()
            
        if section_stack:
            section_stack[-1]["subsections"].append(section)
        else:
            root_sections.append(section)
            
        section_stack.append(section)
    
    return root_sections

def build_section_hierarchy(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a complete hierarchical structure from sections.
    
    Args:
        sections: List of sections to organize
        
    Returns:
        Dictionary with complete hierarchy
    """
    hierarchy = {
        "document": {
            "sections": sections,
            "metadata": {
                "total_sections": len(sections),
                "max_depth": max(s["level"] for s in sections) if sections else 0
            }
        }
    }
    
    # Add breadcrumb paths
    def add_breadcrumbs(section: Dict[str, Any], path: List[str] = None) -> None:
        if path is None:
            path = []
            
        current_path = path + [section["title"]]
        section["breadcrumb"] = current_path
        
        for subsection in section.get("subsections", []):
            add_breadcrumbs(subsection, current_path)
    
    for section in sections:
        add_breadcrumbs(section)
    
    return hierarchy

def get_section_content(section: Dict[str, Any]) -> str:
    """Extract content from a section."""
    content = []
    for token in section.get("content", []):
        if isinstance(token, dict):
            content.append(token.get("content", ""))
    return "\n".join(content) 