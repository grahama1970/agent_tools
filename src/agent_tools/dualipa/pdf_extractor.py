#!/usr/bin/env python3
"""
pdf_extractor.py

This module provides functionality to extract content from PDF documents
and convert it into DuaLipa-compatible blocks for integration with the
extraction pipeline.

Key features:
- PDF document parsing with hierarchical structure detection
- Section and subsection identification
- Extraction of tables, images, and code blocks from PDFs
- Conversion to DuaLipa blocks with proper parent-child relationships
- Integration with the main extraction pipeline

Requirements:
- PyPDF2 or pymupdf for PDF parsing
- Optional: OCR capabilities for scanned documents

Usage:
    from agent_tools.dualipa.pdf_extractor import extract_pdf_blocks
    
    # Extract blocks from a PDF
    blocks = extract_pdf_blocks("/path/to/document.pdf")
    
    # Or integrate with the main extraction pipeline
    from agent_tools.dualipa.extraction import extract_all_blocks
    
    blocks = extract_all_blocks(repo_path, include_pdfs=True)
"""

import re
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except ImportError:
    HAVE_PYMUPDF = False
    try:
        from PyPDF2 import PdfReader
        HAVE_PYPDF2 = True
    except ImportError:
        HAVE_PYPDF2 = False

# Configure logging
logger = logging.getLogger("dualipa.pdf_extractor")

# Regular expressions for structure detection
HEADING_PATTERNS = [
    # Common heading patterns in PDFs
    r'^(\d+\.(?:\d+\.)*)\s+(.+)$',  # Numbered headings (1.1, 1.2.1, etc.)
    r'^(Chapter|Section)\s+(\d+)[\.\:]?\s+(.+)$',  # Chapter/Section headings
    r'^([A-Z][A-Za-z\s]+)$',  # ALL CAPS or Title Case headings
]

# Section level detection based on font size/style
FONT_SIZE_THRESHOLDS = {
    'h1': 18,  # > 18pt
    'h2': 16,  # > 16pt
    'h3': 14,  # > 14pt
    'h4': 12,  # > 12pt
    'h5': 11,  # > 11pt
    'h6': 10,  # > 10pt
}


def detect_pdf_files(repo_path: Path) -> List[Path]:
    """
    Scan a repository for PDF files.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of paths to PDF files
    """
    pdf_files = []
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Skip common directories to ignore
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', 'venv', '__pycache__']]
        
        # Find PDF files
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = Path(root) / file
                pdf_files.append(pdf_path)
    
    logger.info(f"Found {len(pdf_files)} PDF files in repository")
    return pdf_files


def _extract_with_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract content from a PDF using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of page content dictionaries with sections, text, and metadata
    """
    if not HAVE_PYMUPDF:
        logger.error("PyMuPDF not available. Install with 'pip install pymupdf'")
        return []
    
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        
        for page_num, page in enumerate(doc):
            # Extract text with formatting information
            blocks = page.get_text("dict")["blocks"]
            
            # Extract text blocks with formatting
            text_blocks = []
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    lines = []
                    for line in block["lines"]:
                        line_text = ""
                        line_spans = []
                        
                        for span in line["spans"]:
                            span_text = span["text"]
                            line_text += span_text
                            line_spans.append({
                                "text": span_text,
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],  # Bold, italic, etc.
                                "color": span["color"]
                            })
                        
                        lines.append({
                            "text": line_text,
                            "spans": line_spans
                        })
                    
                    text_blocks.append({
                        "text": "".join(line["text"] for line in lines),
                        "lines": lines,
                        "bbox": block["bbox"]
                    })
                    
                elif block["type"] == 1:  # Image block
                    # Extract image metadata
                    text_blocks.append({
                        "type": "image",
                        "bbox": block["bbox"]
                    })
            
            # Detect sections based on font size and style
            sections = detect_sections_from_blocks(text_blocks)
            
            # Extract tables (tables are often not detected as separate blocks)
            tables = detect_tables(page)
            
            # Add page to results
            pages.append({
                "page_num": page_num + 1,
                "text_blocks": text_blocks,
                "sections": sections,
                "tables": tables,
                "page_size": [page.rect.width, page.rect.height]
            })
        
        return pages
    
    except Exception as e:
        logger.error(f"Error extracting PDF content from {pdf_path}: {e}")
        return []


def _extract_with_pypdf2(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract content from a PDF using PyPDF2 (fallback option).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of page content dictionaries with basic text content
    """
    if not HAVE_PYPDF2:
        logger.error("PyPDF2 not available. Install with 'pip install PyPDF2'")
        return []
    
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # Try to detect sections using regex patterns since we don't have formatting info
            sections = detect_sections_from_text(text)
            
            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "sections": sections
            })
        
        return pages
    
    except Exception as e:
        logger.error(f"Error extracting PDF content from {pdf_path}: {e}")
        return []


def detect_sections_from_blocks(text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect sections in PDF text blocks based on font size and style.
    
    Args:
        text_blocks: List of text blocks with formatting information
        
    Returns:
        List of sections with hierarchical structure
    """
    sections = []
    current_section = None
    
    for block in text_blocks:
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            # Skip empty lines
            if not line["text"].strip():
                continue
                
            # Check if this is a potential heading based on formatting
            is_heading = False
            heading_level = 0
            
            # Check font sizes for heading detection
            max_size = 0
            is_bold = False
            
            for span in line["spans"]:
                max_size = max(max_size, span["size"])
                # Check if the span is bold (flag 4 or 16 for bold)
                if span["flags"] & 4 or span["flags"] & 16:
                    is_bold = True
            
            # Determine heading level based on font size
            if max_size > FONT_SIZE_THRESHOLDS["h1"]:
                is_heading = True
                heading_level = 1
            elif max_size > FONT_SIZE_THRESHOLDS["h2"]:
                is_heading = True
                heading_level = 2
            elif max_size > FONT_SIZE_THRESHOLDS["h3"] and is_bold:
                is_heading = True
                heading_level = 3
            elif max_size > FONT_SIZE_THRESHOLDS["h4"] and is_bold:
                is_heading = True
                heading_level = 4
            
            # Also check regex patterns for heading detection
            for pattern in HEADING_PATTERNS:
                match = re.match(pattern, line["text"].strip())
                if match:
                    is_heading = True
                    # Use the numbering to determine level (1.2.3 => level 3)
                    if match.group(1).count('.') > 0:
                        heading_level = min(match.group(1).count('.') + 1, 6)
                    break
            
            if is_heading:
                # Create a new section
                current_section = {
                    "title": line["text"].strip(),
                    "level": heading_level,
                    "content": "",
                    "subsections": []
                }
                sections.append(current_section)
            elif current_section:
                # Append to current section's content
                current_section["content"] += line["text"] + "\n"
    
    # Build hierarchy (nest subsections under their parent sections)
    hierarchical_sections = []
    section_stack = []
    
    for section in sections:
        level = section["level"]
        
        # Pop stack until we find the parent level
        while section_stack and section_stack[-1]["level"] >= level:
            section_stack.pop()
        
        if section_stack:
            # Add as subsection to the parent
            section_stack[-1]["subsections"].append(section)
        else:
            # Top-level section
            hierarchical_sections.append(section)
        
        # Push current section onto stack
        section_stack.append(section)
    
    return hierarchical_sections


def detect_sections_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Detect sections in plain text using regex patterns (for PyPDF2 fallback).
    
    Args:
        text: Plain text content
        
    Returns:
        List of sections with hierarchical structure
    """
    sections = []
    lines = text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a potential heading using regex patterns
        is_heading = False
        heading_level = 0
        heading_title = line
        
        for pattern in HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                is_heading = True
                # Try to determine level from the match
                if len(match.groups()) > 1:
                    if match.group(1).count('.') > 0:
                        heading_level = min(match.group(1).count('.') + 1, 6)
                    elif match.group(1) in ['Chapter', 'CHAPTER']:
                        heading_level = 1
                    elif match.group(1) in ['Section', 'SECTION']:
                        heading_level = 2
                    else:
                        heading_level = 1  # Default to level 1
                    
                    # Use the actual title part
                    heading_title = match.group(len(match.groups()))
                break
        
        # Also check for heading patterns like all caps, followed by lowercase
        if not is_heading and line.isupper() and len(line) > 3:
            is_heading = True
            heading_level = 1
            heading_title = line
        
        if is_heading:
            # Create a new section
            current_section = {
                "title": heading_title.strip(),
                "level": heading_level or 1,  # Default to level 1 if not determined
                "content": "",
                "subsections": []
            }
            sections.append(current_section)
        elif current_section:
            # Append to current section's content
            current_section["content"] += line + "\n"
    
    # Build hierarchy (nest subsections under their parent sections)
    hierarchical_sections = []
    section_stack = []
    
    for section in sections:
        level = section["level"]
        
        # Pop stack until we find the parent level
        while section_stack and section_stack[-1]["level"] >= level:
            section_stack.pop()
        
        if section_stack:
            # Add as subsection to the parent
            section_stack[-1]["subsections"].append(section)
        else:
            # Top-level section
            hierarchical_sections.append(section)
        
        # Push current section onto stack
        section_stack.append(section)
    
    return hierarchical_sections


def detect_tables(page: 'fitz.Page') -> List[Dict[str, Any]]:
    """
    Detect tables in a PDF page (PyMuPDF).
    
    Args:
        page: The PDF page object
        
    Returns:
        List of detected tables with cells and position information
    """
    if not HAVE_PYMUPDF:
        return []
    
    tables = []
    
    try:
        # Use PyMuPDF's built-in table detection
        tab = page.find_tables()
        if tab.tables:
            for idx, table in enumerate(tab.tables):
                cells = []
                rows = table.rows
                cols = table.cols
                
                # Extract cell data
                for r in range(rows):
                    row_cells = []
                    for c in range(cols):
                        rect = table.cell(r, c)
                        text = page.get_text("text", clip=rect)
                        row_cells.append(text.strip())
                    cells.append(row_cells)
                
                tables.append({
                    "id": idx,
                    "bbox": list(table.bbox),
                    "rows": rows,
                    "cols": cols,
                    "cells": cells
                })
    except Exception as e:
        logger.warning(f"Error detecting tables: {e}")
    
    return tables


def extract_pdf_content(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract content from a PDF file with formatting and structure.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF content, sections, and metadata
    """
    # Try PyMuPDF first (better for formatting)
    if HAVE_PYMUPDF:
        pages = _extract_with_pymupdf(pdf_path)
    elif HAVE_PYPDF2:
        # Fall back to PyPDF2 if PyMuPDF is not available
        pages = _extract_with_pypdf2(pdf_path)
    else:
        logger.error("No PDF library available. Install PyMuPDF or PyPDF2")
        return {"error": "No PDF library available", "pages": []}
    
    # Extract metadata about the PDF
    metadata = extract_metadata(pdf_path)
    
    return {
        "file_path": str(pdf_path),
        "file_name": pdf_path.name,
        "metadata": metadata,
        "pages": pages
    }


def extract_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF metadata
    """
    metadata = {
        "title": "",
        "author": "",
        "creator": "",
        "producer": "",
        "creation_date": "",
        "modification_date": "",
        "page_count": 0
    }
    
    try:
        if HAVE_PYMUPDF:
            doc = fitz.open(str(pdf_path))
            metadata.update(doc.metadata)
            metadata["page_count"] = len(doc)
        elif HAVE_PYPDF2:
            reader = PdfReader(str(pdf_path))
            info = reader.metadata
            if info:
                metadata.update({
                    "title": info.title or "",
                    "author": info.author or "",
                    "creator": info.creator or "",
                    "producer": info.producer or "",
                    "creation_date": str(info.creation_date) if hasattr(info, "creation_date") else "",
                    "modification_date": str(info.modification_date) if hasattr(info, "modification_date") else ""
                })
            metadata["page_count"] = len(reader.pages)
    except Exception as e:
        logger.error(f"Error extracting PDF metadata from {pdf_path}: {e}")
    
    return metadata


def convert_to_dualipa_format(pdf_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert PDF content to DuaLipa-compatible blocks with parent-child relationships.
    
    Args:
        pdf_content: Extracted PDF content and structure
        
    Returns:
        List of DuaLipa-compatible blocks
    """
    blocks = []
    
    # Create the main document block
    doc_uuid = str(uuid.uuid4())
    file_path = pdf_content["file_path"]
    file_name = pdf_content["file_name"]
    
    # Use title from metadata or fallback to filename
    title = pdf_content.get("metadata", {}).get("title", "")
    if not title:
        title = file_name
    
    doc_block = {
        "uuid": doc_uuid,
        "id": f"pdf_doc_{file_name.replace('.pdf', '')}",
        "name": title,
        "type": "pdf_document",
        "language": "pdf",
        "content": f"PDF Document: {title}",
        "file_path": file_path,
        "child_uuids": [],
        "metadata": {
            "file_path": file_path,
            "title": title,
            "author": pdf_content.get("metadata", {}).get("author", ""),
            "page_count": pdf_content.get("metadata", {}).get("page_count", 0),
            "creation_date": pdf_content.get("metadata", {}).get("creation_date", "")
        }
    }
    blocks.append(doc_block)
    
    # Process pages
    for page in pdf_content.get("pages", []):
        page_uuid = str(uuid.uuid4())
        page_num = page.get("page_num", 1)
        
        # Create page block
        page_block = {
            "uuid": page_uuid,
            "id": f"pdf_page_{file_name.replace('.pdf', '')}_{page_num}",
            "name": f"Page {page_num}",
            "type": "pdf_page",
            "language": "pdf",
            "content": page.get("text", "") if "text" in page else "\n".join(block.get("text", "") for block in page.get("text_blocks", [])),
            "file_path": file_path,
            "parent_uuid": doc_uuid,
            "child_uuids": [],
            "metadata": {
                "page_number": page_num,
                "page_size": page.get("page_size", [])
            }
        }
        blocks.append(page_block)
        doc_block["child_uuids"].append(page_uuid)
        
        # Process sections hierarchically
        section_blocks = process_sections(page.get("sections", []), file_path, page_uuid, file_name, page_num)
        blocks.extend(section_blocks)
        
        # Add section UUIDs to page's child_uuids
        for section_block in section_blocks:
            if section_block.get("parent_uuid") == page_uuid:
                page_block["child_uuids"].append(section_block["uuid"])
        
        # Process tables
        for i, table in enumerate(page.get("tables", [])):
            table_uuid = str(uuid.uuid4())
            
            # Convert table cells to markdown format for content
            table_content = []
            cells = table.get("cells", [])
            
            if cells:
                # Create header row (using first row)
                header = "| " + " | ".join(cells[0]) + " |"
                separator = "| " + " | ".join(["---"] * len(cells[0])) + " |"
                
                table_content.append(header)
                table_content.append(separator)
                
                # Add data rows (skip first row if it's a header)
                for row in cells[1:]:
                    table_content.append("| " + " | ".join(row) + " |")
            
            table_block = {
                "uuid": table_uuid,
                "id": f"pdf_table_{file_name.replace('.pdf', '')}_{page_num}_{i+1}",
                "name": f"Table {i+1} (Page {page_num})",
                "type": "table",
                "language": "pdf",
                "content": "\n".join(table_content),
                "file_path": file_path,
                "parent_uuid": page_uuid,
                "child_uuids": [],
                "metadata": {
                    "page_number": page_num,
                    "table_index": i,
                    "rows": table.get("rows", 0),
                    "cols": table.get("cols", 0),
                    "bbox": table.get("bbox", [])
                }
            }
            blocks.append(table_block)
            page_block["child_uuids"].append(table_uuid)
    
    return blocks


def process_sections(sections: List[Dict[str, Any]], file_path: str, parent_uuid: str, 
                   file_name: str, page_num: int) -> List[Dict[str, Any]]:
    """
    Process PDF sections recursively to maintain hierarchical structure.
    
    Args:
        sections: List of section dictionaries
        file_path: Path to the PDF file
        parent_uuid: UUID of the parent element
        file_name: Name of the PDF file
        page_num: Page number
        
    Returns:
        List of DuaLipa-compatible section blocks
    """
    blocks = []
    
    for i, section in enumerate(sections):
        section_uuid = str(uuid.uuid4())
        section_title = section.get("title", f"Section {i+1}")
        section_level = section.get("level", 1)
        
        # Create section block
        section_block = {
            "uuid": section_uuid,
            "id": f"pdf_section_{file_name.replace('.pdf', '')}_{page_num}_{i+1}",
            "name": section_title,
            "type": "pdf_section",
            "language": "pdf",
            "content": section.get("content", ""),
            "file_path": file_path,
            "parent_uuid": parent_uuid,
            "child_uuids": [],
            "metadata": {
                "page_number": page_num,
                "section_index": i,
                "header_level": section_level
            }
        }
        blocks.append(section_block)
        
        # Process subsections recursively
        subsection_blocks = process_sections(
            section.get("subsections", []),
            file_path,
            section_uuid,
            file_name,
            page_num
        )
        blocks.extend(subsection_blocks)
        
        # Add subsection UUIDs to parent section's child_uuids
        for subsection_block in subsection_blocks:
            if subsection_block.get("parent_uuid") == section_uuid:
                section_block["child_uuids"].append(subsection_block["uuid"])
    
    return blocks


def extract_pdf_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract content blocks from a PDF file for integration with DuaLipa.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of DuaLipa-compatible blocks
    """
    # Extract PDF content
    pdf_content = extract_pdf_content(pdf_path)
    
    # Convert to DuaLipa blocks
    blocks = convert_to_dualipa_format(pdf_content)
    
    logger.info(f"Extracted {len(blocks)} blocks from PDF: {pdf_path}")
    return blocks


def integrate_pdfs_with_extraction(repo_path: Path, output_blocks: List[Dict]) -> List[Dict]:
    """
    Main integration function to detect PDFs, extract content, and merge with extraction output.
    
    Args:
        repo_path: Path to the repository
        output_blocks: Existing extraction blocks from DuaLipa
        
    Returns:
        Enhanced list of blocks including PDF content
    """
    # Detect PDF files in the repository
    pdf_files = detect_pdf_files(repo_path)
    
    if not pdf_files:
        logger.info("No PDF files found in repository")
        return output_blocks
    
    logger.info(f"Found {len(pdf_files)} PDF files in repository")
    
    # Process each PDF
    for pdf_file in pdf_files:
        # Extract PDF blocks
        pdf_blocks = extract_pdf_blocks(pdf_file)
        
        # Append PDF blocks to output
        output_blocks.extend(pdf_blocks)
        
        logger.info(f"Added {len(pdf_blocks)} PDF blocks from {pdf_file.name} to extraction output")
    
    return output_blocks


def extract_all_blocks_with_pdfs(repo_path: Path) -> List[Dict[str, Any]]:
    """
    Enhanced extraction function that includes PDF documents.
    
    Args:
        repo_path: Directory to extract from
        
    Returns:
        List of extracted blocks including PDF content
    """
    # Try to import the regular extraction function
    try:
        from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
    except ImportError:
        logger.error("Could not import extract_all_blocks from DuaLipa extraction module")
        return []
    
    # Regular extraction
    blocks = extract_all_blocks(repo_path)
    
    # Enhance with PDF content
    enhanced_blocks = integrate_pdfs_with_extraction(repo_path, blocks)
    
    return enhanced_blocks


if __name__ == "__main__":
    import sys
    import json
    
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) != 3:
        print("Usage: python pdf_extractor.py <pdf_file> <output_json_file>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"Error: PDF file {pdf_path} does not exist")
        sys.exit(1)
    
    # Extract blocks
    blocks = extract_pdf_blocks(pdf_path)
    
    # Write output to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, indent=2)
    
    print(f"Extraction completed. Extracted {len(blocks)} blocks.")
    print(f"Output written to: {output_file}")