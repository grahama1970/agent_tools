#!/usr/bin/env python3
"""
extract_sections.py

Official Documentation:
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- SpaCy: https://spacy.io/usage
- pathlib (Python standard library): https://docs.python.org/3/library/pathlib.html

This module extracts section chunks from HTML/markdown text by identifying header tags.
Short sections (under 2 sentences) are merged with adjacent sections. It also collects file and section metadata.
"""

import spacy
from bs4 import BeautifulSoup, Tag
from pathlib import Path
from loguru import logger
import json

# Load SpaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error("SpaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    raise e

def tokenize_sentences(text: str) -> list:
    """
    Use SpaCy to split text into sentences.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_sections_from_html(html: str, file_path: Path) -> list:
    """
    Extract sections from HTML based on header tags.
    
    Args:
        html (str): Cleaned HTML content.
        file_path (Path): Path of the file (for file hierarchy metadata).
    
    Returns:
        list: List of dictionaries representing each section.
    """
    soup = BeautifulSoup(html, "lxml")
    sections = []
    current_section = None

    # For file hierarchy, use relative path info
    file_meta = {
        "file_path": str(file_path),
        "file_name": file_path.stem,
    }

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "img", "table"]):
        if isinstance(element, Tag) and element.name.startswith("h"):
            header_text = element.get_text(strip=True)
            header_level = int(element.name[1])  # Assuming valid h1-h6
            
            # Start a new section for each header
            current_section = {
                "header": header_text,
                "level": header_level,
                "content": "",
                "token_count": 0,
                "file_meta": file_meta,
                "header_hierarchy": [],  # This will be built iteratively
            }
            # For simplicity, we set the hierarchy as a list with the current header (in a real implementation,
            # you would track parent/child relationships across headers)
            current_section["header_hierarchy"].append(f"{'#' * header_level} {header_text}")
            sections.append(current_section)
        elif current_section is not None:
            # Append content to the current section
            if element.name == "p":
                current_section["content"] += "\n" + element.get_text(strip=True)
            elif element.name == "img":
                src = element.get("src", "")
                alt = element.get("alt", "")
                current_section["content"] += f"\n![{alt}]({src})"
            elif element.name == "table":
                # For now, simply include the table's HTML; further processing can occur later
                current_section["content"] += "\n" + str(element)

    # Merge sections that are too short (less than 2 sentences)
    merged_sections = []
    for sec in sections:
        sentences = tokenize_sentences(sec["content"])
        if len(sentences) < 2 and merged_sections:
            # Merge with the previous section
            merged_sections[-1]["content"] += "\n" + sec["content"]
            # Update token count later
        else:
            merged_sections.append(sec)

    # Compute token counts for each section using SpaCy
    for sec in merged_sections:
        doc = nlp(sec["content"])
        sec["token_count"] = len(doc)
    
    return merged_sections

if __name__ == "__main__":
    # Simple usage demo: read a sample HTML file from disk (if available)
    sample_file = Path("sample.html")
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as f:
            html_content = f.read()
        sections = extract_sections_from_html(html_content, sample_file)
        print(json.dumps(sections, indent=2))
    else:
        print("No sample.html file found for demo.")
