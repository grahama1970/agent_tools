#!/usr/bin/env python3
"""
docs_integration.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- uuid: https://docs.python.org/3/library/uuid.html
- typing: https://docs.python.org/3/library/typing.html

This module provides integration between the fetch_docs tool and the DuaLipa extraction module.
It allows documentation from HTML pages to be integrated with extracted code blocks to provide
a comprehensive extraction output.

Input/Output Specifications:

extract_all_blocks_with_docs(repo_path: Path) -> List[Dict[str, Any]]:
    Input:
        - repo_path: Path to repository to extract from
    Output:
        - List of DuaLipa blocks including both code and documentation

integrate_docs_with_extraction(repo_path: Path, output_blocks: List[Dict]) -> List[Dict]:
    Input:
        - repo_path: Path to repository to extract documentation from
        - output_blocks: Existing extraction blocks to enhance
    Output:
        - List of blocks enhanced with documentation

Example usage:
    from agent_tools.dualipa.extraction.docs_integration import extract_all_blocks_with_docs
    from pathlib import Path
    
    # Extract all blocks including documentation
    blocks = extract_all_blocks_with_docs(Path("/path/to/repo"))
    
    # Write to JSON file
    with open("extraction_output.json", "w") as f:
        json.dump(blocks, f, indent=2)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("dualipa.extraction.docs_integration")

def extract_all_blocks_with_docs(repo_path: Path) -> List[Dict[str, Any]]:
    """
    Enhanced extraction function that includes documentation.
    
    Args:
        repo_path: Directory to extract from
        
    Returns:
        List of extracted blocks including documentation
    """
    # Try to import the regular extraction function
    try:
        from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
    except ImportError:
        try:
            # Try alternative import path
            from agent_tools.dualipa.extraction.extractors.code import extract_code_blocks
            from agent_tools.dualipa.extraction.extractors.markdown import extract_markdown_blocks
            
            def extract_all_blocks(path):
                """Fallback extraction function combining code and markdown."""
                # Create output directory for blocks
                output_dir = path / ".dualipa_blocks"
                output_dir.mkdir(exist_ok=True)
                
                # Extract code blocks
                code_blocks = []
                for file_path in path.glob("**/*"):
                    if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.rs']:
                        try:
                            blocks = extract_code_blocks(str(file_path), output_dir)
                            code_blocks.extend(blocks)
                        except Exception as e:
                            logger.error(f"Error extracting from {file_path}: {e}")
                
                # Extract markdown blocks
                md_blocks = []
                for file_path in path.glob("**/*.md"):
                    if file_path.is_file():
                        try:
                            blocks = extract_markdown_blocks(str(file_path), output_dir)
                            md_blocks.extend(blocks)
                        except Exception as e:
                            logger.error(f"Error extracting from {file_path}: {e}")
                
                return code_blocks + md_blocks
        except ImportError:
            logger.error("Could not import extraction functions")
            return []
    
    # Regular extraction
    logger.info(f"Extracting code and markdown blocks from {repo_path}")
    code_blocks = extract_all_blocks(repo_path)
    logger.info(f"Extracted {len(code_blocks)} code and markdown blocks")
    
    # Enhance with documentation
    logger.info("Enhancing with documentation blocks")
    enhanced_blocks = integrate_docs_with_extraction(repo_path, code_blocks)
    logger.info(f"Final extraction includes {len(enhanced_blocks)} blocks")
    
    return enhanced_blocks

def integrate_docs_with_extraction(repo_path: Path, output_blocks: List[Dict]) -> List[Dict]:
    """
    Main integration function to detect docs, download, and merge with extraction output.
    
    Args:
        repo_path: Path to the repository
        output_blocks: Existing extraction blocks from DuaLipa
        
    Returns:
        Enhanced list of blocks including documentation
    """
    # Import fetch_docs functionality
    try:
        from agent_tools.fetch_docs.processor import extract_documentation_from_repo
        
        # Extract documentation blocks
        logger.info(f"Extracting documentation from {repo_path}")
        doc_blocks = extract_documentation_from_repo(repo_path)
        
        if not doc_blocks:
            logger.info("No documentation blocks extracted")
            return output_blocks
        
        logger.info(f"Extracted {len(doc_blocks)} documentation blocks")
        
        # Append documentation blocks to output
        output_blocks.extend(doc_blocks)
        
        logger.info(f"Added {len(doc_blocks)} documentation blocks to extraction output")
        
    except ImportError as e:
        logger.error(f"Could not import fetch_docs processor: {e}")
        logger.warning("Documentation integration skipped")
    except Exception as e:
        logger.error(f"Error during documentation integration: {e}")
    
    return output_blocks

if __name__ == "__main__":
    import json
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract code and documentation from a repository")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--docs-only", action="store_true", help="Extract only documentation (no code)")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    output_file = Path(args.output_file)
    
    if args.docs_only:
        # Import from fetch_docs directly for docs-only extraction
        try:
            from agent_tools.fetch_docs.processor import extract_documentation_from_repo
            output_blocks = extract_documentation_from_repo(repo_path)
        except ImportError:
            logger.error("Could not import processor from fetch_docs")
            output_blocks = []
    else:
        # Full extraction with code and docs
        output_blocks = extract_all_blocks_with_docs(repo_path)
    
    # Write output to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_blocks, f, indent=2)
    
    print(f"Extraction completed. Extracted {len(output_blocks)} blocks.")
    print(f"Output written to: {output_file}")