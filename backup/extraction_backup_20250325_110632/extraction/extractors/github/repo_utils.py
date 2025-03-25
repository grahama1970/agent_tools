"""
GitHub repository utilities for DuaLipa.

This module handles repository-level operations including cloning,
analyzing repository structure, and coordinating extraction across
multiple files.

Key Features:
1. Repository cloning and validation
2. Repository structure analysis
3. Multi-file extraction coordination
4. Repository statistics tracking

Dependencies:
- git: For repository operations
- loguru: For logging
- pathlib: For path handling

Related Files:
- code_extractor.py: Used for code extraction
- markdown_extractor.py: Used for markdown extraction
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from agent_tools.dualipa.extraction.extractors.code.code_extractor import extract_code_blocks
from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor import extract_markdown_blocks
from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict

# Re-export initialize_stats_dict for backward compatibility
__all__ = ['initialize_stats_dict']

def clone_repository(url: str, target_dir: Path) -> Dict[str, Any]:
    """Clone a GitHub repository to the target directory."""
    stats = initialize_stats_dict(source=url, output_dir=target_dir)
    # TODO: Implement repository cloning
    return stats

def analyze_repository(repo_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze a GitHub repository and extract code blocks."""
    if output_dir is None:
        output_dir = repo_dir / "extracted"
    stats = initialize_stats_dict(source=repo_dir, output_dir=output_dir)
    # TODO: Implement repository analysis
    return stats

def verify_repo_structure(repo_path: Path) -> bool:
    """
    Verify repository has valid structure.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if path exists and is directory
        if not repo_path.exists() or not repo_path.is_dir():
            return False
            
        # Check for .git directory
        if not (repo_path / ".git").exists():
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying repository structure: {e}")
        return False

def extract_repository(
    source: str,
    output_dir: Path = None,
    output_path: str = None,  # For backward compatibility
    exclude_patterns: Optional[List[str]] = None,
    extract_documentation: bool = True,  # For backward compatibility
    extract_code: bool = True,  # For backward compatibility
    extract_blocks: bool = True  # For backward compatibility
) -> Dict[str, Any]:
    """
    Extract content from entire repository.
    
    Args:
        source: Repository source (URL or path)
        output_dir: Output directory for extracted content (Path object)
        output_path: Output path for extracted content (string, deprecated)
        exclude_patterns: Patterns to exclude (optional)
        extract_documentation: Whether to extract markdown (deprecated, always True)
        extract_code: Whether to extract code (deprecated, always True)
        extract_blocks: Whether to extract blocks (deprecated, always True)
        
    Returns:
        Repository statistics
    """
    try:
        # Handle backward compatibility with output_path parameter
        if output_dir is None and output_path is not None:
            output_dir = Path(output_path)
        elif output_dir is None:
            output_dir = Path("output")
            
        # Convert output_dir to Path if it's a string
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        # Initialize stats
        stats = init_stats()
        stats["repository"] = {
            "source": source,
            "extraction_started": datetime.now().isoformat()
        }
        
        # Convert source to Path
        source_path = Path(source)
        
        # Special case: if source is a single file, process it directly
        if source_path.is_file():
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Detect language
                language = detect_language(source_path)
                
                # Extract based on file type
                if language == "markdown" and extract_documentation:
                    blocks = extract_markdown_blocks(source_path, output_dir)
                    update_stats(stats, blocks, language)
                    
                elif language != "unknown" and extract_code:
                    blocks = extract_code_blocks(source_path, output_dir)
                    update_stats(stats, blocks, language)
                    
                # Update repository stats
                stats["repository"]["extraction_completed"] = datetime.now().isoformat()
                stats["repository"]["total_files"] = stats["total_files"]
                stats["repository"]["total_blocks"] = stats["total_blocks"]
                stats["repository"]["languages"] = stats["languages"]
                
                return stats
                
            except Exception as e:
                error_msg = f"Error processing {source_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                return stats
        
        # For directories (repositories), verify structure
        if not verify_repo_structure(source_path):
            error_msg = f"Invalid repository structure at {source}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            return stats
            
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all files
        for file_path in source_path.rglob("*"):
            # Skip excluded patterns
            if exclude_patterns and any(file_path.match(p) for p in exclude_patterns):
                continue
                
            # Skip directories and non-files
            if not file_path.is_file():
                continue
                
            try:
                # Detect language
                language = detect_language(file_path)
                
                # Extract based on file type
                if language == "markdown" and extract_documentation:
                    blocks = extract_markdown_blocks(file_path, output_dir)
                    update_stats(stats, blocks, language)
                    
                elif language != "unknown" and extract_code:
                    blocks = extract_code_blocks(file_path, output_dir)
                    update_stats(stats, blocks, language)
                    
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
                
        # Update repository stats
        stats["repository"]["extraction_completed"] = datetime.now().isoformat()
        stats["repository"]["total_files"] = stats["total_files"]
        stats["repository"]["total_blocks"] = stats["total_blocks"]
        stats["repository"]["languages"] = stats["languages"]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error extracting repository: {e}")
        stats["errors"].append(str(e))
        return stats

def usage_example() -> None:
    """Example usage of repository extraction."""
    # Example repository URL
    repo_url = "https://github.com/example/repo.git"
    
    # Set up directories
    repo_dir = Path("temp_repo")
    output_dir = Path("output")
    
    try:
        # Clone repository
        repo_path = clone_repository(repo_url, repo_dir)
        
        # Extract content
        stats = extract_repository(
            str(repo_path),
            output_dir,
            exclude_patterns=["*.pyc", "__pycache__", "*.git*"]
        )
        
        # Print statistics
        print("\nRepository Statistics:")
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Blocks: {stats['total_blocks']}")
        print("\nLanguage Distribution:")
        for lang, count in stats["languages"].items():
            print(f"  {lang}: {count} files")
            
        print("\nErrors:")
        for error in stats["errors"]:
            print(f"  {error}")
            
    finally:
        # Cleanup
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir) 