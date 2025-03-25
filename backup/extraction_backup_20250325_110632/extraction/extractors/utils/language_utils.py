"""
Language detection and utilities for DuaLipa.

This module provides utilities for detecting programming languages,
managing language-specific features, and handling file extensions.

Key Features:
1. Language detection from file extensions
2. Language metadata and capabilities
3. File type validation
4. Language-specific patterns

Dependencies:
- loguru: For logging
- pathlib: For path handling

Related Files:
- code_extractor.py: Uses language detection
- stats_utils.py: Uses language info for stats
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Language definitions
LANGUAGE_INFO = {
    "python": {
        "extensions": [".py", ".pyi", ".pyx"],
        "comment": "#",
        "block_comment": ['"""', "'''"],
        "supports_ast": True,
        "supports_type_hints": True
    },
    "javascript": {
        "extensions": [".js", ".jsx", ".mjs"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True,
        "supports_jsx": True
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True,
        "supports_types": True
    },
    "markdown": {
        "extensions": [".md", ".markdown"],
        "supports_frontmatter": True,
        "supports_code_blocks": True
    },
    "java": {
        "extensions": [".java"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    },
    "c": {
        "extensions": [".c", ".h"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    },
    "cpp": {
        "extensions": [".cpp", ".hpp", ".cc", ".hh"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    },
    "go": {
        "extensions": [".go"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    },
    "ruby": {
        "extensions": [".rb"],
        "comment": "#",
        "block_comment": ["=begin", "=end"],
        "supports_ast": True
    },
    "rust": {
        "extensions": [".rs"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    },
    "php": {
        "extensions": [".php"],
        "comment": "//",
        "block_comment": ["/*", "*/"],
        "supports_ast": True
    }
}

# Language aliases for normalization
LANGUAGE_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "py": "python",
    "rb": "ruby",
    "md": "markdown"
}

def normalize_language(language: str) -> str:
    """
    Normalize language name to standard form.
    
    Args:
        language: Language name or alias
        
    Returns:
        Normalized language name
    """
    try:
        # Convert to lowercase
        lang = language.lower().strip()
        
        # Check aliases first
        if lang in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[lang]
            
        # Check if it's already a known language
        if lang in LANGUAGE_INFO:
            return lang
            
        return "unknown"
        
    except Exception as e:
        logger.error(f"Error normalizing language: {e}")
        return "unknown"

def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Language identifier or "unknown"
    """
    try:
        # Get file extension
        ext = Path(file_path).suffix.lower()
        
        # Check each language
        for language, info in LANGUAGE_INFO.items():
            if ext in info["extensions"]:
                return language
                
        return "unknown"
        
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "unknown"

def get_language_info(language: str) -> Optional[Dict[str, Any]]:
    """
    Get language metadata and capabilities.
    
    Args:
        language: Language identifier
        
    Returns:
        Language info dictionary or None
    """
    try:
        return LANGUAGE_INFO.get(language)
    except Exception as e:
        logger.error(f"Error getting language info: {e}")
        return None

def is_supported_language(file_path: str) -> bool:
    """
    Check if file language is supported.
    
    Args:
        file_path: Path to source file
        
    Returns:
        True if supported, False otherwise
    """
    try:
        language = detect_language(file_path)
        return language != "unknown"
    except Exception as e:
        logger.error(f"Error checking language support: {e}")
        return False

def get_comment_pattern(language: str) -> Optional[str]:
    """
    Get language-specific comment pattern.
    
    Args:
        language: Language identifier
        
    Returns:
        Comment pattern or None
    """
    try:
        info = get_language_info(language)
        if info and "comment" in info:
            return info["comment"]
        return None
    except Exception as e:
        logger.error(f"Error getting comment pattern: {e}")
        return None

def get_block_comment_patterns(language: str) -> Optional[Dict[str, str]]:
    """
    Get language-specific block comment patterns.
    
    Args:
        language: Language identifier
        
    Returns:
        Dictionary with start/end patterns or None
    """
    try:
        info = get_language_info(language)
        if info and "block_comment" in info:
            patterns = info["block_comment"]
            if len(patterns) == 2:
                return {
                    "start": patterns[0],
                    "end": patterns[1]
                }
        return None
    except Exception as e:
        logger.error(f"Error getting block comment patterns: {e}")
        return None

def usage_example() -> None:
    """Example usage of language utilities."""
    # Example files
    files = [
        "example.py",
        "component.tsx",
        "README.md",
        "main.cpp",
        "unknown.xyz"
    ]
    
    print("Language Detection:")
    for file in files:
        language = detect_language(file)
        print(f"{file}: {language}")
        
        if language != "unknown":
            info = get_language_info(language)
            print(f"  Extensions: {info['extensions']}")
            if "comment" in info:
                print(f"  Comment: {info['comment']}")
            if "block_comment" in info:
                print(f"  Block Comment: {info['block_comment']}")
            print(f"  Supported: {is_supported_language(file)}")
            print() 