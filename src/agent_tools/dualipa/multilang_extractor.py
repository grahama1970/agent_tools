"""
Multilanguage code extractor.

This module provides functionality to extract code blocks from files
in different programming languages.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

# Mapping of file extensions to language identifiers
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}

def get_available_languages() -> Set[str]:
    """
    Return the set of supported programming languages.
    
    Returns:
        Set[str]: Set of language identifiers that can be processed
    """
    return set(LANGUAGE_EXTENSIONS.values())

def get_language_for_file(filepath: Union[str, Path]) -> Optional[str]:
    """
    Determine the language of a file based on its extension.
    
    Args:
        filepath: Path to the file, either as string or Path object
        
    Returns:
        str: Language identifier, or None if the language is not supported
    """
    # Convert to Path if string
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    # Get extension (including the dot)
    ext = filepath.suffix.lower()
    
    # Look up in our mapping
    return LANGUAGE_EXTENSIONS.get(ext)

def extract_code_blocks(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract code blocks from a file.
    
    Args:
        filepath: Path to the file, either as string or Path object
        
    Returns:
        List[Dict[str, Any]]: List of extracted code blocks, where each block is a
        dictionary with keys like 'language', 'content', etc.
    """
    # Convert to Path if string
    if isinstance(filepath, str):
        filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        print(f"Warning: File does not exist: {filepath}")
        return []
    
    # Get language
    language = get_language_for_file(filepath)
    if language is None:
        print(f"Warning: Unsupported file type: {filepath}")
        return []
    
    try:
        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a simple block with the entire file content
        blocks = [{
            "language": language,
            "content": content,
            "path": str(filepath),
            "start_line": 1,
            "end_line": content.count('\n') + 1,
            "type": "file",
            "name": filepath.name
        }]
        
        # If this is Python, we can be a bit more sophisticated
        if language == "python":
            # Rudimentary extraction of functions and classes
            import re
            
            # Find all function definitions
            func_matches = re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            for match in func_matches:
                func_name = match.group(1)
                start_pos = match.start()
                
                # Find start line
                start_line = content[:start_pos].count('\n') + 1
                
                # Extract a reasonable chunk of code (simplified approach)
                line_start = content.rfind('\n', 0, start_pos) + 1
                next_def = content.find('\ndef ', start_pos + 1)
                next_class = content.find('\nclass ', start_pos + 1)
                
                if next_def == -1 and next_class == -1:
                    end_pos = len(content)
                elif next_def == -1:
                    end_pos = next_class
                elif next_class == -1:
                    end_pos = next_def
                else:
                    end_pos = min(next_def, next_class)
                
                func_content = content[line_start:end_pos].rstrip()
                end_line = start_line + func_content.count('\n')
                
                blocks.append({
                    "language": language,
                    "content": func_content,
                    "path": str(filepath),
                    "start_line": start_line,
                    "end_line": end_line,
                    "type": "function",
                    "name": func_name
                })
            
            # Find all class definitions
            class_matches = re.finditer(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            for match in class_matches:
                class_name = match.group(1)
                start_pos = match.start()
                
                # Find start line
                start_line = content[:start_pos].count('\n') + 1
                
                # Extract a reasonable chunk of code (simplified approach)
                line_start = content.rfind('\n', 0, start_pos) + 1
                next_def = content.find('\ndef ', start_pos + 1)
                next_class = content.find('\nclass ', start_pos + 1)
                
                if next_def == -1 and next_class == -1:
                    end_pos = len(content)
                elif next_def == -1:
                    end_pos = next_class
                elif next_class == -1:
                    end_pos = next_def
                else:
                    end_pos = min(next_def, next_class)
                
                class_content = content[line_start:end_pos].rstrip()
                end_line = start_line + class_content.count('\n')
                
                blocks.append({
                    "language": language,
                    "content": class_content,
                    "path": str(filepath),
                    "start_line": start_line,
                    "end_line": end_line,
                    "type": "class",
                    "name": class_name
                })
        
        return blocks
        
    except Exception as e:
        print(f"Error extracting code blocks from {filepath}: {e}")
        return []

if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        blocks = extract_code_blocks(file_path)
        print(f"Extracted {len(blocks)} blocks from {file_path}")
        
        for i, block in enumerate(blocks):
            print(f"\nBlock {i+1}:")
            print(f"Type: {block.get('type')}")
            print(f"Name: {block.get('name')}")
            print(f"Language: {block.get('language')}")
            print(f"Lines: {block.get('start_line')}-{block.get('end_line')}")
            print(f"Content preview: {block.get('content')[:100]}...")
    else:
        print("Usage: python multilang_extractor.py <file_path>") 