"""Block metadata creation and management.

This module provides core functionality for creating and managing code block metadata
across all extractors. It serves as the foundation for consistent block creation
and statistics tracking.

Key Features:
1. Code block metadata creation with language-specific detection
2. Statistics dictionary initialization and management
3. Automatic detection of prerequisites and focus areas
4. Standardized block structure for all languages

Dependencies:
- uuid: For unique block identification
- pathlib: For cross-platform path handling
- datetime: For timestamp management

Documentation Links:
- Code Block Format: https://tree-sitter.github.io/tree-sitter/
- AST Documentation: https://docs.python.org/3/library/ast.html
- UUID Documentation: https://docs.python.org/3/library/uuid.html

Related Files:
- python_extractor.py: Uses this for Python block creation
- js_ts_extractor.py: Uses this for JavaScript/TypeScript blocks
- markdown_extractor.py: Uses this for markdown sections

Example Block Structure:
{
    "uuid": "unique-id",
    "id": "file_blockname",
    "type": "code",
    "language": "python",
    "content": "actual code content",
    "qa_generation": {
        "prerequisites": ["Python", "Type hints"],
        "focus_areas": ["Functions", "OOP"]
    }
}
"""

import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

def create_code_block(
    name: str,
    content: str,
    file_path: Path,
    block_type: str = "code",
    language: str = None,
    start_line: int = None,
    end_line: int = None,
    imports: List[str] = None,
    referenced_types: List[str] = None,
    test_file: str = None
) -> Dict[str, Any]:
    """Create a code block with basic reliable metadata."""
    # Initialize with empty lists
    imports = imports or []
    referenced_types = referenced_types or []
    focus_areas = []
    prerequisites = [language.title()] if language else []
    
    # Extract what AST gives us reliably
    if language == "python":
        # Add "Type hints" to prerequisites if we see type annotations
        if ":" in content and "->" in content:
            prerequisites.append("Type hints")
            
        # Add "Flask" to prerequisites if we detect Flask usage
        if 'flask' in content.lower() or 'Flask(' in content:
            prerequisites.append("Flask")
            focus_areas.append("Web development")
            
        # Add "Decorators" if we see any
        if "@" in content:
            prerequisites.append("Decorators")
            
    elif language == "typescript":
        # Add "Type hints" to prerequisites for TypeScript code
        prerequisites.append("Type hints")
        
        # Add "Class design" focus area for TypeScript classes
        if "class" in content and "{" in content:
            focus_areas.append("Class design")
            focus_areas.append("Type system")
            # Also add OOP to prerequisites for TypeScript classes
            prerequisites.append("OOP")
    
    # Add basic focus areas based on block type
    if block_type == "function":
        focus_areas.append("Function implementation")
    elif block_type in ["class", "method"]:
        focus_areas.append("Object-oriented programming")
    
    # Remove duplicates while preserving order
    focus_areas = list(dict.fromkeys(focus_areas))
    prerequisites = list(dict.fromkeys(prerequisites))
    
    return {
        "uuid": str(uuid.uuid4()),
        "id": f"{file_path.stem}_{name.lower()}",
        "type": "code",
        "language": language,
        "title": name,
        "name": name,  # Explicitly include the name field
        "content": content,
        "file_path": str(file_path),
        "breadcrumb": [str(file_path), name],
        "parent_uuid": None,
        "child_uuids": [],
        "dependencies": {
            "imports": imports,
            "referenced_types": referenced_types
        },
        "test_coverage": {
            "test_file": test_file,
            "coverage_percentage": 0
        },
        "version_history": {
            "last_modified": datetime.now().isoformat()
        },
        "qa_generation": {
            "difficulty_levels": ["intermediate"] if prerequisites else ["beginner"],
            "knowledge_prerequisites": prerequisites,
            "focus_areas": focus_areas,
            "qa_examples": []
        },
        "start_line": start_line,
        "end_line": end_line
    }

def initialize_stats_dict(source: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Initialize a statistics dictionary for tracking extraction metrics.
    
    Args:
        source: Source path (file or directory)
        output_dir: Output directory path
        
    Returns:
        Dictionary with initialized statistics fields
    """
    current_time = datetime.now().isoformat()
    
    stats = {
        # Source and output information
        "source": str(source) if source else None,
        "output_dir": str(output_dir) if output_dir else None,
        
        # Timing information
        "start_time": current_time,
        "end_time": None,
        "duration_seconds": 0,
        
        # File and block counts
        "total_files": 0,
        "documentation_files": 0,
        "code_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "skipped_files": 0,
        "error_files": 0,
        
        # Categorization
        "languages": {},  # language -> count
        "file_types": {},  # extension -> count
        
        # Error tracking
        "errors": [],
        
        # Block storage
        "file_blocks": {}  # file_path -> [blocks]
    }
    
    logger.debug(f"Initialized stats dictionary for source: {source}")
    return stats

def verify_block_metadata(block: Dict[str, Any]) -> bool:
    """
    Verify that a block has all required metadata fields.
    
    Args:
        block: Block dictionary to verify
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = {
        "type",  # block type (function, class, method, etc)
        "block_type",  # same as type for consistency
        "content",  # actual code content
        "language",  # programming language
        "source_file",  # original source file
        "output_file",  # where block is saved
        "extracted_at"  # timestamp
    }
    
    return all(field in block for field in required_fields)

def standardize_block_type(block_type: str) -> str:
    """
    Standardize block type names for consistency.
    
    Args:
        block_type: Original block type
        
    Returns:
        Standardized block type
    """
    type_mapping = {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "script": "script",
        "section": "section"
    }
    
    return type_mapping.get(block_type, block_type)

def create_block_metadata(
    block_type: str,
    content: str,
    language: str,
    source_file: Path,
    output_file: Path,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    parent_block: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a metadata dictionary for a code block.
    
    Args:
        block_type: Type of block
        content: Block content
        language: Programming language
        source_file: Source file path
        output_file: Output file path
        line_start: Starting line number
        line_end: Ending line number
        parent_block: Parent block ID
        
    Returns:
        Block metadata dictionary
    """
    block_type = standardize_block_type(block_type)
    
    metadata = {
        "type": block_type,
        "block_type": block_type,
        "content": content,
        "language": language,
        "source_file": str(source_file),
        "output_file": str(output_file),
        "extracted_at": datetime.now().isoformat(),
        "line_range": {
            "start": line_start,
            "end": line_end
        } if line_start and line_end else None,
        "parent_block": parent_block
    }
    
    if not verify_block_metadata(metadata):
        logger.warning(f"Created block metadata missing required fields: {metadata.keys()}")
        
    return metadata

def usage_example() -> None:
    """Example usage of block metadata utilities."""
    # Initialize stats
    stats = initialize_stats_dict(
        source=Path("example.py"),
        output_dir=Path("output")
    )
    
    # Create block metadata
    block = create_block_metadata(
        block_type="function",
        content="def example():\n    pass",
        language="python",
        source_file=Path("example.py"),
        output_file=Path("output/example.py"),
        line_start=1,
        line_end=2
    )
    
    # Verify metadata
    is_valid = verify_block_metadata(block)
    print(f"Block metadata valid: {is_valid}")
    
    # Update stats
    stats["code_blocks"] += 1
    stats["languages"]["python"] = stats["languages"].get("python", 0) + 1
    stats["file_blocks"]["example.py"] = [block]
    
    print("\nStats:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    usage_example() 