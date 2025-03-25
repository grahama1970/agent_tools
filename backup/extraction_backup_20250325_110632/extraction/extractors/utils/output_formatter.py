"""
Output formatting utilities for extraction results.

This module provides functions to format extraction results in different formats:
- JSON: For machine consumption and API responses
- Markdown: For human-readable documentation
- HTML: For web display and reports
"""

import json
import html
from typing import Dict, List, Any, Optional
from datetime import datetime

def format_output_as_json(extraction_data: Dict[str, Any]) -> str:
    """
    Format extraction results as JSON.

    Args:
        extraction_data: Dictionary containing blocks and stats

    Returns:
        JSON string representation of the extraction data
    """
    # Create a copy to avoid modifying the original
    formatted_data = {
        "blocks": [],
        "stats": extraction_data.get("stats", {})
    }

    # Process blocks to ensure QA-compatible format
    blocks = extraction_data.get("blocks", [])
    for block in blocks:
        # Ensure each block has required fields for QA module
        formatted_block = {
            "uuid": block.get("uuid", block.get("id", f"block_{len(formatted_data['blocks'])}")),
            "type": block.get("type", "unknown"),
            "name": block.get("name", "Unnamed Block"),
            "content": block.get("content", ""),
            "language": block.get("language", block.get("metadata", {}).get("language", "unknown"))
        }
        
        # Make sure id matches uuid for backward compatibility
        formatted_block["id"] = formatted_block["uuid"]
        
        # Ensure metadata is correctly structured
        metadata = block.get("metadata", {})
        if not metadata and isinstance(block, dict):
            # If no metadata but block has related fields, construct it
            metadata = {
                "source_file": block.get("path", block.get("source_file", block.get("file", "unknown")))
            }
            
            # Handle line number variations
            if "line_start" in block:
                metadata["line_start"] = block["line_start"]
            elif "start_line" in block:
                metadata["line_start"] = block["start_line"]
            else:
                metadata["line_start"] = 1
                
            if "line_end" in block:
                metadata["line_end"] = block["line_end"]
            elif "end_line" in block:
                metadata["line_end"] = block["end_line"]
            else:
                metadata["line_end"] = metadata["line_start"]
            
            # Include language information
            metadata["language"] = block.get("language", "unknown")
            
            # Add imports if available
            if "imports" in block:
                metadata["imports"] = block["imports"]
        else:
            # Ensure metadata has consistent field names
            if "file" in metadata and "source_file" not in metadata:
                metadata["source_file"] = metadata["file"]
                
            if "path" in metadata and "source_file" not in metadata:
                metadata["source_file"] = metadata["path"]
            
            # Handle line number variations in metadata
            if "start_line" in metadata and "line_start" not in metadata:
                metadata["line_start"] = metadata["start_line"]
                
            if "end_line" in metadata and "line_end" not in metadata:
                metadata["line_end"] = metadata["end_line"]
                
        formatted_block["metadata"] = metadata
        formatted_data["blocks"].append(formatted_block)

    # Add format metadata
    formatted_data["metadata"] = {
        "formatted_at": datetime.now().isoformat(),
        "format_version": "1.0"
    }

    # Convert to JSON with pretty formatting
    return json.dumps(formatted_data, indent=2)

def format_output_as_md(extraction_data: Dict[str, Any]) -> str:
    """
    Format extraction results as Markdown.

    Args:
        extraction_data: Dictionary containing blocks and stats

    Returns:
        Markdown string representation of the extraction data
    """
    blocks = extraction_data.get("blocks", [])
    stats = extraction_data.get("stats", {})
    
    # Start with title
    md_output = "# Extraction Results\n\n"
    
    # Add statistics section
    md_output += "## Statistics\n\n"
    md_output += f"- Total Files: {stats.get('total_files', 0)}\n"
    md_output += f"- Code Files: {stats.get('code_files', 0)}\n"
    md_output += f"- Documentation Files: {stats.get('documentation_files', 0)}\n"
    md_output += f"- Code Blocks: {stats.get('code_blocks', 0)}\n"
    
    # Add extraction time if available
    if "extraction_time" in stats:
        md_output += f"- Extraction Time: {stats.get('extraction_time', 0):.2f} seconds\n"
    
    # Add error counts if available
    if "validation_errors" in stats:
        md_output += f"- Validation Errors: {stats.get('validation_errors', 0)}\n"
    if "verification_errors" in stats:
        md_output += f"- Verification Errors: {stats.get('verification_errors', 0)}\n"
    
    md_output += "\n"
    
    # Add language statistics if available
    if "languages" in stats:
        md_output += "### Languages\n\n"
        for lang, count in stats["languages"].items():
            md_output += f"- {lang}: {count}\n"
        md_output += "\n"
    
    # Add block type statistics if available
    if "block_types" in stats:
        md_output += "### Block Types\n\n"
        for block_type, count in stats["block_types"].items():
            md_output += f"- {block_type}: {count}\n"
        md_output += "\n"
    
    # Add repository information if available
    if "repo_url" in stats:
        md_output += f"Repository: [{stats['repo_url']}]({stats['repo_url']})\n\n"
    
    # Add code blocks section
    md_output += "## Code Blocks\n\n"
    
    # Group blocks by language for better organization
    blocks_by_language = {}
    for block in blocks:
        # Get language from metadata if available, otherwise from block directly
        metadata = block.get("metadata", {})
        language = metadata.get("language", block.get("language", "unknown"))
        
        if language not in blocks_by_language:
            blocks_by_language[language] = []
        blocks_by_language[language].append(block)
    
    # Add blocks for each language
    for language, lang_blocks in blocks_by_language.items():
        md_output += f"### {language.capitalize()} Blocks\n\n"
        
        for block in lang_blocks:
            # Get block metadata
            metadata = block.get("metadata", {})
            
            # Add block information
            name = block.get("name", "Unnamed Block")
            uuid = block.get("uuid", block.get("id", "unknown"))
            block_type = block.get("type", "Unknown Type")
            
            # Get source file from metadata or block
            source_file = metadata.get("source_file", block.get("path", block.get("source_file", "Unknown Path")))
            
            # Get line numbers if available
            line_start = metadata.get("line_start", block.get("line_start", None))
            line_end = metadata.get("line_end", block.get("line_end", None))
            line_info = f" (lines {line_start}-{line_end})" if line_start and line_end else ""
            
            md_output += f"#### {name} ({block_type})\n\n"
            md_output += f"ID: `{uuid}`\n\n"
            md_output += f"File: `{source_file}`{line_info}\n\n"
            
            # Add imports if available
            imports = metadata.get("imports", block.get("imports", []))
            if imports:
                md_output += "Imports:\n```\n"
                for imp in imports:
                    md_output += f"{imp}\n"
                md_output += "```\n\n"
            
            # Add code content with language-specific formatting
            md_output += f"```{language}\n{block.get('content', '')}\n```\n\n"
    
    return md_output

def format_output_as_html(extraction_data: Dict[str, Any]) -> str:
    """
    Format extraction results as HTML.

    Args:
        extraction_data: Dictionary containing blocks and stats

    Returns:
        HTML string representation of the extraction data
    """
    blocks = extraction_data.get("blocks", [])
    stats = extraction_data.get("stats", {})
    
    # Basic HTML template with some styling
    html_output = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h2 {
            margin-top: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        h3 {
            margin-top: 25px;
        }
        h4 {
            margin-top: 20px;
            background-color: #f5f5f5;
            padding: 5px 10px;
            border-left: 4px solid #2196F3;
        }
        pre {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        }
        .stats {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stats-item {
            margin-bottom: 5px;
        }
        .block-info {
            background-color: #f9f9f9;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .file-path {
            font-family: monospace;
            background-color: #eee;
            padding: 2px 5px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Extraction Results</h1>
"""

    # Add statistics section
    html_output += """
    <h2>Statistics</h2>
    <div class="stats">
"""
    html_output += f'        <div class="stats-item"><strong>Total Files:</strong> {stats.get("total_files", 0)}</div>\n'
    html_output += f'        <div class="stats-item"><strong>Code Files:</strong> {stats.get("code_files", 0)}</div>\n'
    html_output += f'        <div class="stats-item"><strong>Documentation Files:</strong> {stats.get("documentation_files", 0)}</div>\n'
    html_output += f'        <div class="stats-item"><strong>Code Blocks:</strong> {stats.get("code_blocks", 0)}</div>\n'
    
    # Add language statistics if available
    if "languages" in stats:
        html_output += '        <div class="stats-item"><strong>Languages:</strong></div>\n'
        html_output += '        <ul>\n'
        for lang, count in stats["languages"].items():
            html_output += f'            <li>{html.escape(lang)}: {count}</li>\n'
        html_output += '        </ul>\n'
    
    # Add repository information if available
    if "repo_url" in stats:
        repo_url = stats["repo_url"]
        html_output += f'        <div class="stats-item"><strong>Repository:</strong> <a href="{html.escape(repo_url)}">{html.escape(repo_url)}</a></div>\n'
    
    html_output += '    </div>\n'
    
    # Add code blocks section
    html_output += '    <h2>Code Blocks</h2>\n'
    
    # Group blocks by language for better organization
    blocks_by_language = {}
    for block in blocks:
        # Get language from metadata if available, otherwise from block directly
        metadata = block.get("metadata", {})
        language = metadata.get("language", block.get("language", "unknown"))
        
        if language not in blocks_by_language:
            blocks_by_language[language] = []
        blocks_by_language[language].append(block)
    
    # Add blocks for each language
    for language, lang_blocks in blocks_by_language.items():
        html_output += f'    <h3>{html.escape(language.capitalize())} Blocks</h3>\n'
        
        for block in lang_blocks:
            # Get block metadata
            metadata = block.get("metadata", {})
            
            # Add block information
            name = html.escape(block.get("name", "Unnamed Block"))
            uuid = html.escape(block.get("uuid", block.get("id", "unknown")))
            block_type = html.escape(block.get("type", "Unknown Type"))
            
            # Get source file from metadata or block
            source_file = html.escape(metadata.get("source_file", 
                block.get("path", block.get("source_file", "Unknown Path"))))
            
            # Get line numbers if available
            line_start = metadata.get("line_start", block.get("line_start", None))
            line_end = metadata.get("line_end", block.get("line_end", None))
            line_info = f" (lines {line_start}-{line_end})" if line_start and line_end else ""
            
            content = html.escape(block.get("content", ""))
            
            html_output += f'    <h4>{name} ({block_type})</h4>\n'
            html_output += f'    <div class="block-info">\n'
            html_output += f'        <div><strong>ID:</strong> <code>{uuid}</code></div>\n'
            html_output += f'        <div><strong>File:</strong> <span class="file-path">{source_file}</span>{line_info}</div>\n'
            
            # Add additional metadata if available
            imports = metadata.get("imports", block.get("imports", []))
            if imports:
                html_output += f'        <div><strong>Imports:</strong></div>\n'
                html_output += f'        <div><pre><code>'
                for imp in imports:
                    html_output += f'{html.escape(imp)}\n'
                html_output += f'</code></pre></div>\n'
            
            html_output += '    </div>\n'
            
            # Add code content with language-specific highlighting
            html_output += f'    <pre><code class="language-{html.escape(language)}">{content}</code></pre>\n\n'
    
    # Close HTML tags
    html_output += """
</body>
</html>
"""
    
    return html_output

def _validate_extraction_data(extraction_data: Dict[str, Any]) -> bool:
    """
    Validate that the extraction data contains required fields.
    
    Args:
        extraction_data: Dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(extraction_data, dict):
        return False
        
    if "blocks" not in extraction_data:
        return False
        
    if not isinstance(extraction_data["blocks"], list):
        return False
        
    return True

def format_output(extraction_data: Dict[str, Any], format_type: str) -> str:
    """
    Format extraction results based on the specified format type.
    
    Args:
        extraction_data: Dictionary containing blocks and stats
        format_type: One of 'json', 'md', 'html'
        
    Returns:
        Formatted string according to format_type
        
    Raises:
        ValueError: If format_type is not supported
    """
    if not _validate_extraction_data(extraction_data):
        raise ValueError("Invalid extraction data format")
        
    format_type = format_type.lower()
    
    if format_type == 'json':
        return format_output_as_json(extraction_data)
    elif format_type == 'md':
        return format_output_as_md(extraction_data)
    elif format_type == 'html':
        return format_output_as_html(extraction_data)
    else:
        raise ValueError(f"Unsupported format type: {format_type}. Supported types: json, md, html")