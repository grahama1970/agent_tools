"""
Enhanced markdown hierarchy parser for extracting section hierarchies.

This module provides functionality to extract sections from markdown content
with hierarchy information, including depth and relative paths.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

def slugify(title: str) -> str:
    """
    Convert a title to a slug for use in filenames and URLs.
    
    Args:
        title: String to slugify
        
    Returns:
        Slugified string
    """
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

def extract_hierarchical_sections(markdown_or_path: str) -> List[Dict[str, Any]]:
    """
    Extract hierarchical sections from markdown content or a file path.
    
    Args:
        markdown_or_path: Either markdown content as a string, or a path to a markdown file
        
    Returns:
        List of section objects with title, content, depth, and path information
    """
    # Check if this is a file path
    if os.path.exists(markdown_or_path) and (
        markdown_or_path.lower().endswith('.md') or 
        markdown_or_path.lower().endswith('.markdown') or 
        markdown_or_path.lower().endswith('.mdown')
    ):
        # It's a file path, read the content
        try:
            with open(markdown_or_path, 'r', encoding='utf-8', errors='replace') as f:
                markdown = f.read()
        except Exception as e:
            print(f"Error reading markdown file {markdown_or_path}: {e}")
            return []
    else:
        # Assume it's markdown content
        markdown = markdown_or_path
    
    # Split the content into lines for processing
    lines = markdown.split('\n')
    
    # Extract all headers first
    headers = []
    for i, line in enumerate(lines):
        header_match = re.match(r'^(#+)\s+(.+)$', line)
        if header_match:
            depth = len(header_match.group(1))
            title = header_match.group(2).strip()
            headers.append({
                'title': title,
                'level': depth,  # Using 'level' as the tests expect
                'depth': depth,  # Keep 'depth' for backward compatibility
                'line_index': i,
                'start_line': i + 1,  # Convert to 1-indexed line number
                'content_lines': []
            })
    
    # If no headers found, return empty list
    if not headers:
        return []
    
    # Assign content to each header
    for i, header in enumerate(headers):
        start_idx = header['line_index']
        if i < len(headers) - 1:
            end_idx = headers[i + 1]['line_index']
        else:
            end_idx = len(lines)
        
        # Include the header line itself in the content
        header['content_lines'] = lines[start_idx:end_idx]
        header['end_line'] = start_idx + len(header['content_lines'])  # Convert to 1-indexed
    
    # Build the path for each header
    for i, header in enumerate(headers):
        path = []
        for prev_header in headers[:i]:
            if prev_header['level'] < header['level']:
                path.append(prev_header['title'])
        header['path'] = path
    
    # Create file_paths for each header
    for header in headers:
        title_slug = slugify(header['title'])
        file_name = f"{title_slug}.md"
        header['file_paths'] = [file_name]
        
        # Add parent paths
        if header['path']:
            parent_path = '/'.join([slugify(p) for p in header['path']])
            header['file_paths'].append(f"{parent_path}/{file_name}")
    
    # Create the final section objects
    sections = []
    for header in headers:
        section = {
            'title': header['title'],
            'level': header['level'],
            'depth': header['depth'],
            'path': header['path'],
            'start_line': header['start_line'],
            'end_line': header['end_line'],
            'content': '\n'.join(header['content_lines']),
            'file_paths': header['file_paths'],
            'subsections': []
        }
        sections.append(section)
    
    # Build the hierarchy by nesting subsections under their parent sections
    top_level_sections = []
    section_map = {}  # Map from (level, title) to section object
    
    # First pass: create a mapping of all sections
    for section in sections:
        key = (section['level'], section['title'])
        section_map[key] = section
    
    # Second pass: build the hierarchy
    for section in sections:
        # Check if this section has a parent
        if section['path']:
            parent_title = section['path'][-1]
            parent_level = section['level'] - 1
            parent_key = (parent_level, parent_title)
            
            # Find parent section and add this section as a subsection
            if parent_key in section_map:
                parent = section_map[parent_key]
                parent['subsections'].append(section)
        else:
            # This is a top-level section
            top_level_sections.append(section)
    
    return top_level_sections

def build_repository_hierarchy(repo_path: str) -> List[Dict[str, Any]]:
    """
    Build a complete hierarchy of markdown files in a repository.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        List of file objects with path, depth, and internal section hierarchies
    """
    hierarchy = []
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Compute file path components
                path_parts = Path(rel_path).parts
                dir_hierarchy = list(path_parts[:-1])  # All but the filename
                
                # Basic file metadata
                file_info = {
                    'path': rel_path,
                    'depth': len(path_parts) - 1,  # Depth in directory structure
                    'name': os.path.splitext(file)[0],
                    'type': '.md',
                    'dir_hierarchy': dir_hierarchy,
                    'full_ancestry': list(path_parts),
                }
                
                # Extract internal structure
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    sections = extract_hierarchical_sections(content)
                    
                    # Structure sections into a nested hierarchy
                    nested_sections = []
                    section_map = {}  # Map from depth+title to section object
                    
                    # First pass: create all section objects with empty children lists
                    for section in sections:
                        section_with_children = section.copy()
                        section_with_children['children'] = []
                        section_key = (section['depth'], section['title'])
                        section_map[section_key] = section_with_children
                    
                    # Second pass: build the hierarchy
                    for section in sections:
                        section_key = (section['depth'], section['title'])
                        section_obj = section_map[section_key]
                        
                        if section['path']:
                            # This has a parent
                            parent_title = section['path'][-1]
                            parent_depth = section['depth'] - 1
                            parent_key = (parent_depth, parent_title)
                            if parent_key in section_map:
                                section_map[parent_key]['children'].append(section_obj)
                        else:
                            # This is a top-level section
                            nested_sections.append(section_obj)
                    
                    file_info['internal_sections'] = nested_sections
                except Exception as e:
                    file_info['internal_sections'] = []
                    file_info['error'] = str(e)
                
                hierarchy.append(file_info)
    
    return hierarchy

def write_hierarchical_sections(sections: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """
    Write hierarchical sections to files.
    
    Args:
        sections: List of section objects from extract_hierarchical_sections
        output_dir: Base directory to write sections to
        
    Returns:
        Dictionary mapping section titles to file paths
    """
    output_files = {}
    output_dir = Path(output_dir)
    
    for section in sections:
        # Get the file path for this section
        if not section['file_paths']:
            continue
            
        # Use the last (deepest) file path
        file_path = section['file_paths'][-1]
        full_path = output_dir / file_path
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create YAML frontmatter
        frontmatter = {
            'title': section['title'],
            'depth': section['depth'],
            'path': section['path'],
        }
        
        # Format frontmatter as YAML
        frontmatter_str = '---\n'
        for key, value in frontmatter.items():
            frontmatter_str += f'{key}: {repr(value)}\n'
        frontmatter_str += '---\n\n'
        
        # Write to file with frontmatter
        with open(full_path, 'w') as f:
            f.write(frontmatter_str + section['content'])
        
        # Record the output file
        output_files[section['title']] = str(full_path)
    
    return output_files

def process_markdown_repository(repo_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process a repository of markdown files into a hierarchical structure.
    
    Args:
        repo_path: Path to repository with markdown files
        output_dir: Directory to write processed markdown files
        
    Returns:
        Dictionary with repository hierarchy and output file mapping
    """
    # Build the complete repository hierarchy
    hierarchy = build_repository_hierarchy(repo_path)
    
    # Extract all sections from all files
    all_sections = []
    for file_info in hierarchy:
        try:
            with open(os.path.join(repo_path, file_info['path']), 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = extract_hierarchical_sections(content)
            
            # Add file path context to each section
            for section in sections:
                section['file_path'] = file_info['path']
                section['dir_hierarchy'] = file_info['dir_hierarchy']
                all_sections.append(section)
        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
    
    # Write all sections with their hierarchy
    output_files = write_hierarchical_sections(all_sections, output_dir)
    
    return {
        'hierarchy': hierarchy,
        'output_files': output_files
    }

# Example usage
if __name__ == "__main__":
    # Simple demo
    markdown = """# Introduction
This is the introduction.

## Getting Started
How to get started.

# Advanced Topics
Advanced topics here.

## Configuration
Configuration details.
"""
    
    sections = extract_hierarchical_sections(markdown)
    print("Extracted sections:")
    for section in sections:
        print(f"Title: {section['title']}")
        print(f"Depth: {section['depth']}")
        print(f"Path: {section['path']}")
        print(f"File paths: {section['file_paths']}")
        print("-----")
    
    print("\nProcessing a repository:")
    # In a real scenario, you would pass your repository path
    # result = process_markdown_repository("path/to/repo", "path/to/output")
    # print(f"Processed {len(result['hierarchy'])} files")
    # print(f"Created {len(result['output_files'])} output files") 