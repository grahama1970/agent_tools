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

def extract_hierarchical_sections(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract hierarchical sections from markdown content.
    
    Args:
        markdown: Markdown content to extract sections from
        
    Returns:
        List of section objects with title, content, depth, and path information
    """
    sections = []
    lines = markdown.split('\n')
    current_section = None
    current_content = []
    
    # Track the current path at each depth
    current_titles = {}
    
    # Track file paths at each depth
    current_file_paths = {}
    
    for line in lines:
        # Check if the line is a header
        header_match = re.match(r'^(#+)\s+(.+)$', line)
        
        if header_match:
            # If we were building a section, finalize it
            if current_section is not None:
                current_section['content'] = '\n'.join(current_content)
                sections.append(current_section)
                
            # Extract header depth and title
            depth = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            # Update the current path
            # Clear any deeper titles from the path
            for d in list(current_titles.keys()):
                if d >= depth:
                    del current_titles[d]
            
            # Set current title at this depth
            current_titles[depth] = title
            
            # Build the path (titles of parent sections)
            path = []
            for d in sorted(current_titles.keys()):
                if d < depth:
                    path.append(current_titles[d])
            
            # Create new section
            current_section = {
                'title': title,
                'depth': depth,
                'path': path,
            }
            
            # Generate file paths
            file_name = f"{slugify(title)}.md"
            file_paths = [file_name]  # Base filename
            
            # Clear any deeper file paths
            for d in list(current_file_paths.keys()):
                if d >= depth:
                    del current_file_paths[d]
            
            # Set the file path at this depth
            if depth == 1:
                # Top-level sections get just the filename
                current_file_paths[depth] = file_name
            else:
                # Find closest parent depth
                parent_depth = None
                for d in sorted(current_file_paths.keys(), reverse=True):
                    if d < depth:
                        parent_depth = d
                        break
                
                if parent_depth is not None:
                    # Get the parent directory name (without .md extension)
                    parent_dir = current_file_paths[parent_depth]
                    if parent_dir.endswith('.md'):
                        parent_dir = parent_dir[:-3]
                    
                    # Create path with the parent directory
                    current_file_paths[depth] = os.path.join(parent_dir, file_name)
                else:
                    # If no parent found, treat as top-level
                    current_file_paths[depth] = file_name
            
            # Generate all file paths (for all levels)
            file_paths = []
            for d in sorted(current_file_paths.keys()):
                if d <= depth:
                    file_paths.append(current_file_paths[d])
            
            current_section['file_paths'] = file_paths
            current_content = [line]  # Start with the header line
        else:
            # Add to current content
            if current_section is not None:
                current_content.append(line)
    
    # Add the final section
    if current_section is not None:
        current_section['content'] = '\n'.join(current_content)
        sections.append(current_section)
    
    return sections

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