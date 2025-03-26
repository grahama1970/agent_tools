#!/usr/bin/env python3
"""
AST-based Code Structure Extractor

This module provides AST-based extraction of code structures using tree-sitter,
integrated with the AI memory system for improved extraction capabilities.

Key Features:
1. Memory-aware extraction with error tracking and learning
2. Language-agnostic interface with specialized extractors
3. Rich structure extraction with nested elements and relationships
4. Fallback mechanisms for robust extraction

Example Usage:
```python
# Initialize with memory integration
extractor = AstExtractor(memory_db_path="extraction_memory.db")

# Extract code structure from a file
result = extractor.extract_file("path/to/file.py")

# Process a directory recursively
results = extractor.extract_directory("path/to/repo", 
                                     languages=['python', 'javascript'])

# Get extraction statistics
stats = extractor.get_statistics()
```

Dependencies:
- tree-sitter: For AST parsing
- tree-sitter-language-pack: Prebuilt language parsers
- AI memory system: For context tracking and error learning
"""

import os
import sys
import glob
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ast_extractor")

# Import tree-sitter utilities
try:
    from tree_sitter import Parser, Node
    from ..utils.tree_sitter_utils import get_parser, get_supported_languages
    from ..utils.tree_sitter_helpers import get_node_text
    TREE_SITTER_AVAILABLE = True
except ImportError:
    logger.warning("Tree-sitter not available. Functionality will be limited.")
    TREE_SITTER_AVAILABLE = False

# Import memory system components if available
try:
    from ...extraction_memory import (
        init_extraction_memory,
        track_extraction_start,
        track_extraction_progress,
        track_extraction_completion,
        record_extraction_error,
        find_similar_errors,
        save_extraction_knowledge,
        find_extraction_knowledge,
        get_extraction_context
    )
    MEMORY_AVAILABLE = True
except ImportError:
    logger.warning("Memory system not available. Extraction will not be memory-aware.")
    MEMORY_AVAILABLE = False


class AstExtractor:
    """
    AST-based code structure extraction using tree-sitter with memory integration.
    
    This extractor provides rich extraction of code structures from source files using
    abstract syntax trees (ASTs) with tree-sitter. It integrates with the AI memory
    system to track extraction progress, learn from errors, and improve extraction
    over time.
    """
    
    def __init__(self, memory_db_path: Optional[str] = None):
        """
        Initialize the AST extractor.
        
        Args:
            memory_db_path: Optional path to the memory database for integration
        """
        # Initialize memory system
        self.memory_available = False
        if memory_db_path and MEMORY_AVAILABLE:
            try:
                init_extraction_memory(memory_db_path)
                self.memory_available = True
                logger.info(f"Memory system initialized with database: {memory_db_path}")
            except Exception as e:
                logger.error(f"Error initializing memory system: {e}")
                
        # Initialize statistics
        self.stats = {
            "files_processed": 0,
            "files_extracted": 0,
            "extraction_errors": 0,
            "languages": {},
            "start_time": time.time()
        }
        
        # Initialize parsers
        self.parsers = {}
        
        # File extension to language mapping
        self.ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp"
        }
        
    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect the programming language of a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name or None if not detected
        """
        ext = os.path.splitext(file_path)[1].lower()
        return self.ext_to_lang.get(ext)
        
    def _get_parser(self, language: str) -> Optional[Parser]:
        """
        Get or initialize a parser for the given language.
        
        Args:
            language: Language name
            
        Returns:
            Parser instance or None if not available
        """
        if not TREE_SITTER_AVAILABLE:
            return None
            
        if language not in self.parsers:
            parser = get_parser(language)
            if parser:
                self.parsers[language] = parser
                
        return self.parsers.get(language)
    
    def extract_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract code structure from a file.
        
        Args:
            file_path: Path to the file to extract
            language: Optional language override
            
        Returns:
            Dictionary with extracted structure
        """
        # Update statistics
        self.stats["files_processed"] += 1
        
        # Determine language if not provided
        if not language:
            language = self._detect_language(file_path)
        
        if not language:
            error_msg = f"Could not detect language for file: {file_path}"
            logger.warning(error_msg)
            
            if self.memory_available:
                record_extraction_error(
                    "language_detection_error",
                    error_msg,
                    file_path,
                    "Specify language explicitly or add extension to mapping",
                    severity=4
                )
                
            self.stats["extraction_errors"] += 1
            return {
                "file_path": file_path,
                "error": "Unknown language"
            }
            
        # Initialize parser
        parser = self._get_parser(language)
        if not parser:
            error_msg = f"No parser available for language: {language}"
            logger.warning(error_msg)
            
            if self.memory_available:
                record_extraction_error(
                    "parser_unavailable",
                    error_msg,
                    file_path,
                    "Install tree-sitter or use fallback extraction method",
                    severity=5
                )
                
            self.stats["extraction_errors"] += 1
            return {
                "file_path": file_path,
                "language": language,
                "error": "No parser available"
            }
            
        # Update language statistics
        if language not in self.stats["languages"]:
            self.stats["languages"][language] = {
                "files": 0,
                "success": 0,
                "errors": 0
            }
        self.stats["languages"][language]["files"] += 1
        
        # Parse file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the content
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Extract structure based on language
            result = None
            if language == 'python':
                result = self._extract_python_ast(file_path, content, tree)
            elif language in ['javascript', 'typescript']:
                result = self._extract_js_ts_ast(file_path, content, tree)
            elif language == 'go':
                result = self._extract_go_ast(file_path, content, tree)
            elif language == 'rust':
                result = self._extract_rust_ast(file_path, content, tree)
            else:
                # Generic extraction for other languages
                result = self._extract_generic_ast(file_path, content, tree, language)
                
            if result:
                # Update statistics on success
                self.stats["files_extracted"] += 1
                self.stats["languages"][language]["success"] += 1
                
                # Track successful extraction pattern
                if self.memory_available:
                    self._track_extraction_success(file_path, language, result)
                    
                return result
            else:
                raise Exception(f"Failed to extract structure from {file_path}")
                
        except Exception as e:
            # Record extraction error
            error_msg = f"Error extracting {language} structure: {str(e)}"
            logger.error(f"{error_msg} from {file_path}")
            
            if self.memory_available:
                record_extraction_error(
                    f"{language}_extraction_error",
                    error_msg,
                    file_path,
                    severity=6
                )
                
                # Try to find similar errors with recovery suggestions
                similar_errors = find_similar_errors(error_msg)
                if isinstance(similar_errors, list) and similar_errors:
                    for error in similar_errors:
                        recovery = error.get("recovery_action")
                        if recovery:
                            logger.info(f"Suggested recovery: {recovery}")
                            
                            # TODO: Implement automatic recovery based on suggestions
            
            # Update statistics
            self.stats["extraction_errors"] += 1
            self.stats["languages"][language]["errors"] += 1
            
            # Return error information
            return {
                "file_path": file_path,
                "language": language,
                "error": str(e)
            }
    
    def extract_directory(self, dir_path: str, 
                         languages: Optional[List[str]] = None,
                         recursive: bool = True,
                         exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract code structures from all files in a directory.
        
        Args:
            dir_path: Path to the directory
            languages: Optional list of languages to process
            recursive: Whether to process subdirectories
            exclude_patterns: Patterns to exclude
            
        Returns:
            Dictionary with statistics and list of extraction results
        """
        if self.memory_available:
            repo_name = os.path.basename(os.path.abspath(dir_path))
            track_extraction_start(repo_name, "ast_extraction", {
                "languages": languages,
                "recursive": recursive,
                "exclude_patterns": exclude_patterns
            })
        
        results = []
        self.stats["start_time"] = time.time()
        
        # Find files to process
        files_to_process = []
        
        if not exclude_patterns:
            exclude_patterns = ["**/node_modules/**", "**/.git/**", "**/__pycache__/**", "**/venv/**"]
            
        # Process files by language
        if languages:
            for lang in languages:
                # Convert language to extensions
                exts = []
                for ext, l in self.ext_to_lang.items():
                    if l == lang:
                        exts.append(ext)
                
                for ext in exts:
                    pattern = f"**/*{ext}" if recursive else f"*{ext}"
                    matched_files = self._find_files(dir_path, pattern, exclude_patterns)
                    files_to_process.extend(matched_files)
        else:
            # Process all known languages
            for ext in self.ext_to_lang.keys():
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                matched_files = self._find_files(dir_path, pattern, exclude_patterns)
                files_to_process.extend(matched_files)
        
        total_files = len(files_to_process)
        logger.info(f"Found {total_files} files to process in {dir_path}")
        
        # Update progress
        if self.memory_available:
            repo_name = os.path.basename(os.path.abspath(dir_path))
            track_extraction_progress(
                repo_name,
                "file_discovery",
                f"Found {total_files} files to process",
                "Start AST extraction",
                {"total_files": total_files}
            )
        
        # Process each file
        for i, file_path in enumerate(files_to_process):
            if i > 0 and i % 10 == 0:
                # Log progress
                logger.info(f"Processed {i}/{total_files} files")
                
                # Update progress in memory
                if self.memory_available:
                    repo_name = os.path.basename(os.path.abspath(dir_path))
                    track_extraction_progress(
                        repo_name,
                        "extraction",
                        f"Processed {i}/{total_files} files",
                        "Continue AST extraction",
                        self.stats
                    )
            
            # Extract file
            result = self.extract_file(file_path)
            results.append(result)
        
        # Finalize statistics
        end_time = time.time()
        self.stats["end_time"] = end_time
        self.stats["duration_seconds"] = end_time - self.stats["start_time"]
        self.stats["total_files"] = total_files
        
        logger.info(f"Extraction completed: {self.stats['files_extracted']} files extracted with {self.stats['extraction_errors']} errors")
        logger.info(f"Total extraction time: {self.stats['duration_seconds']:.2f} seconds")
        
        # Record completion in memory
        if self.memory_available:
            repo_name = os.path.basename(os.path.abspath(dir_path))
            track_extraction_completion(
                repo_name,
                f"Extracted {self.stats['files_extracted']} files with {self.stats['extraction_errors']} errors",
                self.stats
            )
        
        return {
            "stats": self.stats,
            "results": results
        }
    
    def _find_files(self, base_dir: str, pattern: str, exclude_patterns: List[str]) -> List[str]:
        """
        Find files matching a pattern while excluding others.
        
        Args:
            base_dir: Base directory to search
            pattern: Glob pattern to match
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of matching file paths
        """
        base_path = Path(base_dir)
        matched_files = list(base_path.glob(pattern))
        
        if exclude_patterns:
            filtered_files = []
            for file_path in matched_files:
                # Convert to relative path for pattern matching
                rel_path = file_path.relative_to(base_path)
                exclude = False
                
                for exclude_pattern in exclude_patterns:
                    if Path(exclude_pattern).match(str(rel_path)):
                        exclude = True
                        break
                
                if not exclude:
                    filtered_files.append(str(file_path))
                    
            return filtered_files
        else:
            return [str(f) for f in matched_files]
    
    def _track_extraction_success(self, file_path: str, language: str, result: Dict[str, Any]) -> None:
        """
        Track successful extraction pattern in memory.
        
        Args:
            file_path: Path to the file
            language: Programming language
            result: Extraction result
        """
        if not self.memory_available:
            return
            
        # Determine structure types from result
        structure_types = []
        if result.get("classes", []):
            structure_types.append("class")
        if result.get("functions", []):
            structure_types.append("function")
        if result.get("imports", []):
            structure_types.append("import")
        
        # Create summary of complex structures
        complex_structures = []
        
        # Check for nested classes
        for cls in result.get("classes", []):
            if cls.get("inner_classes", []):
                complex_structures.append(f"nested_class:{cls['name']}")
                
        # Check for classes with inheritance
        for cls in result.get("classes", []):
            if cls.get("inherits_from", []):
                complex_structures.append(f"inheritance:{cls['name']}")
        
        # Store extraction pattern for each structure type
        for struct_type in structure_types:
            save_extraction_knowledge(
                f"ast_pattern_{language}_{struct_type}",
                f"# Successful {language} {struct_type} extraction\n\n"
                f"File: {file_path}\n\n"
                f"Complex structures: {', '.join(complex_structures) if complex_structures else 'None'}\n\n"
                f"Example count: {len(result.get(struct_type + 's', []))}",
                summary=f"Successful {language} {struct_type} extraction pattern",
                tags=["ast", "extraction", language, struct_type]
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        # Calculate success rate
        if self.stats["files_processed"] > 0:
            self.stats["success_rate"] = (self.stats["files_extracted"] / self.stats["files_processed"]) * 100
        else:
            self.stats["success_rate"] = 0
            
        return self.stats.copy()
    
    def _extract_python_ast(self, file_path: str, content: str, tree: Any) -> Dict[str, Any]:
        """
        Extract Python code structure from AST.
        
        Args:
            file_path: Path to the Python file
            content: File content
            tree: Tree-sitter parse tree
            
        Returns:
            Dictionary with extracted structure
        """
        if not tree or not tree.root_node:
            raise Exception("Invalid parse tree")
            
        # Initialize result
        result = {
            "file_path": file_path,
            "language": "python",
            "classes": [],
            "functions": [],
            "imports": [],
            "docstring": None
        }
        
        # Function to extract node text
        def get_text(node):
            return content[node.start_byte:node.end_byte]
            
        # Function to extract docstring from a node
        def extract_docstring(node):
            # Look for the first expression statement with a string literal
            for child in node.children:
                if child.type == 'expression_statement':
                    string_child = None
                    # Check if child is a direct string node
                    for grandchild in child.children:
                        if grandchild.type == 'string':
                            string_child = grandchild
                            break
                    
                    if string_child:
                        return get_text(string_child).strip().strip('"\'')
            return None
            
        # Process module-level docstring
        result["docstring"] = extract_docstring(tree.root_node)
        
        # Process module-level nodes
        for node in tree.root_node.children:
            # Import statements
            if node.type == 'import_statement' or node.type == 'import_from_statement':
                result["imports"].append({
                    "text": get_text(node).strip(),
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1]
                })
                
            # Class definitions
            elif node.type == 'class_definition':
                # Get class name
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if not name_node:
                    continue
                    
                class_name = get_text(name_node)
                
                # Get inheritance
                inherits_from = []
                argument_list = None
                for child in node.children:
                    if child.type == 'argument_list':
                        argument_list = child
                        break
                
                if argument_list:
                    for child in argument_list.children:
                        if child.type == 'identifier':
                            inherits_from.append(get_text(child))
                
                # Create class structure
                class_struct = {
                    "name": class_name,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1],
                    "docstring": extract_docstring(node),
                    "methods": [],
                    "attributes": [],
                    "inner_classes": [],
                    "inherits_from": inherits_from
                }
                
                # Find class body block
                body_node = None
                for child in node.children:
                    if child.type == 'block':
                        body_node = child
                        break
                
                if body_node:
                    # Process class body
                    for body_child in body_node.children:
                        # Function definition (method)
                        if body_child.type == 'function_definition':
                            method_name = None
                            for method_child in body_child.children:
                                if method_child.type == 'identifier':
                                    method_name = get_text(method_child)
                                    break
                            
                            if method_name:
                                # Handle decorators
                                decorators = []
                                for sibling in body_child.children:
                                    if sibling.type == 'decorator':
                                        decorators.append(get_text(sibling).strip())
                                
                                # Extract parameters
                                parameters = []
                                params_node = None
                                for param_child in body_child.children:
                                    if param_child.type == 'parameters':
                                        params_node = param_child
                                        break
                                
                                if params_node:
                                    for param_child in params_node.children:
                                        if param_child.type == 'identifier':
                                            parameters.append(get_text(param_child))
                                
                                # Add method to class
                                class_struct["methods"].append({
                                    "name": method_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1],
                                    "decorators": decorators,
                                    "parameters": parameters,
                                    "docstring": extract_docstring(body_child)
                                })
                        
                        # Inner class definition
                        elif body_child.type == 'class_definition':
                            inner_name_node = None
                            for inner_child in body_child.children:
                                if inner_child.type == 'identifier':
                                    inner_name_node = inner_child
                                    break
                            
                            if inner_name_node:
                                inner_name = get_text(inner_name_node)
                                class_struct["inner_classes"].append({
                                    "name": inner_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1],
                                    "docstring": extract_docstring(body_child)
                                })
                        
                        # Class variable assignment
                        elif body_child.type == 'expression_statement' and '=' in get_text(body_child):
                            # Simplistic attribute detection - could be improved
                            attr_text = get_text(body_child).strip()
                            if not attr_text.startswith('self.'):
                                # Class-level attribute
                                if '=' in attr_text:
                                    attr_name = attr_text.split('=')[0].strip()
                                    class_struct["attributes"].append({
                                        "name": attr_name,
                                        "line": body_child.start_point[0] + 1,
                                        "column": body_child.start_point[1]
                                    })
                
                # Add class to result
                result["classes"].append(class_struct)
                
            # Function definitions
            elif node.type == 'function_definition':
                # Get function name
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if not name_node:
                    continue
                    
                func_name = get_text(name_node)
                
                # Handle decorators
                decorators = []
                for child in node.children:
                    if child.type == 'decorator':
                        decorators.append(get_text(child).strip())
                
                # Extract parameters
                parameters = []
                params_node = None
                for child in node.children:
                    if child.type == 'parameters':
                        params_node = child
                        break
                
                if params_node:
                    for param_child in params_node.children:
                        if param_child.type == 'identifier':
                            parameters.append(get_text(param_child))
                
                # Add function to result
                result["functions"].append({
                    "name": func_name,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1],
                    "decorators": decorators,
                    "parameters": parameters,
                    "docstring": extract_docstring(node)
                })
        
        return result
    
    def _extract_js_ts_ast(self, file_path: str, content: str, tree: Any) -> Dict[str, Any]:
        """
        Extract JavaScript or TypeScript code structure from AST.
        
        Args:
            file_path: Path to the JS/TS file
            content: File content
            tree: Tree-sitter parse tree
            
        Returns:
            Dictionary with extracted structure
        """
        if not tree or not tree.root_node:
            raise Exception("Invalid parse tree")
            
        language = "typescript" if file_path.endswith((".ts", ".tsx")) else "javascript"
            
        # Initialize result
        result = {
            "file_path": file_path,
            "language": language,
            "classes": [],
            "functions": [],
            "interfaces": [],
            "imports": [],
            "exports": []
        }
        
        # Function to extract node text
        def get_text(node):
            return content[node.start_byte:node.end_byte]
        
        # Extract imports and exports
        from ..utils.tree_sitter_utils import extract_js_ts_imports_exports
        imports, exports = extract_js_ts_imports_exports(content, tree)
        
        for imp in imports:
            result["imports"].append({
                "text": imp,
                "line": 0,  # Would require mapping back to original line
                "column": 0
            })
            
        for exp in exports:
            result["exports"].append({
                "text": exp,
                "line": 0,  # Would require mapping back to original line
                "column": 0
            })
        
        # Process module-level nodes
        for node in tree.root_node.children:
            # Class declarations
            if node.type == 'class_declaration' or node.type == 'class':
                # Get class name
                name_node = node.child_by_field_name('name')
                if not name_node:
                    continue
                    
                class_name = get_text(name_node)
                
                # Get inheritance
                inherits_from = []
                extends_node = node.child_by_field_name('extends')
                
                if extends_node:
                    inherits_from.append(get_text(extends_node))
                
                # Get implemented interfaces
                implements = []
                implements_node = node.child_by_field_name('implements')
                
                if implements_node:
                    implements_text = get_text(implements_node)
                    # Simple parsing - could be improved with proper node traversal
                    if implements_text.startswith('implements '):
                        implements_list = implements_text[11:].split(',')
                        implements = [i.strip() for i in implements_list]
                
                # Create class structure
                class_struct = {
                    "name": class_name,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1],
                    "methods": [],
                    "properties": [],
                    "inherits_from": inherits_from,
                    "implements": implements
                }
                
                # Find class body
                body_node = node.child_by_field_name('body')
                
                if body_node:
                    # Process class body
                    for body_child in body_node.children:
                        # Method definitions
                        if body_child.type == 'method_definition':
                            name_node = body_child.child_by_field_name('name')
                            if name_node:
                                method_name = get_text(name_node)
                                
                                # Extract parameters
                                parameters = []
                                params_node = body_child.child_by_field_name('parameters')
                                
                                if params_node:
                                    for param_child in params_node.children:
                                        if param_child.type == 'identifier':
                                            parameters.append(get_text(param_child))
                                
                                # Determine if static
                                is_static = False
                                for child in body_child.children:
                                    if child.type == 'static' or get_text(child) == 'static':
                                        is_static = True
                                        break
                                
                                # Add method to class
                                class_struct["methods"].append({
                                    "name": method_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1],
                                    "parameters": parameters,
                                    "static": is_static
                                })
                        
                        # Property definitions
                        elif body_child.type in ('public_field_definition', 'field_definition'):
                            name_node = body_child.child_by_field_name('name')
                            if name_node:
                                prop_name = get_text(name_node)
                                
                                # Determine if static
                                is_static = False
                                for child in body_child.children:
                                    if child.type == 'static' or get_text(child) == 'static':
                                        is_static = True
                                        break
                                
                                # Add property to class
                                class_struct["properties"].append({
                                    "name": prop_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1],
                                    "static": is_static
                                })
                
                # Add class to result
                result["classes"].append(class_struct)
            
            # Interface declarations (TypeScript only)
            elif node.type == 'interface_declaration':
                # Get interface name
                name_node = node.child_by_field_name('name')
                if not name_node:
                    continue
                    
                interface_name = get_text(name_node)
                
                # Get extended interfaces
                extends_from = []
                extends_node = node.child_by_field_name('extends')
                
                if extends_node:
                    extends_text = get_text(extends_node)
                    # Simple parsing - could be improved
                    if extends_text.startswith('extends '):
                        extends_list = extends_text[8:].split(',')
                        extends_from = [e.strip() for e in extends_list]
                
                # Create interface structure
                interface_struct = {
                    "name": interface_name,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1],
                    "methods": [],
                    "properties": [],
                    "extends_from": extends_from
                }
                
                # Find interface body
                body_node = node.child_by_field_name('body')
                
                if body_node:
                    # Process interface body
                    for body_child in body_node.children:
                        # Method signatures
                        if body_child.type == 'method_signature':
                            name_node = body_child.child_by_field_name('name')
                            if name_node:
                                method_name = get_text(name_node)
                                
                                # Add method to interface
                                interface_struct["methods"].append({
                                    "name": method_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1]
                                })
                        
                        # Property signatures
                        elif body_child.type == 'property_signature':
                            name_node = body_child.child_by_field_name('name')
                            if name_node:
                                prop_name = get_text(name_node)
                                
                                # Add property to interface
                                interface_struct["properties"].append({
                                    "name": prop_name,
                                    "line": body_child.start_point[0] + 1,
                                    "column": body_child.start_point[1]
                                })
                
                # Add interface to result
                result["interfaces"].append(interface_struct)
            
            # Function declarations
            elif node.type == 'function_declaration' or node.type == 'function':
                # Get function name
                name_node = node.child_by_field_name('name')
                if not name_node:
                    continue
                    
                func_name = get_text(name_node)
                
                # Extract parameters
                parameters = []
                params_node = node.child_by_field_name('parameters')
                
                if params_node:
                    for param_child in params_node.children:
                        if param_child.type == 'identifier':
                            parameters.append(get_text(param_child))
                
                # Add function to result
                result["functions"].append({
                    "name": func_name,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1],
                    "parameters": parameters
                })
            
            # Arrow functions / variable function declarations
            elif node.type in ('lexical_declaration', 'variable_declaration'):
                for child in node.children:
                    if child.type == 'variable_declarator':
                        # Check if this is a function assignment
                        name_node = child.child_by_field_name('name')
                        value_node = child.child_by_field_name('value')
                        
                        if name_node and value_node and value_node.type in ('arrow_function', 'function'):
                            func_name = get_text(name_node)
                            
                            # Extract parameters
                            parameters = []
                            params_node = value_node.child_by_field_name('parameters')
                            
                            if params_node:
                                for param_child in params_node.children:
                                    if param_child.type == 'identifier':
                                        parameters.append(get_text(param_child))
                            
                            # Add function to result
                            result["functions"].append({
                                "name": func_name,
                                "line": node.start_point[0] + 1,
                                "column": node.start_point[1],
                                "parameters": parameters,
                                "type": "arrow_function" if value_node.type == 'arrow_function' else "function_expression"
                            })
        
        return result
    
    def _extract_go_ast(self, file_path: str, content: str, tree: Any) -> Dict[str, Any]:
        """
        Extract Go code structure from AST.
        
        Args:
            file_path: Path to the Go file
            content: File content
            tree: Tree-sitter parse tree
            
        Returns:
            Dictionary with extracted structure
        """
        # Basic implementation - will be enhanced in future
        if not tree or not tree.root_node:
            raise Exception("Invalid parse tree")
            
        # Initialize result
        result = {
            "file_path": file_path,
            "language": "go",
            "package": "",
            "imports": [],
            "structs": [],
            "interfaces": [],
            "functions": []
        }
        
        # Function to extract node text
        def get_text(node):
            return content[node.start_byte:node.end_byte]
        
        # Process package declaration
        for node in tree.root_node.children:
            if node.type == 'package_clause':
                package_name = ""
                for child in node.children:
                    if child.type == 'package_identifier':
                        package_name = get_text(child)
                        break
                
                result["package"] = package_name.strip()
                break
        
        # Process imports
        for node in tree.root_node.children:
            if node.type == 'import_declaration':
                for child in node.children:
                    if child.type == 'import_spec' or child.type == 'import_spec_list':
                        import_path = get_text(child).strip()
                        result["imports"].append({
                            "text": import_path,
                            "line": node.start_point[0] + 1,
                            "column": node.start_point[1]
                        })
        
        # Process struct declarations
        for node in tree.root_node.children:
            if node.type == 'type_declaration':
                for child in node.children:
                    if child.type == 'type_spec':
                        name_node = None
                        type_node = None
                        
                        for grandchild in child.children:
                            if grandchild.type == 'type_identifier':
                                name_node = grandchild
                            elif grandchild.type == 'struct_type':
                                type_node = grandchild
                        
                        if name_node and type_node:
                            struct_name = get_text(name_node)
                            
                            # Create struct structure
                            struct_struct = {
                                "name": struct_name,
                                "line": child.start_point[0] + 1,
                                "column": child.start_point[1],
                                "fields": []
                            }
                            
                            # Process fields
                            field_list_node = None
                            for field_child in type_node.children:
                                if field_child.type == 'field_declaration_list':
                                    field_list_node = field_child
                                    break
                            
                            if field_list_node:
                                for field_decl in field_list_node.children:
                                    if field_decl.type == 'field_declaration':
                                        field_name = ""
                                        for field_name_node in field_decl.children:
                                            if field_name_node.type == 'field_identifier':
                                                field_name = get_text(field_name_node)
                                                break
                                        
                                        if field_name:
                                            struct_struct["fields"].append({
                                                "name": field_name,
                                                "line": field_decl.start_point[0] + 1,
                                                "column": field_decl.start_point[1]
                                            })
                            
                            # Add struct to result
                            result["structs"].append(struct_struct)
        
        # Process function declarations
        for node in tree.root_node.children:
            if node.type == 'function_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier' or child.type == 'function_name':
                        name_node = child
                        break
                
                if name_node:
                    func_name = get_text(name_node)
                    
                    # Create function structure
                    func_struct = {
                        "name": func_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                        "parameters": []
                    }
                    
                    # Process parameters
                    param_list_node = None
                    for child in node.children:
                        if child.type == 'parameter_list':
                            param_list_node = child
                            break
                    
                    if param_list_node:
                        for param_decl in param_list_node.children:
                            if param_decl.type == 'parameter_declaration':
                                param_name = ""
                                for param_name_node in param_decl.children:
                                    if param_name_node.type == 'identifier':
                                        param_name = get_text(param_name_node)
                                        break
                                
                                if param_name:
                                    func_struct["parameters"].append(param_name)
                    
                    # Add function to result
                    result["functions"].append(func_struct)
        
        return result
    
    def _extract_rust_ast(self, file_path: str, content: str, tree: Any) -> Dict[str, Any]:
        """
        Extract Rust code structure from AST.
        
        Args:
            file_path: Path to the Rust file
            content: File content
            tree: Tree-sitter parse tree
            
        Returns:
            Dictionary with extracted structure
        """
        # Basic implementation - will be enhanced in future
        if not tree or not tree.root_node:
            raise Exception("Invalid parse tree")
            
        # Initialize result
        result = {
            "file_path": file_path,
            "language": "rust",
            "modules": [],
            "structs": [],
            "enums": [],
            "traits": [],
            "implementations": [],
            "functions": []
        }
        
        # Function to extract node text
        def get_text(node):
            return content[node.start_byte:node.end_byte]
        
        # Process module declarations
        for node in tree.root_node.children:
            if node.type == 'mod_item':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    mod_name = get_text(name_node)
                    result["modules"].append({
                        "name": mod_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1]
                    })
        
        # Process struct declarations
        for node in tree.root_node.children:
            if node.type == 'struct_item':
                name_node = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                
                if name_node:
                    struct_name = get_text(name_node)
                    
                    # Create struct structure
                    struct_struct = {
                        "name": struct_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                        "fields": []
                    }
                    
                    # Process fields
                    field_list_node = None
                    for child in node.children:
                        if child.type == 'field_declaration_list':
                            field_list_node = child
                            break
                    
                    if field_list_node:
                        for field_decl in field_list_node.children:
                            if field_decl.type == 'field_declaration':
                                field_name = ""
                                for field_name_node in field_decl.children:
                                    if field_name_node.type == 'field_identifier':
                                        field_name = get_text(field_name_node)
                                        break
                                
                                if field_name:
                                    struct_struct["fields"].append({
                                        "name": field_name,
                                        "line": field_decl.start_point[0] + 1,
                                        "column": field_decl.start_point[1]
                                    })
                    
                    # Add struct to result
                    result["structs"].append(struct_struct)
        
        # Process trait declarations
        for node in tree.root_node.children:
            if node.type == 'trait_item':
                name_node = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                
                if name_node:
                    trait_name = get_text(name_node)
                    
                    # Create trait structure
                    trait_struct = {
                        "name": trait_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                        "methods": []
                    }
                    
                    # Process trait body
                    body_node = None
                    for child in node.children:
                        if child.type == 'declaration_list':
                            body_node = child
                            break
                    
                    if body_node:
                        for method_node in body_node.children:
                            if method_node.type == 'function_item' or method_node.type == 'function_signature_item':
                                method_name = ""
                                for method_name_node in method_node.children:
                                    if method_name_node.type == 'identifier':
                                        method_name = get_text(method_name_node)
                                        break
                                
                                if method_name:
                                    trait_struct["methods"].append({
                                        "name": method_name,
                                        "line": method_node.start_point[0] + 1,
                                        "column": method_node.start_point[1]
                                    })
                    
                    # Add trait to result
                    result["traits"].append(trait_struct)
        
        # Process function declarations
        for node in tree.root_node.children:
            if node.type == 'function_item':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    func_name = get_text(name_node)
                    
                    # Create function structure
                    func_struct = {
                        "name": func_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                        "parameters": []
                    }
                    
                    # Process parameters
                    param_list_node = None
                    for child in node.children:
                        if child.type == 'parameters':
                            param_list_node = child
                            break
                    
                    if param_list_node:
                        for param_decl in param_list_node.children:
                            if param_decl.type == 'parameter':
                                param_name = ""
                                for param_name_node in param_decl.children:
                                    if param_name_node.type == 'identifier':
                                        param_name = get_text(param_name_node)
                                        break
                                
                                if param_name:
                                    func_struct["parameters"].append(param_name)
                    
                    # Add function to result
                    result["functions"].append(func_struct)
        
        return result
    
    def _extract_generic_ast(self, file_path: str, content: str, tree: Any, language: str) -> Dict[str, Any]:
        """
        Extract generic code structure from AST for any supported language.
        This is a fallback method with limited functionality.
        
        Args:
            file_path: Path to the file
            content: File content
            tree: Tree-sitter parse tree
            language: Programming language
            
        Returns:
            Dictionary with extracted structure
        """
        if not tree or not tree.root_node:
            raise Exception("Invalid parse tree")
            
        # Initialize result
        result = {
            "file_path": file_path,
            "language": language,
            "functions": [],
            "classes": [],
            "imports": []
        }
        
        # Function to extract node text
        def get_text(node):
            return content[node.start_byte:node.end_byte]
        
        # Function to recursively find nodes by type
        def find_nodes_by_type(node, type_name):
            found = []
            if node.type == type_name:
                found.append(node)
            
            for child in node.children:
                found.extend(find_nodes_by_type(child, type_name))
            
            return found
        
        # Try to find common node types across languages
        function_types = ['function_declaration', 'function_definition', 'function_item', 'function', 'method_definition']
        class_types = ['class_declaration', 'class_definition', 'class', 'struct_item', 'struct_definition']
        import_types = ['import_statement', 'import_declaration', 'import', 'using_declaration']
        
        # Find functions
        for type_name in function_types:
            for node in find_nodes_by_type(tree.root_node, type_name):
                # Try to extract function name using common patterns
                name_node = None
                for child in node.children:
                    if child.type in ['identifier', 'function_name', 'name', 'type_identifier']:
                        name_node = child
                        break
                
                if name_node:
                    func_name = get_text(name_node)
                    result["functions"].append({
                        "name": func_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1]
                    })
        
        # Find classes
        for type_name in class_types:
            for node in find_nodes_by_type(tree.root_node, type_name):
                # Try to extract class name using common patterns
                name_node = None
                for child in node.children:
                    if child.type in ['identifier', 'class_name', 'name', 'type_identifier']:
                        name_node = child
                        break
                
                if name_node:
                    class_name = get_text(name_node)
                    result["classes"].append({
                        "name": class_name,
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1]
                    })
        
        # Find imports
        for type_name in import_types:
            for node in find_nodes_by_type(tree.root_node, type_name):
                import_text = get_text(node)
                result["imports"].append({
                    "text": import_text,
                    "line": node.start_point[0] + 1,
                    "column": node.start_point[1]
                })
        
        return result


# Simple command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AST-based Code Structure Extractor")
    
    parser.add_argument(
        "target",
        help="File or directory to extract"
    )
    
    parser.add_argument(
        "--memory-db",
        help="Path to memory database",
        default="extraction_memory.db"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path",
        default="ast_extraction_result.json"
    )
    
    parser.add_argument(
        "--languages",
        help="Languages to extract (comma-separated)",
        default="python,javascript,typescript"
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AstExtractor(args.memory_db)
    
    # Process target
    target = args.target
    languages = args.languages.split(',') if args.languages else None
    
    import json
    
    if os.path.isfile(target):
        # Extract single file
        result = extractor.extract_file(target)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        print(f"Extraction result saved to {args.output}")
        print(f"Statistics: {extractor.get_statistics()}")
        
    elif os.path.isdir(target):
        # Extract directory
        results = extractor.extract_directory(target, languages)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"Extraction results saved to {args.output}")
        print(f"Statistics: {json.dumps(extractor.get_statistics(), indent=2)}")
        
    else:
        print(f"Target not found: {target}")