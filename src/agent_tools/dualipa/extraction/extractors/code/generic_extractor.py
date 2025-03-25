"""
Generic code extraction for DuaLipa.

This module provides pattern-based extraction for languages without dedicated parsers,
using regex patterns and brace matching for basic code block detection.

Key Features:
1. Pattern-based extraction
2. Multi-language support
3. Brace matching
4. Indentation tracking

Dependencies:
- loguru: For logging (https://github.com/Delgan/loguru)
- re: For regex pattern matching (https://docs.python.org/3/library/re.html)
- textwrap: For text formatting (https://docs.python.org/3/library/textwrap.html)

Documentation Links:
- Regular Expressions: https://docs.python.org/3/howto/regex.html
- Loguru: https://loguru.readthedocs.io/
- Python Text Processing: https://docs.python.org/3/library/text.html

Input/Output Specifications:

extract_generic_blocks(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    Input:
        - file_path: Path to source file
    Output:
        - Tuple containing:
            1. List of dictionaries:
                - uuid: str
                - type: str (function, class)
                - name: str
                - content: str
                - metadata: Dict[str, Any]
                    - line_start: int
                    - line_end: int
                    - imports: List[str]
                    - language: str
            2. Statistics dictionary:
                - total_files: int
                - total_blocks: int
                - languages: Dict[str, Dict]
                - block_types: Dict[str, int]
                - errors: List[str]
                - classes: int
                - functions: int
                - imports: int
                - file_blocks: Dict[str, List]

_extract_block_content(content: str, start_pos: int) -> Optional[str]:
    Input:
        - content: Source code
        - start_pos: Starting position of block
    Output:
        - Block content if successful, None otherwise

_extract_imports(content: str, language: str) -> List[str]:
    Input:
        - content: Source code
        - language: Programming language
    Output:
        - List of import statements

Related Files:
- python_extractor.py: AST-based extraction
- js_ts_extractor.py: Tree-sitter based extraction
"""

import re
import textwrap
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Pattern
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info

# Common patterns for various languages
PATTERNS = {
    "function": {
        "c": r"(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{",
        "cpp": r"(?:virtual\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?{",
        "java": r"(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(?:\{|throws)",
        "go": r"func\s+(\w+)\s*\([^)]*\)\s*(?:\([^)]*\))?\s*{",
        "ruby": r"(?:def)\s+(\w+)(?:\([^)]*\))?\s*(?:do|\n|$|{)",
        "php": r"(?:function|public function|private function|protected function)\s+(\w+)\s*\([^)]*\)\s*{",
        "rust": r"(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^{]+)?\s*{",
        "javascript": r"(?:function\s+)(\w+)\s*\([^)]*\)\s*{|(?:const|let|var)\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*{|(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
        "typescript": r"(?:function\s+)(\w+)(?:<[^>]*>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*{|(?:const|let|var)\s+(\w+)(?:<[^>]*>)?\s*=\s*function\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*{|(?:const|let|var)\s+(\w+)(?:<[^>]*>)?\s*=\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*=>\s*{",
    },
    "class": {
        "c": r"(?:class|struct)\s+(\w+)(?:\s*:\s*\w+)?\s*{",
        "cpp": r"(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*{",
        "java": r"(?:public\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
        "go": r"type\s+(\w+)\s+struct\s*{",
        "ruby": r"class\s+(\w+)(?:\s*<\s*\w+)?\s*(?:do|\n|$|{)",
        "php": r"(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]+)?\s*{",
        "rust": r"(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*{",
        "javascript": r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*{",
        "typescript": r"class\s+(\w+)(?:<[^>]*>)?(?:\s+extends\s+\w+(?:<[^>]*>)?)?(?:\s+implements\s+[^{]+)?\s*{|interface\s+(\w+)(?:<[^>]*>)?\s*(?:extends\s+[^{]+)?\s*{",
    }
}

def extract_generic_blocks(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract code blocks using pattern matching.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Tuple of (extracted blocks, statistics)
    """
    try:
        # Initialize stats
        stats = init_stats()
        blocks = []
        
        # Verify file exists
        if not Path(file_path).exists():
            stats["errors"].append(f"File not found: {file_path}")
            return [], stats
            
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Detect language
        language = detect_language(file_path)
        if language == "unknown":
            stats["errors"].append(f"Unknown language for file: {file_path}")
            return [], stats
            
        # Get language info
        info = get_language_info(language)
        if not info:
            stats["errors"].append(f"Unsupported language: {language}")
            return [], stats
            
        # Track imports
        imports = _extract_imports(content, language)
        stats["imports"] = len(imports)
        
        # Extract functions
        if language in PATTERNS["function"]:
            pattern = PATTERNS["function"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_content = _extract_block_content(content, match.start())
                if func_content:
                    stats["functions"] = stats.get("functions", 0) + 1
                    
                    # Extract docstring
                    docstring = _extract_generic_docstring(content, match.start())
                    
                    blocks.append({
                        "uuid": str(uuid.uuid4()),
                        "type": "function",
                        "name": match.group(1),
                        "content": textwrap.dedent(func_content),
                        "doc_string": docstring or "No documentation provided",
                        "metadata": {
                            "line_start": content.count('\n', 0, match.start()) + 1,
                            "line_end": content.count('\n', 0, match.start() + len(func_content)) + 1,
                            "imports": imports.copy(),
                            "language": language,
                            "has_docstring": docstring is not None
                        }
                    })
                    
        # Extract classes
        if language in PATTERNS["class"]:
            pattern = PATTERNS["class"][language]
            for match in re.finditer(pattern, content, re.MULTILINE):
                class_content = _extract_block_content(content, match.start())
                if class_content:
                    stats["classes"] = stats.get("classes", 0) + 1
                    
                    # Extract docstring
                    docstring = _extract_generic_docstring(content, match.start())
                    
                    blocks.append({
                        "uuid": str(uuid.uuid4()),
                        "type": "class",
                        "name": match.group(1),
                        "content": textwrap.dedent(class_content),
                        "doc_string": docstring or "No documentation provided",
                        "metadata": {
                            "line_start": content.count('\n', 0, match.start()) + 1,
                            "line_end": content.count('\n', 0, match.start() + len(class_content)) + 1,
                            "imports": imports.copy(),
                            "language": language,
                            "has_docstring": docstring is not None
                        }
                    })
                    
        # Update stats
        stats.update({
            "total_blocks": len(blocks),
            "file_blocks": {file_path: blocks}
        })
        
        return blocks, stats
        
    except Exception as e:
        logger.error(f"Error extracting blocks from {file_path}: {e}")
        return [], stats

def _extract_generic_docstring(content: str, start_pos: int) -> Optional[str]:
    """Extract potential docstring above a code block.
    
    Args:
        content: Source code
        start_pos: Starting position of block
        
    Returns:
        Docstring if found, None otherwise
    """
    try:
        # Look for comment lines above the block
        line_start = content.rfind('\n', 0, start_pos)
        if line_start == -1:  # Handle case where block is at the start of file
            line_start = 0
        else:
            line_start += 1  # Move past the newline
            
        # Get the line where the block starts
        block_line = content.count('\n', 0, start_pos) + 1
        
        # Check for JavaDoc or similar comment styles (/** ... */)
        text_before = content[max(0, start_pos - 500):start_pos]
        javadoc_pattern = r'/\*\*([^*]|\*[^/])*\*/'  # Match javadoc comment
        match = list(re.finditer(javadoc_pattern, text_before, re.DOTALL))
        if match:
            last_match = match[-1]
            # Clean up comment
            comment = last_match.group(0)
            lines = [line.strip().lstrip('* ') for line in comment.splitlines()[1:-1]]
            return '\n'.join(line for line in lines if line)
            
        # Check for single-line comments above the block
        if block_line > 1:
            comment_lines = []
            line_num = block_line - 1
            comment_symbol = None
            
            # Go up to 10 lines before or until a non-comment line is found
            while line_num > 0 and line_num >= block_line - 10:
                line_start = content.rfind('\n', 0, line_start - 1)
                if line_start == -1:  # Handle first line
                    line_start = 0
                else:
                    line_start += 1  # Move past the newline
                    
                line = content[line_start:content.find('\n', line_start)].strip()
                
                # Detect comment style if not already determined
                if comment_symbol is None:
                    if line.startswith('//'):  # C-style
                        comment_symbol = '//'
                    elif line.startswith('#'):  # Python/Ruby style
                        comment_symbol = '#'
                    elif line.startswith('--'):  # SQL/Lua style
                        comment_symbol = '--'
                    elif line.startswith(';'):  # Assembly/Lisp style
                        comment_symbol = ';'
                    else:  # Not a comment line
                        break
                
                # Check if this is still a comment
                if line.startswith(comment_symbol):
                    comment_lines.insert(0, line[len(comment_symbol):].strip())
                    line_num -= 1
                else:  # If not a comment, stop
                    break
                    
            if comment_lines:
                return '\n'.join(comment_lines)
                
        return None
    except Exception as e:
        logger.error(f"Error extracting generic docstring: {e}")
        return None

def _extract_block_content(content: str, start_pos: int) -> Optional[str]:
    """
    Extract content of a code block using brace matching.
    
    Args:
        content: Source code
        start_pos: Starting position of block
        
    Returns:
        Block content if successful, None otherwise
    """
    try:
        brace_count = 0
        in_string = False
        string_char = None
        pos = start_pos
        
        # Find opening brace
        while pos < len(content):
            if content[pos] == '{':
                brace_count = 1
                break
            pos += 1
            
        if brace_count == 0:
            return None
            
        start = start_pos
        pos += 1
        
        # Match braces
        while pos < len(content) and brace_count > 0:
            char = content[pos]
            
            # Handle strings
            if char in ('"', "'") and content[pos-1] != '\\':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    
            # Count braces if not in string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
            pos += 1
            
        # Extract content
        if brace_count == 0:
            return content[start:pos]
            
        return None
        
    except Exception as e:
        logger.error(f"Error extracting block content: {e}")
        return None

def _extract_imports(content: str, language: str) -> List[str]:
    """
    Extract import statements based on language.
    
    Args:
        content: Source code
        language: Programming language
        
    Returns:
        List of import statements
    """
    imports = []
    
    try:
        # Language-specific import patterns
        patterns = {
            "c": r'#include\s*[<"]([^>"]+)[>"]',
            "cpp": r'#include\s*[<"]([^>"]+)[>"]',
            "java": r'import\s+[\w.]+(?:\s*\*)?;',
            "go": r'import\s+(?:"[^"]+"|`[^`]+`|\([^)]+\))',
            "ruby": r'require\s+[\'"][^\'"]+[\'"]',
            "php": r'(?:require|include|use)\s+[\'"][^\'"]+[\'"];?',
            "rust": r'use\s+[\w:]+(?:\s*\*)?;'
        }
        
        if language in patterns:
            pattern = patterns[language]
            imports = [m.group(0) for m in re.finditer(pattern, content)]
            
    except Exception as e:
        logger.error(f"Error extracting imports: {e}")
        
    return imports

def usage_example() -> None:
    """Example usage of generic code extraction."""
    # Example C++ file
    cpp_content = textwrap.dedent('''
    #include <iostream>
    #include <string>
    
    class Person {
    private:
        std::string name;
        int age;
        
    public:
        Person(const std::string& n, int a) : name(n), age(a) {}
        
        void greet() const {
            std::cout << "Hello, " << name << "!" << std::endl;
        }
    };
    
    int main() {
        Person person("Alice", 30);
        person.greet();
        return 0;
    }
    ''')
    
    # Save to temp file
    with open('temp.cpp', 'w') as f:
        f.write(cpp_content)
        
    # Extract blocks
    blocks, stats = extract_generic_blocks('temp.cpp')
    
    print(f"Found {len(blocks)} blocks:")
    for block in blocks:
        print(f"\nType: {block['type']}")
        print(f"Name: {block['name']}")
        print("Content:\n")
        print(textwrap.indent(block['content'], "    "))
        
    print("\nStatistics:")
    print(f"Classes: {stats.get('classes', 0)}")
    print(f"Functions: {stats.get('functions', 0)}")
    print(f"Imports: {stats.get('imports', 0)}")
    
    # Cleanup
    import os
    os.remove('temp.cpp')

if __name__ == "__main__":
    print("Running generic extractor usage example...")
    usage_example()
    print("Done!") 