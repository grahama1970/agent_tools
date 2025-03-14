"""
Language detection module for DuaLipa.

This module provides utilities to detect programming languages
based on file extensions and content. It's used by the code extractor
to properly categorize extracted code.

Official Documentation References:
- re: https://docs.python.org/3/library/re.html
- pathlib: https://docs.python.org/3/library/pathlib.html
- loguru: https://loguru.readthedocs.io/en/stable/
"""

import re
import os
import sys
import tempfile
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from pathlib import Path
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Common file extensions and their corresponding languages
LANGUAGE_EXTENSIONS: Dict[str, str] = {
    # Python
    '.py': 'python',
    '.pyw': 'python',
    '.ipynb': 'jupyter',
    
    # JavaScript family
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.jsx': 'jsx',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.vue': 'vue',
    '.svelte': 'svelte',
    
    # Web technologies
    '.html': 'html',
    '.htm': 'html',
    '.xhtml': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.less': 'less',
    
    # C family
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    '.c++': 'cpp',
    '.h++': 'cpp',
    
    # C#, Java, and other JVM languages
    '.cs': 'csharp',
    '.java': 'java',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    '.sc': 'scala',
    '.groovy': 'groovy',
    
    # Ruby
    '.rb': 'ruby',
    '.erb': 'erb',
    '.rake': 'ruby',
    
    # PHP
    '.php': 'php',
    '.phtml': 'php',
    
    # Go
    '.go': 'go',
    
    # Rust
    '.rs': 'rust',
    
    # Swift
    '.swift': 'swift',
    
    # Shell scripts
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'bash',
    '.fish': 'fish',
    '.bat': 'batch',
    '.cmd': 'batch',
    '.ps1': 'powershell',
    
    # Configuration and data formats
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.toml': 'toml',
    '.ini': 'ini',
    '.xml': 'xml',
    '.csv': 'csv',
    
    # Documentation
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.rst': 'rst',
    '.tex': 'tex',
    '.txt': 'text',
    
    # Other languages
    '.sql': 'sql',
    '.r': 'r',
    '.perl': 'perl',
    '.pl': 'perl',
    '.lua': 'lua',
    '.dart': 'dart',
    '.f': 'fortran',
    '.f77': 'fortran',
    '.f90': 'fortran',
    '.hs': 'haskell',
    '.lhs': 'haskell',
    '.elm': 'elm',
    '.clj': 'clojure',
    '.ex': 'elixir',
    '.exs': 'elixir',
}

# Shebang patterns for scripts
SHEBANG_PATTERNS: Dict[str, str] = {
    r'^#!.*\bpython\b': 'python',
    r'^#!.*\bruby\b': 'ruby',
    r'^#!.*\bnode\b': 'javascript',
    r'^#!.*\bbash\b': 'bash',
    r'^#!.*\bzsh\b': 'bash',
    r'^#!.*\bsh\b': 'bash',
    r'^#!.*\bperl\b': 'perl',
    r'^#!.*\bphp\b': 'php',
}


def detect_language(file_path: Union[str, Path], content: Optional[str] = None) -> str:
    """
    Detect the programming language of a file based on its extension and/or content.
    
    Args:
        file_path: Path to the file
        content: Optional file content (if already loaded)
        
    Returns:
        Detected language name (lowercase)
    """
    try:
        # Convert to Path object if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # First check by file extension
        file_ext = file_path.suffix.lower()
        if file_ext in LANGUAGE_EXTENSIONS:
            language = LANGUAGE_EXTENSIONS[file_ext]
            logger.debug(f"Detected language for {file_path.name}: {language} (by extension)")
            return language
        
        # No content provided, try to open the file
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1024)  # Read just the beginning for shebang
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {str(e)}")
                return "unknown"
        
        # Check for shebang
        first_line = content.split('\n')[0] if content else ""
        for pattern, language in SHEBANG_PATTERNS.items():
            if re.match(pattern, first_line):
                logger.debug(f"Detected language for {file_path.name}: {language} (by shebang)")
                return language
        
        # Additional heuristics could be added here
        # ...
        
        # If nothing is detected, use a simple fallback based on filename
        filename = file_path.name.lower()
        if 'makefile' in filename:
            return 'makefile'
        elif 'dockerfile' in filename:
            return 'dockerfile'
        
        logger.debug(f"Could not detect language for {file_path.name}, returning 'unknown'")
        return "unknown"
        
    except Exception as e:
        logger.error(f"Error detecting language for {file_path}: {str(e)}")
        return "unknown"


def get_file_language_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed language information for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with language information:
        - language: Detected language
        - extension: File extension
        - is_binary: Whether the file is likely binary
        - category: General category (code, data, documentation, etc.)
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        language = detect_language(file_path)
        extension = file_path.suffix.lower()
        
        # Determine if the file is likely binary
        is_binary = False
        try:
            # Try to open as text, if it fails it's likely binary
            with open(file_path, 'r', encoding='utf-8') as f:
                is_binary = not all(c.isprintable() or c.isspace() for c in f.read(1024))
        except UnicodeDecodeError:
            is_binary = True
        except Exception as e:
            logger.warning(f"Error checking if {file_path} is binary: {str(e)}")
        
        # Determine category
        category = "unknown"
        if language in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp', 'ruby', 'go', 'rust']:
            category = "code"
        elif language in ['markdown', 'rst', 'tex', 'text']:
            category = "documentation"
        elif language in ['json', 'yaml', 'toml', 'xml', 'csv', 'ini']:
            category = "data"
        elif language in ['html', 'css', 'scss', 'sass', 'less']:
            category = "web"
        elif language in ['bash', 'batch', 'powershell', 'fish']:
            category = "script"
        
        return {
            "language": language,
            "extension": extension,
            "is_binary": is_binary,
            "category": category,
            "filename": file_path.name
        }
        
    except Exception as e:
        logger.error(f"Error getting language info for {file_path}: {str(e)}")
        return {
            "language": "unknown",
            "extension": "",
            "is_binary": False,
            "category": "unknown",
            "filename": str(file_path).split('/')[-1] if isinstance(file_path, str) else file_path.name,
            "error": str(e)
        }


def is_supported_language(file_path: Union[str, Path]) -> bool:
    """
    Check if a file's language is supported by the code extractor.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the language is supported, False otherwise
    """
    try:
        language = detect_language(file_path)
        
        # List of supported languages
        supported_languages = {
            'python', 'javascript', 'typescript', 'jsx', 'tsx', 'java', 
            'cpp', 'c', 'csharp', 'go', 'rust', 'ruby', 'markdown', 'html',
            'css', 'bash', 'sql', 'php'
        }
        
        return language in supported_languages
        
    except Exception as e:
        logger.error(f"Error checking if language is supported for {file_path}: {str(e)}")
        return False


def demo_language_detection() -> None:
    """Demonstrate the language detection functionality with examples.
    
    This function shows how to use the main components of the language detection module:
    1. Detecting languages based on file extensions
    2. Getting detailed language information for files
    3. Checking if languages are supported
    
    Returns:
        None - prints results to the console
    """
    try:
        logger.info("Language Detection Demo")
        logger.info("======================")
        
        # Create temporary directory for the demo
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample files for testing
            files = {
                "python_script.py": '#!/usr/bin/env python3\n\nprint("Hello from Python")',
                "javascript_module.js": 'const greet = (name) => `Hello, ${name}!`;\nexport default greet;',
                "markdown_doc.md": '# Markdown Document\n\nThis is a test document.',
                "cpp_program.cpp": '#include <iostream>\n\nint main() {\n  std::cout << "Hello from C++\\n";\n  return 0;\n}',
                "shell_script.sh": '#!/bin/bash\n\necho "Hello from Bash"',
                "html_page.html": '<!DOCTYPE html>\n<html>\n<body>\n<h1>Hello from HTML</h1>\n</body>\n</html>',
                "makefile": 'all:\n\techo "This is a Makefile"',
                "config.json": '{\n  "name": "Test Config",\n  "version": 1.0\n}',
                "unknown_file.xyz": 'This file has an unknown extension'
            }
            
            # Write sample files
            file_paths = []
            for filename, content in files.items():
                file_path = temp_path / filename
                with open(file_path, 'w') as f:
                    f.write(content)
                file_paths.append(file_path)
            
            # 1. Basic language detection
            logger.info("\n1. Basic language detection:")
            for file_path in file_paths:
                language = detect_language(file_path)
                logger.info(f"  {file_path.name} → {language}")
            
            # 2. Detailed language information
            logger.info("\n2. Detailed language information:")
            for file_path in file_paths[:4]:  # Show details for first few files
                info = get_file_language_info(file_path)
                logger.info(f"  {file_path.name}:")
                logger.info(f"    Language: {info['language']}")
                logger.info(f"    Extension: {info['extension']}")
                logger.info(f"    Category: {info['category']}")
                logger.info(f"    Binary: {info['is_binary']}")
            
            # 3. Check supported languages
            logger.info("\n3. Checking supported languages:")
            for file_path in file_paths:
                supported = is_supported_language(file_path)
                logger.info(f"  {file_path.name} → {'Supported' if supported else 'Not supported'}")
            
            # 4. Test language detection from content
            logger.info("\n4. Language detection from content:")
            contents = {
                "Python with shebang": "#!/usr/bin/env python\nimport os\nprint(os.getcwd())",
                "Bash script": "#!/bin/bash\necho 'Hello World'",
                "JavaScript code": "function hello() {\n  console.log('Hello World');\n}\nhello();",
                "Plain text": "This is just plain text without any specific language markers."
            }
            
            for name, content in contents.items():
                # Use a temporary file with .txt extension so extension doesn't help
                temp_file = temp_path / "temp.txt"
                with open(temp_file, 'w') as f:
                    f.write(content)
                
                language = detect_language(temp_file, content)
                logger.info(f"  {name} → {language}")
        
        logger.info("\nLanguage Detection Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in language detection demo: {e}")


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_language_detection()
    
    # Process files if paths are provided as arguments
    if len(sys.argv) > 1:
        logger.info("\nProcessing files from arguments:")
        
        for file_path in sys.argv[1:]:
            path = Path(file_path)
            
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                continue
                
            logger.info(f"\nAnalyzing: {path}")
            
            if path.is_file():
                # Get detailed language information
                info = get_file_language_info(path)
                
                # Print information
                logger.info(f"Language: {info['language']}")
                logger.info(f"Extension: {info['extension']}")
                logger.info(f"Category: {info['category']}")
                logger.info(f"Binary: {info['is_binary']}")
                logger.info(f"Supported: {is_supported_language(path)}")
                
                # Show file preview if it's not binary
                if not info['is_binary']:
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            preview = f.read(200)
                        logger.info("File preview:")
                        logger.info("---")
                        logger.info(preview + ("..." if len(preview) >= 200 else ""))
                        logger.info("---")
                    except Exception as e:
                        logger.error(f"Could not read file: {e}")
            
            elif path.is_dir():
                # Process directory
                logger.info("Processing directory...")
                
                file_count = 0
                language_stats = {}
                
                for file in path.glob('**/*'):
                    if file.is_file():
                        file_count += 1
                        language = detect_language(file)
                        
                        if language in language_stats:
                            language_stats[language] += 1
                        else:
                            language_stats[language] = 1
                
                # Print statistics
                logger.info(f"Total files: {file_count}")
                logger.info("Language distribution:")
                
                for language, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / file_count) * 100 if file_count > 0 else 0
                    logger.info(f"  {language}: {count} files ({percentage:.1f}%)")
            
            else:
                logger.error(f"Path is neither a file nor a directory: {path}") 