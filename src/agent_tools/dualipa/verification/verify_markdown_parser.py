#!/usr/bin/env python3
"""
Verify markdown parsing functionality.

This script tests the markdown parsing functionality in the DuaLipa library,
which extracts code blocks, sections, and other elements from markdown files.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the required modules
try:
    from agent_tools.dualipa.markdown_parser import (
        parse_markdown,
        extract_code_blocks,
        extract_sections,
        extract_tables
    )
    print("Successfully imported markdown parser modules")
except ImportError as e:
    print(f"Error importing markdown parser modules: {e}")
    sys.exit(1)

def print_header(text, underline='='):
    """Print a header with underline."""
    print(f"\n{text}")
    print(underline * len(text))

def get_test_markdown():
    """Return a test markdown document with various elements."""
    return """# Markdown Test Document

This is a test markdown document with various elements like **bold text**, *italic text*, and `inline code`.

## Code Blocks

Here's a Python code block:

```python
def hello_world():
    print("Hello, World!")
    return "Hello, World!"

# Another function
def add(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b
```

And here's a JavaScript code block:

```javascript
function helloWorld() {
    console.log("Hello, World!");
    return "Hello, World!";
}

// Another function
function add(a, b) {
    // Add two numbers and return the result
    return a + b;
}
```

## Tables

Here's a simple table:

| Name | Age | Occupation |
|------|-----|------------|
| John | 30  | Developer  |
| Jane | 25  | Designer   |
| Bob  | 35  | Manager    |

## Nested Lists

- Item 1
  - Subitem 1.1
  - Subitem 1.2
    - Subsubitem 1.2.1
- Item 2
  - Subitem 2.1
- Item 3

## Blockquotes

> This is a blockquote
> It can span multiple lines
>
> > And can be nested
>
> Back to the first level

## Math Formulas

Inline formula: $E = mc^2$

Display formula:

$$
\\frac{d}{dx}\\left( \\int_{0}^{x} f(u)\\,du\\right)=f(x)
$$

## Mixed Code Blocks

Here's a mixed code block:

```
This is not a specific language
Just some code
```

```bash
# This is a bash script
echo "Hello, World!"
```

```sql
-- This is SQL
SELECT * FROM users WHERE age > 18;
```

## That's all!

End of the test document.
"""

def verify_parse_markdown():
    """Verify full markdown parsing."""
    print_header("Testing full markdown parsing", "-")
    
    test_markdown = get_test_markdown()
    
    try:
        print("Parsing markdown document...")
        parsed = parse_markdown(test_markdown)
        
        # Print parsed information
        print(f"Parsed document has {len(parsed)} elements:")
        element_types = {}
        for element in parsed:
            element_type = element.get('type', 'unknown')
            if element_type not in element_types:
                element_types[element_type] = 0
            element_types[element_type] += 1
        
        # Print distribution of element types
        for element_type, count in element_types.items():
            print(f"  {element_type}: {count}")
        
        # Print a few examples of elements
        print("\nExample elements:")
        for i, element in enumerate(parsed[:3]):  # Show only first 3 elements
            print(f"\nElement {i+1}:")
            print(f"  Type: {element.get('type', 'unknown')}")
            content = element.get('text', element.get('content', ''))
            print(f"  Content: {content[:50]}..." if content else "  Content: None")
        
        return len(parsed) > 0 and len(element_types) > 1
    except Exception as e:
        print(f"❌ Error during markdown parsing: {str(e)}")
        return False

def verify_extract_code_blocks():
    """Verify extraction of code blocks from markdown."""
    print_header("Testing code block extraction", "-")
    
    test_markdown = get_test_markdown()
    
    try:
        print("Extracting code blocks from markdown...")
        code_blocks = extract_code_blocks(test_markdown)
        
        # Print code block information
        print(f"Extracted {len(code_blocks)} code blocks:")
        languages = {}
        for block in code_blocks:
            language = block.get('language', 'unknown')
            if language not in languages:
                languages[language] = 0
            languages[language] += 1
        
        # Print distribution of languages
        for language, count in languages.items():
            print(f"  {language}: {count}")
        
        # Print a few examples of code blocks
        print("\nExample code blocks:")
        for i, block in enumerate(code_blocks[:3]):  # Show only first 3 blocks
            print(f"\nBlock {i+1}:")
            print(f"  Language: {block.get('language', 'unknown')}")
            print(f"  Content length: {len(block.get('content', ''))}")
            print(f"  First line: {block.get('content', '').splitlines()[0] if block.get('content', '') else ''}")
        
        # Verify we found the expected code blocks
        expected_languages = {'python', 'javascript', 'bash', 'sql', None}  # None for unspecified language
        found_languages = set(languages.keys())
        found_expected = expected_languages.issubset(found_languages)
        print(f"\nFound all expected languages: {'✅' if found_expected else '❌'}")
        
        return len(code_blocks) >= 5 and found_expected
    except Exception as e:
        print(f"❌ Error during code block extraction: {str(e)}")
        return False

def verify_extract_sections():
    """Verify extraction of sections from markdown."""
    print_header("Testing section extraction", "-")
    
    test_markdown = get_test_markdown()
    
    try:
        print("Extracting sections from markdown...")
        sections = extract_sections(test_markdown)
        
        # Print section information
        print(f"Extracted {len(sections)} sections:")
        for i, section in enumerate(sections):
            print(f"\nSection {i+1}:")
            print(f"  Level: {section.get('level', 0)}")
            print(f"  Title: {section.get('title', 'Untitled')}")
            print(f"  Content length: {len(section.get('content', ''))}")
            print(f"  First line: {section.get('content', '').splitlines()[0] if section.get('content', '') else ''}")
        
        # Verify we found the expected sections
        expected_titles = {'Markdown Test Document', 'Code Blocks', 'Tables', 'Nested Lists', 
                          'Blockquotes', 'Math Formulas', 'Mixed Code Blocks', "That's all!"}
        found_titles = {section.get('title', '') for section in sections}
        found_expected = expected_titles.issubset(found_titles)
        print(f"\nFound all expected sections: {'✅' if found_expected else '❌'}")
        
        return len(sections) >= 8 and found_expected
    except Exception as e:
        print(f"❌ Error during section extraction: {str(e)}")
        return False

def verify_extract_tables():
    """Verify extraction of tables from markdown."""
    print_header("Testing table extraction", "-")
    
    test_markdown = get_test_markdown()
    
    try:
        print("Extracting tables from markdown...")
        tables = extract_tables(test_markdown)
        
        # Print table information
        print(f"Extracted {len(tables)} tables:")
        for i, table in enumerate(tables):
            print(f"\nTable {i+1}:")
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            print(f"  Headers: {', '.join(headers)}")
            print(f"  Rows: {len(rows)}")
            for j, row in enumerate(rows[:2]):  # Show only first 2 rows
                print(f"  Row {j+1}: {row}")
        
        # Verify we found the expected table structure
        if tables:
            first_table = tables[0]
            headers = first_table.get('headers', [])
            rows = first_table.get('rows', [])
            expected_headers = ['Name', 'Age', 'Occupation']
            headers_match = headers == expected_headers
            expected_rows_count = 3
            rows_count_match = len(rows) == expected_rows_count
            
            print(f"\nTable headers match expected: {'✅' if headers_match else '❌'}")
            print(f"Table rows count match expected: {'✅' if rows_count_match else '❌'}")
            
            return headers_match and rows_count_match
        else:
            print("❌ No tables found")
            return False
    except Exception as e:
        print(f"❌ Error during table extraction: {str(e)}")
        return False

def main():
    """Run all verification tests."""
    print_header("Markdown Parser Verification")
    
    # Run all verification tests
    parse_markdown_success = verify_parse_markdown()
    extract_code_blocks_success = verify_extract_code_blocks()
    extract_sections_success = verify_extract_sections()
    extract_tables_success = verify_extract_tables()
    
    # Calculate overall success
    all_success = (
        parse_markdown_success and
        extract_code_blocks_success and
        extract_sections_success and
        extract_tables_success
    )
    
    # Print summary
    print_header("Verification Summary")
    print(f"Parse Markdown: {'✅' if parse_markdown_success else '❌'}")
    print(f"Extract Code Blocks: {'✅' if extract_code_blocks_success else '❌'}")
    print(f"Extract Sections: {'✅' if extract_sections_success else '❌'}")
    print(f"Extract Tables: {'✅' if extract_tables_success else '❌'}")
    print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 