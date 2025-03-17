import tempfile
from src.agent_tools.dualipa.markdown_hierarchy import extract_hierarchical_sections

# Create markdown with tables
table_content = """# Document with Tables

## Table Section

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

## Another Section

Content without table.
"""

with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
    f.write(table_content)
    f.flush()
    
    # Extract sections
    sections = extract_hierarchical_sections(f.name)
    
    print(f"Sections found: {len(sections)}")
    
    for section in sections:
        print(f"\nTitle: {section['title']}")
        print(f"Content length: {len(section['content'])}")
        print(f"Content: {section['content']}")
        
        # Check subsections
        if 'subsections' in section and section['subsections']:
            for subsection in section['subsections']:
                print(f"\n  Sub-Title: {subsection['title']}")
                print(f"  Sub-Content length: {len(subsection['content'])}")
                print(f"  Sub-Content: {subsection['content']}") 