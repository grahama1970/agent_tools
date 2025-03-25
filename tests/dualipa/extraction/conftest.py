"""
Test fixtures for extraction module.

This file contains pytest fixtures used across multiple test files.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_code_file(tmp_path):
    """Create a sample Python file for testing."""
    file_path = tmp_path / "sample.py"
    with open(file_path, "w") as f:
        f.write("""
def example_function():
    \"\"\"Example function for testing.\"\"\"
    return 42

class ExampleClass:
    \"\"\"Example class for testing.\"\"\"
    
    def __init__(self):
        self.value = 10
        
    def get_value(self):
        \"\"\"Get the value.\"\"\"
        return self.value
""")
    return file_path

@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a sample Markdown file for testing."""
    file_path = tmp_path / "sample.md"
    with open(file_path, "w") as f:
        f.write("""# Sample Markdown
        
This is a sample markdown file for testing.

## Section 1

Some content in section 1.

```python
def example():
    return "Hello World"
```

## Section 2

More content in section 2.
""")
    return file_path
