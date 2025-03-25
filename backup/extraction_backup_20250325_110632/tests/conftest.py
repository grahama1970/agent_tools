"""
Common test configuration and fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import os
import sys
import pytest
from pathlib import Path
import tempfile
import shutil

# Configure path correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test repository paths
REPOS_DIR = project_root / "test_repos"
REQUESTS_REPO = REPOS_DIR / "requests"
REACT_REPO = REPOS_DIR / "react"
RUST_ANALYZER_REPO = REPOS_DIR / "rust-analyzer"

@pytest.fixture(scope="session")
def project_root():
    """Fixture to provide the project root path."""
    return project_root

@pytest.fixture(scope="session")
def test_repos_dir():
    """Fixture to provide the test repositories directory."""
    return REPOS_DIR

@pytest.fixture(scope="session")
def requests_repo():
    """Fixture to provide the requests repository path."""
    if not REQUESTS_REPO.exists():
        pytest.skip("Requests repository not available")
    return REQUESTS_REPO

@pytest.fixture(scope="session")
def react_repo():
    """Fixture to provide the React repository path."""
    if not REACT_REPO.exists():
        pytest.skip("React repository not available")
    return REACT_REPO

@pytest.fixture(scope="session")
def rust_analyzer_repo():
    """Fixture to provide the Rust Analyzer repository path."""
    if not RUST_ANALYZER_REPO.exists():
        pytest.skip("Rust Analyzer repository not available")
    return RUST_ANALYZER_REPO

@pytest.fixture
def temp_dir():
    """Fixture to provide a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def stats_dict():
    """Fixture to provide an initialized stats dictionary."""
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats
    return init_stats()

@pytest.fixture
def python_sample():
    """Fixture to provide sample Python code."""
    return '''
def hello_world():
    """Simple function that returns a greeting."""
    return "Hello, world!"

class SampleClass:
    """A sample class with methods."""
    
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        return f"Hello, {self.name}!"
'''

@pytest.fixture
def javascript_sample():
    """Fixture to provide sample JavaScript code."""
    return '''
/**
 * Simple function that returns a greeting.
 */
function helloWorld() {
    return "Hello, world!";
}

/**
 * A sample class for greeting.
 */
class Greeter {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return `Hello, ${this.name}!`;
    }
}

// Export the function and class
module.exports = {
    helloWorld,
    Greeter
};
'''

@pytest.fixture
def markdown_sample():
    """Fixture to provide sample Markdown content."""
    return '''
# Sample Document

This is a sample markdown document.

## Code Section

Here's a Python code block:

```python
def example_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

And a JavaScript block:

```javascript
function example() {
    return "Hello, world!";
}
```
''' 