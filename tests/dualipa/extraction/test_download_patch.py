"""
Patch download_site function for testing.

This module provides a patched version of the download_site function
that doesn't actually try to download any files, but instead creates
mock files for testing.
"""

import os
import sys
from pathlib import Path


def mock_download_site(root_url: str, output_dir: str, recursive: bool = True) -> None:
    """
    Mock version of download_site that doesn't actually download anything.
    
    Instead, it creates a minimal directory structure with sample HTML files
    for testing.
    
    Args:
        root_url: URL that would normally be downloaded
        output_dir: Directory where mock files will be created
        recursive: Ignored in the mock version
    """
    print(f"Mock downloading site: {root_url}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse the URL to create a sensible directory structure
    if "readthedocs.io" in root_url or "readthedocs.org" in root_url:
        # Create ReadTheDocs mock structure
        site_parts = root_url.split('/')
        if len(site_parts) >= 3:
            domain = site_parts[2]  # e.g., python.readthedocs.io
            
            # Create domain directory
            domain_dir = output_path / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Add other path components
            current_dir = domain_dir
            for part in site_parts[3:]:
                if part:
                    current_dir = current_dir / part
                    current_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a mock index.html
            with open(current_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head><title>Python Documentation (Mock)</title></head>
<body>
  <div class="content">
    <h1>Python Documentation</h1>
    <p>This is a mock documentation page for {root_url}</p>
    <h2>Python Tutorial</h2>
    <p>Python is an easy to learn, powerful programming language.</p>
    <pre><code class="language-python">
    def hello():
        print("Hello, World!")
        
    hello()
    </code></pre>
    <h2>Python Modules</h2>
    <p>Python has a rich standard library of modules.</p>
    <h3>The math Module</h3>
    <p>The math module provides mathematical functions.</p>
    <table>
      <tr><th>Function</th><th>Description</th></tr>
      <tr><td>math.sqrt(x)</td><td>Return the square root of x</td></tr>
      <tr><td>math.pi</td><td>Mathematical constant π</td></tr>
    </table>
  </div>
</body>
</html>""")
                
            print(f"Created mock ReadTheDocs file: {current_dir / 'index.html'}")
    
    elif "arangodb.com" in root_url:
        # Create ArangoDB mock structure
        site_parts = root_url.split('/')
        if len(site_parts) >= 3:
            domain = site_parts[2]  # e.g., docs.arangodb.com
            
            # Create domain directory
            domain_dir = output_path / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Add other path components
            current_dir = domain_dir
            for part in site_parts[3:]:
                if part:
                    current_dir = current_dir / part
                    current_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a mock index.html for ArangoDB
            with open(current_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head><title>ArangoDB Documentation (Mock)</title></head>
<body>
  <div class="content">
    <h1>ArangoDB Query Language (AQL)</h1>
    <p>This is a mock documentation page for {root_url}</p>
    <h2>AQL Functions</h2>
    <p>AQL provides a variety of built-in functions.</p>
    <pre><code class="language-javascript">
    FOR doc IN collection
      FILTER doc.value > 10
      RETURN doc
    </code></pre>
    <h2>AQL Syntax</h2>
    <p>AQL has a SQL-like syntax.</p>
    <h3>Keywords</h3>
    <p>The following keywords are reserved in AQL.</p>
    <table>
      <tr><th>Keyword</th><th>Description</th></tr>
      <tr><td>FOR</td><td>Iteration</td></tr>
      <tr><td>FILTER</td><td>Filtering</td></tr>
      <tr><td>RETURN</td><td>Result projection</td></tr>
    </table>
  </div>
</body>
</html>""")
                
            print(f"Created mock ArangoDB file: {current_dir / 'index.html'}")
    
    else:
        # Create a generic mock index.html
        site_parts = root_url.split('/')
        if len(site_parts) >= 3:
            domain = site_parts[2]
            
            # Create domain directory
            domain_dir = output_path / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a mock index.html
            with open(domain_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head><title>Mock Documentation</title></head>
<body>
  <div class="content">
    <h1>Documentation</h1>
    <p>This is a mock documentation page for {root_url}</p>
    <h2>Section 1</h2>
    <p>Content for section 1.</p>
    <h2>Section 2</h2>
    <p>Content for section 2.</p>
  </div>
</body>
</html>""")
                
            print(f"Created mock generic file: {domain_dir / 'index.html'}")
    
    print("Mock download completed successfully.")