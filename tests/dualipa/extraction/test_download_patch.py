"""
Patch download_site functions for testing.

This module provides patched versions of the download_site and 
download_site_with_playwright functions that don't actually try to 
download any files, but instead create mock files for testing.
"""

import os
import sys
import json
from pathlib import Path


def mock_download_site(root_url: str, output_dir: str, recursive: bool = True, use_playwright: bool = False) -> bool:
    """
    Mock version of download_site that doesn't actually download anything.
    
    Instead, it creates a minimal directory structure with sample HTML files
    for testing.
    
    Args:
        root_url: URL that would normally be downloaded
        output_dir: Directory where mock files will be created
        recursive: Ignored in the mock version
        use_playwright: Whether to simulate Playwright usage
        
    Returns:
        bool: True to indicate success
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
    
    # Return success
    return True


def mock_download_site_with_playwright(
    url: str, 
    output_dir: str, 
    wait_time: int = 5,
    recursive: bool = False,
    max_depth: int = 2,
    timeout: int = 30000
) -> dict:
    """
    Mock version of download_site_with_playwright that doesn't use a browser.
    
    Instead, it creates a minimal directory structure with sample HTML files
    that simulates what Playwright would download, including JavaScript-rendered
    content.
    
    Args:
        url: URL that would normally be downloaded
        output_dir: Directory where mock files will be created
        wait_time: Ignored in the mock version
        recursive: Whether to create mock linked pages
        max_depth: How many levels of links to simulate
        timeout: Ignored in the mock version
        
    Returns:
        dict: Statistics about the mock download process
    """
    print(f"Mock Playwright downloading site: {url}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get domain from URL
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    base_path = parsed_url.path
    
    # Statistics
    stats = {
        "success": True,
        "pages_downloaded": 0,
        "errors": 0,
        "pages": {}
    }
    
    # Create domain directory
    site_dir = output_path / domain
    if base_path and base_path != "/":
        for part in base_path.strip("/").split("/"):
            if part:
                site_dir = site_dir / part
    
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mock index.html with JavaScript-rendered content
    html_content = ""
    is_recursive = False
    
    if "readthedocs.io" in url or "readthedocs.org" in url:
        # Create ReadTheDocs mock structure
        html_content = f"""<!DOCTYPE html>
<html>
<head><title>Python Documentation (JavaScript Rendered)</title></head>
<body>
  <div class="content">
    <h1>Python Documentation</h1>
    <p>This is a mock JavaScript-rendered documentation page for {url}</p>
    <div id="js-content" class="js-rendered">
      <h2>Python Tutorial</h2>
      <p>Python is an easy to learn, powerful programming language.</p>
      <pre><code class="language-python">
      async def hello():
          await asyncio.sleep(1)
          print("Hello, World!")
          
      await hello()
      </code></pre>
    </div>
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
  <div class="navigation">
    <a href="/tutorial/index.html">Tutorial</a>
    <a href="/library/index.html">Library</a>
  </div>
</body>
</html>"""
        is_recursive = True
        
    elif "arangodb.com" in url:
        # Create ArangoDB mock structure
        html_content = f"""<!DOCTYPE html>
<html>
<head><title>ArangoDB Documentation (JavaScript Rendered)</title></head>
<body>
  <div class="content">
    <h1>ArangoDB Query Language (AQL)</h1>
    <p>This is a mock JavaScript-rendered documentation page for {url}</p>
    <div id="js-content" class="js-rendered">
      <h2>AQL Functions</h2>
      <p>AQL provides a variety of built-in functions.</p>
      <pre><code class="language-javascript">
      // Modern JavaScript features rendered by Playwright
      async function getData() {{
          const cursor = await db.query(`
              FOR doc IN collection
                FILTER doc.value > 10
                RETURN doc
          `);
          return await cursor.all();
      }}
      </code></pre>
    </div>
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
  <div class="navigation">
    <a href="/aql/functions.html">Functions</a>
    <a href="/aql/operations.html">Operations</a>
  </div>
</body>
</html>"""
        is_recursive = True
        
    else:
        # Create a generic mock index.html with JavaScript rendering
        html_content = f"""<!DOCTYPE html>
<html>
<head><title>Mock JavaScript Documentation</title></head>
<body>
  <div class="content">
    <h1>Documentation</h1>
    <p>This is a mock JavaScript-rendered documentation page for {url}</p>
    <div id="js-content" class="js-rendered">
      <!-- This content would only be visible after JavaScript execution -->
      <h2>Dynamically Rendered Section</h2>
      <p>This section was rendered with JavaScript.</p>
      <pre><code class="language-javascript">
      // Dynamic JavaScript example
      const data = await fetch('/api/data');
      const result = await data.json();
      renderContent(result);
      </code></pre>
    </div>
    <h2>Static Section</h2>
    <p>This section would be visible without JavaScript.</p>
  </div>
  <div class="navigation">
    <a href="/page1.html">Page 1</a>
    <a href="/page2.html">Page 2</a>
  </div>
</body>
</html>"""
        is_recursive = True
    
    # Save the HTML
    html_file = site_dir / "index.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    stats["pages_downloaded"] += 1
    stats["pages"][url] = {
        "path": str(html_file),
        "size": len(html_content)
    }
    
    print(f"Created mock Playwright file: {html_file}")
    
    # Create mock resources directory and files
    resource_dir = site_dir / "resources"
    resource_dir.mkdir(exist_ok=True)
    
    # Create mock CSS
    css_file = resource_dir / "style.css"
    with open(css_file, 'w', encoding='utf-8') as f:
        f.write("""/* Mock CSS file for JavaScript-rendered page */
.js-rendered {
  display: block;
  padding: 10px;
  border: 1px solid #ccc;
  background-color: #f8f8f8;
}

h1 {
  color: #0066cc;
}

pre {
  background-color: #eee;
  padding: 10px;
}""")
    
    # Create mock JavaScript
    js_file = resource_dir / "script.js"
    with open(js_file, 'w', encoding='utf-8') as f:
        f.write("""// Mock JavaScript file for rendered page
document.addEventListener('DOMContentLoaded', function() {
  console.log('JavaScript executed on page load');
  
  // Dynamic content generation would happen here
  const jsContent = document.getElementById('js-content');
  if (jsContent) {
    console.log('Found JS content section');
  }
});
""")
    
    print(f"Created mock resources in: {resource_dir}")
    
    # If recursive and max_depth > 0, create linked pages
    if recursive and max_depth > 0 and is_recursive:
        # Create a few linked pages based on the navigation links
        subpages = []
        
        if "readthedocs.io" in url or "readthedocs.org" in url:
            subpages = [
                {"path": "tutorial/index.html", "title": "Python Tutorial"},
                {"path": "library/index.html", "title": "Python Library Reference"}
            ]
        elif "arangodb.com" in url:
            subpages = [
                {"path": "aql/functions.html", "title": "AQL Functions"},
                {"path": "aql/operations.html", "title": "AQL Operations"}
            ]
        else:
            subpages = [
                {"path": "page1.html", "title": "Page 1"},
                {"path": "page2.html", "title": "Page 2"}
            ]
        
        # Create up to 2 subpages
        for subpage in subpages[:2]:
            subpage_path = subpage["path"]
            subpage_title = subpage["title"]
            subpage_url = f"{parsed_url.scheme}://{domain}/{subpage_path}"
            
            # Create directory structure for subpage
            subpage_dir = output_path / domain
            for part in subpage_path.split('/')[:-1]:
                if part:
                    subpage_dir = subpage_dir / part
                    subpage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subpage HTML
            subpage_html = f"""<!DOCTYPE html>
<html>
<head><title>{subpage_title} (JavaScript Rendered)</title></head>
<body>
  <div class="content">
    <h1>{subpage_title}</h1>
    <p>This is a mock JavaScript-rendered subpage at {subpage_url}</p>
    <div id="js-content" class="js-rendered">
      <h2>JavaScript Content</h2>
      <p>This content was rendered with JavaScript.</p>
    </div>
    <h2>Static Content</h2>
    <p>This is static content.</p>
  </div>
  <div class="navigation">
    <a href="/index.html">Main Page</a>
  </div>
</body>
</html>"""
            
            # Save subpage
            subpage_file = subpage_dir / subpage_path.split('/')[-1]
            with open(subpage_file, 'w', encoding='utf-8') as f:
                f.write(subpage_html)
            
            stats["pages_downloaded"] += 1
            stats["pages"][subpage_url] = {
                "path": str(subpage_file),
                "size": len(subpage_html)
            }
            
            print(f"Created mock subpage: {subpage_file}")
    
    # Save stats
    stats_file = output_path / "download_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("Mock Playwright download completed successfully.")
    
    return stats