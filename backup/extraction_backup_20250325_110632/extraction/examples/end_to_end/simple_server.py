#!/usr/bin/env python3
"""
Simple HTTP Server for Documentation Extraction Validation.

This script serves the current directory with a simple HTTP server,
making the validation reports accessible via a web browser.
"""

import os
import sys
import http.server
import socketserver
import logging
from pathlib import Path
import threading
import webbrowser
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_server")

# Simple HTTP request handler
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with logging and markdown support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info("%s - %s", self.address_string(), format % args)
    
    def do_GET(self):
        """Handle GET requests."""
        # Check if this is a markdown file
        if self.path.endswith('.md'):
            # Convert markdown to HTML
            try:
                try:
                    import markdown
                    has_markdown = True
                except ImportError:
                    has_markdown = False
                
                file_path = self.path.lstrip('/')
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if has_markdown:
                        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
                    else:
                        # Simple fallback for markdown if the package is not available
                        html_content = f"<pre>{content}</pre>"
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    # Wrap the markdown content in a simple HTML page
                    html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>{file_path}</title>
                        <style>
                            body {{
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                margin: 0;
                                padding: 20px;
                                color: #333;
                                max-width: 1200px;
                                margin: 0 auto;
                            }}
                            h1, h2, h3 {{
                                color: #2c3e50;
                            }}
                            pre {{
                                background-color: #f5f5f5;
                                padding: 10px;
                                border-radius: 5px;
                                overflow-x: auto;
                            }}
                            code {{
                                background-color: #f5f5f5;
                                padding: 2px 4px;
                                border-radius: 3px;
                            }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin-bottom: 20px;
                            }}
                            th, td {{
                                padding: 8px;
                                text-align: left;
                                border-bottom: 1px solid #ddd;
                            }}
                            th {{
                                background-color: #f2f2f2;
                            }}
                            .back-link {{
                                margin-bottom: 20px;
                            }}
                            .back-link a {{
                                color: #007bff;
                                text-decoration: none;
                            }}
                            .back-link a:hover {{
                                text-decoration: underline;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="back-link">
                            <a href="/">← Back to Home</a>
                        </div>
                        {html_content}
                    </body>
                    </html>
                    """
                    
                    self.wfile.write(html.encode('utf-8'))
                    return
            except Exception as e:
                logger.error(f"Error handling markdown file: {e}")
        
        # For all other files, use the default handler
        return super().do_GET()


def open_browser():
    """Open a browser after a short delay."""
    time.sleep(1)
    webbrowser.open('http://localhost:8000')


def main():
    """Main function."""
    try:
        PORT = 8000
        
        # Check if we're running in Docker
        in_docker = os.path.exists('/.dockerenv')
        
        # Change to script directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if validation reports exist
        if os.path.exists('./test_results_dashboard/summary.html'):
            index_url = './test_results_dashboard/summary.html'
        else:
            index_url = './index.html'
            
            # Create a simple index.html if it doesn't exist
            if not os.path.exists(index_url):
                with open(index_url, 'w', encoding='utf-8') as f:
                    f.write("""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Documentation Extraction Validation</title>
                        <style>
                            body {
                                font-family: Arial, sans-serif;
                                margin: 40px;
                                line-height: 1.6;
                            }
                            h1 {
                                color: #333;
                            }
                            ul {
                                list-style-type: square;
                            }
                            a {
                                color: #0066cc;
                                text-decoration: none;
                            }
                            a:hover {
                                text-decoration: underline;
                            }
                        </style>
                    </head>
                    <body>
                        <h1>Documentation Extraction Validation</h1>
                        <p>No validation reports found. Please run validation first.</p>
                        
                        <h2>Available Files:</h2>
                        <ul>
                            <li><a href="./VALIDATION_FRAMEWORK.md">Validation Framework Documentation</a></li>
                            <li><a href="./CHANGELOG.md">Changelog</a></li>
                        </ul>
                    </body>
                    </html>
                    """)
        
        # Setup server
        Handler = CustomHandler
        
        # Create server
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Server started at http://localhost:{PORT}/")
            
            if not in_docker:
                # Open browser if not in Docker
                threading.Thread(target=open_browser).start()
                
            # Keep serving
            httpd.serve_forever()
    
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()