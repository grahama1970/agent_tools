# Report Server Module

The `report_server` module provides utilities for serving HTML reports using Docker/Docker Compose or Python's built-in HTTP server. It's designed to be reusable by various components in the agent_tools package that need to display HTML-based reports.

## Features

- **Docker Integration**: Serves HTML reports using Nginx in a Docker container
- **Docker Compose Support**: Manages containers with Docker Compose
- **Fallback Python Server**: Alternative built-in Python HTTP server
- **Auto-Detection**: Detects WSL2 and network configurations automatically
- **Terminal Hyperlinks**: Creates clickable links that work in modern terminals

## Usage

### Basic Usage

```python
from agent_tools.report_server import serve_with_docker, create_terminal_link

# Start a Docker server to serve a directory
success, url = serve_with_docker('/path/to/reports', port=8765)

if success:
    # Create terminal hyperlinks to important files
    summary_link = create_terminal_link(f"{url}/summary.html", "Open Summary")
    print(f"Report available at: {summary_link}")
```

### With Docker Compose

```python
# Docker Compose is automatically used if available
success, url = serve_with_docker('/path/to/reports', port=8765, use_compose=True)
```

### Managing the Server

```python
from agent_tools.report_server import stop_docker_server

# Stop the server when done
stop_docker_server()
```

## Environment Detection

The module automatically detects:

- WSL2 environment
- Tailscale networking
- Docker/Docker Compose availability
- Local IP addresses

## Requirements

- Docker (optional but recommended)
- Docker Compose (optional)
- Python 3.6+

## Architecture

The module is organized into:

- `docker_server.py`: Docker and Docker Compose integration
- `utils.py`: Utility functions for networking and terminal links
- `docker-compose.yml`: Container configuration

## Example Integration

This module is used by the fetch_docs transparent testing utilities to serve HTML verification reports.

```python
# In a test runner script
from agent_tools.report_server import serve_with_docker

# Run tests and generate reports
results_dir = run_tests()

# Serve the reports
serve_with_docker(results_dir)
```