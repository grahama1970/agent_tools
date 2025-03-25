# Documentation Extraction Verification

This document explains how to run the transparent verification tests for the fetch_docs integration with dualipa. These tests download HTML documentation from websites, process it through the extraction pipeline, and create human-readable verification artifacts.

## Key Features

- Downloads and saves the original HTML from documentation sites
- Extracts and saves the processed blocks in JSON format
- Creates HTML reports with side-by-side comparisons of input and output
- Shows statistics on extracted blocks (sections, code blocks, tables, etc.)
- Provides shell commands for further inspection of results
- **NEW**: Integrated web server for remote access to HTML reports
- **NEW**: Automatic Tailscale IP detection for secure report access

## Available Test Scripts

### Combined Test Runner

The `run_transparent_tests.py` script runs all tests in sequence and creates a single summary report:

#### Basic Usage

```bash
python run_transparent_tests.py --output-dir test_results
```

This will:
1. Run both ArangoDB and ReadTheDocs extraction tests
2. Create a combined summary report
3. Save all results in one organized directory
4. Attempt to open the summary in your browser (local machine only)

#### Remote Access with Web Server

For remote access (e.g., from WSL2 or a server), you have two options:

#### 1. Docker-based Approach (Recommended)

Use the `--docker-serve` option to serve results using a Docker container with Nginx:

```bash
python run_transparent_tests.py --output-dir test_results --docker-serve
```

This will:
1. Run all tests as normal
2. Start a Docker Nginx container to host the report files
3. Display URLs for accessing the reports
4. Container will run in the background until explicitly stopped

You can also specify a custom port:

```bash
python run_transparent_tests.py --output-dir test_results --docker-serve --port 8080
```

**Advantages of Docker-based approach:**
- Works consistently across all environments (Linux, macOS, WSL2)
- Automatically handles WSL2-Windows networking
- No manual port forwarding required
- Better performance for serving HTML files
- Container continues running even after the script exits

**Managing the Docker container:**

With Docker Compose (recommended):
```bash
# Stop the container
docker-compose down

# View logs
docker-compose logs

# Restart with different settings
PORT=9000 RESULTS_DIR=/path/to/results docker-compose up -d
```

Without Docker Compose:
```bash
# Stop the container
docker stop dualipa-test-reports

# View logs
docker logs dualipa-test-reports
```

#### 2. Built-in Python Server

Use the `--serve` option to use the built-in Python HTTP server:

```bash
python run_transparent_tests.py --output-dir test_results --serve
```

This will:
1. Run all tests as normal
2. Start a Python HTTP server to host the report files
3. Display URLs for accessing the reports, including:
   - Local network URL
   - Tailscale URL (if Tailscale is installed)
4. Keep the server running until you press Ctrl+C

You can also specify a custom port:

```bash
python run_transparent_tests.py --output-dir test_results --serve --port 8080
```

### Individual Test Scripts

You can also run tests individually:

#### ArangoDB Test

Tests extraction from ArangoDB's AQL documentation:

```bash
python test_arangodb_extraction_transparent.py --output-dir test_results/arangodb
```

#### ReadTheDocs Test

Tests extraction from Python's ReadTheDocs documentation:

```bash
python test_readthedocs_extraction_transparent.py --output-dir test_results/readthedocs
```

## Verification Process

The test output includes several artifacts for human verification:

1. **Original HTML:** The raw HTML downloaded from the documentation site
2. **Extracted Blocks:** The JSON output from the extraction pipeline
3. **HTML Summary:** A visual report showing:
   - Statistics on extracted blocks
   - Side-by-side comparisons of HTML input and JSON output
   - Links to all generated files
   - Sample commands for further inspection

## Understanding the Results

The HTML reports are designed to help you verify that:

1. **All block types are extracted:** Documentation site, pages, sections, code blocks, and tables
2. **Hierarchical relationships are preserved:** Parent-child relationships between blocks
3. **Content is properly extracted:** Headers, code samples, tables, etc.

## Example Commands for Inspection

After running a test, you can use these commands to inspect the results:

```bash
# Count blocks by type
cat test_results/arangodb/arangodb_blocks.json | grep "type" | sort | uniq -c

# Check all code blocks
cat test_results/arangodb/arangodb_blocks.json | jq '.[] | select(.type == "code_block")'

# Examine section hierarchy
cat test_results/arangodb/arangodb_blocks.json | jq '.[] | select(.type == "doc_section") | {name: .name, header_level: .metadata.header_level}'
```

## Requirements

- Python 3.8+
- BeautifulSoup4 (for HTML parsing)
- lxml (required HTML parser for BeautifulSoup)
- Dependencies from agent_tools.fetch_docs and agent_tools.dualipa modules

### Environment Setup

Ensure your environment is properly set up:

```bash
# Install required dependencies
uv add beautifulsoup4 lxml loguru spacy

# Download spacy model if needed
python -m spacy download en_core_web_sm

# Set PYTHONPATH to include both src and tests directories
export PYTHONPATH=/path/to/agent_tools/src:/path/to/agent_tools/tests
```

**Important**: The tests expect absolute imports from the agent_tools package. Make sure all dependent modules can be accessed through the PYTHONPATH.

## Troubleshooting

### General Issues

If downloads fail, the tests will create fallback HTML files with minimal content to allow testing to continue.

If you encounter the error "Could not import download_site function", make sure the fetch_docs module is in your Python path, or use the local download_site_patch.py which is included as a fallback.

### Remote Access with Web Server

The test runner includes an integrated web server that works in both regular Linux environments and WSL2. To start the server, use the `--serve` option:

```bash
python run_transparent_tests.py --output-dir test_results --serve
```

#### Using in WSL2 Environments

When running in WSL2, the test runner automatically detects this environment and provides special instructions:

1. **Accessing from the same Windows machine**:
   - Use the URLs provided under "Windows host access"
   - These use the WSL2 VM's IP address which is accessible from the Windows host

2. **Accessing from other computers on your network**:
   - If the Windows host IP is detected, those URLs will be shown
   - You may need to set up port forwarding on Windows (instructions will be displayed)

3. **Port forwarding setup (if needed)**:
   From a Windows PowerShell (Admin), run commands like:
   ```powershell
   # Replace 8765 with the actual port number displayed
   # Replace 172.x.x.x with the actual WSL2 IP displayed
   netsh interface portproxy add v4tov4 listenport=8765 listenaddress=0.0.0.0 connectport=8765 connectaddress=172.x.x.x
   
   # Add firewall rule
   New-NetFirewallRule -DisplayName "WSL2 Port 8765" -Direction Inbound -LocalPort 8765 -Action Allow -Protocol TCP
   ```

### Troubleshooting Remote Access

#### Docker-based Approach (Recommended)

1. **Docker container fails to start**:
   - Check if Docker is running with `docker ps`
   - Verify permission to access the results directory
   - Check if the port is already in use with `sudo lsof -i :PORT`
   - Try a different port with `--port XXXX`

2. **Cannot access the Docker container from Windows when using WSL2**:
   - Ensure Docker Desktop is running on Windows
   - Check Docker Desktop settings to confirm WSL2 integration is enabled
   - Try accessing via localhost instead of IP address

3. **Docker container stops unexpectedly**:
   - Check the container logs with `docker logs dualipa-test-reports`
   - Restart the container with `docker start dualipa-test-reports`
   - Recreate the container using the docker-serve.sh script directly:
     ```bash
     ./docker-serve.sh --directory /path/to/results --port 8765
     ```

#### Built-in Python Server

1. **Cannot access server from another machine**:
   - Ensure you're using the correct IP address (local network IP or Tailscale IP)
   - Check if firewall rules are blocking the port
   - Try specifying a different port with `--port XXXX`
   - Consider using the Docker-based approach which handles networking better

2. **Tailscale links not appearing**:
   - Verify Tailscale is installed and running with `tailscale status`
   - Ensure you're logged into your Tailscale account
   - Check connectivity with `ping your-tailscale-ip` from another device

3. **Server terminates unexpectedly**:
   - Run with `python run_transparent_tests.py --serve > server.log 2>&1 &` to keep it running in the background
   - Check `server.log` for any error messages
   - Consider switching to the Docker-based approach which is more reliable

4. **WSL2-specific issues**:
   - For WSL2, the Docker-based approach is strongly recommended as it handles Windows-WSL2 networking automatically
   - If using the Python server, ensure Windows Firewall allows connections to the WSL2 instance
   - If using VS Code Remote, forward the port in VS Code's "Ports" tab
   - Try accessing via the Windows host's Tailscale IP if direct access fails