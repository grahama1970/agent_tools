# Playwright Support for JavaScript-Rendered Sites

This document explains how to use the Playwright support for fetching JavaScript-rendered websites in the DuaLipa extraction pipeline.

## Overview

Modern websites often use JavaScript to render content dynamically, which can make traditional web scraping tools like `wget` ineffective. The Playwright integration in the fetch_docs module addresses this issue by providing a headless browser automation solution that can:

1. Fully render JavaScript-based websites
2. Handle modern SPA (Single Page Application) frameworks
3. Wait for dynamic content to load
4. Navigate through JavaScript-rendered page structures
5. Extract content that is generated or modified by client-side scripts

## Requirements

To use the Playwright functionality, you need to install:

```bash
# Install Python package
pip install playwright

# Install browser dependencies
playwright install
```

On Linux systems, you may need additional dependencies. If you see warnings about missing libraries, install them with:

```bash
# Ubuntu/Debian
sudo apt-get install libgtk-3-0 libnotify4 libnss3 libxss1 libasound2

# CentOS/RHEL
sudo yum install gtk3 libnotify nss libXScrnSaver alsa-lib
```

## Usage

### Command-Line Arguments

The testing scripts now support a `--playwright` flag to enable Playwright-based downloading:

```bash
# Run ArangoDB test with Playwright
python test_arangodb_extraction_transparent.py --playwright

# Run all tests with Playwright
python run_transparent_tests.py --playwright
```

### Programmatic Usage

To use Playwright programmatically in your code:

```python
from agent_tools.fetch_docs.download_site import download_site, download_site_with_playwright

# Option 1: Let the system decide when to use Playwright
download_site("https://example.com", "output_dir", use_playwright=True)

# Option 2: Use Playwright directly
stats = download_site_with_playwright(
    "https://example.com", 
    "output_dir",
    wait_time=5,           # Seconds to wait for JavaScript rendering
    recursive=True,        # Follow links on the page
    max_depth=2,           # How deep to follow links
    timeout=30000          # Timeout in milliseconds
)
```

## Implementation Details

The Playwright implementation includes:

1. **Graceful Fallback**: If Playwright download fails, it falls back to wget
2. **Resource Handling**: Downloads CSS and JavaScript files needed for rendering
3. **Recursive Downloading**: Can follow links to download multiple pages
4. **Content Waiting**: Waits for network activity to complete before capturing content
5. **Error Handling**: Comprehensive error handling and logging

## Testing

A dedicated test script is provided to verify Playwright functionality:

```bash
python test_playwright_fetch.py https://example.com
```

This will download the specified URL using Playwright and save diagnostic information to help troubleshoot any issues.

## Known Limitations

1. **Browser Dependencies**: Playwright requires system libraries that may not be available in all environments
2. **Performance**: Playwright is slower than wget for simple static sites
3. **Memory Usage**: Higher memory requirements compared to wget
4. **External Resources**: Some resources may still fail to load due to CORS restrictions

## When to Use Playwright

Playwright should be used when:

1. The target site heavily depends on JavaScript for content rendering
2. Traditional wget-based scraping returns incomplete content
3. The site uses modern frameworks like React, Vue, or Angular
4. You need to interact with the page before extracting content

For simple static sites, wget is faster and more efficient.

## Debugging

If you encounter issues with Playwright:

1. Check the `playwright_stats.json` file in the output directory for error details
2. Increase the `wait_time` parameter to allow more time for JavaScript rendering
3. Look for missing system dependencies in console messages
4. Try a direct test with `test_playwright_fetch.py` to isolate the issue

## Future Improvements

Planned enhancements to the Playwright integration:

1. Better content interaction capabilities (clicking, scrolling)
2. Improved handling of authentication-required sites
3. Support for capturing screenshots for verification
4. Performance optimizations for large sites