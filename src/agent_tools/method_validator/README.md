# ✅ Method Validator - AI Agent's API Discovery Tool

A specialized tool for AI agents to autonomously analyze Python packages, discover existing methods, and validate APIs before implementing new solutions. This tool helps prevent redundant code creation and method hallucination by identifying existing functionality in non-standard packages.

## For Users

To ensure the AI agent properly validates methods and checks for duplicated functionality, always use this specific prompt format:

```
TOOL: method_validator - Implement [task description]
```

Example:

```
TOOL: method_validator - Write a function to extract tables from a webpage using Playwright
```

Why this format?

- Forces the agent to use the method validator tool
- Prevents method hallucination
- Ensures verification before code presentation
- Avoids duplicating existing functionality
- Cursor's .cursorrules is (often) ignored by Cursor 🤷‍♂️

The agent will:

1. Draft the implementation (without showing it)
2. Verify all methods exist
3. Check for duplicate functionality
4. Only present code after verification
5. Include verification markers:
   ```
   ✓ All methods verified to exist
   ✓ No duplicate functionality found
   Here's the implementation:
   [code]
   ```

## For AI Agents

When a user requests code implementation:

1. Draft your implementation (do not show yet)
2. Extract ALL methods you plan to use
3. Run verification:
   ```bash
   method-validator package_name --list-all  # Check for existing functionality
   method-validator package_name --method method_name  # For each method
   ```
4. If verification fails:
   - Remove non-existent methods
   - Fix duplicated functionality
5. Only present code after verification with confirmation markers

## Features

- **Smart Package Analysis**: Automatically filters out standard library and common utility packages
- **Method Discovery**: Quick scanning of available methods with categorization
- **Detailed Analysis**: In-depth examination of method signatures, parameters, and return types
- **Method Relationships**: Automatic detection of related methods (sync/async variants, specialized versions)
- **Smart Categorization**: Automatic grouping of methods by functionality, provider, and execution mode
- **Exception Analysis**: Identifies and prioritizes relevant error handling patterns
- **Machine-Readable Output**: JSON format support for automated processing
- **Optimized Caching**: SQLite-based caching with zlib compression for efficient storage
- **Progress Visualization**: Real-time progress bars for human observers

## Installation

### As a Development Tool

```bash
# Clone the repository
git clone https://github.com/your-org/method-validator.git
cd method-validator

# Install in development mode
uv pip install -e .
```

### As a Package

```bash
# Install directly from PyPI
uv pip install method-validator

# Or using uv for better dependency resolution
uv pip install method-validator
```

## Usage

### For AI Agents

```python
from method_validator import MethodAnalyzer

def agent_function():
    analyzer = MethodAnalyzer()

    # Quick check if functionality exists
    methods = analyzer.quick_scan("target_package")

    # Get detailed info about a specific method
    method_info = analyzer.deep_analyze("target_package", "method_name")

    # Access categorization and relationships
    categories = method_info["categories"]  # e.g., ["async", "completion", "provider:openai"]
    related_methods = method_info["related_methods"]  # e.g., sync/async variants

    # Make intelligent decisions based on relationships
    if "async" in categories:
        sync_variant = next((r["name"] for r in related_methods if r["type"] == "sync_variant"), None)
        if sync_variant:
            # Use sync variant if needed
            pass
```

### Command Line

```bash
# Basic method analysis
method-validator package_name --method method_name --json

# List all available methods
method-validator package_name --list-all --json

# Get exception information
method-validator package_name --exceptions-only --json

# Cache management
method-validator --cache-stats  # Show cache statistics
method-validator --cache-clear  # Clear the cache
method-validator --compression-level 9  # Set maximum compression
method-validator --recompress  # Recompress all cached data
```

### Command Line Options

- `--method`: Analyze a specific method
- `--list-all`: Show all available methods
- `--by-category`: Group methods by category
- `--show-exceptions`: Show detailed exception information
- `--exceptions-only`: Focus on exception analysis
- `--json`: Output in JSON format for machine consumption
- `--timing`: Show execution timing statistics

### Cache Management Options

- `--cache-stats`: Show detailed cache statistics
- `--cache-clear`: Clear the entire cache
- `--cache-migrate`: Migrate from old cache location
- `--cache-max-age`: Set maximum cache entry age in days
- `--cache-max-size`: Set maximum cache size in MB
- `--compression-level`: Set compression level (0-9)
- `--recompress`: Recompress all cached data

## Example Output

```json
{
  "method_info": {
    "name": "example_method",
    "signature": "(param1: str, param2: Optional[int] = None) -> Dict[str, Any]",
    "summary": "Example method description",
    "parameters": {
      "param1": {
        "type": "str",
        "required": true,
        "description": "First parameter description"
      }
    },
    "categories": ["async", "completion", "provider:openai"],
    "related_methods": [
      {
        "name": "example_method_sync",
        "type": "sync_variant"
      },
      {
        "name": "example_method_streaming",
        "type": "specialized_variant"
      }
    ],
    "exceptions": [
      {
        "type": "ValueError",
        "description": "When invalid input is provided"
      }
    ]
  }
}
```

## Key Features for AI Agents

1. **Autonomous Operation**:

   - Smart filtering of packages
   - Machine-readable output format
   - Persistent compressed caching
   - Automatic method categorization
   - Relationship detection between methods

2. **Focused Analysis**:

   - Prioritizes relevant methods and parameters
   - Filters out internal/private methods
   - Highlights commonly used parameters
   - Groups related functionality
   - Identifies method variants and alternatives

3. **Error Handling Intelligence**:

   - Identifies custom exceptions
   - Prioritizes well-documented error cases
   - Provides exception hierarchy information

4. **Performance Optimizations**:
   - SQLite-based persistent caching
   - zlib compression for efficient storage
   - Source code change detection
   - Smart categorization caching
   - Automatic cache cleanup

## Method Categories

The tool automatically categorizes methods based on:

1. **Operation Type**:

   - `completion`: Completion-related operations
   - `embedding`: Embedding generation
   - `streaming`: Streaming operations

2. **Execution Mode**:

   - `async`: Asynchronous methods
   - `sync`: Synchronous methods

3. **Provider Specific**:

   - `provider:openai`
   - `provider:anthropic`
   - `provider:vertex`
   - etc.

4. **Special Features**:
   - `batch`: Batch processing
   - `chat`: Chat-based operations
   - `text`: Text-based operations

## Method Relationships

The tool identifies several types of relationships:

1. **Execution Variants**:

   - Sync/Async pairs
   - Streaming variants
   - Batch variants

2. **Specialized Versions**:
   - Provider-specific implementations
   - Feature-specific variants
   - Optimized versions

## Best Practices

- Only analyze non-standard packages directly relevant to the task
- Use `--json` flag for machine-readable output
- Leverage exception analysis for robust error handling
- Focus on well-documented and commonly used methods
- Cache is stored in `analysis/method_analysis_cache.db` within the package

## Analysis Directory Structure

```
method_validator/
├── analysis/                      # Analysis and cache directory
│   ├── method_analysis_cache.db   # SQLite cache with compression
│   ├── package1_analysis.json     # Analysis results
│   └── package2_analysis.json
├── examples/                      # Example integrations
│   └── example_integration.py
└── ...
```

## Cache Management

The tool uses an optimized caching system with the following features:

1. **Compressed Storage**:

   - Uses zlib compression (levels 0-9)
   - Automatic compression for entries > 1KB
   - Configurable compression levels
   - Recompression support for existing entries

2. **Cleanup Policies**:

   - Age-based cleanup (default: 30 days)
   - Size-based cleanup (default: 100MB)
   - Source code change detection
   - Automatic invalid entry removal

3. **Statistics and Monitoring**:

   - Entry counts and sizes
   - Per-package statistics
   - Compression ratios
   - Age distribution

4. **Management Commands**:
   - Cache statistics viewing
   - Manual cleanup options
   - Compression level control
   - Cache migration support

## Limitations

- Does not analyze standard library packages
- Skips common utility packages (requests, urllib3, etc.)
- Limited to Python packages installed in the environment
- Cache is local to each installation of the tool

## Contributing

Contributions to improve the tool's AI agent capabilities are welcome. Please ensure any changes maintain or enhance the tool's autonomous operation capabilities.

## License

[MIT License](LICENSE)

## Agent-Tool Interaction Flow

Here's a typical interaction between a user and an AI agent using the Method Validator:

```
USER: Write a script that uses LiteLLM's acompletion function to generate text asynchronously.

AGENT: Let me help you write that script. First, I should verify the exact signature and capabilities
of LiteLLM's acompletion function to ensure we use it correctly.

[INTERNAL THOUGHT: Need to check if this method exists and understand its parameters]
```

1. **Initial Method Discovery**:

```python
# Agent uses Method Validator to verify method existence
analyzer = MethodAnalyzer()
methods = analyzer.quick_scan("litellm")

# Deep analyze the specific method (uses cache if available)
method_info = analyzer.deep_analyze("litellm", "acompletion")
```

2. **Cached Analysis Result**:

```json
{
  "name": "acompletion",
  "signature": "(model: str, messages: List[Dict], api_key: Optional[str] = None, ...) -> AsyncResponse",
  "categories": ["async", "completion", "core"],
  "parameters": {
    "model": {
      "type": "str",
      "required": true,
      "description": "Name of the model to use"
    },
    "messages": {
      "type": "List[Dict]",
      "required": true,
      "description": "List of message dictionaries"
    }
  },
  "related_methods": [
    {
      "name": "completion",
      "type": "sync_variant"
    }
  ]
}
```

3. **Agent Implementation**:

```
AGENT: I've analyzed the LiteLLM package and found the acompletion function. Here's a script using
the correct parameters:
```

```python
import asyncio
from litellm import acompletion

async def generate_text(prompt: str, model: str = "gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = await acompletion(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
```

4. **User Request for Enhancement**:

```
USER: Yes, add error handling. Also, can we stream the response?

AGENT: Let me check the streaming capabilities and exception patterns in LiteLLM...
```

5. **Additional Analysis**:

```python
# Agent looks for streaming variants
methods = [
    name for name, summary, cats in analyzer.quick_scan("litellm")
    if "stream" in name.lower() and "async" in cats
]

# Analyze streaming functionality
stream_info = analyzer.deep_analyze("litellm", "acompletion_stream")
```

6. **Enhanced Implementation**:

```python
async def generate_text_stream(prompt: str, model: str = "gpt-3.5-turbo") -> AsyncGenerator[str, None]:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            stream=True  # Found from analysis
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except APIError as e:  # Exception types from analysis
        print(f"API Error: {e}")
        raise
    except APIConnectionError as e:
        print(f"Connection Error: {e}")
        raise
```

### Key Benefits of This Flow:

1. **Verification Before Implementation**:

   - Agent verifies method existence before writing code
   - Prevents hallucination of non-existent methods
   - Ensures correct parameter usage

2. **Intelligent Feature Discovery**:

   - Finds related methods (sync/async variants)
   - Discovers additional capabilities (streaming)
   - Identifies required exception handling

3. **Performance Optimization**:

   - Uses cached analysis results for quick verification
   - Avoids repeated package analysis
   - Makes informed decisions about method variants

4. **Error Prevention**:
   - Uses correct parameter types and structures
   - Implements proper exception handling
   - Avoids duplicating existing functionality

### When Agents Use the Tool:

1. **Initial Implementation**:

   - Before writing new functions
   - When using unfamiliar packages
   - To verify method existence

2. **Feature Enhancement**:

   - When adding new capabilities
   - Looking for alternative implementations
   - Finding related functionality

3. **Error Handling**:

   - Adding exception handling
   - Understanding error patterns
   - Finding package-specific errors

4. **Code Review**:
   - Verifying correct usage
   - Finding better alternatives
   - Checking for duplicated functionality
