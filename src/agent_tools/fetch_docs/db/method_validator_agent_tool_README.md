## 🛡️ STIX Data Import (Work in Progress)

## Getting Started

See [docs/DEV_SETUP.md](docs/DEV_SETUP.md) for environment setup instructions. For STIX-specific development, refer to [docs/STIX_IMPORT.md](docs/STIX_IMPORT.md).

The `sparta_stix_import.py` module provides experimental support for processing STIX-formatted cybersecurity intelligence data. This standalone feature is being developed in a feature branch before merging to main.

**Key Functionality**:

- 🚩 **STIX Data Pipeline**:
  - Downloads STIX JSON from configured URL
  - Normalizes and cleans raw STIX data
  - Stores objects/relationships in ArangoDB
  - Generates text summaries and QA pairs

**Current Capabilities**:

```text
1. Data Ingestion → 2. DB Insertion → 3. Relationship Mapping → 4. Knowledge Enhancement
```

**Prerequisites**:

```bash
# Requires ArangoDB running with configured credentials
export ARANGO_ROOT_PASSWORD=your_password_here
```

**Usage**:

```python
# From project root with active venv
python -m src.sparta_stix_import
```

⚠️ **Important Notes**:

- Currently depends on `.env` configuration
- Requires manual DB schema setup
- Async processing still being optimized
- Tested with Sparta STIX v2.1 samples

## Monorepo Structure

For details about the monorepo architecture and module organization, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

# Method Validator - AI Agent's API Discovery Tool

A specialized tool designed for AI agents to autonomously analyze Python packages, discover existing methods, and validate APIs before implementing new solutions. This tool helps prevent redundant code creation by identifying existing functionality in non-standard packages.

## Features

- **Smart Package Analysis**: Automatically filters out standard library and common utility packages
- **Method Discovery**: Quick scanning of available methods with categorization
- **Detailed Analysis**: In-depth examination of method signatures, parameters, and return types
- **Exception Analysis**: Identifies and prioritizes relevant error handling patterns
- **Machine-Readable Output**: JSON format support for automated processing
- **Virtual Environment Support**: Automatic detection and use of project virtual environments

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install loguru
```

## Usage

### For AI Agents

```python
# Basic method analysis
python method_validator.py package_name --method method_name --json

# List all available methods
python method_validator.py package_name --list-all --json

# Get exception information
python method_validator.py package_name --exceptions-only --json
```

### Command Line Options

- `--method`: Analyze a specific method
- `--list-all`: Show all available methods
- `--by-category`: Group methods by category
- `--show-exceptions`: Show detailed exception information
- `--exceptions-only`: Focus on exception analysis
- `--json`: Output in JSON format for machine consumption
- `--venv-path`: Specify virtual environment path (auto-detected by default)

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
   - Auto-detection of virtual environments
   - Smart filtering of packages
   - Machine-readable output format

2. **Focused Analysis**:
   - Prioritizes relevant methods and parameters
   - Filters out internal/private methods
   - Highlights commonly used parameters

3. **Error Handling Intelligence**:
   - Identifies custom exceptions
   - Prioritizes well-documented error cases
   - Provides exception hierarchy information

## Best Practices

- Only analyze non-standard packages directly relevant to the task
- Use `--json` flag for machine-readable output
- Leverage exception analysis for robust error handling
- Focus on well-documented and commonly used methods

## Limitations

- Does not analyze standard library packages
- Skips common utility packages (requests, urllib3, etc.)
- Limited to Python packages installed in the virtual environment

## Contributing

Contributions to improve the tool's AI agent capabilities are welcome. Please ensure any changes maintain or enhance the tool's autonomous operation capabilities.

## License

[MIT License](LICENSE)
