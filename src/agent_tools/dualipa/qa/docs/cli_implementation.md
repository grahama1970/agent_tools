# CLI Implementation

This document describes the Command Line Interface (CLI) implementation for the DuaLipa QA Generation Module. The CLI provides a user-friendly way to interact with the QA generation system directly from the command line.

## Overview

The CLI is implemented in the `__main__.py` module, enabling direct execution from the command line. It provides comprehensive options for controlling the QA generation process, configuring system parameters, and managing output.

## Usage

### Basic Usage

```bash
python -m agent_tools.dualipa.qa <input_file> <output_file>
```

### Full Options

```bash
python -m agent_tools.dualipa.qa [options] <input_file> <output_file>

Options:
  --model=<model>                 LLM model to use (default: gpt-3.5-turbo)
  --workers=<num>                 Number of worker threads (default: auto)
  --max-concurrent=<num>          Maximum concurrent requests (default: 4)
  --max-pairs=<num>               Maximum QA pairs per section (default: 5)
  --temperature=<temp>            Temperature for generation (default: 0.7)
  --bi-ratio=<ratio>              Bidirectional generation ratio (default: 0.3)
  --no-bidirectional              Disable bidirectional generation
  --no-monitoring                 Disable metrics collection
  --cache-dir=<dir>               Cache directory (default: system tmp)
  --verbose                       Enable verbose logging
  --debug                         Enable debug logging
  --version                       Show version and exit
  --help                          Show this help message and exit
```

## Examples

### Basic Processing

Process an extraction JSON file with default settings:

```bash
python -m agent_tools.dualipa.qa input.json output.json
```

### Advanced Configuration

Process with specific model, worker count, and pair limit:

```bash
python -m agent_tools.dualipa.qa \
  --model=gpt-4 \
  --workers=8 \
  --max-pairs=10 \
  input.json output.json
```

### Performance Tuning

Configure for high-performance processing of large documents:

```bash
python -m agent_tools.dualipa.qa \
  --workers=16 \
  --max-concurrent=20 \
  --cache-dir=/path/to/cache \
  --no-bidirectional \
  large_input.json output.json
```

### Debugging

Enable detailed debugging output:

```bash
python -m agent_tools.dualipa.qa \
  --debug \
  --model=gpt-3.5-turbo \
  input.json output.json
```

## Implementation Details

### Command-Line Argument Parsing

The CLI uses Python's `argparse` library to parse command-line arguments. This provides a robust framework for defining options, validating inputs, and generating help documentation.

```python
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="dualipa-qa",
        description="Generate Q&A pairs from extraction JSON"
    )
    
    # Add arguments...
    
    return parser.parse_args()
```

### Configuration Handling

Arguments are converted into appropriate configuration objects for the QA generation system:

```python
# Create configuration from CLI arguments
config = QAGenerationConfig(
    model=args.model,
    temperature_range=[args.temperature],
    max_concurrent_requests=args.max_concurrent,
    max_qa_pairs_per_section=args.max_pairs,
    bidirectional_ratio=args.bi_ratio
)

# If workers was specified, set it explicitly
if args.workers is not None:
    config.worker_count = args.workers
```

### Error Handling

The CLI implements robust error handling to provide clear feedback on issues:

1. **Input Validation**: Validates that input files exist before processing
2. **Output Directory Creation**: Creates output directories if needed
3. **Exception Handling**: Catches and reports errors with appropriate exit codes
4. **Debug Mode**: Provides detailed stack traces in debug mode

### Logging Integration

The CLI integrates with Python's logging system to provide configurable output verbosity:

```python
# Configure logging based on verbosity
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
elif args.verbose:
    logging.getLogger().setLevel(logging.INFO)
else:
    logging.getLogger().setLevel(logging.WARNING)
```

### Performance Monitoring

The CLI includes integration with the performance monitoring system to provide detailed metrics:

```python
# Output metrics if monitoring enabled
if not args.no_monitoring:
    try:
        metrics = get_processing_metrics()
        logger.info(f"Performance metrics: "
                    f"cache_hit_rate={metrics.get('cache_hit_rate', 0):.2f}, "
                    f"worker_utilization={metrics.get('worker_utilization', 0):.2f}%")
    except ImportError:
        pass
```

## Design Decisions

### 1. Standalone Module Approach

The CLI is implemented as a `__main__.py` module, making it directly executable as a Python module. This approach has several advantages:

- **Easy Discoverability**: Users can run the module directly
- **Proper Importing**: Ensures all imports work correctly
- **Package Integration**: Works seamlessly as part of the larger package

### 2. Comprehensive Option Set

The CLI provides a comprehensive set of configuration options, giving users fine-grained control over the QA generation process:

- **Core Parameters**: Model selection, worker count, pair limits
- **Advanced Features**: Bidirectional generation, monitoring, caching
- **Operational Controls**: Verbosity, debugging, help

### 3. Sensible Defaults

All parameters have sensible defaults to ensure ease of use while maintaining flexibility:

- **Model**: gpt-3.5-turbo (good balance of quality and speed)
- **Workers**: Automatically determined based on system resources
- **Max Pairs**: 5 pairs per section (reasonable for most content)
- **Temperature**: 0.7 (balanced creativity and consistency)

### 4. Progressive Disclosure

The help system is designed with progressive disclosure in mind:

- **Basic Usage**: Simple input/output pattern for common cases
- **Available Options**: Complete list of options for advanced users
- **Detailed Help**: Specific help for each option with examples

## Testing Strategy

The CLI implementation is tested with several approaches:

1. **Functional Testing**: Verifies the CLI correctly processes arguments and calls the appropriate functions
2. **Integration Testing**: Tests end-to-end processing with actual files
3. **Parameter Validation**: Tests handling of invalid parameters
4. **Error Handling**: Verifies proper error messages and exit codes

The primary test file is `test_cli.py`, which contains tests for CLI execution, parameter validation, and version output.

## Conclusion

The CLI implementation provides a user-friendly, flexible interface to the DuaLipa QA Generation Module. It balances ease of use with comprehensive configuration options, making it suitable for both simple and complex use cases.

The design follows best practices for command-line applications, providing progressive disclosure, sensible defaults, and robust error handling. The implementation is thoroughly tested to ensure reliability and correctness.