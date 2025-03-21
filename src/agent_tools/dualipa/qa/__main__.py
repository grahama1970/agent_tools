#!/usr/bin/env python3
"""Command-line interface for the DuaLipa QA Generation Module.

This module provides a command-line interface for generating Q&A pairs
from extraction JSON files, with various configuration options.

Official documentation:
- argparse: https://docs.python.org/3/library/argparse.html
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
- logging: https://docs.python.org/3/library/logging.html
- os: https://docs.python.org/3/library/os.html
- sys: https://docs.python.org/3/library/sys.html
- time: https://docs.python.org/3/library/time.html
- pathlib: https://docs.python.org/3/library/pathlib.html

Expected input/output:
- parse_arguments: Parses command-line arguments, returns parsed args
- main: The main CLI entry point, returns exit code (0 for success, 1 for failure)

Usage:
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
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog="dualipa-qa",
        description="Generate Q&A pairs from extraction JSON"
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        help="Input extraction JSON file path"
    )
    parser.add_argument(
        "output_file",
        help="Output file path for generated Q&A pairs"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker threads (default: auto)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent requests (default: 4)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Maximum QA pairs per section (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--bi-ratio",
        type=float,
        default=0.3,
        help="Bidirectional generation ratio (default: 0.3)"
    )
    parser.add_argument(
        "--no-bidirectional",
        action="store_true",
        help="Disable bidirectional generation"
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable metrics collection"
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory (default: system tmp)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"DuaLipa QA Generator v{__version__}",
        help="Show version and exit"
    )
    
    return parser.parse_args()


async def main():
    """Run the QA generation process from CLI arguments."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging based on verbosity
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Validate output directory exists
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            sys.exit(1)
    
    # Import QA module components
    try:
        from agent_tools.dualipa.qa.processor import process_extraction_json
        from agent_tools.dualipa.qa.models.config import QAGenerationConfig
        
        # Initialize cache if path provided
        if args.cache_dir:
            from agent_tools.dualipa.qa.utils.cache import initialize_cache
            initialize_cache(args.cache_dir)
            logger.info(f"Using cache directory: {args.cache_dir}")
    except ImportError as e:
        logger.error(f"Failed to import QA module: {e}")
        sys.exit(1)
    
    # Configure performance monitoring if available
    if not args.no_monitoring:
        try:
            from agent_tools.dualipa.qa.utils.performance import performance_tracker
            performance_tracker.reset()
            performance_tracker.start("cli_execution")
            
            logger.info("Performance monitoring enabled")
        except ImportError:
            logger.warning("Performance monitoring not available")
    
    # Create configuration
    temperature_range = [args.temperature]
    
    config = QAGenerationConfig(
        model=args.model,
        temperature_range=temperature_range,
        max_concurrent_requests=args.max_concurrent,
        max_qa_pairs_per_section=args.max_pairs,
        bidirectional_ratio=args.bi_ratio
    )
    
    # If workers was specified, set it explicitly
    if args.workers is not None:
        config.worker_count = args.workers
    
    # Log configuration
    logger.info(f"Processing file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Configuration: model={config.model}, "
                f"workers={getattr(config, 'worker_count', 'auto')}, "
                f"max_pairs={config.max_qa_pairs_per_section}, "
                f"bidirectional={not args.no_bidirectional}")
    
    # Process the extraction JSON
    try:
        start_time = time.time()
        
        # Run the processing
        result = await process_extraction_json(
            input_data=input_path,
            output_file=output_path,
            config=config,
            enable_bidirectional=not args.no_bidirectional,
            enable_monitoring=not args.no_monitoring
        )
        
        elapsed_time = time.time() - start_time
        
        # Success output
        qa_count = len(result.qa_pairs)
        logger.info(f"Successfully generated {qa_count} Q&A pairs "
                    f"in {elapsed_time:.2f} seconds")
        
        # Output metrics if monitoring enabled
        if not args.no_monitoring:
            try:
                from agent_tools.dualipa.qa.monitoring import get_processing_metrics
                metrics = get_processing_metrics()
                
                logger.info(f"Performance metrics: "
                            f"cache_hit_rate={metrics.get('cache_hit_rate', 0):.2f}, "
                            f"worker_utilization={metrics.get('worker_utilization', 0):.2f}%")
                
                # End CLI performance tracking
                try:
                    cli_time = performance_tracker.end("cli_execution")
                    logger.info(f"Total CLI execution time: {cli_time:.2f}s")
                    
                    # Output detailed metrics in debug mode
                    if args.debug:
                        all_metrics = performance_tracker.get_metrics()
                        logger.debug(f"Detailed performance metrics: {json.dumps(all_metrics, indent=2)}")
                except:
                    pass
            except ImportError:
                pass
        
        return 0
    except Exception as e:
        logger.error(f"Error processing extraction file: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)