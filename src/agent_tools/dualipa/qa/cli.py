"""Command-line interface for QA generation.

This module provides a CLI for the QA generation pipeline.

Usage:
    python -m dualipa.qa.cli input.json [--output output.json] [--temps 0.3 0.5 0.7]

Official documentation:
- argparse: https://docs.python.org/3/library/argparse.html
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import os
import json
import logging
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from .processor import process_extraction_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from extraction JSON"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to output JSON file (default: qa_output.json)",
        default="qa_output.json"
    )
    
    parser.add_argument(
        "--temps", "-t",
        type=float,
        nargs="+",
        help="Temperature range for generation (default: 0.3 0.5 0.7)",
        default=[0.3, 0.5, 0.7]
    )
    
    parser.add_argument(
        "--min-reasoning", "-m",
        type=int,
        help="Minimum words in reasoning (default: 15)",
        default=15
    )
    
    parser.add_argument(
        "--similarity", "-s",
        type=float,
        help="Similarity threshold for deduplication (default: 0.85)",
        default=0.85
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Check output path
    output_path = Path(args.output)
    if output_path.exists():
        logger.warning(f"Output file already exists, will be overwritten: {output_path}")
    
    try:
        # Log start
        logger.info(f"Processing {input_path} with temperatures {args.temps}")
        
        # Process extraction JSON
        response = await process_extraction_json(
            input_data=input_path,
            output_file=output_path,
            temps=args.temps,
            min_reasoning_words=args.min_reasoning,
            similarity_threshold=args.similarity
        )
        
        # Log results
        logger.info(f"Generated {len(response.qa_pairs)} QA pairs")
        logger.info(f"Forward pairs: {response.generation_metadata['forward_pairs']}")
        logger.info(f"Reverse pairs: {response.generation_metadata['reverse_pairs']}")
        logger.info(f"Processing time: {response.generation_metadata['processing_time_seconds']} seconds")
        logger.info(f"Output written to {output_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Error processing extraction JSON: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)