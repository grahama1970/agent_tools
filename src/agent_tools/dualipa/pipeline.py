"""
DuaLipa Pipeline: End-to-end pipeline for question-answer pair generation.

This module provides a complete pipeline for:
1. Extracting code and documentation from repositories
2. Generating QA pairs from the extracted content
3. Training a model on the generated data (optional)

Official Documentation References:
- Click: https://click.palletsprojects.com/
- Rich: https://rich.readthedocs.io/
- Loguru: https://loguru.readthedocs.io/en/stable/
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger
from rich.syntax import Syntax

# Import pipeline components
try:
    # Try package imports first (installed mode)
    from agent_tools.dualipa.code_extractor import extract_repository
    from agent_tools.dualipa.format_dataset import format_for_lora
    from agent_tools.dualipa.llm_generator import check_litellm_available
    from agent_tools.dualipa.train_lora import train_lora
    from agent_tools.dualipa import __version__
except ImportError:
    # Fallback to relative imports (development mode)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from agent_tools.dualipa.code_extractor import extract_repository
    from agent_tools.dualipa.format_dataset import format_for_lora
    from agent_tools.dualipa.llm_generator import check_litellm_available
    from agent_tools.dualipa.train_lora import train_lora
    
    # For version
    __version__ = "0.1.0"

# Create rich console
console = Console()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


class Pipeline:
    """DuaLipa Pipeline for end-to-end QA generation and model training.
    
    This class orchestrates the entire pipeline from repository extraction
    to model training, with configurable options at each stage.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the pipeline with configuration options.
        
        Args:
            debug: Enable debug mode with verbose logging
        """
        self.debug = debug
        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
            logger.debug("Debug mode enabled for DuaLipa pipeline")
        
        self.stats = {
            "extract": {},
            "format": {},
            "train": {}
        }
    
    async def run(self, 
               repo_path: str, 
               output_dir: str,
               extract_kwargs: Optional[Dict[str, Any]] = None,
               format_kwargs: Optional[Dict[str, Any]] = None,
               train_kwargs: Optional[Dict[str, Any]] = None,
               run_extract: bool = True,
               run_format: bool = True,
               run_train: bool = False) -> Dict[str, Any]:
        """Run the complete DuaLipa pipeline.
        
        Args:
            repo_path: Path to the repository or GitHub URL
            output_dir: Base output directory for all stages
            extract_kwargs: Keyword arguments for extraction stage
            format_kwargs: Keyword arguments for formatting stage
            train_kwargs: Keyword arguments for training stage
            run_extract: Whether to run the extraction stage
            run_format: Whether to run the formatting stage
            run_train: Whether to run the training stage
            
        Returns:
            Dictionary with statistics from each pipeline stage
            
        Raises:
            Exception: If any pipeline stage fails
        """
        console.print(Panel(f"[bold blue]DuaLipa Pipeline v{__version__}[/]", 
                            title="Starting Pipeline"))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Default kwargs
        extract_kwargs = extract_kwargs or {}
        format_kwargs = format_kwargs or {}
        train_kwargs = train_kwargs or {}
        
        # Set up paths for each stage
        extract_output = os.path.join(output_dir, "extracted")
        extract_json = os.path.join(extract_output, "extraction.json")
        
        format_output = os.path.join(output_dir, "formatted")
        format_json = os.path.join(format_output, "qa_dataset.json")
        
        train_output = os.path.join(output_dir, "models")
        
        # 1. Extract stage
        if run_extract:
            console.print(Panel(f"[bold blue]Stage 1: Extracting from repository[/]", 
                                title="Extract Stage"))
            try:
                os.makedirs(extract_output, exist_ok=True)
                self.stats["extract"] = extract_repository(
                    source=repo_path,
                    output_path=extract_output,
                    **extract_kwargs
                )
                self._print_extract_stats()
            except Exception as e:
                console.print(f"[bold red]Error during extraction stage:[/] {str(e)}")
                if self.debug:
                    import traceback
                    console.print(traceback.format_exc())
                raise
        else:
            console.print("[yellow]Skipping extraction stage...[/]")
        
        # 2. Format stage
        if run_format:
            console.print(Panel(f"[bold blue]Stage 2: Formatting QA pairs[/]", 
                                title="Format Stage"))
            
            try:
                os.makedirs(format_output, exist_ok=True)
                
                # Check if we need to use previous extraction output
                input_file = extract_json
                if not run_extract:
                    input_file = extract_kwargs.get('input_file', extract_json)
                    if not os.path.exists(input_file):
                        console.print(f"[bold red]Error:[/] Input file {input_file} doesn't exist. Extraction must be run first.")
                        return self.stats
                
                # Check LLM availability
                use_llm = format_kwargs.get('use_llm', False)
                if use_llm:
                    llm_available = check_litellm_available()
                    if not llm_available:
                        console.print("[yellow]Warning: LiteLLM not available. Falling back to basic generation.[/]")
                        format_kwargs['use_llm'] = False
                
                self.stats["format"] = format_for_lora(
                    input_file=input_file,
                    output_file=format_json,
                    **format_kwargs
                )
                self._print_format_stats()
            except Exception as e:
                console.print(f"[bold red]Error during formatting stage:[/] {str(e)}")
                if self.debug:
                    import traceback
                    console.print(traceback.format_exc())
                raise
        else:
            console.print("[yellow]Skipping formatting stage...[/]")
        
        # 3. Train stage (optional)
        if run_train:
            console.print(Panel(f"[bold blue]Stage 3: Training LoRA model[/]", 
                                title="Train Stage"))
            
            try:
                os.makedirs(train_output, exist_ok=True)
                
                # Check if we need to use previous formatting output
                dataset_path = format_json
                if not run_format:
                    dataset_path = format_kwargs.get('output_file', format_json)
                    if not os.path.exists(dataset_path):
                        console.print(f"[bold red]Error:[/] Dataset file {dataset_path} doesn't exist. Formatting must be run first.")
                        return self.stats
                
                # Run training
                train_lora(
                    dataset_path=dataset_path,
                    output_dir=train_output,
                    **train_kwargs
                )
                
                self.stats["train"] = {
                    "dataset_path": dataset_path,
                    "output_dir": train_output,
                    "model_name": train_kwargs.get("model_name", "unsloth/Mistral-7B"),
                    "num_train_epochs": train_kwargs.get("num_train_epochs", 3)
                }
                self._print_train_stats()
            except Exception as e:
                console.print(f"[bold red]Error during training stage:[/] {str(e)}")
                if self.debug:
                    import traceback
                    console.print(traceback.format_exc())
                raise
        else:
            console.print("[yellow]Skipping training stage...[/]")
        
        console.print("[bold green]Pipeline completed successfully![/]")
        return self.stats
    
    def _print_extract_stats(self):
        """Print statistics from the extraction stage."""
        stats = self.stats["extract"]
        if not stats:
            return
            
        table = Table(title="Extraction Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total files processed", str(stats.get("total_files", 0)))
        table.add_row("Code files extracted", str(stats.get("code_files", 0)))
        table.add_row("Documentation files extracted", str(stats.get("documentation_files", 0)))
        table.add_row("Code blocks extracted", str(stats.get("code_blocks", 0)))
        
        console.print(table)
    
    def _print_format_stats(self):
        """Print statistics from the formatting stage."""
        stats = self.stats["format"]
        if not stats:
            return
            
        table = Table(title="Formatting Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total items processed", str(stats.get("total_items_processed", 0)))
        table.add_row("Total QA pairs generated", str(stats.get("total_qa_pairs", 0)))
        table.add_row("Code items", str(stats.get("code_items", 0)))
        table.add_row("Documentation items", str(stats.get("documentation_items", 0)))
        
        if stats.get("llm_generated_pairs"):
            table.add_row("LLM-generated pairs", str(stats.get("llm_generated_pairs", 0)))
            table.add_row("Reverse-generated pairs", str(stats.get("reverse_generated_pairs", 0)))
        else:
            table.add_row("Basic-generated pairs", str(stats.get("basic_generated_pairs", 0)))
        
        table.add_row("Validated pairs", str(stats.get("validated_pairs", 0)))
        
        console.print(table)
    
    def _print_train_stats(self):
        """Print statistics from the training stage."""
        stats = self.stats["train"]
        if not stats:
            return
            
        table = Table(title="Training Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        console.print(table)


def run_pipeline(repo_path: str, 
                output_dir: str,
                extract_kwargs: Optional[Dict[str, Any]] = None,
                format_kwargs: Optional[Dict[str, Any]] = None,
                train_kwargs: Optional[Dict[str, Any]] = None,
                run_extract: bool = True,
                run_format: bool = True,
                run_train: bool = False,
                debug: bool = False) -> Dict[str, Any]:
    """Convenience function to run the complete pipeline.
    
    Args:
        repo_path: Path to the repository or GitHub URL
        output_dir: Base output directory for all stages
        extract_kwargs: Keyword arguments for extraction stage
        format_kwargs: Keyword arguments for formatting stage
        train_kwargs: Keyword arguments for training stage
        run_extract: Whether to run the extraction stage
        run_format: Whether to run the formatting stage
        run_train: Whether to run the training stage
        debug: Enable debug mode with verbose logging
        
    Returns:
        Dictionary with statistics from each pipeline stage
    """
    pipeline = Pipeline(debug=debug)
    return asyncio.run(pipeline.run(
        repo_path=repo_path,
        output_dir=output_dir,
        extract_kwargs=extract_kwargs,
        format_kwargs=format_kwargs,
        train_kwargs=train_kwargs,
        run_extract=run_extract,
        run_format=run_format,
        run_train=run_train
    ))


def demo_pipeline():
    """Demonstrate the pipeline with a small example repository."""
    console.print(Panel("[bold blue]DuaLipa Pipeline Demonstration[/]", 
                        title="Pipeline Demo"))
    
    # Example GitHub repository for demonstration
    repo_path = "https://github.com/psf/requests"
    output_dir = "dualipa_demo_output"
    
    console.print(f"Running pipeline on: [bold]{repo_path}[/]")
    console.print(f"Output directory: [bold]{output_dir}[/]")
    
    # Run with limited scope for demo purposes
    extract_kwargs = {
        "max_files": 10,
        "include_patterns": ["*.py", "*.md"],
        "extract_documentation": True,
        "extract_code": True
    }
    
    format_kwargs = {
        "use_llm": False,  # Don't use LLM for demo
        "max_pairs_per_item": 5
    }
    
    # Run pipeline without training for demo
    stats = run_pipeline(
        repo_path=repo_path,
        output_dir=output_dir,
        extract_kwargs=extract_kwargs,
        format_kwargs=format_kwargs,
        run_extract=True,
        run_format=True,
        run_train=False
    )
    
    console.print("[bold green]Pipeline demonstration completed![/]")
    console.print(f"Output directory: [bold]{os.path.abspath(output_dir)}[/]")


def show_usage():
    """Show usage examples for the pipeline module.
    
    This function prints examples of how to use the pipeline module both 
    programmatically and from the command line.
    """
    console.print(Panel("[bold blue]DuaLipa Pipeline Usage Examples[/]", 
                        title="Usage Guide"))
    
    # Python API usage example
    console.print("\n[bold yellow]Python API Usage:[/]")
    python_example = """
# Basic usage - Extract and Format
from agent_tools.dualipa.pipeline import run_pipeline

stats = run_pipeline(
    repo_path="https://github.com/username/repo",
    output_dir="output_directory", 
    extract_kwargs={"max_files": 500},
    format_kwargs={"use_llm": True}
)

# Advanced usage - With all options
from agent_tools.dualipa.pipeline import run_pipeline

stats = run_pipeline(
    repo_path="https://github.com/username/repo",
    output_dir="output_directory",
    # Extraction options
    extract_kwargs={
        "max_files": 1000,
        "include_patterns": ["*.py", "*.md", "*.js"],
        "exclude_patterns": ["*test*", "*node_modules*"],
        "extract_documentation": True,
        "extract_code": True
    },
    # Formatting options
    format_kwargs={
        "use_llm": True,
        "max_pairs_per_item": 10,
        "llm_model": "gpt-3.5-turbo",
        "include_reverse_pairs": True
    },
    # Training options
    train_kwargs={
        "model_name": "unsloth/Mistral-7B", 
        "num_train_epochs": 3,
        "learning_rate": 2e-4
    },
    # Pipeline control
    run_extract=True,
    run_format=True,
    run_train=True,
    debug=True
)

# Access statistics
print(f"Total files processed: {stats['extract'].get('total_files', 0)}")
print(f"QA pairs generated: {stats['format'].get('total_qa_pairs', 0)}")
"""
    console.print(Syntax(python_example, "python", theme="monokai", line_numbers=True))
    
    # CLI usage example
    console.print("\n[bold yellow]Command Line Usage:[/]")
    cli_example = """
# Basic pipeline run (extract and format)
dualipa pipeline https://github.com/username/repo output_dir/

# Advanced pipeline with options
dualipa pipeline https://github.com/username/repo output_dir/ \\
    --use-llm \\
    --max-files 500 \\
    --max-pairs 10 \\
    --run-train

# Skip extraction (if already done)
dualipa pipeline https://github.com/username/repo output_dir/ \\
    --skip-extract \\
    --use-llm

# Only run extraction
dualipa pipeline https://github.com/username/repo output_dir/ \\
    --skip-format

# Standalone extract and format commands
dualipa extract https://github.com/username/repo extracted_data/
dualipa format extracted_data/extraction.json formatted_data.json --use-llm
"""
    console.print(Syntax(cli_example, "bash", theme="monokai"))
    
    # Additional tips
    console.print("\n[bold yellow]Tips:[/]")
    console.print("• For LLM integration, set your API key: [cyan]export OPENAI_API_KEY='your-key-here'[/]")
    console.print("• Use [cyan]--debug[/] flag for verbose logging")
    console.print("• See documentation for more options: [cyan]dualipa --help[/]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DuaLipa Pipeline")
    parser.add_argument("repo_path", nargs="?", help="Repository path or GitHub URL")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction stage")
    parser.add_argument("--skip-format", action="store_true", help="Skip formatting stage")
    parser.add_argument("--run-train", action="store_true", help="Run training stage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-files", type=int, default=1000, help="Maximum files to extract")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for QA generation")
    parser.add_argument("--show-usage", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    # Show usage if requested or if no arguments provided
    if args.show_usage or (args.repo_path is None and args.output_dir is None):
        show_usage()
        sys.exit(0)
    
    # Run pipeline with command line arguments
    extract_kwargs = {"max_files": args.max_files}
    format_kwargs = {"use_llm": args.use_llm}
    
    run_pipeline(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        extract_kwargs=extract_kwargs,
        format_kwargs=format_kwargs,
        run_extract=not args.skip_extract,
        run_format=not args.skip_format,
        run_train=args.run_train,
        debug=args.debug
    ) 