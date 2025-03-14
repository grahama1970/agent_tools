"""
DuaLipa: Dual LLM-Informed Python Automation.

A toolkit for generating high-quality question-answer pairs from code repositories 
and documentation, using LLMs to enhance the depth and quality of generated QA pairs.

Official Documentation References:
- Click: https://click.palletsprojects.com/
- Rich: https://rich.readthedocs.io/
- Loguru: https://loguru.readthedocs.io/en/stable/
- LiteLLM: https://docs.litellm.ai/docs/
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz
"""

import sys
from pathlib import Path
import os
import json
import tempfile
import shutil
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
import logging
from typing import List, Optional, Dict, Any, Union
from loguru import logger

# Import module functions
from .code_extractor import extract_repository, demo_code_extractor
from .format_dataset import format_for_lora, demo_format_dataset
from .llm_generator import (
    check_litellm_available, 
    generate_code_related_qa_pairs,
    generate_markdown_related_qa_pairs,
    generate_reverse_qa_pairs,
    demo_llm_generator
)
from .github_utils import demo_github_utils
from .language_detection import demo_language_detection
from .markdown_parser import demo_markdown_parser
from .qa_validator import demo_qa_validation, validate_and_enhance_qa_pairs
from .pipeline import run_pipeline, demo_pipeline

__version__ = "0.1.0"

# Create Rich console for formatting
console = Console()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


@click.group()
@click.version_option(version=__version__, prog_name="DuaLipa")
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose logging')
@click.pass_context
def cli(ctx, debug):
    """DuaLipa: Dual LLM-Informed Python Automation.
    
    This toolkit generates high-quality question-answer pairs from code repositories 
    and documentation, using LLMs to enhance the depth and quality of generated QA pairs.
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)
    
    # Store debug setting
    ctx.obj['DEBUG'] = debug
    
    # Set up debug mode if requested
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled")


@cli.command()
@click.argument('repo_path', type=click.Path(exists=False))
@click.argument('output_dir', type=click.Path())
@click.option('--max-files', type=int, default=1000, help='Maximum number of files to extract')
@click.option('--include', multiple=True, default=[], help='Glob patterns to include')
@click.option('--exclude', multiple=True, default=[], help='Glob patterns to exclude')
@click.option('--no-docs', is_flag=True, help='Skip documentation extraction')
@click.option('--no-code', is_flag=True, help='Skip code extraction')
@click.pass_context
def extract(ctx, repo_path, output_dir, max_files, include, exclude, no_docs, no_code):
    """Extract code from a repository.
    
    REPO_PATH is the path to the repository or GitHub URL.
    OUTPUT_DIR is the directory where extracted data will be saved.
    """
    console.print(Panel(f"[bold blue]Extracting from repository:[/] [green]{repo_path}[/]", 
                        title="DuaLipa Extractor"))
    
    try:
        stats = extract_repository(
            source=repo_path, 
            output_path=output_dir,
            max_files=max_files,
            include_patterns=include if include else None,
            exclude_patterns=exclude if exclude else None,
            extract_documentation=not no_docs,
            extract_code=not no_code
        )
        
        # Display statistics
        table = Table(title="Extraction Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total files processed", str(stats.get("total_files", 0)))
        table.add_row("Code files extracted", str(stats.get("code_files", 0)))
        table.add_row("Documentation files extracted", str(stats.get("documentation_files", 0)))
        table.add_row("Code blocks extracted", str(stats.get("code_blocks", 0)))
        
        console.print(table)
        
        # Show languages
        if stats.get("languages"):
            lang_table = Table(title="Languages Detected")
            lang_table.add_column("Language", style="cyan")
            lang_table.add_column("Count", style="green")
            
            for lang, count in stats.get("languages", {}).items():
                lang_table.add_row(lang, str(count))
            
            console.print(lang_table)
        
        # Show errors if any
        if stats.get("errors"):
            console.print("[bold yellow]Warnings/Errors during extraction:[/]")
            for error in stats.get("errors", [])[:5]:  # Show max 5 errors
                console.print(f"- {error}")
            
            if len(stats.get("errors", [])) > 5:
                console.print(f"... and {len(stats.get('errors', [])) - 5} more")
        
        console.print(f"[bold green]✓[/] Extraction completed successfully.")
        console.print(f"[bold]Output:[/] {os.path.join(output_dir, 'extraction_stats.json')}")
        
    except Exception as e:
        console.print(f"[bold red]Error during extraction:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--use-llm', is_flag=True, help='Use LLM for enhanced generation')
@click.option('--max-pairs', type=int, default=10, 
              help='Maximum number of QA pairs to generate per item')
@click.pass_context
def format(ctx, input_file, output_file, use_llm, max_pairs):
    """Format extracted data into QA pairs.
    
    INPUT_FILE is the path to the extracted data JSON file.
    OUTPUT_FILE is the path where the formatted dataset will be saved.
    """
    console.print(Panel(f"[bold blue]Formatting data:[/] [green]{input_file}[/]", 
                        title="DuaLipa Formatter"))
    
    try:
        # Check if LLM is available if requested
        if use_llm:
            llm_available = check_litellm_available()
            if not llm_available:
                console.print("[yellow]Warning: LiteLLM not available or API key not set. Falling back to basic generation.[/]")
                use_llm = False
        
        stats = format_for_lora(
            input_file=input_file, 
            output_file=output_file,
            use_llm=use_llm,
            max_pairs_per_item=max_pairs
        )
        
        # Display statistics
        table = Table(title="Formatting Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total items processed", str(stats.get("total_items_processed", 0)))
        table.add_row("Total QA pairs generated", str(stats.get("total_qa_pairs", 0)))
        table.add_row("Code items", str(stats.get("code_items", 0)))
        table.add_row("Documentation items", str(stats.get("documentation_items", 0)))
        
        if use_llm:
            table.add_row("LLM-generated pairs", str(stats.get("llm_generated_pairs", 0)))
            table.add_row("Reverse-generated pairs", str(stats.get("reverse_generated_pairs", 0)))
        else:
            table.add_row("Basic-generated pairs", str(stats.get("basic_generated_pairs", 0)))
        
        table.add_row("Validated pairs", str(stats.get("validated_pairs", 0)))
        
        console.print(table)
        
        # Show errors if any
        if stats.get("errors"):
            console.print("[bold yellow]Warnings/Errors during formatting:[/]")
            for error in stats.get("errors", [])[:5]:  # Show max 5 errors
                console.print(f"- {error}")
            
            if len(stats.get("errors", [])) > 5:
                console.print(f"... and {len(stats.get('errors', [])) - 5} more")
        
        console.print(f"[bold green]✓[/] Formatting completed successfully.")
        console.print(f"[bold]Output:[/] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error during formatting:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('repo_path', type=click.Path(exists=False))
@click.argument('output_dir', type=click.Path())
@click.option('--skip-extract', is_flag=True, help='Skip extraction stage')
@click.option('--skip-format', is_flag=True, help='Skip formatting stage')
@click.option('--run-train', is_flag=True, help='Run training stage')
@click.option('--max-files', type=int, default=1000, help='Maximum number of files to extract')
@click.option('--use-llm', is_flag=True, help='Use LLM for QA generation')
@click.option('--max-pairs', type=int, default=5, help='Maximum QA pairs per item')
@click.pass_context
def pipeline(ctx, repo_path, output_dir, skip_extract, skip_format, run_train, max_files, use_llm, max_pairs):
    """Run the complete DuaLipa pipeline.
    
    This command runs the entire pipeline from extraction to formatting and optional training.
    
    REPO_PATH is the path to the repository or GitHub URL.
    OUTPUT_DIR is the directory where all outputs will be saved.
    """
    console.print(Panel(f"[bold blue]Running DuaLipa Pipeline:[/] [green]{repo_path}[/]", 
                        title="DuaLipa Pipeline"))
    
    try:
        # Setup parameters
        extract_kwargs = {
            "max_files": max_files
        }
        
        format_kwargs = {
            "use_llm": use_llm,
            "max_pairs_per_item": max_pairs
        }
        
        # Run the pipeline
        stats = run_pipeline(
            repo_path=repo_path,
            output_dir=output_dir,
            extract_kwargs=extract_kwargs,
            format_kwargs=format_kwargs,
            run_extract=not skip_extract,
            run_format=not skip_format,
            run_train=run_train,
            debug=ctx.obj.get('DEBUG', False)
        )
        
        # Display final summary
        summary_table = Table(title="Pipeline Summary")
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Status", style="green")
        
        summary_table.add_row("Extraction", 
                             "✓ Completed" if stats["extract"] else "⨯ Skipped")
        summary_table.add_row("Formatting", 
                             "✓ Completed" if stats["format"] else "⨯ Skipped")
        summary_table.add_row("Training", 
                             "✓ Completed" if stats["train"] else "⨯ Skipped")
        
        console.print(summary_table)
        
        # Show total QA pairs
        if stats["format"]:
            qa_count = stats["format"].get("total_qa_pairs", 0)
            console.print(f"[bold green]Total QA Pairs Generated:[/] {qa_count}")
        
        console.print(f"[bold green]✓[/] Pipeline completed successfully.")
        console.print(f"[bold]Output Directory:[/] {os.path.abspath(output_dir)}")
        
    except Exception as e:
        console.print(f"[bold red]Error running pipeline:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('component', type=click.Choice(
    ["extract", "format", "github", "language", "markdown", "llm", "qa", "pipeline", "all"]
))
@click.pass_context
def demo(ctx, component):
    """Run demonstration for a component.
    
    COMPONENT is the component to demonstrate ('extract', 'format', 'github', 
    'language', 'markdown', 'llm', 'qa', 'pipeline', or 'all').
    """
    console.print(Panel(f"[bold blue]Running demonstration for:[/] [green]{component}[/]", 
                        title="DuaLipa Demonstrator"))
    
    try:
        run_demo(component)
        console.print(f"[bold green]✓[/] Demonstration completed successfully.")
    except Exception as e:
        console.print(f"[bold red]Error during demonstration:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def run_demo(component: str):
    """Run demonstration for a specific component.
    
    Args:
        component: The component to demonstrate
    """
    console.print(f"Running demonstration for component: [bold]{component}[/]")
    
    if component in ["extract", "all"]:
        console.print("\n[bold blue]=== Code Extractor Demonstration ===[/]")
        demo_code_extractor()
    
    if component in ["github", "all"]:
        console.print("\n[bold blue]=== GitHub Utilities Demonstration ===[/]")
        demo_github_utils()
    
    if component in ["language", "all"]:
        console.print("\n[bold blue]=== Language Detection Demonstration ===[/]")
        demo_language_detection()
    
    if component in ["markdown", "all"]:
        console.print("\n[bold blue]=== Markdown Parser Demonstration ===[/]")
        demo_markdown_parser()
    
    if component in ["llm", "all"]:
        console.print("\n[bold blue]=== LLM Generator Demonstration ===[/]")
        demo_llm_generator()
    
    if component in ["qa", "all"]:
        console.print("\n[bold blue]=== QA Validator Demonstration ===[/]")
        demo_qa_validation()
    
    if component in ["format", "all"]:
        console.print("\n[bold blue]=== Dataset Formatter Demonstration ===[/]")
        demo_format_dataset()
        
    if component in ["pipeline", "all"]:
        console.print("\n[bold blue]=== End-to-End Pipeline Demonstration ===[/]")
        demo_pipeline()


@cli.command()
@click.option('--check-llm', is_flag=True, help='Check if LLM is available')
@click.pass_context
def info(ctx, check_llm):
    """Display information about the DuaLipa package."""
    console.print(Panel("[bold blue]DuaLipa: Dual LLM-Informed Python Automation[/]", 
                       title="Package Information"))
    
    console.print("[bold]Version:[/]", __version__)
    console.print("[bold]Path:[/]", os.path.dirname(os.path.abspath(__file__)))
    
    # Get modules information
    modules = [
        ("code_extractor.py", "Extract code from repositories"),
        ("format_dataset.py", "Convert extracted data to QA pairs"),
        ("github_utils.py", "GitHub repository interactions"),
        ("language_detection.py", "Programming language detection"),
        ("llm_generator.py", "LLM-based QA pair generation"),
        ("markdown_parser.py", "Markdown content parsing"),
        ("qa_validator.py", "QA pair validation and enhancement"),
        ("pipeline.py", "End-to-end pipeline for extraction to training")
    ]
    
    table = Table(title="Modules")
    table.add_column("Module", style="cyan")
    table.add_column("Purpose", style="green")
    
    for module, purpose in modules:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module)
        exists = "✓" if os.path.exists(file_path) else "✗"
        table.add_row(f"{exists} {module}", purpose)
    
    console.print(table)
    
    # Check LLM availability if requested
    if check_llm:
        console.print("\n[bold]LLM Configuration:[/]")
        llm_available = check_litellm_available()
        if llm_available:
            console.print("[bold green]✓[/] LLM integration is available and configured")
        else:
            console.print("[bold yellow]![/] LLM integration is not available or not configured")
            console.print("  Make sure you have installed [bold]litellm[/] and set API keys.")
            console.print("  Example: [cyan]export OPENAI_API_KEY='your-key-here'[/]")


def demo_main() -> None:
    """Run a comprehensive demonstration of DuaLipa functionality.
    
    This function demonstrates all major components:
    1. GitHub repository interaction
    2. Language detection
    3. Code extraction
    4. QA pair generation and validation
    5. Dataset formatting
    6. End-to-end pipeline
    
    Returns:
        None - prints results to the console
    """
    console.print(Panel("[bold blue]DuaLipa: Dual LLM-Informed Python Automation[/]\n" +
                        "Comprehensive Demonstration", title="DuaLipa Demo"))
    
    console.print("\n[bold]Available Components:[/]")
    console.print("1. GitHub Repository Utilities")
    console.print("2. Language Detection")
    console.print("3. Code Extraction")
    console.print("4. Markdown Parsing")
    console.print("5. LLM-based QA Generation")
    console.print("6. QA Validation")
    console.print("7. Dataset Formatting")
    console.print("8. End-to-End Pipeline")
    
    console.print("\n[bold yellow]Starting demonstrations...[/]")
    
    # Run each component's demo function
    try:
        console.print("\n[bold blue]=== GitHub Utilities Demonstration ===[/]")
        demo_github_utils()
    except Exception as e:
        console.print(f"[bold red]Error in GitHub utilities demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== Language Detection Demonstration ===[/]")
        demo_language_detection()
    except Exception as e:
        console.print(f"[bold red]Error in language detection demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== Markdown Parser Demonstration ===[/]")
        demo_markdown_parser()
    except Exception as e:
        console.print(f"[bold red]Error in markdown parser demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== LLM Generator Demonstration ===[/]")
        demo_llm_generator()
    except Exception as e:
        console.print(f"[bold red]Error in LLM generator demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== QA Validator Demonstration ===[/]")
        demo_qa_validation()
    except Exception as e:
        console.print(f"[bold red]Error in QA validator demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== Code Extractor Demonstration ===[/]")
        demo_code_extractor()
    except Exception as e:
        console.print(f"[bold red]Error in code extractor demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== Dataset Formatter Demonstration ===[/]")
        demo_format_dataset()
    except Exception as e:
        console.print(f"[bold red]Error in dataset formatter demo: {e}[/]")
    
    try:
        console.print("\n[bold blue]=== End-to-End Pipeline Demonstration ===[/]")
        demo_pipeline()
    except Exception as e:
        console.print(f"[bold red]Error in pipeline demo: {e}[/]")
    
    console.print("\n[bold green]Comprehensive demonstration completed![/]")
    console.print("\nFor detailed documentation and usage examples, visit:")
    console.print("https://github.com/yourusername/dualipa")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_main()
