"""
Unified CLI for DuaLipa: Integrated Extraction, QA Generation, and Training.

This CLI provides a single interface for all DuaLipa modules, allowing users
to perform extraction, QA generation, and training operations with a consistent
interface. It's primarily designed for agent use but is also human-friendly.

Usage:
    # Extract from a repository
    python -m dualipa_cli extract /path/to/repo /path/to/output
    
    # Generate QA pairs from extraction output
    python -m dualipa_cli qa /path/to/extraction.json /path/to/qa_output.json
    
    # Validate extraction output for QA compatibility
    python -m dualipa_cli validate /path/to/extraction.json [--convert --output qa_input.json]
    
    # Run the complete pipeline
    python -m dualipa_cli pipeline /path/to/repo /path/to/output_dir
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from agent_tools.method_validator.analyzer import validate_method

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dualipa.cli")

# Package version
__version__ = "0.2.0"

# Create Rich console for formatting
console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="DuaLipa")
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose logging')
@click.pass_context
def cli(ctx, debug):
    """DuaLipa: Dual LLM-Informed Python Automation.
    
    This toolkit extracts code and documentation from repositories,
    generates high-quality question-answer pairs, and provides
    tools for fine-tuning models with the resulting dataset.
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)
    
    # Store debug setting
    ctx.obj['DEBUG'] = debug
    
    # Set up debug mode if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--max-files', type=int, default=1000, help='Maximum number of files to extract')
@click.option('--include', multiple=True, default=[], help='Glob patterns to include')
@click.option('--exclude', multiple=True, default=[], help='Glob patterns to exclude')
@click.option('--format', type=click.Choice(['standard', 'deepseek']), default='standard', 
              help='Output format: standard or deepseek')
@click.pass_context
def extract(ctx, repo_path, output_dir, max_files, include, exclude, format):
    """Extract code and documentation from a repository.
    
    REPO_PATH is the path to the repository to extract from.
    OUTPUT_DIR is the directory where extracted data will be saved.
    """
    console.print(Panel(f"[bold blue]Extracting from repository:[/] [green]{repo_path}[/]", 
                        title="DuaLipa Extractor"))
    
    try:
        # Try to import the new extraction module - using method_validator to validate first
        try:
            # Validate extraction methods exist
            extraction_methods = [
                ("agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks", "extract_all_blocks"),
                ("agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks", "find_source_files"),
                ("agent_tools.dualipa.extraction.examples.end_to_end.hierarchy_analyzer", "analyze_hierarchies"),
                ("agent_tools.dualipa.extraction.examples.end_to_end.hierarchy_analyzer", "enrich_blocks_with_hierarchy"),
                ("agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter", "create_qa_compatible_blocks"),
                ("agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter", "create_qa_compatible_output")
            ]
            
            all_valid = True
            for package, method in extraction_methods:
                is_valid, message = validate_method(package, method)
                if not is_valid:
                    console.print(f"[bold yellow]Warning:[/] {method} not found: {message}")
                    all_valid = False
            
            if not all_valid:
                console.print("[bold yellow]Required extraction methods are not all available.[/]")
                console.print("Falling back to original extraction functionality...")
                raise ImportError("Extraction method validation failed")
                
            # If all methods validated, import them directly
            from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks, find_source_files
            from agent_tools.dualipa.extraction.examples.end_to_end.hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
            from agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
        except ImportError:
            console.print("[bold yellow]Warning:[/] Could not import new extraction modules directly.")
            console.print("Falling back to original extraction functionality...")
            
            # Fall back to original extraction
            from .code_extractor import extract_repository
            
            # Run the original extract_repository function
            stats = extract_repository(
                source=repo_path, 
                output_path=output_dir,
                max_files=max_files,
                include_patterns=include if include else None,
                exclude_patterns=exclude if exclude else None,
                extract_documentation=True,
                extract_code=True
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
            
            console.print(f"[bold green]✓[/] Extraction completed successfully.")
            console.print(f"[bold]Output:[/] {os.path.join(output_dir, 'extraction_output.json')}")
            return
        
        # Continue with new extraction module
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all source files
        console.print("Finding source files...")
        source_files = find_source_files(Path(repo_path))
        
        # Filter files based on include/exclude patterns if provided
        if include:
            import fnmatch
            included_files = []
            for pattern in include:
                included_files.extend([f for f in source_files 
                                      if fnmatch.fnmatch(str(f), pattern)])
            source_files = included_files
        
        if exclude:
            import fnmatch
            excluded_patterns = [fnmatch.translate(p) for p in exclude]
            import re
            excluded_res = [re.compile(p) for p in excluded_patterns]
            source_files = [f for f in source_files 
                           if not any(r.match(str(f)) for r in excluded_res)]
        
        # Limit number of files if specified
        if max_files > 0 and len(source_files) > max_files:
            console.print(f"Limiting extraction to {max_files} files (out of {len(source_files)})")
            source_files = source_files[:max_files]
        
        # Display file count
        console.print(f"Found {len(source_files)} source files to process")
        
        # Extract blocks
        console.print("Extracting blocks...")
        blocks = extract_all_blocks(Path(repo_path))
        console.print(f"Extracted {len(blocks)} blocks")
        
        # Analyze hierarchies
        console.print("Analyzing hierarchies...")
        hierarchies = analyze_hierarchies(blocks)
        console.print(f"Analyzed {len(hierarchies)} file hierarchies")
        
        # Enrich blocks with hierarchy
        console.print("Enriching blocks with hierarchy information...")
        enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
        console.print(f"Enriched {len(enriched_blocks)} blocks")
        
        # Create QA-compatible blocks
        console.print("Creating QA-compatible blocks...")
        qa_blocks = create_qa_compatible_blocks(enriched_blocks)
        console.print(f"Created {len(qa_blocks)} QA-compatible blocks")
        
        # Create QA-compatible output
        console.print("Creating QA-compatible output...")
        output = create_qa_compatible_output(qa_blocks)
        
        # Write output to file
        output_file = output_path / f"extraction_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        # Display statistics
        table = Table(title="Extraction Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Count different types of blocks
        block_types = {}
        for block in blocks:
            block_type = block.get("type", "unknown")
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        table.add_row("Total files processed", str(block_types.get("file", 0)))
        table.add_row("Total blocks extracted", str(len(blocks)))
        table.add_row("Code blocks", str(block_types.get("function", 0) + 
                                         block_types.get("class", 0) + 
                                         block_types.get("method", 0)))
        table.add_row("Markdown sections", str(block_types.get("section", 0)))
        table.add_row("Output format", format)
        
        console.print(table)
        
        # Display format-specific information
        if isinstance(output, list):
            console.print(f"[bold]Deepseek Format:[/] {len(output)} sections")
            # Count elements inside sections
            table_count = sum(len(s.get("tables", [])) for s in output)
            code_count = sum(len(s.get("code", [])) for s in output)
            image_count = sum(len(s.get("images", [])) for s in output)
            console.print(f"  • {table_count} tables, {code_count} code blocks, {image_count} images")
        else:
            console.print(f"[bold]Standard Format:[/] {len(output.get('sections', []))} sections")
        
        console.print(f"[bold green]✓[/] Extraction completed successfully.")
        console.print(f"[bold]Output:[/] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error during extraction:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--temps', '-t', type=float, multiple=True, default=[0.3, 0.5, 0.7],
              help='Temperature values for generation')
@click.option('--min-reasoning', '-m', type=int, default=15,
              help='Minimum words in reasoning')
@click.option('--similarity', '-s', type=float, default=0.85,
              help='Similarity threshold for deduplication')
@click.option('--bidirectional/--no-bidirectional', default=True,
              help='Enable/disable bidirectional generation')
@click.pass_context
def qa(ctx, input_file, output_file, temps, min_reasoning, similarity, bidirectional):
    """Generate QA pairs from extraction output.
    
    INPUT_FILE is the path to the extraction output JSON file.
    OUTPUT_FILE is the path where the QA pairs will be saved.
    """
    console.print(Panel(f"[bold blue]Generating QA Pairs:[/] [green]{input_file}[/]", 
                        title="DuaLipa QA Generator"))
    
    try:
        # Try to import QA modules - using method_validator to validate first
        try:
            # Validate QA methods exist
            qa_methods = [
                ("agent_tools.dualipa.qa.processor", "process_extraction_json"),
                ("agent_tools.dualipa.qa.models.qa_models", "QAGenerationConfig")
            ]
            
            all_valid = True
            for package, method in qa_methods:
                is_valid, message = validate_method(package, method)
                if not is_valid:
                    console.print(f"[bold yellow]Warning:[/] {method} not found: {message}")
                    all_valid = False
            
            if not all_valid:
                console.print("[bold yellow]Required QA methods are not available.[/]")
                console.print("Attempting to use subprocess as fallback...")
                raise ImportError("QA method validation failed")
                
            # If all methods validated, import them directly
            from agent_tools.dualipa.qa.processor import process_extraction_json
            from agent_tools.dualipa.qa.models.qa_models import QAGenerationConfig
        except ImportError:
            console.print("[bold yellow]Warning:[/] Could not import QA modules directly.")
            console.print("Make sure the package is installed or the path is correct.")
            console.print("Attempting to use subprocess to run the QA CLI...")
            
            # Fall back to subprocess
            import subprocess
            args = [
                "python", "-m", "agent_tools.dualipa.qa.cli",
                input_file,
                f"--output={output_file}",
                f"--temps={' '.join(str(t) for t in temps)}",
                f"--min-reasoning={min_reasoning}",
                f"--similarity={similarity}"
            ]
            if ctx.obj.get('DEBUG'):
                args.append("--verbose")
            if not bidirectional:
                args.append("--no-bidirectional")
                
            console.print(f"Running command: {' '.join(args)}")
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            console.print(result.stdout)
            if result.stderr:
                console.print("[bold yellow]Warnings/Errors:[/]")
                console.print(result.stderr)
            
            console.print(f"[bold green]✓[/] QA generation completed.")
            console.print(f"[bold]Output:[/] {output_file}")
            return
        
        # If import succeeded, use the modules directly
        console.print("Setting up QA generation...")
        
        # Create configuration
        config = QAGenerationConfig(
            temperature_range=list(temps),
            min_reasoning_words=min_reasoning,
            similarity_threshold=similarity
        )
        
        # Process extraction JSON asynchronously
        async def run_qa():
            console.print(f"Processing with temperatures: {temps}")
            response = await process_extraction_json(
                input_data=input_file,
                output_file=output_file,
                config=config,
                enable_bidirectional=bidirectional
            )
            return response
        
        # Run the async function
        response = asyncio.run(run_qa())
        
        # Display statistics
        table = Table(title="QA Generation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total QA pairs", str(len(response.qa_pairs)))
        table.add_row("Forward pairs", str(response.generation_metadata.get("forward_pairs", 0)))
        table.add_row("Reverse pairs", str(response.generation_metadata.get("reverse_pairs", 0)))
        table.add_row("Processing time", f"{response.generation_metadata.get('processing_time_seconds', 0):.2f} seconds")
        table.add_row("Temperatures", ", ".join(f"{t:.1f}" for t in temps))
        table.add_row("Bidirectional", "Yes" if bidirectional else "No")
        
        console.print(table)
        
        # Display sample QA pair
        if response.qa_pairs:
            console.print("[bold]Sample QA Pair:[/]")
            pair = response.qa_pairs[0]
            console.print(f"[bold cyan]Q:[/] {pair.question}")
            console.print(f"[bold green]A:[/] {pair.answer}")
            console.print(f"[bold yellow]Reasoning:[/] {pair.reasoning}")
        
        console.print(f"[bold green]✓[/] QA generation completed successfully.")
        console.print(f"[bold]Output:[/] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error during QA generation:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--max-files', type=int, default=1000, help='Maximum number of files to extract')
@click.option('--temps', '-t', type=float, multiple=True, default=[0.3, 0.5, 0.7],
              help='Temperature values for QA generation')
@click.option('--min-reasoning', '-m', type=int, default=15,
              help='Minimum words in reasoning')
@click.option('--similarity', '-s', type=float, default=0.85,
              help='Similarity threshold for deduplication')
@click.option('--bidirectional/--no-bidirectional', default=True,
              help='Enable/disable bidirectional generation')
@click.option('--skip-extraction', is_flag=True, help='Skip extraction stage')
@click.option('--skip-qa', is_flag=True, help='Skip QA generation stage')
@click.option('--run-train', is_flag=True, help='Run training stage')
@click.pass_context
def pipeline(ctx, repo_path, output_dir, max_files, temps, min_reasoning, 
             similarity, bidirectional, skip_extraction, skip_qa, run_train):
    """Run the complete DuaLipa pipeline.
    
    This command runs extraction, QA generation, and optionally training
    in a single workflow.
    
    REPO_PATH is the path to the repository to process.
    OUTPUT_DIR is the directory where outputs will be saved.
    """
    console.print(Panel(f"[bold blue]Running DuaLipa Pipeline:[/] [green]{repo_path}[/]", 
                        title="DuaLipa Pipeline"))
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        extraction_file = output_path / "extraction_output.json"
        qa_file = output_path / "qa_output.json"
        
        # Track stages completion
        stages = {
            "extraction": False,
            "qa": False,
            "train": False
        }
        
        # Run extraction if not skipped
        if not skip_extraction:
            console.print("[bold blue]Stage 1: Extraction[/]")
            ctx.invoke(
                extract,
                repo_path=repo_path,
                output_dir=output_dir,
                max_files=max_files,
                format='standard'  # Always use standard format for pipeline
            )
            stages["extraction"] = True
            
            # Update extraction file path - find the latest extraction file
            extraction_files = list(output_path.glob("extraction_output_*.json"))
            if extraction_files:
                latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
                extraction_file = latest_file
        else:
            console.print("[bold yellow]Skipping extraction stage[/]")
            # Check if there's an existing extraction file
            extraction_files = list(output_path.glob("extraction_output_*.json"))
            if extraction_files:
                latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
                extraction_file = latest_file
                console.print(f"Using existing extraction file: {extraction_file}")
            else:
                console.print("[bold red]No existing extraction file found![/]")
                console.print("Either run extraction first or provide an extraction file")
                sys.exit(1)
        
        # Validate extraction output is QA-compatible
        console.print("\n[bold blue]Validating Extraction Output[/]")
        try:
            # Import validation functions
            try:
                sys.path.insert(0, "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end")
                from validate_qa_compatibility import load_json_file, validate_qa_compatibility
                from validate_qa_compatibility import convert_deepseek_to_qa_format, adapt_standard_format
                
                # Load file
                data = load_json_file(Path(extraction_file))
                
                # Validate
                validation_results = validate_qa_compatibility(data)
                
                # If not valid, convert it
                qa_input_file = extraction_file
                if not validation_results["valid"]:
                    console.print("[bold yellow]Extraction output is not QA-compatible.[/]")
                    console.print("Converting to QA-compatible format...")
                    
                    # Convert the data
                    qa_input = None
                    if isinstance(data, list):
                        qa_input = convert_deepseek_to_qa_format(data)
                    elif isinstance(data, dict):
                        qa_input = adapt_standard_format(data)
                    
                    if qa_input:
                        # Save to a new file
                        qa_input_file = output_path / f"qa_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(qa_input_file, 'w', encoding='utf-8') as f:
                            json.dump(qa_input, f, indent=2)
                        console.print(f"[bold green]✓[/] Saved QA-compatible format to {qa_input_file}")
                else:
                    console.print("[bold green]✓[/] Extraction output is already QA-compatible.")
            except ImportError:
                console.print("[bold yellow]Warning:[/] Could not import validation module.")
                console.print("Skipping validation step, proceeding with original extraction file.")
                qa_input_file = extraction_file
        except Exception as e:
            console.print(f"[bold yellow]Warning: Validation failed - {str(e)}[/]")
            console.print("Proceeding with original extraction file.")
            qa_input_file = extraction_file
            
        # Run QA generation if not skipped
        if not skip_qa:
            console.print("\n[bold blue]Stage 2: QA Generation[/]")
            ctx.invoke(
                qa,
                input_file=str(qa_input_file),
                output_file=str(qa_file),
                temps=temps,
                min_reasoning=min_reasoning,
                similarity=similarity,
                bidirectional=bidirectional
            )
            stages["qa"] = True
        else:
            console.print("[bold yellow]Skipping QA generation stage[/]")
        
        # Run training if requested
        if run_train:
            console.print("\n[bold blue]Stage 3: Training[/]")
            console.print("[bold yellow]Training stage not implemented yet[/]")
            console.print("This will be implemented in a future version")
        
        # Display pipeline summary
        console.print("\n[bold blue]Pipeline Summary[/]")
        table = Table(title="Pipeline Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        
        for stage, completed in stages.items():
            status = "[bold green]✓ Completed[/]" if completed else "[bold yellow]⨯ Skipped[/]"
            table.add_row(stage.capitalize(), status)
        
        console.print(table)
        
        console.print(f"[bold green]✓[/] Pipeline completed successfully.")
        console.print(f"[bold]Output Directory:[/] {output_path.absolute()}")
        
    except Exception as e:
        console.print(f"[bold red]Error running pipeline:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display information about DuaLipa components."""
    console.print(Panel("[bold blue]DuaLipa: Dual LLM-Informed Python Automation[/]", 
                       title="Package Information"))
    
    console.print("[bold]Version:[/]", __version__)
    
    # Detect components
    components = [
        {
            "name": "Extraction Module",
            "path": "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end",
            "files": [
                "extraction_blocks.py",
                "hierarchy_analyzer.py",
                "qa_formatter.py",
                "validation.py"
            ]
        },
        {
            "name": "QA Validation Module",
            "path": "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end",
            "files": [
                "validate_qa_compatibility.py",
                "test_extraction_qa_integration.py"
            ]
        },
        {
            "name": "QA Generation Module",
            "path": "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/qa",
            "files": [
                "processor.py",
                "models/qa_models.py",
                "llm/generation.py",
                "cli.py"
            ]
        }
    ]
    
    for component in components:
        table = Table(title=component["name"])
        table.add_column("File", style="cyan")
        table.add_column("Status", style="green")
        
        all_exist = True
        for file in component["files"]:
            file_path = os.path.join(component["path"], file)
            exists = os.path.exists(file_path)
            status = "✓" if exists else "✗"
            all_exist = all_exist and exists
            table.add_row(file, status)
        
        console.print(table)
        if all_exist:
            console.print(f"[bold green]✓[/] {component['name']} is available")
        else:
            console.print(f"[bold yellow]![/] {component['name']} is partially available")
    
    console.print("\n[bold]Usage Examples:[/]")
    console.print("[cyan]# Extract from a repository[/]")
    console.print("dualipa extract ./my_repo ./output")
    console.print("\n[cyan]# Validate extraction output for QA compatibility[/]")
    console.print("dualipa validate ./output/extraction_output.json --convert")
    console.print("\n[cyan]# Generate QA pairs[/]")
    console.print("dualipa qa ./output/extraction_output.json ./output/qa_output.json")
    console.print("\n[cyan]# Run complete pipeline[/]")
    console.print("dualipa pipeline ./my_repo ./output")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--convert', is_flag=True, help='Convert to QA-compatible format and save')
@click.option('--output', type=str, help='Path to save converted output (only with --convert)')
@click.pass_context
def validate(ctx, input_file, convert, output):
    """Validate extraction output for QA compatibility.
    
    INPUT_FILE is the path to the extraction output JSON file to validate.
    """
    console.print(Panel(f"[bold blue]Validating extraction output:[/] [green]{input_file}[/]", 
                        title="DuaLipa Validator"))
    
    try:
        # Try to import validation module
        try:
            # First validate the methods exist using method_validator
            package_path = "agent_tools.dualipa.extraction.examples.end_to_end.validate_qa_compatibility"
            methods_to_validate = [
                ("load_json_file", "Loading JSON files"),
                ("validate_qa_compatibility", "Validating extraction output"),
                ("convert_deepseek_to_qa_format", "Converting DeepSeek format"),
                ("adapt_standard_format", "Adapting standard format")
            ]
            
            all_valid = True
            for method_name, purpose in methods_to_validate:
                is_valid, message = validate_method(package_path, method_name)
                if not is_valid:
                    console.print(f"[bold yellow]Warning:[/] Method for {purpose} not found: {message}")
                    all_valid = False
            
            if not all_valid:
                console.print("[bold yellow]Some validation methods are not available.[/]")
                console.print("Attempting to use subprocess as fallback...")
                raise ImportError("Method validation failed")
                
            # If all methods validated, import them directly
            sys.path.insert(0, "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end")
            from validate_qa_compatibility import load_json_file, validate_qa_compatibility
            from validate_qa_compatibility import convert_deepseek_to_qa_format, adapt_standard_format
        except ImportError:
            console.print("[bold yellow]Warning:[/] Could not import validation module directly.")
            console.print("Attempting to use subprocess to run the validation script...")
            
            # Fall back to subprocess
            import subprocess
            cmd = ["python", "/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end/validate_qa_compatibility.py", 
                   input_file]
            
            if convert:
                cmd.append("--convert")
                if output:
                    cmd.append("--output")
                    cmd.append(output)
            
            console.print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            console.print(result.stdout)
            if result.stderr:
                console.print("[bold yellow]Warnings/Errors:[/]")
                console.print(result.stderr)
            
            return
        
        # Load the input file
        input_path = Path(input_file)
        data = load_json_file(input_path)
        if data is None:
            console.print("[bold red]Error:[/] Failed to load JSON data from file")
            sys.exit(1)
        
        # Validate QA compatibility
        results = validate_qa_compatibility(data)
        
        # Report results
        validation_table = Table(title="Validation Results")
        validation_table.add_column("Metric", style="cyan")
        validation_table.add_column("Value", style="green")
        
        validation_table.add_row("Valid", "✓ Yes" if results["valid"] else "✗ No")
        validation_table.add_row("QA Compatible", "✓ Yes" if results["qa_compatible"] else "✗ No")
        validation_table.add_row("Sections Count", str(results["sections_count"]))
        validation_table.add_row("Errors", str(len(results["errors"])))
        validation_table.add_row("Warnings", str(len(results["warnings"])))
        
        console.print(validation_table)
        
        # Show errors and warnings
        if results["errors"]:
            console.print("[bold red]Errors:[/]")
            for error in results["errors"]:
                console.print(f"  • {error}")
        
        if results["warnings"]:
            console.print("[bold yellow]Warnings:[/]")
            for warning in results["warnings"]:
                console.print(f"  • {warning}")
        
        # Create and save the QA-compatible format if requested
        if convert:
            if not output:
                output_path = input_path.with_name(f"{input_path.stem}_qa_compatible.json")
            else:
                output_path = Path(output)
            
            # Convert to QA input format if needed
            qa_input = None
            
            if isinstance(data, list):
                qa_input = convert_deepseek_to_qa_format(data)
            elif isinstance(data, dict):
                qa_input = adapt_standard_format(data)
            
            if qa_input:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_input, f, indent=2)
                console.print(f"[bold green]✓[/] Saved QA-compatible format to {output_path}")
        
        # Show result
        if results["valid"]:
            console.print("[bold green]✓[/] Validation successful. The extraction output is QA-compatible.")
        else:
            console.print("[bold red]✗[/] Validation failed. The extraction output requires conversion.")
            if not convert:
                console.print("[bold yellow]Tip:[/] Use the --convert flag to convert to QA-compatible format")
        
    except Exception as e:
        console.print(f"[bold red]Error during validation:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('component', type=click.Choice(
    ["extract", "qa", "validate", "pipeline", "all"]
))
@click.pass_context
def demo(ctx, component):
    """Run demonstration for a component.
    
    COMPONENT is the component to demonstrate ('extract', 'qa', 'validate', 'pipeline', or 'all').
    """
    console.print(Panel(f"[bold blue]Running demonstration for:[/] [green]{component}[/]", 
                        title="DuaLipa Demonstrator"))
    
    # Use a sample repository for demo purposes
    sample_repo = Path(__file__).parent / "examples" / "sample_repo"
    if not sample_repo.exists():
        console.print("[bold yellow]Sample repository not found.[/]")
        console.print("Creating a minimal sample repository...")
        sample_repo.mkdir(parents=True, exist_ok=True)
        
        # Create a simple Python file
        with open(sample_repo / "example.py", "w") as f:
            f.write("""
def hello_world():
    \"\"\"Print a greeting.\"\"\"
    print("Hello, World!")

class Example:
    \"\"\"An example class.\"\"\"
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        \"\"\"Greet the user.\"\"\"
        return f"Hello, {self.name}!"
""")
        
        # Create a simple Markdown file
        with open(sample_repo / "README.md", "w") as f:
            f.write("""# Sample Repository

This is a sample repository for DuaLipa demos.

## Features

- Python code
- Markdown documentation
- Examples of functions and classes

## Usage

```python
from example import hello_world

hello_world()  # Prints "Hello, World!"
```

## Table Example

| Name | Description |
|------|-------------|
| hello_world | A simple greeting function |
| Example | A class with greeting capabilities |
""")
    
    # Create a temporary directory for output
    import tempfile
    output_dir = Path(tempfile.mkdtemp(prefix="dualipa_demo_"))
    
    try:
        if component in ["extract", "all"]:
            console.print("\n[bold blue]=== Extraction Demonstration ===[/]")
            ctx.invoke(
                extract,
                repo_path=str(sample_repo),
                output_dir=str(output_dir),
                max_files=10,
                format='standard'
            )
        
        if component in ["validate", "all"]:
            # Skip if no extraction file exists
            extraction_files = list(output_dir.glob("extraction_output_*.json"))
            if not extraction_files and component == "validate":
                console.print("[bold yellow]No extraction output available.[/]")
                console.print("Running extraction first...")
                ctx.invoke(
                    extract,
                    repo_path=str(sample_repo),
                    output_dir=str(output_dir),
                    max_files=10,
                    format='standard'
                )
                extraction_files = list(output_dir.glob("extraction_output_*.json"))
            
            if extraction_files:
                console.print("\n[bold blue]=== Validation Demonstration ===[/]")
                latest_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
                ctx.invoke(
                    validate,
                    input_file=str(latest_file),
                    convert=True,
                    output=str(output_dir / "qa_compatible_output.json")
                )
                
        if component in ["qa", "all"]:
            # Skip if no extraction file exists
            extraction_files = list(output_dir.glob("extraction_output_*.json"))
            qa_compat_files = list(output_dir.glob("qa_compatible_output.json"))
            
            if not extraction_files and component == "qa":
                console.print("[bold yellow]No extraction output available.[/]")
                console.print("Running extraction first...")
                ctx.invoke(
                    extract,
                    repo_path=str(sample_repo),
                    output_dir=str(output_dir),
                    max_files=10,
                    format='standard'
                )
                extraction_files = list(output_dir.glob("extraction_output_*.json"))
            
            input_file = None
            if qa_compat_files:
                # Use already validated file if available
                input_file = qa_compat_files[0]
            elif extraction_files:
                # Otherwise use latest extraction file
                input_file = max(extraction_files, key=lambda x: x.stat().st_mtime)
            
            if input_file:
                console.print("\n[bold blue]=== QA Generation Demonstration ===[/]")
                ctx.invoke(
                    qa,
                    input_file=str(input_file),
                    output_file=str(output_dir / "qa_output.json"),
                    temps=[0.5],
                    min_reasoning=10,
                    similarity=0.85,
                    bidirectional=True
                )
        
        if component in ["pipeline", "all"]:
            console.print("\n[bold blue]=== Pipeline Demonstration ===[/]")
            ctx.invoke(
                pipeline,
                repo_path=str(sample_repo),
                output_dir=str(output_dir / "pipeline"),
                max_files=10,
                temps=[0.5],
                min_reasoning=10,
                similarity=0.85,
                bidirectional=True
            )
        
        console.print(f"[bold green]✓[/] Demonstration completed successfully.")
        console.print(f"[bold]Output Directory:[/] {output_dir}")
        
    except Exception as e:
        console.print(f"[bold red]Error during demonstration:[/] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()