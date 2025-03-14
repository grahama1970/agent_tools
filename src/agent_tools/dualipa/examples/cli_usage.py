#!/usr/bin/env python3
"""
Example usage of the DuaLipa CLI with the new Click interface.

This script demonstrates how to use the CLI to perform the core functions:
1. Extract code from a repository
2. Format the extracted data into QA pairs
3. Train a LoRA model

Official Documentation References:
- Click: https://click.palletsprojects.com/
- Rich: https://rich.readthedocs.io/
"""

import os
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

# Example repository (small and public for quick demonstration)
EXAMPLE_REPO = "https://github.com/huggingface/transformers"
TEMP_DIR = Path("./dualipa_example_output")


def run_command(command):
    """Run a shell command and print the result."""
    console.print(f"[bold blue]Running command:[/] [green]{command}[/]")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        console.print("[bold green]Command completed successfully![/]")
        if result.stdout:
            console.print("[bold]Output:[/]")
            console.print(result.stdout)
    else:
        console.print("[bold red]Command failed![/]")
        if result.stderr:
            console.print("[bold]Error:[/]")
            console.print(result.stderr)
    
    return result.returncode == 0


def main():
    """Run the example CLI commands."""
    # Create output directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    console.print(f"[bold]Created output directory: {TEMP_DIR}[/]")
    
    console.print("\n[bold cyan]DuaLipa CLI Example[/]\n", style="bold", justify="center")
    
    # Show help
    console.print("\n[bold yellow]1. Show DuaLipa Help[/]")
    run_command("python -m agent_tools.dualipa --help")
    
    # Show command-specific help
    console.print("\n[bold yellow]2. Show Extract Command Help[/]")
    run_command("python -m agent_tools.dualipa extract --help")
    
    # Extract from repository (limiting to a small subset for demo)
    console.print("\n[bold yellow]3. Extract Code from Repository[/]")
    extract_cmd = f"python -m agent_tools.dualipa extract {EXAMPLE_REPO} {TEMP_DIR / 'extracted'} --extensions .md --ignore __pycache__ docs examples tests"
    run_command(extract_cmd)
    
    # Format the extracted data
    console.print("\n[bold yellow]4. Format Extracted Data into QA Pairs[/]")
    format_cmd = f"python -m agent_tools.dualipa format {TEMP_DIR / 'extracted' / 'extracted_data.json'} {TEMP_DIR / 'formatted_data.json'}"
    run_command(format_cmd)
    
    # Show train help (skip actual training as it requires significant resources)
    console.print("\n[bold yellow]5. Show Train Command Help[/]")
    run_command("python -m agent_tools.dualipa train --help")
    
    # Debug mode demonstration
    console.print("\n[bold yellow]6. Run Debug Mode[/]")
    run_command("python -m agent_tools.dualipa --debug debug markdown")
    
    console.print("\n[bold green]Example completed![/]", style="bold", justify="center")
    console.print(f"Output files can be found in: {TEMP_DIR}")


if __name__ == "__main__":
    main() 