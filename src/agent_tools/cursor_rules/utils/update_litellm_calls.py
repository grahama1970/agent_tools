"""
Script to update litellm.acompletion calls to use acompletion_with_retry.

This script scans the codebase for instances of litellm.acompletion and
provides instructions on how to update them to use the new acompletion_with_retry
function with tenacity retries.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

from loguru import logger


def find_python_files(root_dir: str) -> List[Path]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(os.path.join(root, file)))
    return python_files


def find_litellm_acompletion_calls(file_path: Path) -> List[Tuple[int, str]]:
    """Find all lines containing litellm.acompletion calls in the given file."""
    matches = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if "litellm.acompletion" in line:
                matches.append((i, line.strip()))
    return matches


def check_for_import(file_path: Path) -> bool:
    """Check if the file already imports acompletion_with_retry."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        return (
            "from sparta.utils.litellm_utils import acompletion_with_retry" in content
        )


def generate_update_instructions(
    file_path: Path, matches: List[Tuple[int, str]], has_import: bool
) -> str:
    """Generate instructions for updating the file."""
    if not matches:
        return ""

    instructions = f"\n\nFile: {file_path}\n"

    if not has_import:
        instructions += "1. Add the following import:\n"
        instructions += (
            "   from sparta.utils.litellm_utils import acompletion_with_retry\n\n"
        )

    instructions += "2. Replace the following litellm.acompletion calls:\n\n"

    for line_num, line in matches:
        instructions += f"   Line {line_num}: {line}\n"
        # Generate the replacement line
        replacement = line.replace("litellm.acompletion", "acompletion_with_retry")
        instructions += f"   With: {replacement}\n\n"

    return instructions


def main():
    """Main function to scan the codebase and generate update instructions."""
    # Get the project root directory
    project_root = Path(
        __file__
    ).parent.parent.parent  # Assuming this script is in src/sparta/utils

    logger.info(f"Scanning Python files in {project_root}")

    # Find all Python files
    python_files = find_python_files(project_root)
    logger.info(f"Found {len(python_files)} Python files")

    # Files that need updates
    files_to_update = []
    all_instructions = ""

    # Check each file for litellm.acompletion calls
    for file_path in python_files:
        # Skip the litellm_utils.py file itself
        if file_path.name == "litellm_utils.py":
            continue

        matches = find_litellm_acompletion_calls(file_path)
        if matches:
            has_import = check_for_import(file_path)
            instructions = generate_update_instructions(file_path, matches, has_import)
            if instructions:
                files_to_update.append(file_path)
                all_instructions += instructions

    # Print summary
    logger.info(f"Found {len(files_to_update)} files that need updates")

    if files_to_update:
        logger.info("Update instructions:")
        print(all_instructions)

        # Save instructions to a file
        output_file = project_root / "litellm_update_instructions.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Instructions for updating litellm.acompletion calls\n\n")
            f.write(
                "The following files contain litellm.acompletion calls that should be updated to use acompletion_with_retry:\n"
            )
            f.write(all_instructions)

        logger.info(f"Instructions saved to {output_file}")
    else:
        logger.info("No files need updates")


if __name__ == "__main__":
    main()
