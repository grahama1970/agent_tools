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

# Package version
__version__ = "0.1.0"

# Core module imports for external use
from .code_extractor import extract_repository
from .format_dataset import format_for_lora
from .llm_generator import generate_code_related_qa_pairs, generate_markdown_related_qa_pairs
from .qa_validator import validate_and_enhance_qa_pairs
from .utils import format_string, ensure_directory
from .cli import main, cli, demo_main

# CLI entrypoint
if __name__ == "__main__":
    from .cli import demo_main
    demo_main()
