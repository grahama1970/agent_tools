"""DuaLipa LLM Q&A Generation Module.

This module provides functionality for generating question-answer pairs from content.

Usage:
    ```python
    from agent_tools.dualipa.qa import process_extraction_json
    
    # Generate QA pairs from content
    response = await process_extraction_json(
        input_data="extraction.json",
        output_file="qa_pairs.json"
    )
    ```
"""

from .models.qa_models import QAPair, QAResponse
from .processor import process_extraction_json

__all__ = ["process_extraction_json", "QAPair", "QAResponse"]