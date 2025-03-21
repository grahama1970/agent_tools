"""Test QA processor with minimal pipeline.

This module tests the minimal pipeline real data test for the MVP.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

import json
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mark test as async
pytestmark = pytest.mark.asyncio


async def test_minimal_pipeline_real_data(sample_extraction_json, tmp_path):
    """Test the minimal pipeline with real data (MVP test).
    
    Input: Real JSON.
    Expect: One Q&A pair written to file.
    
    As specified in Task 0.2 of task.md:
    - Input: Real JSON.
    - Expect: One Q&A pair written to file.
    """
    # Import the function to test
    try:
        from agent_tools.dualipa.qa.processor import process_extraction_json
        from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Set up output path
    output_file = tmp_path / "qa_output.json"
    
    # Create a mock for the batch_process_sections function to return mock QA pairs
    mock_qa_pair = QAPair(
        question="What does this module do?",
        answer="This module provides QA generation capabilities. It extracts questions and answers from documentation.",
        reasoning="Based on the module's name and structure, it appears to be focused on QA generation. Oh wait?! Looking at the code more carefully, it specifically handles extraction from documentation sections and code samples.",
        direction=Direction.FORWARD
    )
    
    # Mock the batch processing to avoid actual LLM calls
    with patch('agent_tools.dualipa.qa.utils.batch_processing.batch_process_sections') as mock_batch:
        # Return a mock result for each section
        mock_batch.return_value = [[mock_qa_pair] for _ in range(len(sample_extraction_json["sections"]))]
        
        # Run the process
        response = await process_extraction_json(
            input_data=sample_extraction_json,
            output_file=output_file
        )
    
    # Verify that the response is returned
    assert response is not None
    
    # Verify that the file was written
    assert output_file.exists()
    
    # Read the file and check content
    with open(output_file, 'r') as f:
        output_json = json.load(f)
        assert "qa_pairs" in output_json
        assert len(output_json["qa_pairs"]) >= 1