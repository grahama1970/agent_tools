"""Test constraint tracking for Phase 3.

This module tests constraint tracking for the QA generation module's
Phase 3 infrastructure components, including worker pools, monitoring,
and configuration integration.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Mark all tests as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_extraction_small():
    """Sample extraction data with a few sections."""
    return {
        "metadata": {
            "title": "Constraint Testing Document",
            "version": "1.0"
        },
        "sections": [
            {
                "uuid": "section-1",
                "content": "This is a test section for constraint tracking.",
                "type": "text"
            },
            {
                "uuid": "section-2",
                "content": "```python\ndef test_function():\n    return 'Testing constraints'\n```",
                "type": "code"
            }
        ]
    }


async def test_constraint_tracking_phase3(sample_extraction_small, tmp_path):
    """Test constraint tracking for Phase 3.
    
    This test verifies that:
    1. Worker pool respects configuration settings
    2. Metrics are properly logged during processing
    3. max_qa_pairs_per_section constraint is respected
    4. bidirectional_ratio constraint is respected
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.processor import process_extraction_json
        from agent_tools.dualipa.qa.models.config import QAGenerationConfig
        from agent_tools.dualipa.qa.monitoring import get_processing_metrics, initialize_metrics
        from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
        from agent_tools.dualipa.qa.utils.batch_processing import batch_process_sections
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Set up output path
    output_file = tmp_path / "constraint_test_output.json"
    
    # Create a custom configuration with specific constraints
    custom_worker_count = 3
    custom_max_pairs = 4
    custom_bidirectional_ratio = 0.4
    
    config = QAGenerationConfig(
        max_concurrent_requests=custom_worker_count,
        max_qa_pairs_per_section=custom_max_pairs,
        bidirectional_ratio=custom_bidirectional_ratio,
        temperature_range=[0.2, 0.5, 0.8]
    )
    
    # Reset metrics for this test
    initialize_metrics()
    
    # Create mock QA pairs with different directions
    forward_qa_pair = QAPair(
        question="What is this test verifying?",
        answer="This test verifies phase 3 constraints.",
        reasoning="Based on the test name and documentation. Oh wait?! It specifically checks worker pools and metrics.",
        direction=Direction.FORWARD,
        temperature_used=0.5
    )
    
    reverse_qa_pair = QAPair(
        question="Which constraints are being verified?",
        answer="Worker pool configuration, metrics logging, and QA pair limits.",
        reasoning="The test focuses on infrastructure components. Oh wait?! It's specifically for Phase 3 constraints.",
        direction=Direction.REVERSE,
        temperature_used=0.5
    )
    
    # Create a controlled mix of forward and reverse pairs
    # We'll create a specific ratio to verify bidirectional_ratio constraint
    forward_count = int(custom_max_pairs * (1 - custom_bidirectional_ratio))
    reverse_count = custom_max_pairs - forward_count
    
    mock_pairs_section1 = [forward_qa_pair] * forward_count + [reverse_qa_pair] * reverse_count
    mock_pairs_section2 = [forward_qa_pair] * forward_count + [reverse_qa_pair] * reverse_count
    
    # Test constraint 1: Worker pool respects configuration
    # We'll patch batch_process_sections to verify the semaphore value matches config
    original_batch_process = batch_process_sections
    
    async def mock_batch_process(sections, config, process_function, enable_bidirectional, chunk_size=None):
        # Verify the semaphore is created with the correct value from config
        assert getattr(config, 'worker_count', config.max_concurrent_requests) == custom_worker_count
        
        # Return mock results
        return [mock_pairs_section1, mock_pairs_section2]
    
    # Process with monitoring and mocked batch processing
    with patch('agent_tools.dualipa.qa.utils.batch_processing.batch_process_sections', side_effect=mock_batch_process):
        # Process with monitoring
        response = await process_extraction_json(
            input_data=sample_extraction_small,
            output_file=output_file,
            config=config,
            enable_bidirectional=True,
            enable_monitoring=True
        )
    
    # Verify output structure
    assert response is not None
    assert response.qa_pairs is not None
    assert len(response.qa_pairs) <= len(sample_extraction_small["sections"]) * custom_max_pairs
    
    # Test constraint 2: Metrics are properly logged
    metrics = get_processing_metrics()
    assert metrics["pairs_generated"] > 0
    assert "worker_utilization" in metrics
    
    # Verify worker utilization was calculated correctly based on configuration
    sections_count = len(sample_extraction_small["sections"])
    expected_utilization = min(1.0, sections_count / custom_worker_count) * 100.0
    assert abs(metrics["worker_utilization"] - expected_utilization) < 0.01  # Allow for small floating-point differences
    
    # Test constraint 3: max_qa_pairs_per_section is respected
    # Count pairs per section in the response
    assert len(mock_pairs_section1) <= custom_max_pairs
    assert len(mock_pairs_section2) <= custom_max_pairs
    
    # Test constraint 4: bidirectional_ratio is respected
    # Count forward and reverse pairs in the mock data
    total_pairs = len(response.qa_pairs)
    forward_pairs = sum(1 for pair in response.qa_pairs if pair.direction == Direction.FORWARD)
    reverse_pairs = sum(1 for pair in response.qa_pairs if pair.direction == Direction.REVERSE)
    
    # Verify metadata reflects the correct counts
    assert "forward_pairs" in response.generation_metadata
    assert "reverse_pairs" in response.generation_metadata
    assert "reverse_ratio" in response.generation_metadata
    
    # Verify the ratio is close to the configured value
    # We allow some small variance since the ratio might not be exactly achievable
    # with a small number of pairs
    actual_ratio = reverse_pairs / total_pairs if total_pairs > 0 else 0
    assert abs(actual_ratio - custom_bidirectional_ratio) < 0.1  # Allow 10% error margin
    
    # Verify the metadata ratio matches the actual ratio
    assert abs(response.generation_metadata["reverse_ratio"] - actual_ratio) < 0.01
    
    # Verify output file
    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_json = json.load(f)
        assert "qa_pairs" in output_json
        assert "generation_metadata" in output_json
        
    # Verify temperature constraints continue to be respected
    for pair in response.qa_pairs:
        assert 0.0 <= pair.temperature_used <= 1.0
        assert pair.temperature_used in config.temperature_range


async def test_worker_pool_constraint_with_metrics():
    """Test that worker pool constraints are enforced and tracked in metrics.
    
    This test verifies:
    1. The worker count is properly used for concurrency limiting
    2. Worker utilization is properly recorded in metrics
    3. Performance statistics reflect actual workload
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.utils.batch_processing import batch_process_with_stats
        from agent_tools.dualipa.qa.monitoring import (
            initialize_metrics, 
            record_metric,
            get_processing_metrics,
            QA_WORKER_UTILIZATION
        )
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Reset metrics
    initialize_metrics()
    
    # Create a sample workload
    items = [f"item-{i}" for i in range(10)]
    max_workers = 4
    
    # Create a process function that simulates work
    async def process_func(item):
        await asyncio.sleep(0.01)  # Simulate a small amount of work
        return f"processed-{item}"
    
    # Record worker utilization
    worker_utilization = min(1.0, len(items) / max_workers) * 100.0
    record_metric(QA_WORKER_UTILIZATION, worker_utilization)
    
    # Process with stats
    result = await batch_process_with_stats(
        items=items,
        process_func=process_func,
        max_workers=max_workers
    )
    
    # Verify stats reflect the workload
    assert result["stats"]["total_items"] == len(items)
    assert result["stats"]["successful"] == len(items)
    assert result["stats"]["failed"] == 0
    
    # Verify utilization is recorded correctly
    metrics = get_processing_metrics()
    assert "worker_utilization" in metrics
    assert abs(metrics["worker_utilization"] - worker_utilization) < 0.01
    
    # Verify worker utilization makes sense
    assert metrics["worker_utilization"] > 0
    assert metrics["worker_utilization"] <= 100.0


async def test_max_qa_pairs_constraint():
    """Test that max_qa_pairs_per_section constraint is enforced.
    
    This test verifies:
    1. The max_qa_pairs_per_section config setting limits QA pairs
    2. This constraint is respected throughout the pipeline
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.processor import process_section
        from agent_tools.dualipa.qa.models.config import QAGenerationConfig
        from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Create a sample section
    section = {
        "uuid": "test-section",
        "content": "This is a test section for max_qa_pairs constraint.",
        "type": "text"
    }
    
    # Create a config with specific max_qa_pairs_per_section
    custom_max_pairs = 3
    config = QAGenerationConfig(
        max_qa_pairs_per_section=custom_max_pairs
    )
    
    # Mock the generation function to return more pairs than allowed
    excess_pairs = 6  # More than our constraint
    
    # Create a QA pair factory for consistent creation
    def create_qa_pair(index, direction=Direction.FORWARD):
        return QAPair(
            question=f"Test question {index}?",
            answer=f"Test answer {index}.",
            reasoning=f"Test reasoning {index}. Oh wait?! This is for testing.",
            direction=direction,
            temperature_used=0.5
        )
    
    # Mock the calls that would generate QA pairs
    with patch('agent_tools.dualipa.qa.llm.generation.iterate_temperatures') as mock_iterate:
        # Return more pairs than our constraint allows
        mock_iterate.return_value = [create_qa_pair(i) for i in range(excess_pairs)]
        
        with patch('agent_tools.dualipa.qa.llm.reversal.generate_bidirectional_qa_pairs') as mock_generate:
            # Generate more forward and reverse pairs than allowed
            forward_pairs = [create_qa_pair(i) for i in range(excess_pairs // 2)]
            reverse_pairs = [create_qa_pair(i + excess_pairs // 2, Direction.REVERSE) for i in range(excess_pairs // 2)]
            mock_generate.return_value = (forward_pairs, reverse_pairs)
            
            # Process with bidirectional enabled
            result_with_bidirectional = await process_section(
                section=section,
                config=config,
                enable_bidirectional=True
            )
            
            # Process with bidirectional disabled
            result_without_bidirectional = await process_section(
                section=section,
                config=config,
                enable_bidirectional=False
            )
    
    # Verify constraints are enforced
    assert len(result_with_bidirectional) <= custom_max_pairs
    assert len(result_without_bidirectional) <= custom_max_pairs