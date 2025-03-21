"""Test monitoring functionality for QA pipeline.

This module tests the monitoring system, metrics collection, and alerting
capabilities for the QA generation pipeline.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- prometheus_client: https://github.com/prometheus/client_python
- logging: https://docs.python.org/3/library/logging.html
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html

Expected input/output:
- test_process_metadata_with_metrics: Tests metrics collection during processing
  Input: Sample extraction data and processing config
  Output: Metrics stored in Prometheus format
- test_error_alerting: Tests alerting when error count exceeds threshold
  Input: Simulated errors above threshold
  Output: Alert triggered via alerting system
- test_cache_metrics_integration: Tests cache metrics integration with monitoring
  Input: Cache hits/misses and batch processing
  Output: Combined metrics showing cache effectiveness
"""

import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import logging

# Mark all tests as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_extraction_full():
    """Full sample extraction data with multiple sections."""
    return {
        "metadata": {
            "title": "Test Document",
            "version": "1.0"
        },
        "sections": [
            {
                "uuid": "section-1",
                "content": "This is a test section with important information about testing.",
                "type": "text"
            },
            {
                "uuid": "section-2",
                "content": "```python\ndef example():\n    return 'This is a code example'\n```",
                "type": "code"
            },
            {
                "uuid": "section-3",
                "content": "## Testing Best Practices\nAlways write tests before implementing.",
                "type": "markdown"
            }
        ]
    }


async def test_process_extraction_json_full_real(sample_extraction_full, tmp_path):
    """Test full pipeline processing with real data.
    
    Input: Complete extraction JSON with multiple section types
    Expect: All sections processed with QA pairs, metrics collected
    """
    # Import the functions to test
    try:
        from agent_tools.dualipa.qa.processor import process_extraction_json
        from agent_tools.dualipa.qa.models.config import QAGenerationConfig
        from agent_tools.dualipa.qa.monitoring import get_processing_metrics, initialize_metrics
        from agent_tools.dualipa.qa.models.qa_models import QAPair, Direction
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Reset metrics for this test
    initialize_metrics()
    
    # Set up output path
    output_file = tmp_path / "qa_full_output.json"
    
    # Create configuration with monitoring enabled
    config = QAGenerationConfig(
        max_concurrent_requests=2,
        temperature_range=[0.3, 0.5],
        max_qa_pairs_per_section=5,
        bidirectional_ratio=0.3
    )
    
    # Create a mock for the batch_process_sections function to return mock QA pairs
    mock_qa_pairs = [
        QAPair(
            question="What is the main point of this content?",
            answer="The content describes testing best practices.",
            reasoning="Based on analyzing the content structure. It specifically mentions writing tests before implementation.",
            direction=Direction.FORWARD
        ),
        QAPair(
            question="Why is test-driven development important?",
            answer="Because it ensures code quality and correctness from the beginning.",
            reasoning="The document emphasizes writing tests before implementing functionality, which is a core TDD principle.",
            direction=Direction.FORWARD
        ),
        QAPair(
            question="What is a code example in the document?",
            answer="A Python function called 'example' that returns 'This is a code example'.",
            reasoning="The document contains a code block with a simple Python function definition.",
            direction=Direction.REVERSE
        )
    ]
    
    # Mock the batch processing to avoid actual LLM calls
    with patch('agent_tools.dualipa.qa.utils.batch_processing.batch_process_sections') as mock_batch:
        # Return mock results for each section
        mock_batch.return_value = [mock_qa_pairs for _ in range(len(sample_extraction_full["sections"]))]
        
        # Process with monitoring
        response = await process_extraction_json(
            input_data=sample_extraction_full,
            output_file=output_file,
            config=config,
            enable_bidirectional=True,
            enable_monitoring=True
        )
    
    # Verify response structure
    assert response is not None
    assert response.qa_pairs is not None
    assert len(response.qa_pairs) > 0
    assert response.generation_metadata is not None
    assert "sections_processed" in response.generation_metadata
    assert response.generation_metadata["sections_processed"] == len(sample_extraction_full["sections"])
    
    # Verify output file
    assert output_file.exists()
    with open(output_file, 'r') as f:
        output_json = json.load(f)
        assert "qa_pairs" in output_json
        assert "generation_metadata" in output_json
        assert len(output_json["qa_pairs"]) > 0
    
    # Verify metrics were recorded and available in metadata
    assert "metrics" in response.generation_metadata
    
    # Check metrics directly
    metrics = get_processing_metrics()
    assert metrics["pairs_generated"] > 0
    
    # Ensure worker utilization was recorded
    assert "worker_utilization" in metrics
    assert metrics["worker_utilization"] > 0


async def test_process_metadata_with_metrics():
    """Test metrics collection during processing.
    
    Input: Processing pipeline execution
    Expect: Metrics stored in Prometheus format
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.monitoring import (
            initialize_metrics, 
            record_metric, 
            get_processing_metrics,
            QA_GENERATION_TIME,
            QA_PAIRS_GENERATED,
            QA_ERRORS_TOTAL,
            QA_PROCESSING_DURATION,
            MetricsContextManager
        )
    except ImportError:
        pytest.fail("monitoring.py module is missing required components")
    
    # Reset metrics for clean test
    initialize_metrics()
    
    # Record some basic metrics
    record_metric(QA_PAIRS_GENERATED, 5, {"section_type": "text"})
    record_metric(QA_PAIRS_GENERATED, 3, {"section_type": "code"})
    
    # Record metrics using context manager
    with MetricsContextManager(QA_GENERATION_TIME, {"section_type": "text"}):
        await asyncio.sleep(0.01)  # Simulate work
    
    with MetricsContextManager(QA_GENERATION_TIME, {"section_type": "code"}):
        await asyncio.sleep(0.01)  # Simulate work
    
    # Record overall processing time
    with MetricsContextManager(QA_PROCESSING_DURATION):
        await asyncio.sleep(0.02)  # Simulate overall processing
    
    # Get metrics
    metrics = get_processing_metrics()
    
    # Verify metrics structure
    assert "pairs_generated" in metrics
    assert "error_count" in metrics
    assert metrics["pairs_generated"] == 8  # 5 + 3
    assert metrics["error_count"] == 0
    
    # Verify generation time metrics exist
    assert "generation_time" in metrics
    assert metrics["generation_time"] > 0  # Should be populated from context manager
    
    # Verify the generation_time_by_type was correctly populated
    assert "generation_time_by_type" in metrics
    assert "text" in metrics["generation_time_by_type"]
    assert "code" in metrics["generation_time_by_type"]
    assert metrics["generation_time_by_type"]["text"] > 0
    assert metrics["generation_time_by_type"]["code"] > 0


async def test_error_alerting():
    """Test alerting when error count exceeds threshold.
    
    Input: Simulated errors above threshold
    Expect: Alert triggered via alerting system
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.monitoring import (
            initialize_metrics,
            record_metric,
            QA_ERRORS_TOTAL,
            check_alert_conditions,
            ALERT_ERROR_THRESHOLD
        )
    except ImportError:
        pytest.fail("monitoring.py module is missing required components")
    
    # Reset metrics for clean test
    initialize_metrics()
    
    # Mock the alert function
    with patch('agent_tools.dualipa.qa.monitoring.send_alert') as mock_alert:
        # Record errors below threshold
        for _ in range(ALERT_ERROR_THRESHOLD - 1):
            record_metric(QA_ERRORS_TOTAL, 1)
        
        # Check alerts - should not trigger
        check_alert_conditions()
        mock_alert.assert_not_called()
        
        # Add one more error to exceed threshold
        record_metric(QA_ERRORS_TOTAL, 1)
        
        # Check alerts - should trigger
        check_alert_conditions()
        mock_alert.assert_called_once()


async def test_cache_metrics_integration():
    """Test cache metrics integration with monitoring.
    
    Input: Cache hits/misses and batch processing
    Expect: Combined metrics showing cache effectiveness
    """
    # Import the modules to test
    try:
        from agent_tools.dualipa.qa.monitoring import (
            initialize_metrics,
            record_metric,
            get_processing_metrics,
            integrate_cache_metrics,
            QA_CACHE_HIT,
            QA_CACHE_MISS
        )
    except ImportError:
        pytest.fail("Required modules are missing")
    
    # Mock cache stats with controlled values
    mock_cache_stats = {"hits": 15, "misses": 5}
    
    # Create patch to return mock stats and avoid actual cache calls
    with patch('agent_tools.dualipa.qa.utils.cache.get_cache_stats', return_value=mock_cache_stats):
        # Reset metrics
        initialize_metrics()
        
        # Integrate cache metrics (will pull stats from the mock)
        integrate_cache_metrics()
        
        # Get metrics for verification
        metrics = get_processing_metrics()
        
        # Verify cache metrics are included
        assert "cache_hit_rate" in metrics
        assert metrics["cache_hit_rate"] == 15 / (15 + 5)  # 75%
        assert "cache_hits" in metrics
        assert metrics["cache_hits"] == 15
        assert "cache_misses" in metrics
        assert metrics["cache_misses"] == 5
        
        # Check alert condition for high cache hit rate
        # (should not alert since hit rate is good)
        from agent_tools.dualipa.qa.monitoring import check_alert_conditions, ALERT_CACHE_HIT_RATE_THRESHOLD
        
        with patch('agent_tools.dualipa.qa.monitoring.send_alert') as mock_alert:
            check_alert_conditions()
            mock_alert.assert_not_called()
            
        # Test with poor cache hit rate
        mock_cache_stats = {"hits": 1, "misses": 19}
        
        with patch('agent_tools.dualipa.qa.utils.cache.get_cache_stats', return_value=mock_cache_stats):
            # Re-initialize metrics
            initialize_metrics()
            integrate_cache_metrics()
            
            # Verify updated cache metrics
            metrics = get_processing_metrics()
            assert metrics["cache_hit_rate"] == 1 / 20  # 5%
            
            # Should trigger alert if below threshold
            with patch('agent_tools.dualipa.qa.monitoring.send_alert') as mock_alert:
                check_alert_conditions()
                # Only alert if we have enough data (total requests > 10)
                assert mock_alert.called