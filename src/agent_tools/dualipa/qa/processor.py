"""Processing pipeline for QA generation.

This module provides the implementation of the QA generation pipeline.
It started as a minimal MVP and has been enhanced with additional features.

Official documentation:
- asyncio: https://docs.python.org/3/library/asyncio.html
- json: https://docs.python.org/3/library/json.html
"""

import json
import asyncio
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

from .models.qa_models import QAPair, QAResponse, Direction
from .models.config import QAGenerationConfig, DEFAULT_TEMPERATURE_RANGE
from .utils.validation import validate_input_json, normalize_input_json
from .utils.security import sanitize_input_json
from .llm.generation import iterate_temperatures

logger = logging.getLogger(__name__)


async def process_section(
    section: Dict[str, Any], 
    config: Optional[QAGenerationConfig] = None,
    enable_bidirectional: bool = True
) -> List[QAPair]:
    """Process a single section to generate QA pairs.
    
    Args:
        section: Section data from extraction JSON
        config: Optional QA generation configuration
        enable_bidirectional: Whether to generate reversed QA pairs
        
    Returns:
        List of QA pairs generated from the section (both forward and reverse)
        limited by max_qa_pairs_per_section in config
    """
    if config is None:
        config = QAGenerationConfig()
    
    # Extract section data
    content = section.get("content", "")
    content_type = section.get("type", "text")
    section_uuid = section.get("uuid")
    
    # Determine max pairs to generate based on config
    max_pairs = config.max_qa_pairs_per_section
    
    if enable_bidirectional:
        # Use bidirectional generation
        from .llm.reversal import generate_bidirectional_qa_pairs
        
        forward_pairs, reverse_pairs = await generate_bidirectional_qa_pairs(
            content=content,
            content_type=content_type,
            config=config,
            section_uuid=section_uuid
        )
        
        # Apply bidirectional ratio for selecting pairs
        bidirectional_ratio = config.bidirectional_ratio
        forward_count = min(len(forward_pairs), int(max_pairs * (1 - bidirectional_ratio)))
        reverse_count = min(len(reverse_pairs), max_pairs - forward_count)
        
        # Trim to respect max_qa_pairs_per_section
        limited_forward = forward_pairs[:forward_count]
        limited_reverse = reverse_pairs[:reverse_count]
        
        # Combine limited forward and reverse pairs
        return limited_forward + limited_reverse
    else:
        # Use standard temperature iteration for forward-only generation
        from .llm.generation import iterate_temperatures
        
        all_pairs = await iterate_temperatures(
            content=content,
            content_type=content_type,
            temps=config.temperature_range,
            section_uuid=section_uuid,
            max_pairs_per_temp=3,
            max_concurrent_requests=config.max_concurrent_requests
        )
        
        # Limit to max_qa_pairs_per_section
        return all_pairs[:max_pairs]


async def process_extraction_with_config(
    input_data: Dict[str, Any],
    config: Optional[QAGenerationConfig] = None,
    output_file: Optional[Union[str, Path]] = None,
    enable_bidirectional: bool = True,
    enable_monitoring: bool = True
) -> QAResponse:
    """Process extraction JSON with configuration.
    
    Args:
        input_data: Input JSON data
        config: Optional QA generation configuration
        output_file: Optional path to write output JSON
        enable_bidirectional: Whether to generate reversed QA pairs
        enable_monitoring: Whether to record metrics during processing
        
    Returns:
        QA response with generated pairs
    """
    # Set up monitoring if enabled
    if enable_monitoring:
        try:
            from .monitoring import (
                MetricsContextManager, 
                record_metric, 
                QA_PROCESSING_DURATION,
                QA_PAIRS_GENERATED
            )
            metrics_context = MetricsContextManager(QA_PROCESSING_DURATION)
        except ImportError:
            logger.warning("Monitoring module not available, processing without metrics")
            enable_monitoring = False
            metrics_context = None
    else:
        metrics_context = None
    
    # Use metrics context manager if monitoring enabled
    if metrics_context:
        metrics_context.__enter__()
    
    try:
        if config is None:
            config = QAGenerationConfig()
        
        # Sanitize and normalize input
        sanitized_data = sanitize_input_json(input_data)
        normalized_data = normalize_input_json(sanitized_data)
        
        # Extract sections
        sections = normalized_data.get("sections", [])
        logger.info(f"Processing {len(sections)} sections" + 
                    f" with bidirectional={enable_bidirectional}")
        
        # Process each section using batch processing with worker pools
        from .utils.batch_processing import batch_process_sections
        
        # Record worker utilization if monitoring enabled
        if enable_monitoring:
            from .monitoring import record_metric, QA_WORKER_UTILIZATION
            worker_count = getattr(config, 'worker_count', config.max_concurrent_requests)
            if len(sections) > 0:
                utilization = min(1.0, len(sections) / worker_count) * 100.0
                record_metric(QA_WORKER_UTILIZATION, utilization)
        
        section_results = await batch_process_sections(
            sections=sections,
            config=config,
            process_function=process_section,
            enable_bidirectional=enable_bidirectional
        )
    
        # Combine all QA pairs
        all_qa_pairs = []
        forward_count = 0
        reverse_count = 0
        
        for result in section_results:
            all_qa_pairs.extend(result)
            
            # Count forward and reverse pairs
            for pair in result:
                if pair.direction == Direction.FORWARD:
                    forward_count += 1
                elif pair.direction == Direction.REVERSE:
                    reverse_count += 1
        
        # Calculate reverse ratio for metadata
        total_pairs = len(all_qa_pairs)
        reverse_ratio = 0.0
        if total_pairs > 0:
            reverse_ratio = reverse_count / total_pairs
        
        # Record generation metrics if monitoring enabled
        if enable_monitoring:
            record_metric(QA_PAIRS_GENERATED, total_pairs)
            
            # Record detailed metrics by section type
            section_type_counts = {}
            for section in sections:
                section_type = section.get("type", "unknown")
                if section_type not in section_type_counts:
                    section_type_counts[section_type] = 0
                section_type_counts[section_type] += 1
            
            for section_type, count in section_type_counts.items():
                record_metric(QA_PAIRS_GENERATED, count, {"section_type": section_type})
        
        # Create response with monitoring metadata if enabled
        metadata = {
            "model_used": config.model,
            "temperature_range": config.temperature_range,
            "sections_processed": len(sections),
            "total_qa_pairs": total_pairs,
            "forward_pairs": forward_count,
            "reverse_pairs": reverse_count,
            "reverse_ratio": reverse_ratio,
            "bidirectional_enabled": enable_bidirectional
        }
        
        # Add monitoring status to metadata
        if enable_monitoring:
            metadata["monitoring_enabled"] = True
            
            # Integrate cache metrics if available
            try:
                from .monitoring import integrate_cache_metrics, get_processing_metrics
                integrate_cache_metrics()
                metrics = get_processing_metrics()
                metadata["metrics"] = {
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0),
                    "worker_utilization": metrics.get("worker_utilization", 0),
                    "generation_time": metrics.get("generation_time", 0)
                }
            except ImportError:
                pass
        
        # Create response
        response = QAResponse(
            qa_pairs=all_qa_pairs,
            generation_metadata=metadata
        )
        
        # Write output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(response.model_dump(), f, indent=2)
        
        logger.info(f"Generated {total_pairs} QA pairs " +
                    f"({forward_count} forward, {reverse_count} reverse)")
        
        # Check alert conditions if monitoring enabled
        if enable_monitoring:
            try:
                from .monitoring import check_alert_conditions
                check_alert_conditions()
            except ImportError:
                pass
        
        return response
    except Exception as e:
        # Record error if monitoring enabled
        if enable_monitoring:
            try:
                from .monitoring import record_metric, QA_ERRORS_TOTAL
                record_metric(QA_ERRORS_TOTAL, 1, {"error_type": type(e).__name__})
            except ImportError:
                pass
        
        # Re-raise the exception
        raise
    finally:
        # Cleanup metrics context if needed
        if metrics_context:
            metrics_context.__exit__(None, None, None)




async def process_extraction_json(
    input_data: Union[Dict[str, Any], str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[QAGenerationConfig] = None,
    enable_bidirectional: bool = True,
    enable_monitoring: bool = True
) -> QAResponse:
    """Process extraction JSON to generate QA pairs.
    
    This function loads the input data, processes it to generate QA pairs,
    and returns the results. It can optionally write the results to a file.
    
    Args:
        input_data: Input JSON data or file path
        output_file: Optional path to write output JSON
        config: Optional QA generation configuration
        enable_bidirectional: Whether to generate reversed QA pairs
        enable_monitoring: Whether to collect metrics during processing
        
    Returns:
        QA response with generated pairs
    """
    # Handle input format
    if isinstance(input_data, (str, Path)):
        with open(input_data, 'r') as f:
            input_json = json.load(f)
    else:
        input_json = input_data
    
    # Create default config if not provided
    if config is None:
        config = QAGenerationConfig()
    
    # Use the full implementation with monitoring
    return await process_extraction_with_config(
        input_data=input_json,
        config=config,
        output_file=output_file,
        enable_bidirectional=enable_bidirectional,
        enable_monitoring=enable_monitoring
    )