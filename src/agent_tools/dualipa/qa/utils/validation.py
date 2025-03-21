"""Validation utilities for QA generation.

This module provides utilities for validating QA pairs and responses,
as well as input JSON validation and normalization. It provides functions
for validating against both schema requirements and business rules, handling
normalization of inputs, and validating configuration parameters.

Official documentation:
- pydantic: https://docs.pydantic.dev/latest/
- json: https://docs.python.org/3/library/json.html
- datetime: https://docs.python.org/3/library/datetime.html
- logging: https://docs.python.org/3/library/logging.html
- uuid: https://docs.python.org/3/library/uuid.html

Expected input/output:
- validate_qa_pair: Takes a QAPair object, returns True if valid, False otherwise
- validate_qa_response: Takes a QAResponse object, returns True if valid, False otherwise
- validate_input_json: Takes input JSON data, returns True if valid, False otherwise
- normalize_input_json: Takes input JSON data, returns normalized data with defaults
- validate_temperature_range: Takes list of temperatures, returns validated list
"""

import json
import logging
import datetime
import uuid
import textwrap
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union, Callable
from agent_tools.dualipa.qa.models.qa_models import QAPair, QAResponse
from agent_tools.dualipa.qa.models.config import MIN_REASONING_WORDS, DEFAULT_CONFIDENCE_THRESHOLD

# Constants for validation
logger = logging.getLogger(__name__)

# Default values for missing fields
DEFAULT_EXTRACTION_FOCUS = "general content"
DEFAULT_SUMMARY_INSTRUCTIONS = "Generate question-answer pairs about the content"

# Required fields for different parts of input
SECTION_REQUIRED_FIELDS = ["uuid", "type", "content"]
SECTION_OPTIONAL_FIELDS = {
    "extraction_focus": DEFAULT_EXTRACTION_FOCUS,
    "summary_instructions": DEFAULT_SUMMARY_INSTRUCTIONS
}
METADATA_REQUIRED_FIELDS = ["model_used"]
METADATA_OPTIONAL_FIELDS = {
    "timestamp": lambda: datetime.datetime.now().isoformat()
}

# Validators for QA pairs
PAIR_VALIDATORS = [
    lambda p: "?" in p.question,
    lambda p: "Oh wait?!" in p.reasoning,
    lambda p: len(p.answer.split()) >= 5,
    lambda p: p.confidence_score is None or p.confidence_score >= DEFAULT_CONFIDENCE_THRESHOLD,
    lambda p: len(p.reasoning.split()) >= MIN_REASONING_WORDS
]

# Validators for QA responses
RESPONSE_METADATA_REQUIRED = ["model_used", "temperature_range", "timestamp"]


def validate_qa_pair(pair: QAPair) -> bool:
    """Validate a QA pair against business rules.
    
    This validates beyond what Pydantic handles, checking business rules
    like reasoning quality, answer length, and confidence scores.
    
    Args:
        pair: The QA pair to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Run all validators defined in PAIR_VALIDATORS
        for validator in PAIR_VALIDATORS:
            if not validator(pair):
                return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating QA pair: {e}")
        return False


def validate_input_json(input_data: Dict[str, Any], raise_on_error: bool = False) -> bool:
    """Validate input JSON structure.
    
    Validates the structure of extraction JSON data, checking for required
    fields and proper types.
    
    Args:
        input_data: The input JSON data
        raise_on_error: If True, raise ValueError on validation failure
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    error_messages = []
    
    # Check for required top-level fields
    if not all(field in input_data for field in ["sections", "extraction_metadata"]):
        msg = "Missing required fields in input JSON (require 'sections' and 'extraction_metadata')"
        error_messages.append(msg)
        logger.error(msg)
        
    # Check sections structure
    sections = input_data.get("sections", [])
    if not sections or not isinstance(sections, list):
        msg = "Sections must be a non-empty list"
        error_messages.append(msg)
        logger.error(msg)
    
    # Validate each section
    for i, section in enumerate(sections):
        if not all(field in section for field in SECTION_REQUIRED_FIELDS):
            missing = set(SECTION_REQUIRED_FIELDS) - set(section.keys())
            msg = f"Section {i} missing required fields: {', '.join(missing)}"
            error_messages.append(msg)
            logger.error(msg)
    
    # Validate extraction metadata
    metadata = input_data.get("extraction_metadata", {})
    if not isinstance(metadata, dict):
        msg = "extraction_metadata must be a dictionary"
        error_messages.append(msg)
        logger.error(msg)
    elif not all(field in metadata for field in METADATA_REQUIRED_FIELDS):
        missing = set(METADATA_REQUIRED_FIELDS) - set(metadata.keys())
        msg = f"Metadata missing required fields: {', '.join(missing)}"
        error_messages.append(msg)
        logger.error(msg)
    
    # Raise or return
    if error_messages and raise_on_error:
        raise ValueError("\n".join(error_messages))
        
    return len(error_messages) == 0


def validate_qa_response(response: QAResponse) -> bool:
    """Validate a QA response.
    
    Validates the full QA response against business rules, including checking
    that all QA pairs are valid and the metadata is complete.
    
    Args:
        response: The QA response to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check for required fields
        if not hasattr(response, "qa_pairs") or not hasattr(response, "generation_metadata"):
            logger.error("QA response missing required attributes")
            return False
            
        # Check QA pairs
        if not response.qa_pairs:
            logger.warning("QA response contains no pairs")
            return False
            
        # Check that all QA pairs are valid
        invalid_pairs = [i for i, pair in enumerate(response.qa_pairs) if not validate_qa_pair(pair)]
        if invalid_pairs:
            logger.error(f"Invalid QA pairs at indices: {invalid_pairs}")
            return False
            
        # Check metadata
        metadata = response.generation_metadata
        missing_fields = [field for field in RESPONSE_METADATA_REQUIRED if field not in metadata]
        if missing_fields:
            logger.error(f"Missing required metadata fields: {missing_fields}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating QA response: {e}")
        return False


def validate_temperature_range(temps: List[float]) -> List[float]:
    """Validate and normalize temperature range.
    
    Ensures temperatures are within valid range and sorts them.
    
    Args:
        temps: List of temperatures
        
    Returns:
        Validated list of temperatures
    """
    # Filter out invalid temperatures
    valid_temps = [t for t in temps if 0.0 <= t <= 1.0]
    
    # If no valid temperatures, use default
    if not valid_temps:
        logger.warning("No valid temperatures provided, using default")
        return [0.5]
    
    # Sort temperatures in ascending order
    valid_temps.sort()
        
    return valid_temps


def normalize_input_json(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize input JSON by adding default values for missing fields.
    
    This function adds default values for optional fields and ensures
    all required fields are present with correct types.
    
    Args:
        input_data: The input JSON data
        
    Returns:
        Normalized JSON data
    """
    # Make a deep copy to avoid modifying the input
    normalized = deepcopy(input_data)
    
    # Ensure top-level structure
    if "sections" not in normalized:
        normalized["sections"] = []
    if "extraction_metadata" not in normalized:
        normalized["extraction_metadata"] = {}
    
    # Normalize extraction metadata
    metadata = normalized["extraction_metadata"]
    for field, default_value in METADATA_OPTIONAL_FIELDS.items():
        if field not in metadata:
            # Handle callable defaults (like timestamps)
            if callable(default_value):
                metadata[field] = default_value()
            else:
                metadata[field] = default_value
    
    # Normalize each section
    for section in normalized["sections"]:
        # Generate UUID if missing
        if "uuid" not in section:
            section["uuid"] = str(uuid.uuid4())
            
        # Add defaults for optional fields
        for field, default_value in SECTION_OPTIONAL_FIELDS.items():
            if field not in section:
                section[field] = default_value
    
    return normalized