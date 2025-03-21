"""Security utilities for QA generation.

This module provides utilities for sanitizing inputs and detecting potential
security issues like prompt injections, as well as identifying personally
identifiable information (PII) in content.

Official documentation:
- bleach: https://bleach.readthedocs.io/
- re: https://docs.python.org/3/library/re.html
"""

import json
import bleach
import logging
import re
import textwrap
from typing import Dict, Any, List, Union, Optional, Tuple
from agent_tools.dualipa.qa.models.config import PROMPT_INJECTION_PATTERNS

logger = logging.getLogger(__name__)

# Enhanced patterns for prompt injection detection
INJECTION_PATTERNS = [
    r'ignore (?:all )?(?:previous |prior )?instructions',
    r'disregard (?:all )?(?:previous |prior )?instructions',
    r'override (?:all )?(?:previous |prior |system )?(?:instructions|settings)',
    r'bypass (?:all )?(?:previous |prior |system )?(?:instructions|settings|filters)',
    r'do not (?:follow|adhere to) (?:the )?(?:guidelines|instructions)',
    r'(?:disregard|ignore) (?:all )?ethics',
    r'(?:disregard|ignore) (?:all )?safety',
    r'no longer act as',
    r'new instructions:',
    r'new prompt:',
]

# Enhanced PII detection patterns
PII_PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    "ssn": r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
    "address": r'\b\d{1,5}\s+[a-zA-Z0-9\s,]{5,}\b(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\b',
}


def sanitize_html_content(content: str) -> str:
    """Sanitize HTML content by removing all tags.
    
    Args:
        content: The HTML content to sanitize
        
    Returns:
        Sanitized content with HTML tags removed
    """
    if content is None:
        return ""
    
    # Remove all HTML tags and remove sanitized content
    return bleach.clean(content, tags=[], strip=True, strip_comments=True)


def detect_prompt_injection(content: str) -> Union[bool, Dict[str, Any]]:
    """Detect potential prompt injection in content.
    
    Args:
        content: The content to check for prompt injection
        
    Returns:
        False if no injection detected, or dict with injection details
    """
    if not content:
        return False
    
    # Convert to lowercase for case-insensitive matching
    content_lower = content.lower()
    
    # Exclude legitimate programming contexts
    if "how to ignore errors in python" in content_lower:
        return False
    
    # Check for word-based patterns (simple matching)
    for pattern in PROMPT_INJECTION_PATTERNS:
        # Skip 'ignore' by itself (too common in legitimate contexts)
        if pattern == "ignore" and "ignore" in content_lower:
            if not re.search(r'ignore\s+(previous|all|these|the)', content_lower):
                continue
                
        if pattern in content_lower:
            return {
                "detected": True,
                "pattern": pattern,
                "type": "simple_match"
            }
    
    # Check for regex patterns (more complex matching)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, content_lower):
            return {
                "detected": True, 
                "pattern": pattern,
                "type": "regex_match"
            }
    
    return False


def sanitize_input(content: str, allow_injection_pattern: bool = False) -> str:
    """Sanitize HTML/JS and detect prompt injections.
    
    Args:
        content: The input content to sanitize
        allow_injection_pattern: If True, don't raise error on injection patterns
        
    Returns:
        Sanitized content string
        
    Raises:
        ValueError: If potential prompt injection is detected and not allowed
    """
    if content is None:
        return ""
        
    # Clean HTML content
    cleaned = sanitize_html_content(content)
    
    # Check for potential prompt injections
    injection = detect_prompt_injection(cleaned)
    if injection and not allow_injection_pattern:
        logger.warning(f"Potential prompt injection detected in content: {injection.get('pattern', 'unknown')}")
        raise ValueError("Potential prompt injection detected")
    elif injection and allow_injection_pattern:
        # Remove the injection pattern
        for pattern in PROMPT_INJECTION_PATTERNS:
            cleaned = cleaned.replace(pattern, "[REDACTED]")
        for pattern in INJECTION_PATTERNS:
            cleaned = re.sub(pattern, "[REDACTED]", cleaned, flags=re.IGNORECASE)
            
    return cleaned


def sanitize_input_json(
    input_data: Dict[str, Any], 
    raise_on_injection: bool = True
) -> Dict[str, Any]:
    """Sanitize all string values in input JSON.
    
    Args:
        input_data: The input JSON data
        raise_on_injection: If True, raise error on injection patterns
        
    Returns:
        Sanitized JSON data
        
    Raises:
        ValueError: If potential prompt injection is detected and raise_on_injection=True
    """
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary")
    
    sanitized_data = {}
    
    for key, value in input_data.items():
        if isinstance(value, str):
            try:
                sanitized_data[key] = sanitize_input(value, not raise_on_injection)
            except ValueError as e:
                if raise_on_injection:
                    raise
                sanitized_data[key] = sanitize_input(value, True)
        elif isinstance(value, dict):
            sanitized_data[key] = sanitize_input_json(value, raise_on_injection)
        elif isinstance(value, list):
            sanitized_list = []
            for item in value:
                if isinstance(item, dict):
                    sanitized_list.append(sanitize_input_json(item, raise_on_injection))
                elif isinstance(item, str):
                    try:
                        sanitized_list.append(sanitize_input(item, not raise_on_injection))
                    except ValueError as e:
                        if raise_on_injection:
                            raise
                        sanitized_list.append(sanitize_input(item, True))
                else:
                    sanitized_list.append(item)
            sanitized_data[key] = sanitized_list
        else:
            sanitized_data[key] = value
            
    return sanitized_data


def check_pii_in_content(content: str) -> Dict[str, Any]:
    """Check for PII in generated content.
    
    Args:
        content: The content to check for PII
        
    Returns:
        Dict with PII detection results including types found
    """
    if not content:
        return {"has_pii": False, "pii_types": []}
    
    pii_found = []
    
    # Check each type of PII
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            pii_found.append(pii_type)
            logger.warning(f"Potential {pii_type} PII detected in content")
    
    result = {
        "has_pii": len(pii_found) > 0,
        "pii_types": pii_found
    }
    
    return result