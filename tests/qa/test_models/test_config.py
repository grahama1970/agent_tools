"""Test configuration management.

This module tests configuration loading and validation for QA generation.

Official documentation:
- pytest: https://docs.pytest.org/en/stable/
- pydantic: https://docs.pydantic.dev/latest/
- PyYAML: https://pyyaml.org/wiki/PyYAML
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path

from agent_tools.dualipa.qa.models.config import load_config, QAConfig, validate_temperature_range


def test_load_config_real(tmp_path):
    """Test loading a real configuration file.
    
    This test creates a real YAML configuration file
    and verifies that it can be loaded correctly.
    """
    # Create a test config file
    config_path = tmp_path / "qa_config.yaml"
    
    config_data = {
        "llm": {
            "model": "gpt-4-turbo",
            "api_key_env": "OPENAI_API_KEY",
            "fallback_model": "gpt-3.5-turbo"
        },
        "generation": {
            "temperature_range": [0.3, 0.5, 0.7],
            "max_tokens": 1000,
            "top_p": 0.95
        },
        "processing": {
            "max_concurrent_requests": 5,
            "deduplication_threshold": 0.85,
            "cache_duration_hours": 24
        },
        "security": {
            "enable_pii_detection": True,
            "log_sanitization": True,
            "prompt_injection_checks": True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Load the config
    config = load_config(config_path)
    
    # Verify it's a QAConfig instance
    assert isinstance(config, QAConfig)
    
    # Verify values were loaded correctly
    assert config.llm.model == "gpt-4-turbo"
    assert config.generation.temperature_range == [0.3, 0.5, 0.7]
    assert config.processing.max_concurrent_requests == 5
    assert config.security.enable_pii_detection is True


def test_load_config_env_vars(tmp_path, monkeypatch):
    """Test loading a configuration with environment variables.
    
    This test ensures that environment variables are correctly
    resolved in the configuration.
    """
    # Set environment variables
    monkeypatch.setenv("QA_MODEL", "gpt-4-turbo-preview")
    monkeypatch.setenv("QA_MAX_TOKENS", "2000")
    
    # Create a test config file with env vars
    config_path = tmp_path / "qa_config_env.yaml"
    
    config_data = {
        "llm": {
            "model": "${QA_MODEL}",
            "max_tokens": "${QA_MAX_TOKENS}"
        },
        "generation": {
            "temperature_range": [0.3, 0.5, 0.7]
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_data, f)
    
    # Load the config
    config = load_config(config_path)
    
    # Verify environment variables were resolved
    assert config.llm.model == "gpt-4-turbo-preview"
    assert config.llm.max_tokens == 2000


def test_validate_temperature_range():
    """Test temperature range validation.
    
    This test verifies that temperature ranges are properly validated.
    """
    # Valid temperature ranges
    valid_ranges = [
        [0.0, 0.5, 1.0],
        [0.3, 0.7],
        [0.5]
    ]
    
    for temp_range in valid_ranges:
        result = validate_temperature_range(temp_range)
        assert result is True
    
    # Invalid temperature ranges
    invalid_ranges = [
        [-0.1, 0.5, 1.0],  # Negative value
        [0.3, 1.1],        # Value > 1.0
        [],                # Empty list
        [0.7, 0.5, 0.3]    # Not in ascending order
    ]
    
    for temp_range in invalid_ranges:
        with pytest.raises(ValueError):
            validate_temperature_range(temp_range)