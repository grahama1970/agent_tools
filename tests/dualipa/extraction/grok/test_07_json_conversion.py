"""
Tests for converting extracted blocks into ordered JSON for LLM consumption.
Depends on: All extraction methods producing consistent output.
"""

import os
import sys
import tempfile
import json
import pytest
from pathlib import Path

# Configure the Path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Fail loudly if dependencies are missing
try:
    from agent_tools.dualipa.code_extractor import extract_repository, run_test
    from agent_tools.dualipa.utils import initialize_stats_dict
except ImportError as e:
    raise ImportError(f"Required code extractor modules not available: {e}")

def test_json_conversion():
    """Test the full extraction pipeline with JSON output."""
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+') as temp_file:
        temp_file.write("""
def greet(name):
    return f"Hello, {name}!"
    
class Calculator:
    def add(self, a, b):
        return a + b
""")
        temp_file.flush()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            stats = initialize_stats_dict()  # From original
            
            # Run the full extraction pipeline (using extract_repository from original)
            result = extract_repository(
                source=str(temp_file.name),
                output_path=str(output_dir)
            )
            
            # Verify result structure from <DOCUMENT>
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "code_files" in result, "Result should contain code_files count"
            assert "code_blocks" in result, "Result should contain code_blocks count"
            
            # Check JSON output file exists
            json_path = output_dir / "extraction_stats.json"
            assert json_path.exists(), "Stats JSON not created"
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Original assertions
            assert "extraction" in data, "Extraction stats missing"
            assert data["extraction"]["blocks"]["total"] >= 2, "Expected at least 2 blocks"
            assert "errors" in data, "Errors field missing"
            
            # <DOCUMENT> enhancements
            assert "total_files" in data, "JSON should include total_files"
            assert "code_blocks" in data, "JSON should include code_blocks"
            assert data["total_files"] == 1, "Expected 1 file processed"
            print(f"Extracted {data['extraction']['blocks']['total']} code blocks")