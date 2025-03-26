"""Test end-to-end extraction functionality.

This module tests the end-to-end extraction process, ensuring:
1. Blocks are extracted correctly
2. Hierarchies are analyzed
3. Output is compatible with QA module requirements
"""

import sys
import pytest
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the script we want to test
sys.path.append(str(project_root / "src"))
try:
    # Import functions directly from main module
    from agent_tools.dualipa.extraction.examples.end_to_end.main import main
    
    # Import required functions directly
    from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import find_source_files, extract_all_blocks
    from agent_tools.dualipa.extraction.examples.end_to_end.hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from agent_tools.dualipa.extraction.examples.end_to_end.validation import validate_qa_output
    
    # Create a wrapper object to organize all imported functions
    class EndToEndExtraction:
        def __init__(self):
            self.find_source_files = find_source_files
            self.extract_all_blocks = extract_all_blocks
            self.analyze_hierarchies = analyze_hierarchies
            self.enrich_blocks_with_hierarchy = enrich_blocks_with_hierarchy
            self.create_qa_compatible_blocks = create_qa_compatible_blocks
            self.create_qa_compatible_output = create_qa_compatible_output
            self.validate_qa_output = validate_qa_output
            self.main = main
    
    end_to_end_extraction = EndToEndExtraction()
    
except ImportError as e:
    pytest.fail(f"Failed to import extraction modules: {e}")


@pytest.fixture
def sample_python_dir():
    """Create a temporary directory with sample Python files."""
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = Path(temp_dir.name)
    
    # Create a simple Python module with classes and functions
    module_path = dir_path / "sample_module.py"
    with open(module_path, 'w') as f:
        f.write("""
from typing import List, Optional

class DataProcessor:
    def __init__(self, data: List[str]):
        self.data = data
        
    def process(self) -> List[str]:
        return [item.strip().lower() for item in self.data]
        
    def get_stats(self) -> Dict[str, int]:
        return {
            "total": len(self.data),
            "empty": sum(1 for item in self.data if not item.strip()),
            "unique": len(set(self.data))
        }

def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        return f.readlines()
        
def save_data(data: List[str], file_path: str) -> None:
    with open(file_path, 'w') as f:
        f.writelines(data)
""")
    
    # Create a second file for validation
    utils_path = dir_path / "utils.py"
    with open(utils_path, 'w') as f:
        f.write("""
import os
import json
from typing import Dict, Any

def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return json.load(f)
        
def write_json(data: Dict[str, Any], file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def ensure_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
""")
    
    yield dir_path
    
    # Cleanup
    temp_dir.cleanup()


def test_find_source_files(sample_python_dir):
    """Test that source files are found correctly."""
    files = end_to_end_extraction.find_source_files(sample_python_dir, extensions=[".py"])
    assert len(files) == 2
    assert any(file.name == "sample_module.py" for file in files)
    assert any(file.name == "utils.py" for file in files)


def test_extract_all_blocks(sample_python_dir):
    """Test extraction of code blocks from all files."""
    blocks = end_to_end_extraction.extract_all_blocks(sample_python_dir)
    assert len(blocks) > 0
    
    # Check that we have blocks for both files
    file_paths = {block.get("file_path") for block in blocks}
    assert len(file_paths) == 2
    
    # Check that we have blocks for classes and functions
    block_types = {block.get("type") for block in blocks}
    assert "class" in block_types
    assert "function" in block_types


def test_analyze_hierarchies(sample_python_dir):
    """Test analysis of code hierarchies."""
    blocks = end_to_end_extraction.extract_all_blocks(sample_python_dir)
    hierarchies = end_to_end_extraction.analyze_hierarchies(blocks)
    
    assert len(hierarchies) == 2
    
    # Check for specific hierarchies in sample_module.py
    sample_module_path = str(sample_python_dir / "sample_module.py")
    if sample_module_path in hierarchies:
        hierarchy = hierarchies[sample_module_path]
        assert hierarchy["language"] == "python"
        assert "DataProcessor" in hierarchy.get("classes", {})
        assert "load_data" in hierarchy.get("functions", {})
        assert "save_data" in hierarchy.get("functions", {})


def test_create_qa_compatible_blocks(sample_python_dir):
    """Test creation of QA-compatible blocks."""
    blocks = end_to_end_extraction.extract_all_blocks(sample_python_dir)
    hierarchies = end_to_end_extraction.analyze_hierarchies(blocks)
    enriched_blocks = end_to_end_extraction.enrich_blocks_with_hierarchy(blocks, hierarchies)
    qa_blocks = end_to_end_extraction.create_qa_compatible_blocks(enriched_blocks)
    
    assert len(qa_blocks) > 0
    
    # Check required fields for QA module
    required_fields = [
        "uuid", "type", "content", "extraction_focus", "summary_instructions", 
        "parent_uuid", "child_uuids", "breadcrumb"
    ]
    
    for block in qa_blocks:
        for field in required_fields:
            assert field in block, f"Missing field: {field}"


def test_end_to_end_extraction(sample_python_dir):
    """Test the complete end-to-end extraction process."""
    output_file = sample_python_dir / "output.json"
    
    # Run the script programmatically
    sys.argv = ["end_to_end_extraction.py", str(sample_python_dir), str(output_file)]
    end_to_end_extraction.main()
    
    # Check that the output file was created
    assert output_file.exists()
    
    # Load and validate the output
    with open(output_file, 'r') as f:
        output = json.load(f)
    
    assert "sections" in output
    assert "extraction_metadata" in output
    assert len(output["sections"]) > 0
    
    # Validate that output matches QA module requirements
    assert end_to_end_extraction.validate_qa_output(output)
    
    # Check that sections have all required fields
    required_section_fields = [
        "uuid", "type", "content", "extraction_focus", "summary_instructions"
    ]
    
    for section in output["sections"]:
        for field in required_section_fields:
            assert field in section, f"Missing field: {field}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])