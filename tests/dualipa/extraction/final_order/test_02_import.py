import pytest
import sys
from pathlib import Path

# Configure paths properly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_import_dualipa():
    try:
        from agent_tools.dualipa import __version__
        print(f"Successfully imported dualipa, version: {__version__}")
        assert __version__ is not None
    except ImportError as e:
        print(f"Import error: {e}")
        
        # Add debugging information
        print(f"Python path: {sys.path}")
        
        import os
        try:
            import agent_tools
            print(f"agent_tools found at: {agent_tools.__file__}")
            agent_tools_dir = os.path.dirname(agent_tools.__file__)
            print(f"Contents of {agent_tools_dir}:")
            for item in os.listdir(agent_tools_dir):
                print(f"  - {item}")
        except ImportError:
            print("Could not import agent_tools at all")
            
        assert False, f"Failed to import dualipa: {e}"

def test_imports():
    """Test importing the needed modules."""
    try:
        import agent_tools.dualipa
        from agent_tools.dualipa.code_extractor import initialize_stats_dict
        from agent_tools.dualipa.code_extractor import _extract_python_blocks
        from agent_tools.dualipa.code_extractor import _extract_js_ts_blocks
        from agent_tools.dualipa.code_extractor import _extract_markdown_blocks
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
        
def test_stats_dictionary_consistency():
    """Test that the stats dictionary is consistent across all modules that use it."""
    # Import all modules that should use the stats dictionary
    try:
        from agent_tools.dualipa.code_extractor import initialize_stats_dict
        from agent_tools.dualipa.extract_repo import initialize_stats_dict as repo_init_stats
        from agent_tools.dualipa.code_hierarchy import initialize_stats_dict as hierarchy_init_stats
    except ImportError as e:
        pytest.fail(f"Failed to import initialize_stats_dict from modules: {e}")
    
    # Create stats dictionaries with each function
    base_stats = initialize_stats_dict(source="test_source", output_dir=Path("/tmp"))
    
    # Verify all required fields exist in the base stats dictionary
    required_fields = [
        # Source and output information
        "source", "repo_url", "output_path",
        # Timing information
        "start_time", "end_time", "duration_seconds",
        # File and block counts
        "total_files", "documentation_files", "code_files", 
        "code_blocks", "doc_blocks", "skipped_files", "error_files",
        # Categorization
        "languages", "file_types",
        # Error tracking
        "errors",
        # Block storage
        "file_blocks"
    ]
    
    # Verify all required fields exist
    for field in required_fields:
        assert field in base_stats, f"Missing required field '{field}' in stats dictionary"
    
    # Verify that these are all the same implementation (should point to the same function)
    import inspect
    base_source = inspect.getsource(initialize_stats_dict)
    try:
        repo_source = inspect.getsource(repo_init_stats)
        assert repo_source == base_source, "extract_repo.py is not importing the standardized initialize_stats_dict function"
    except TypeError:
        pass  # This is acceptable if it's just an imported reference
        
    try:
        hierarchy_source = inspect.getsource(hierarchy_init_stats)
        assert hierarchy_source == base_source, "code_hierarchy.py is not importing the standardized initialize_stats_dict function"
    except TypeError:
        pass  # This is acceptable if it's just an imported reference
    
    # Verify that the direct imports all use the same function (by reference)
    assert initialize_stats_dict is repo_init_stats, "extract_repo.py is not using the same initialize_stats_dict function"
    assert initialize_stats_dict is hierarchy_init_stats, "code_hierarchy.py is not using the same initialize_stats_dict function"
    
    print("Stats dictionary consistency verified across all modules")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
