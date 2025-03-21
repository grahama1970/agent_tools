import pytest
import sys
from pathlib import Path
import inspect

# Configure paths properly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_import_dualipa():
    """Test that we can import the dualipa package."""
    try:
        import agent_tools.dualipa
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import dualipa: {e}")

def test_imports():
    """Test that we can import key functions."""
    try:
        from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict
        assert callable(initialize_stats_dict)
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_stats_dictionary_consistency():
    """Test that initialize_stats_dict is used consistently."""
    try:
        from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict
        from agent_tools.dualipa.extraction.extractors.github.repo_utils import initialize_stats_dict as repo_init_stats
        from agent_tools.dualipa.extraction.extractors.code.hierarchy import initialize_stats_dict as hierarchy_init_stats
    except ImportError as e:
        pytest.fail(f"Failed to import initialize_stats_dict from modules: {e}")

    # Test that the function works
    base_stats = initialize_stats_dict(source="test_source", output_dir=Path("/tmp"))
    assert isinstance(base_stats, dict)
    assert "source" in base_stats
    assert "output_dir" in base_stats
    assert "languages" in base_stats
    assert "file_blocks" in base_stats
    assert "code_blocks" in base_stats
    assert "errors" in base_stats

    # Test that all modules use the same function
    base_source = inspect.getsource(initialize_stats_dict)
    try:
        repo_source = inspect.getsource(repo_init_stats)
    except TypeError:
        pytest.fail("extract_repo.py is not importing the standardized initialize_stats_dict function")

    try:
        hierarchy_source = inspect.getsource(hierarchy_init_stats)
    except TypeError:
        pytest.fail("code_hierarchy.py is not importing the standardized initialize_stats_dict function")

    # Compare function sources
    assert repo_source == base_source, "extract_repo.py is not importing the standardized initialize_stats_dict function"
    assert hierarchy_source == base_source, "code_hierarchy.py is not importing the standardized initialize_stats_dict function"

    # Compare function objects directly
    assert initialize_stats_dict is repo_init_stats, "extract_repo.py is not using the same initialize_stats_dict function"
    assert initialize_stats_dict is hierarchy_init_stats, "code_hierarchy.py is not using the same initialize_stats_dict function"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
