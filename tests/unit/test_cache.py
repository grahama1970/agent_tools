"""Unit tests for the method validator cache functionality."""

import pytest
import tempfile
from unittest.mock import patch
from pathlib import Path
from typing import Any, Callable, Generator
from agent_tools.method_validator.cache import AnalysisCache, get_cache_dir, TimingStats

@pytest.fixture
def temp_cache() -> Generator[AnalysisCache, None, None]:
    """Create a temporary cache for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = AnalysisCache()
        # Override the default cache path for testing
        cache.db_path = Path(tmpdir) / "test_cache.db"
        cache._init_db()
        yield cache

def dummy_method() -> None:
    """Dummy method for testing."""
    pass

def test_cache_initialization(temp_cache: AnalysisCache) -> None:
    """Test that cache initializes correctly."""
    assert temp_cache.db_path.exists()
    assert temp_cache.max_age_days == 30
    assert temp_cache.max_size_mb == 100
    assert temp_cache.compression_level == 6

def test_cache_set_and_get(temp_cache: AnalysisCache) -> None:
    """Test setting and getting values from cache."""
    # Test data
    package_name = "test_package"
    method_name = "test_method"
    result = ("description", "summary", ["category1", "category2"])

    # Set in cache
    temp_cache.set(package_name, method_name, dummy_method, result)

    # Get from cache
    cached_result = temp_cache.get(package_name, method_name, dummy_method)
    assert cached_result == result

def test_cache_clear(temp_cache: AnalysisCache) -> None:
    """Test clearing the cache."""
    # Add test data
    package_name = "test_package"
    method_name = "test_method"
    result = ("description", "summary", ["category"])
    
    temp_cache.set(package_name, method_name, dummy_method, result)
    assert temp_cache.get(package_name, method_name, dummy_method) == result
    
    # Clear cache
    temp_cache.clear()
    assert temp_cache.get(package_name, method_name, dummy_method) is None

def test_cache_cleanup(temp_cache: AnalysisCache) -> None:
    """Test cache cleanup based on age and size."""
    # Override max age and size for testing
    temp_cache.max_age_days = 0  # Immediate expiry
    temp_cache.max_size_mb = 1  # Small size limit

    # Add test data
    package_name = "test_package"
    method_names = [f"method_{i}" for i in range(5)]
    result = ("description", "summary", ["category"])

    # Mock time to simulate aging entries
    with patch('time.time') as mock_time:
        # Add multiple entries with different timestamps
        for i, method_name in enumerate(method_names):
            mock_time.return_value = 1000 + (i * 3600)  # Each entry 1 hour apart
            temp_cache.set(package_name, method_name, dummy_method, result)
        
        # Set current time to future
        mock_time.return_value = 1000 + (24 * 3600)  # 24 hours later
        
        # Trigger cleanup
        temp_cache._cleanup_if_needed()

        # Verify old entries are removed
        for method_name in method_names:
            assert temp_cache.get(package_name, method_name, dummy_method) is None

def test_cache_compression(temp_cache: AnalysisCache) -> None:
    """Test data compression in cache."""
    # Create large test data
    large_data = ("description", "summary", ["category"] * 1000)
    package_name = "test_package"
    method_name = "test_method"

    # Set compression level
    temp_cache.compression_level = 9
    temp_cache.set(package_name, method_name, dummy_method, large_data)

    # Verify data can be retrieved
    cached_result = temp_cache.get(package_name, method_name, dummy_method)
    assert cached_result == large_data

def test_get_all_methods(temp_cache: AnalysisCache) -> None:
    """Test retrieving all methods for a package."""
    package_name = "test_package"
    methods = {
        "method1": ("desc1", "sum1", ["cat1"]),
        "method2": ("desc2", "sum2", ["cat2"]),
        "method3": ("desc3", "sum3", ["cat3"])
    }

    # Add multiple methods
    for method_name, result in methods.items():
        temp_cache.set(package_name, method_name, dummy_method, result)

    # Get all methods
    cached_methods = temp_cache.get_all_methods(package_name)
    assert len(cached_methods) == len(methods)
    for result in cached_methods:
        assert result in methods.values()

def test_timing_stats() -> None:
    """Test TimingStats functionality."""
    stats = TimingStats("test_op")
    assert stats.operation == "test_op"
    assert stats.total_time == 0.0
    assert stats.calls == 0
    assert stats.average_time == 0.0

    # Update stats
    stats.total_time = 10.0
    stats.calls = 5
    assert stats.average_time == 2.0

def test_get_cache_dir() -> None:
    """Test cache directory creation and access."""
    cache_dir = get_cache_dir()
    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert "method_validator" in str(cache_dir)

def test_batch_operations(temp_cache: AnalysisCache) -> None:
    """Test batch set operations."""
    package_name = "test_package"
    entries = [
        (package_name, f"method_{i}", dummy_method, (f"desc_{i}", f"sum_{i}", [f"cat_{i}"]))
        for i in range(3)
    ]

    # Set batch entries
    temp_cache.set_batch(entries)

    # Verify all entries
    for package_name, method_name, _, result in entries:
        cached = temp_cache.get(package_name, method_name, dummy_method)
        assert cached == result 