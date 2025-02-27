"""Cache and timing utilities for method validator."""

import time
import sqlite3
import pickle
import hashlib
import inspect
import zlib
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from loguru import logger
from .analyzer import MethodAnalyzer

# Get the method_validator package root directory
PACKAGE_ROOT = Path(__file__).parent

# Compression level (0-9, where 0 is no compression and 9 is maximum)
DEFAULT_COMPRESSION_LEVEL = 6


@dataclass
class TimingStats:
    """Container for timing statistics of operations."""

    operation: str
    total_time: float = 0.0
    calls: int = 0

    @property
    def average_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0


class TimingManager:
    """Manages timing statistics for method validator operations."""

    def __init__(self):
        self.stats: Dict[str, TimingStats] = defaultdict(lambda: TimingStats(""))
        self.enabled = True

    @contextmanager
    def measure(self, operation: str, description: str = ""):
        """Context manager to measure execution time of an operation."""
        if not self.enabled:
            yield
            return

        start_time = time.time()
        try:
            if description:
                logger.debug(f"Starting {operation}: {description}")
            yield
        finally:
            elapsed = time.time() - start_time
            if operation not in self.stats:
                self.stats[operation] = TimingStats(operation)
            self.stats[operation].total_time += elapsed
            self.stats[operation].calls += 1
            if description:
                logger.debug(f"Completed {operation} in {elapsed:.2f}s")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of timing statistics."""
        return {
            op: {
                "total_time": stat.total_time,
                "calls": stat.calls,
                "average_time": stat.average_time,
            }
            for op, stat in self.stats.items()
        }


class AnalysisCache:
    """Persistent cache for method analysis results using SQLite with compression."""

    def __init__(
        self,
        max_age_days: int = 30,
        max_size_mb: int = 100,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ):
        """Initialize the cache with cleanup and compression settings.

        Args:
            max_age_days: Maximum age of cache entries in days (default: 30)
            max_size_mb: Maximum size of cache in megabytes (default: 100)
            compression_level: zlib compression level 0-9 (default: 6)
        """
        self.max_age_days = max_age_days
        self.max_size_mb = max_size_mb
        self.compression_level = compression_level

        # Store cache in the analysis directory within the package
        cache_dir = PACKAGE_ROOT / "analysis"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "method_analysis_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS method_cache (
                    package_name TEXT,
                    method_name TEXT,
                    source_hash TEXT,
                    result BLOB,
                    timestamp REAL,
                    size INTEGER,
                    compressed BOOLEAN,
                    PRIMARY KEY (package_name, method_name)
                )
            """
            )
            # Add index on timestamp for faster cleanup queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON method_cache(timestamp)"
            )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data, level=self.compression_level)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress zlib compressed data."""
        return zlib.decompress(data)

    def _get_source_hash(self, obj) -> str:
        """Get hash of method source code to detect changes."""
        try:
            source = inspect.getsource(obj)
            return hashlib.sha256(source.encode()).hexdigest()
        except (TypeError, OSError):
            return ""

    def get(self, package_name: str, method_name: str, obj) -> Optional[Any]:
        """Get cached result if it exists and is valid."""
        self._cleanup_if_needed()  # Opportunistic cleanup
        source_hash = self._get_source_hash(obj)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT result, source_hash, compressed FROM method_cache WHERE package_name = ? AND method_name = ?",
                (package_name, method_name),
            ).fetchone()

            if row and row[1] == source_hash:
                data = row[0]
                if row[2]:  # If compressed
                    data = self._decompress_data(data)
                return pickle.loads(data)
        return None

    def set(self, package_name: str, method_name: str, obj, result: Any):
        """Cache analysis result with compression."""
        source_hash = self._get_source_hash(obj)
        result_blob = pickle.dumps(result)

        # Compress if the data is large enough to benefit
        should_compress = len(result_blob) > 1024  # Only compress if > 1KB
        if should_compress:
            result_blob = self._compress_data(result_blob)

        size = len(result_blob)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO method_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    package_name,
                    method_name,
                    source_hash,
                    result_blob,
                    time.time(),
                    size,
                    should_compress,
                ),
            )

    def _cleanup_if_needed(self):
        """Perform cache cleanup if size or age limits are exceeded."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current cache size
                total_size = conn.execute(
                    "SELECT COALESCE(SUM(size), 0) FROM method_cache"
                ).fetchone()[0]

                if total_size > self.max_size_mb * 1024 * 1024:
                    # Delete oldest entries until under size limit
                    conn.execute(
                        """
                        DELETE FROM method_cache 
                        WHERE rowid IN (
                            SELECT rowid FROM method_cache 
                            ORDER BY timestamp ASC 
                            LIMIT -1 OFFSET ?
                        )
                    """,
                        (self.max_size_mb * 1024 * 1024 // 1000,),
                    )  # Rough estimate of entries to keep

                # Delete old entries
                max_age = time.time() - (self.max_age_days * 24 * 60 * 60)
                conn.execute("DELETE FROM method_cache WHERE timestamp < ?", (max_age,))

                conn.commit()
        except Exception as e:
            logger.debug(f"Cache cleanup failed: {e}")

    def clear(self):
        """Clear the entire cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM method_cache")
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache clear failed: {e}")

    def recompress_all(self, new_level: Optional[int] = None) -> Dict[str, Any]:
        """Recompress all cached data with new compression level."""
        if new_level is not None:
            self.compression_level = new_level

        stats = {"total": 0, "compressed": 0, "size_before": 0, "size_after": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all entries
                rows = conn.execute(
                    "SELECT package_name, method_name, result, compressed FROM method_cache"
                ).fetchall()

                for pkg, method, data, was_compressed in rows:
                    stats["total"] += 1
                    original_size = len(data)
                    stats["size_before"] += original_size

                    # Decompress if needed
                    if was_compressed:
                        data = self._decompress_data(data)

                    # Always try to compress
                    if len(data) > 1024:  # Only compress if > 1KB
                        compressed_data = self._compress_data(data)
                        new_size = len(compressed_data)
                        stats["size_after"] += new_size
                        stats["compressed"] += 1

                        # Update the entry
                        conn.execute(
                            """
                            UPDATE method_cache 
                            SET result = ?, size = ?, compressed = 1
                            WHERE package_name = ? AND method_name = ?
                            """,
                            (compressed_data, new_size, pkg, method),
                        )
                    else:
                        stats["size_after"] += len(data)

                conn.commit()

        except Exception as e:
            logger.error(f"Recompression failed: {e}")

        return stats


# Create global instances
timing = TimingManager()
analysis_cache = AnalysisCache()


def agent_analyze_package(package_name: str, target_functionality: str):
    analyzer = MethodAnalyzer()

    # First, quick scan to find relevant methods
    methods = analyzer.quick_scan(package_name)

    # Filter methods based on target functionality
    relevant_methods = [
        name
        for name, summary, categories in methods
        if target_functionality.lower() in summary.lower()
    ]

    # Deep analyze the most promising methods
    results = []
    for method_name in relevant_methods[:3]:  # Top 3 matches
        info = analyzer.deep_analyze(package_name, method_name)
        if info:
            results.append(info)

    return results


# Agent wants to write code for LiteLLM completion
analyzer = MethodAnalyzer()

# First, quick scan to find relevant methods
methods = analyzer.quick_scan("litellm")
completion_methods = [
    name for name, summary, _ in methods if "completion" in name.lower()
]

# Deep analyze promising methods
for method_name in completion_methods:
    info = analyzer.deep_analyze("litellm", method_name)
    if info:
        # Check method categories
        if "completion" in info["categories"]:
            # Found a completion method

            # Check if async or sync needed
            if "async" in info["categories"]:
                # Look for sync variant if needed
                sync_variant = next(
                    (
                        r["name"]
                        for r in info["related_methods"]
                        if r["type"] == "sync_variant"
                    ),
                    None,
                )
