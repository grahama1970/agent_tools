"""Cache and timing utilities for method validator."""

import time
import sqlite3
import pickle
import hashlib
import inspect
import zlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, cast
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from loguru import logger
from .utils import timing, TimingManager
import importlib
from types import ModuleType

# Cache directory in user's home
CACHE_DIR = Path.home() / ".cache" / "method_validator"
DB_PATH = CACHE_DIR / "method_analysis_cache.db"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_db() -> sqlite3.Connection:
    """Get database connection with proper settings."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    return conn

def init_db() -> None:
    """Initialize the database schema."""
    logger.debug(f"Initializing database at {DB_PATH}")
    
    try:
        with get_db() as conn:
            logger.debug("Creating table if not exists")
            conn.execute("""
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
            """)
            
            # Log table info for debugging
            cursor = conn.execute("PRAGMA table_info(method_cache)")
            columns = cursor.fetchall()
            logger.debug(f"method_cache columns: {columns}")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def store_result(
    package_name: str,
    method_name: str,
    source_hash: str,
    result: Any,
    compress: bool = True
) -> None:
    """Store analysis result in cache."""
    try:
        serialized = pickle.dumps(result)
        size = len(serialized)
        
        with get_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO method_cache
                (package_name, method_name, source_hash, result, timestamp, size, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (package_name, method_name, source_hash, serialized, time.time(), size, compress)
            )
    except Exception as e:
        logger.error(f"Failed to store result: {e}")

def get_result(
    package_name: str,
    method_name: str,
    source_hash: Optional[str] = None
) -> Optional[Tuple[Any, float]]:
    """
    Get cached result if available and valid.
    Returns (result, timestamp) tuple or None if not found/invalid.
    """
    try:
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT result, timestamp, source_hash
                FROM method_cache
                WHERE package_name = ? AND method_name = ?
                """,
                (package_name, method_name)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
                
            result_blob, timestamp, cached_hash = row
            
            # Validate source hash if provided
            if source_hash and cached_hash != source_hash:
                return None
                
            try:
                result = pickle.loads(result_blob)
                return result, timestamp
            except:
                return None
                
    except Exception as e:
        logger.error(f"Failed to get result: {e}")
        return None

def clear_cache() -> None:
    """Clear all cached results."""
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM method_cache")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache."""
    try:
        with get_db() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size) as total_size,
                    MIN(timestamp) as oldest_entry,
                    MAX(timestamp) as newest_entry
                FROM method_cache
            """)
            row = cursor.fetchone()
            
            if not row:
                return {}
                
            return {
                "total_entries": row[0],
                "total_size_bytes": row[1],
                "oldest_entry_timestamp": row[2],
                "newest_entry_timestamp": row[3]
            }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {}

def get_cache_dir() -> Path:
    """Get the appropriate cache directory for the current platform.
    
    Returns:
        Path to the cache directory, created if it doesn't exist.
        Windows: %LOCALAPPDATA%/method_validator
        Unix: ~/.cache/method_validator or $XDG_CACHE_HOME/method_validator
    """
    if os.name == 'nt':  # Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', '~/.cache'))
    else:  # Unix-like
        base_dir = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache'))
    
    base_dir = base_dir.expanduser()
    cache_dir = base_dir / 'method_validator'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# Compression level (0-9, where 0 is no compression and 9 is maximum)
DEFAULT_COMPRESSION_LEVEL = 6

# Lazy loading of analyzer to avoid circular imports
_analyzer = None


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from .analyzer import MethodAnalyzer

        _analyzer = MethodAnalyzer()
    return _analyzer


@dataclass
class TimingStats:
    """Container for timing statistics of operations."""

    operation: str
    total_time: float = 0.0
    calls: int = 0

    @property
    def average_time(self) -> float:
        return self.total_time / self.calls if self.calls > 0 else 0.0


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

        # Use platform-specific cache directory
        self.db_path = get_cache_dir() / "method_analysis_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        logger.debug(f"Initializing database at {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            # Only create the table if it doesn't exist
            logger.debug("Creating table if not exists")
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

            # Verify schema
            columns = conn.execute("PRAGMA table_info(method_cache)").fetchall()
            logger.debug(f"method_cache columns: {columns}")

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
        """Cache analysis result."""
        source_hash = self._get_source_hash(obj)
        result_blob = pickle.dumps(result)
        size = len(result_blob)
        timestamp = time.time()
        compressed = False

        logger.debug(f"Setting cache for {package_name}.{method_name}")
        logger.debug(
            f"Values: package_name={package_name}, method_name={method_name}, hash={source_hash[:8]}..., size={size}, timestamp={timestamp}, compressed={compressed}"
        )
        logger.debug(f"Result tuple: {result[:2]}")  # Only log first two elements for brevity
        logger.debug(f"Table schema: {self._get_table_schema()}")

        with sqlite3.connect(self.db_path) as conn:
            try:
                # Insert the values
                insert_sql = """
                    INSERT OR REPLACE INTO method_cache 
                    (package_name, method_name, source_hash, result, timestamp, size, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                values = (
                    package_name,
                    method_name,
                    source_hash,
                    result_blob,
                    timestamp,
                    size,
                    compressed,
                )
                logger.debug(f"Inserting with SQL: {insert_sql}")
                logger.debug(f"Values: {values[:3]}, <blob>, {values[4:]}")  # Don't log the blob
                conn.execute(insert_sql, values)
                conn.commit()  # Add explicit commit
                logger.debug(f"Successfully inserted/updated cache entry for {package_name}.{method_name}")
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
                # Log table schema at point of failure
                columns = conn.execute("PRAGMA table_info(method_cache)").fetchall()
                logger.error(f"Current table schema: {columns}")
                raise

    def _get_table_schema(self) -> str:
        """Get the current table schema."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='method_cache'"
            ).fetchone()[0]

    def _cleanup_if_needed(self):
        """Perform cache cleanup if size or age limits are exceeded."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current cache size
                total_size = conn.execute(
                    "SELECT COALESCE(SUM(length(result)), 0) FROM method_cache"
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
                            SET result = ?, compressed = 1
                            WHERE package_name = ? AND method_name = ?
                            """,
                            (compressed_data, pkg, method),
                        )
                    else:
                        stats["size_after"] += len(data)

                conn.commit()

        except Exception as e:
            logger.error(f"Recompression failed: {e}")

        return stats

    def get_all_methods(self, package_name: str) -> List[Tuple[str, str, List[str]]]:
        """Get all cached methods for a package."""
        self._cleanup_if_needed()
        
        try:
            # Time the database query
            query_start = time.time()
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT method_name, result, compressed
                    FROM method_cache 
                    WHERE package_name = ?
                    """,
                    (package_name,)
                ).fetchall()
            query_time = time.time() - query_start
            logger.info(f"Database query completed in {query_time:.3f}s")

            # Time the decompression and processing
            process_start = time.time()
            results = []
            for method_name, data, is_compressed in rows:
                try:
                    if is_compressed:
                        data = self._decompress_data(data)
                    result = cast(Tuple[str, str, List[str]], pickle.loads(data))
                    results.append(result)
                except Exception as e:
                    logger.debug(f"Error processing cached data for {method_name}: {e}")
                    continue
            
            process_time = time.time() - process_start
            logger.info(f"Data processing completed in {process_time:.3f}s")
            logger.info(f"Total methods retrieved: {len(results)}")
            
            return results
        except Exception as e:
            logger.error(f"Cache retrieval error for package {package_name}: {e}")
            return []

    def test_performance(self, package_name: str) -> Dict[str, float]:
        """Test database retrieval performance."""
        perf_stats = {}
        
        # Test 1: Simple count query
        start = time.time()
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM method_cache WHERE package_name = ?",
                (package_name,)
            ).fetchone()[0]
        perf_stats['count_query_time'] = time.time() - start
        
        # Test 2: Full retrieval without decompression
        start = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT method_name, result, compressed FROM method_cache WHERE package_name = ?",
                (package_name,)
            ).fetchall()
        perf_stats['raw_retrieval_time'] = time.time() - start
        
        # Test 3: Full retrieval with decompression
        start = time.time()
        results = []
        for _, data, is_compressed in rows:
            try:
                if is_compressed:
                    data = self._decompress_data(data)
                result = pickle.loads(data)
                results.append(result)
            except Exception:
                continue
        perf_stats['full_processing_time'] = time.time() - start
        
        logger.info(f"Performance test results for {package_name}:")
        logger.info(f"Found {count} cached methods")
        logger.info(f"Count query time: {perf_stats['count_query_time']:.3f}s")
        logger.info(f"Raw retrieval time: {perf_stats['raw_retrieval_time']:.3f}s")
        logger.info(f"Processing time: {perf_stats['full_processing_time']:.3f}s")
        
        return perf_stats

    def set_batch(self, entries: List[Tuple[str, str, Any, Any]]):
        """Cache multiple analysis results in a single transaction.
        
        Args:
            entries: List of tuples containing (package_name, method_name, obj, result)
        """
        with sqlite3.connect(self.db_path) as conn:
            try:
                # Prepare all entries
                values = []
                for package_name, method_name, obj, result in entries:
                    source_hash = self._get_source_hash(obj)
                    result_blob = pickle.dumps(result)
                    size = len(result_blob)
                    timestamp = time.time()
                    compressed = False
                    values.append((
                        package_name,
                        method_name,
                        source_hash,
                        result_blob,
                        timestamp,
                        size,
                        compressed
                    ))

                # Insert all entries in a single transaction
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO method_cache 
                    (package_name, method_name, source_hash, result, timestamp, size, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    values
                )
                conn.commit()
                logger.debug(f"Successfully inserted/updated {len(entries)} cache entries in batch")
            except sqlite3.Error as e:
                logger.error(f"Database error during batch insert: {e}")
                raise


# Create global instances
analysis_cache = AnalysisCache()


def agent_analyze_package(package_name: str, target_functionality: str):
    """Analyze a package using the agent-based approach."""
    analyzer = get_analyzer()
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
