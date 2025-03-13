"""Performance monitoring utilities for fetch-page."""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
from loguru import logger
import psutil
import os
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OperationStats:
    """Statistics for a single operation."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    memory_start: int = 0
    memory_end: Optional[int] = None
    cpu_percent: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Calculate operation duration in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def memory_used(self) -> int:
        """Calculate memory used in bytes."""
        if self.memory_end is None:
            return 0
        return self.memory_end - self.memory_start

class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize the performance monitor."""
        self.operations: Dict[str, OperationStats] = {}
        self.process = psutil.Process(os.getpid())

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.process.memory_info().rss

    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        return self.process.cpu_percent()

    @contextmanager
    def track(self, operation_name: str):
        """
        Context manager to track an operation's performance.
        
        Args:
            operation_name: Name of the operation to track
        """
        # Initialize stats
        stats = OperationStats(
            name=operation_name,
            start_time=time.time(),
            memory_start=self._get_memory_usage()
        )
        self.operations[operation_name] = stats

        try:
            yield
            # Update stats on successful completion
            stats.end_time = time.time()
            stats.memory_end = self._get_memory_usage()
            stats.cpu_percent = self._get_cpu_percent()
            stats.success = True
        except Exception as e:
            # Update stats on error
            stats.end_time = time.time()
            stats.memory_end = self._get_memory_usage()
            stats.cpu_percent = self._get_cpu_percent()
            stats.success = False
            stats.error = str(e)
            raise

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations."""
        return {
            op_name: {
                "duration": stats.duration,
                "memory_used": stats.memory_used,
                "cpu_percent": stats.cpu_percent,
                "success": stats.success,
                "error": stats.error
            }
            for op_name, stats in self.operations.items()
        }

    def display_stats(self) -> None:
        """Display performance statistics."""
        logger.info("\nPerformance Statistics:")
        logger.info("-" * 80)
        
        for op_name, stats in self.operations.items():
            logger.info(f"\nOperation: {op_name}")
            logger.info(f"Duration: {stats.duration:.2f}s")
            logger.info(f"Memory Used: {stats.memory_used / 1024 / 1024:.2f} MB")
            logger.info(f"CPU Usage: {stats.cpu_percent:.1f}%")
            if not stats.success:
                logger.error(f"Error: {stats.error}")
        
        logger.info("-" * 80)

    def reset(self) -> None:
        """Reset all tracked statistics."""
        self.operations.clear() 