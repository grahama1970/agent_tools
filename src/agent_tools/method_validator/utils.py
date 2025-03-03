"""Utility functions and classes for method validator."""

import time
from typing import Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from loguru import logger
import re


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


# Create global timing instance
timing = TimingManager()

def should_auto_execute_command(command: str) -> bool:
    """Determine if a command should be executed automatically without user confirmation.
    
    Args:
        command: The command to check
        
    Returns:
        True if the command should be executed automatically, False otherwise
    """
    # List of regex patterns for commands that can be executed automatically
    auto_patterns = [
        # Git commands
        r'^git (status|add|commit|push|pull|fetch|branch|checkout|merge|rebase|log|diff|clone|init)',
        # Build/compilation commands
        r'^(make|cmake|gcc|g\+\+|javac|mvn|gradle|npm run build|yarn build)',
        # Pytest commands
        r'^pytest\s+(?:(?!rm|remove|delete|drop|truncate|format|clean).)*$',  # Allow pytest but not destructive flags
        # Method validator commands
        r'^method-validator\s+\S+\s+(?:--method\s+\S+\s+--quick|--list-all)$'  # Allow quick validation and list-all commands
    ]
    
    return any(re.match(pattern, command) for pattern in auto_patterns)
