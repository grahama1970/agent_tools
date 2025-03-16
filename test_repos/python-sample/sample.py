#!/usr/bin/env python3
"""
Sample Python file with well-structured code for extraction testing.
"""

import os
import sys
from typing import List, Dict, Any, Optional


def calculate_sum(numbers: List[int]) -> int:
    """
    Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of integers to sum
        
    Returns:
        The sum of all numbers in the list
    """
    return sum(numbers)


def find_max(numbers: List[int]) -> Optional[int]:
    """
    Find the maximum value in a list of numbers.
    
    Args:
        numbers: List of integers to search
        
    Returns:
        The maximum value or None if the list is empty
    """
    if not numbers:
        return None
    return max(numbers)


class DataProcessor:
    """A class for processing data with various methods."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            name: Name of the processor
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.processed_items = 0
        
    def process_item(self, item: Any) -> Dict[str, Any]:
        """
        Process a single item.
        
        Args:
            item: The item to process
            
        Returns:
            A dictionary with processed data
        """
        self.processed_items += 1
        
        return {
            "original": item,
            "processed_by": self.name,
            "process_id": self.processed_items,
            "config_used": self.config
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            A dictionary with processor statistics
        """
        return {
            "processor_name": self.name,
            "items_processed": self.processed_items,
            "config": self.config
        }


class AdvancedProcessor(DataProcessor):
    """An advanced data processor with additional capabilities."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None, 
                 advanced_mode: bool = False):
        """
        Initialize the AdvancedProcessor.
        
        Args:
            name: Name of the processor
            config: Optional configuration dictionary
            advanced_mode: Whether to enable advanced processing
        """
        super().__init__(name, config)
        self.advanced_mode = advanced_mode
        
    def process_item(self, item: Any) -> Dict[str, Any]:
        """
        Process a single item with advanced processing if enabled.
        
        Args:
            item: The item to process
            
        Returns:
            A dictionary with processed data
        """
        result = super().process_item(item)
        
        if self.advanced_mode:
            result["advanced_processing"] = True
            result["advanced_result"] = f"Advanced: {item}"
            
        return result


if __name__ == "__main__":
    # Example usage
    numbers = [1, 2, 3, 4, 5]
    print(f"Sum: {calculate_sum(numbers)}")
    print(f"Max: {find_max(numbers)}")
    
    processor = DataProcessor("Basic Processor")
    result = processor.process_item("test")
    print(f"Basic processing result: {result}")
    
    advanced = AdvancedProcessor("Advanced Processor", advanced_mode=True)
    advanced_result = advanced.process_item("test")
    print(f"Advanced processing result: {advanced_result}") 