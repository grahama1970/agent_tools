from typing import List, Any, Optional
from loguru import logger

def process_data(data: List[Any], should_filter: bool = False) -> Optional[List[Any]]:
    """Process a list of data items with optional filtering.
    
    Args:
        data: List of items to process
        should_filter: Whether to filter the data
        
    Returns:
        Processed data list or None if an error occurs
    """
    result: List[Any] = []
    filtered: List[Any] = []
    
    try:
        logger.info("Processing data...")
        data.process_items()
        
        for item in data:
            if should_filter:
                processed = item.nonexistent_method()
                result.append(processed)
            else:
                filtered.append(item)
                
        logger.debug("Returning results")
        return result if result else filtered
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None 