from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


def load_aql_queries(
    directory: Path, include_queries: Optional[List[str]] = None
) -> List[Dict]:
    """Load AQL queries from .aql files in the specified directory."""
    example_queries = []
    for file_path in directory.glob("*.aql"):
        query_name = file_path.stem
        if include_queries and query_name not in include_queries:
            continue  # Skip queries not in the include list
        try:
            with open(file_path, "r") as f:
                query_content = f.read()
                example_queries.append({"name": query_name, "query": query_content})
        except Exception as e:
            logger.error(f"Failed to load AQL query from {file_path}: {e}")
    return example_queries
