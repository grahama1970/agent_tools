from typing import List, Dict, Optional
from loguru import logger
from arango.exceptions import ArangoError

from smolagent.utils.spacy_utils import count_tokens, truncate_text_by_tokens


# Function to get sample rows from a collection
def get_sample_rows(db, collection_name: str, limit: int = 5) -> List[Dict]:
    """Get sample rows from a collection, excluding ArangoDB system fields."""
    try:
        cursor = db.aql.execute(
            f"FOR doc IN {collection_name} SORT RAND() LIMIT {limit} RETURN doc"
        )

        # Remove ArangoDB system fields from each document
        cleaned_rows = []
        for doc in cursor:
            cleaned_doc = {k: v for k, v in doc.items() if not k.startswith("_")}
            cleaned_rows.append(cleaned_doc)

        return cleaned_rows
    except Exception as e:
        logger.error(f"Error sampling rows from collection {collection_name}: {e}")
        return []


def truncate_sample_data(sample_rows: List[Dict], max_tokens: int = 50) -> List[Dict]:
    """Truncate long fields in sample documents for LLM comprehension."""

    def truncate_value(value, key: Optional[str] = None):
        """Truncate values for LLM processing while preserving meaningful content."""
        if value is None:
            return value

        # Handle embeddings (lists)
        if key == "embedding" and isinstance(value, list):
            if len(value) <= 5:
                return value
            # For large arrays (like embeddings), show first 3 values and last value
            return [
                *value[:3],  # First 3 values
                f"... ({len(value)} total values) ...",  # Length indicator
                value[-1],  # Last value
            ]

        # Handle base64 strings
        if isinstance(value, str) and value.startswith(
            ("data:image", "data:application")
        ):
            if len(value) <= 40:
                return value
            return f"{value[:30]}...({len(value)} chars)...{value[-10:]}"

        # Handle JSON/dict fields
        if isinstance(value, (dict, list)):
            text_value = str(value)
            if count_tokens(text_value) > max_tokens:
                return truncate_text_by_tokens(text_value, max_tokens)
            return value

        # Handle long text strings
        if isinstance(value, str) and count_tokens(value) > max_tokens:
            return truncate_text_by_tokens(value, max_tokens)

        return value

    results = [{k: truncate_value(v, k) for k, v in row.items()} for row in sample_rows]
    return results
