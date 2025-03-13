"""
Schema definitions for Cursor Rules Database.
"""
import json
from pathlib import Path

def load_schema(schema_name: str) -> dict:
    """Load a schema file from the schemas directory."""
    schema_path = Path(__file__).parent / f"{schema_name}.json"
    with open(schema_path) as f:
        return json.load(f)

# Load schemas
DB_SCHEMA = load_schema("db_schema")
AI_KNOWLEDGE_SCHEMA = load_schema("ai_knowledge_schema")

__all__ = ['DB_SCHEMA', 'AI_KNOWLEDGE_SCHEMA', 'load_schema'] 