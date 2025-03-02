from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import regex as re
from loguru import logger
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError

# Move to a Shared Project-wide folder in the project shared
from search_tool.shared_utils.spacy_utils import count_tokens, truncate_text_by_tokens


def initialize_database(config: Dict[str, Any]):
    """
    Sets up and connects to the ArangoDB client, ensuring the database is created if it doesn't exist.

    Args:
        config (dict): Either a standalone `arango_config` dictionary or a larger `config` dictionary
                    containing `arango_config` as a nested field.

    Returns:
        db: The connected ArangoDB database instance or None if an error occurs.
    """
    try:
        # Handle both standalone `arango_config` and nested `arango_config` cases
        if "arango_config" in config:
            arango_config = config["arango_config"]
        else:
            arango_config = config

        # Extract configuration values with defaults
        arango_config = config.get("arango_config", {})
        hosts = arango_config.get("hosts", ["http://localhost:8529"])
        db_name = arango_config.get("db_name", "verifaix")
        username = arango_config.get("username", "root")
        password = arango_config.get("password", "openSesame")

        # Initialize the ArangoDB client
        client = ArangoClient(hosts=hosts)

        # Connect to the database
        db = client.db(db_name, username=username, password=password)
        logger.info(f"Connected to database '{db_name}'.")
        return db

    except ArangoError as e:
        logger.error(f"ArangoDB error: {e}")
        return None
    except ServerConnectionError as e:
        logger.error(f"Failed to connect to ArangoDB server: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def insert_schema_into_collection(db, schema: dict, sparse_schema: dict, collection_name: str = "database_schema") -> bool:
    """
    Inserts the schema into the specified collection. Creates the collection if it doesn't exist.

    Args:
        db: The ArangoDB database object.
        schema: The complete schema to insert (as a dictionary).
        sparse_schema: The sparse schema to insert (as a dictionary).
        collection_name: The name of the schema collection (default: "database_schema").

    Returns:
        bool: True if the schema was inserted successfully, False otherwise.
    """
    try:
        # Create the collection if it doesn't exist
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            logger.info(f"Created collection '{collection_name}'.")

        # Insert the schema document
        schema_collection = db.collection(collection_name)
        schema_document = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  
            "complete_schema": schema,  # The complete schema
            "sparse_schema": sparse_schema  # The sparse schema for low context LLMs
        }
        schema_collection.insert(schema_document)
        logger.info(f"Schema inserted into collection '{collection_name}'.")
        return True
    
    except ArangoError as e:
        logger.error(f"ArangoDB operation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while inserting schema into '{collection_name}': {e}")
        return False

def verify_relationship(
    db, 
    source_collection: str, 
    target_collection: str, 
    source_field: str  # Renamed from 'field' to 'source_field'
) -> bool:
    """
    Verify if a relationship exists between two collections.
    - `source_collection`: The collection containing the reference field (e.g., `microsoft_issues`).
    - `target_collection`: The collection containing the target key (e.g., `microsoft_products`).
    - `source_field`: The field in the source collection that references the `_key` in the target collection (e.g., `product_id`).
    """
    try:
        # Construct the AQL query
        query = f"""
        FOR source IN {source_collection}
            FILTER source.{source_field} != null
            FOR target IN {target_collection}
                FILTER target._key == source.{source_field}
                LIMIT 1
                RETURN 1
        """
        
        # Log the query for debugging
        logger.debug(f"Executing AQL query:\n{query}")
        
        # Execute the query asynchronously
        cursor = db.aql.execute(query)
        
        # Return True if at least one result is found, otherwise False
        return bool(list(cursor))
    
    except Exception as e:
        logger.debug(f"Failed to verify relationship between {source_collection} and {target_collection}: {e}")
        return False
    
    

def find_matching_collection(field_base: str, collections: List[Dict], config: Dict) -> str | None:
    """Find a matching collection for a given field base name."""
    pattern = rf".*{re.escape(field_base)}s?$"  # Match field_base or field_base+'s' at end, case-insensitive
    
    for collection in collections:
        if re.search(pattern, collection["name"], re.IGNORECASE):
            return collection["name"]
    return None

def discover_relationships(db, config: Dict) -> List[Dict]:
    """Discover relationships between collections by analyzing edge collections and field patterns."""
    relationships = []
    collections = db.collections()
    
    # Find edge collection relationships
    relationships.extend(discover_edge_relationships(db, collections))
    
    # Find field-based relationships. 
    # Ignore system collections and excluded collections
    filtered_collections = [
        c for c in collections 
        if not c["name"].startswith('_') 
        and c["name"] not in config.get("excluded_collections", [])
    ]
    relationships.extend(discover_field_relationships(db, filtered_collections, config))
    
    return relationships

def discover_edge_relationships(db, collections: List[Dict]) -> List[Dict]:
    """Discover relationships from edge collections. TODO: NOT DONE YET"""
    
    relationships = []
    for collection in db.collections():
        collection_name = collection["name"]
        if collection_name.startswith('_') or collection["type"] != "edge":
            continue
            
        try:
            sample = get_sample_rows(db, collection_name, limit=1)
            if sample:
                from_collection = sample[0]["_from"].split("/")[0]
                to_collection = sample[0]["_to"].split("/")[0]
                if not (from_collection.startswith('_') or to_collection.startswith('_')):
                    relationships.append({
                        "from": from_collection,
                        "to": to_collection,
                        "type": "edge",
                        "via": collection_name
                    })
        except Exception as e:
            logger.error(f"Error analyzing edge collection {collection_name}: {e}")
    return relationships


def discover_field_relationships(db, collections: List[Dict], config: Dict) -> List[Dict]:
    """Discover relationships from field patterns."""
    relationships = []
    id_patterns = ['_id$', '_idfk$', '_foreign_key$', '_fk$', 'id$']
    
    
    for collection in collections:
        collection_name = collection["name"]
        collection_type = collection["type"]
        try:
            sample = get_sample_rows(db, collection_name, limit=1)
            if not sample:
                continue

            for field in sample[0].keys():
                if field.startswith('_'):
                    continue
                    
                for pattern in id_patterns:
                    if re.search(pattern, field, re.IGNORECASE):
                        # Remove the matched suffix to get base name
                        field_base = re.sub(pattern, '', field, flags=re.IGNORECASE)
                        
                        # Find matching collection
                        if target_collection := find_matching_collection(field_base, collections, config):
                            # Verify relationship exists in data
                            if  verify_relationship(
                                db, 
                                source_collection = collection_name, # microsoft_issue 
                                target_collection = target_collection, # microsoft_products", 
                                source_field = field
                            ):
                                relationships.append({
                                    "from": collection_name,
                                    "to": target_collection,
                                    "type": "foreign_key",
                                    "via": field
                                })
                                logger.info(f"Verified relationship: {collection_name}.{field} -> {target_collection}")
                                break  # Found valid relationship for this field
                            
        except Exception as e:
            logger.error(f"Error analyzing collection {collection_name} for relationships: {e}")
            
    return relationships


def load_aql_queries(directory: Path, include_queries: Optional[List[str]] = None) -> List[Dict]:
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
                value[-1]    # Last value
            ]
            
        # Handle base64 strings
        if isinstance(value, str) and value.startswith(('data:image', 'data:application')):
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

# this function takes in *args, **kwargs and returns the key you want to use for caching
# Not used for now
def custom_get_cache_key(*args, **kwargs):
    """Generate a custom cache key based on model, messages, temperature, and logit_bias."""
    key = (
        kwargs.get("model", "") 
        + str(kwargs.get("messages", "")) 
        + str(kwargs.get("temperature", 0.3)) 
        + str(kwargs.get("logit_bias", ""))
    )
    print("Key for cache:", key)
    return key




if __name__ == "__main__":
    config = {
        "arango_config": {
            "db_name": "test_db",
            "username": "root",
            "password": "root"
        }
    }
    db = initialize_database(config)
    print(db)

