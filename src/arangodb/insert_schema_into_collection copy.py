from datetime import datetime, timezone
from loguru import logger
from arango.exceptions import ArangoError


def insert_schema_into_collection(
    db, schema: dict, sparse_schema: dict, collection_name: str = "database_schema"
) -> bool:
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
            "sparse_schema": sparse_schema,  # The sparse schema for low context LLMs
        }
        schema_collection.insert(schema_document)
        logger.info(f"Schema inserted into collection '{collection_name}'.")
        return True

    except ArangoError as e:
        logger.error(f"ArangoDB operation error: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error while inserting schema into '{collection_name}': {e}"
        )
        return False
