import regex as re
from loguru import logger
from typing import Dict, Any
from arango import ArangoClient
from arango.exceptions import ArangoError, ServerConnectionError

# Move to a Shared Project-wide folder in the project shared
from smolagent.utils.spacy_utils import count_tokens, truncate_text_by_tokens


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
        # logger.success(f"Connected to database '{db_name}'.")
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


# from arango import ArangoClient
# from loguru import logger
# from smolagent.database.initialize_database import initialize_database  # Shared database initialization

# # Initialize Loguru Logger
# logger.add("arangodb_view.log", rotation="1 MB", level="INFO")


# def is_numeric_list(value):
#     """
#     Helper function to check if a list contains only numbers (int/float).
#     """
#     return isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)


# def get_collection_fields(db, collection_name):
#     """
#     Fetch field names dynamically from a collection,
#     excluding numeric fields and lists of numbers.
#     """
#     try:
#         cursor = db.collection(collection_name).all(limit=10)  # Fetch sample docs
#         fields = {}

#         for doc in cursor:
#             for key, value in doc.items():
#                 # Exclude numeric fields and numeric lists
#                 if isinstance(value, (int, float)) or is_numeric_list(value):
#                     continue

#                 # Store field type
#                 fields[key] = type(value)

#         logger.info(f"Extracted fields for collection {collection_name}: {fields}")
#         return fields

#     except Exception as e:
#         logger.error(f"Error retrieving fields from {collection_name}: {e}")
#         return {}


# def create_view(db, config):
#     """
#     Create an ArangoSearch View dynamically based on field types.
#     """
#     try:
#         view_name = config["view_name"]
#         collections = config["collections"]

#         # Build dynamic view properties
#         view_properties = {"links": {}}

#         for collection in collections:
#             fields = get_collection_fields(db, collection)

#             # Construct field mappings dynamically
#             field_mappings = {
#                 field: {"analyzers": ["text_en"]}
#                 for field, dtype in fields.items()
#                 if field != "embedding" and dtype not in (int, float, list)
#             }

#             # Ensure 'embedding' field gets 'identity' analyzer
#             if "embedding" in fields:
#                 field_mappings["embedding"] = {"analyzers": ["identity"]}

#             view_properties["links"][collection] = {"fields": field_mappings}

#         # Create or update the ArangoSearch view
#         if db.has_arangosearch_view(view_name):
#             db.update_arangosearch_view(view_name, properties=view_properties)
#             logger.info(f"Updated existing ArangoSearch view: {view_name}")
#         else:
#             db.create_arangosearch_view(view_name, properties=view_properties)
#             logger.info(f"Created new ArangoSearch view: {view_name}")

#     except Exception as e:
#         logger.error(f"Error creating/updating view {config['view_name']}: {e}")


# def main():
#     """
#     Main function: Initializes the database and creates the ArangoSearch view.
#     """
#     config = {
#         "arango_db": "verifaix",
#         "arango_host": "http://localhost:8529",
#         "arango_username": "root",
#         "arango_password": "openSesame",
#         "collections": ["microsoft_issues", "microsoft_support", "microsoft_products"],
#         "view_name": "microsoft_view",
#     }

#     # Initialize database connection
#     db = initialize_database(config)

#     if db:
#         create_view(db, config)
#     else:
#         logger.error("Database connection failed. Exiting script.")


# if __name__ == "__main__":
#     main()
