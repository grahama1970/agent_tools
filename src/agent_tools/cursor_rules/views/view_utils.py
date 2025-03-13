import asyncio
from arango import ArangoClient
from arango.database import StandardDatabase
from loguru import logger
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def delete_analyzer_if_exists(db: StandardDatabase, analyzer_name: str) -> None:
    """Delete an analyzer if it exists."""
    try:
        existing_analyzers = db.analyzers()
        target_analyzer_name = f'{db.name}::{analyzer_name}'
        analyzer_exists = any(a.get('name') == target_analyzer_name for a in existing_analyzers)
        if analyzer_exists:
            db.delete_analyzer(analyzer_name, force=True)
            logger.info(f"Analyzer '{analyzer_name}' deleted successfully.")     
    except Exception as e:
        logger.error(f"Failed to check existing analyzers: {e}")


def create_analyzer(db: StandardDatabase, analyzer_config: dict) -> None:
    """Create a custom text analyzer."""
    delete_analyzer_if_exists(db, analyzer_config["analyzer_name"])

    try:
        db.create_analyzer(
            name=analyzer_config["analyzer_name"],
            analyzer_type="text",
            properties=analyzer_config["properties"],
            features=analyzer_config["features"]
        )
        logger.info(f"Analyzer '{analyzer_config['analyzer_name']}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create analyzer: {e}")


def create_view(db: StandardDatabase, view_config: dict) -> None:
    """Create an ArangoSearch view from configuration.
    
    Args:
        db: ArangoDB database instance
        view_config: Complete view configuration including name, type, and properties
    """
    try:
        # Delete existing view if it exists
        view_name = view_config["name"]
        existing_views = db.views()
        view_exists = any(view['name'] == view_name for view in existing_views)

        if view_exists:
            db.delete_view(view_name)
            logger.info(f"Existing view '{view_name}' deleted.")

        # Create view directly from config
        db.create_view(
            name=view_name,
            view_type=view_config["type"],
            properties=view_config["properties"]
        )
        logger.info(f"View '{view_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create view: {e}")


def create_vector_index(db: StandardDatabase, collection_name: str) -> None:
    """Create a vector index on the 'embedding' field using the native vector type."""
    try:
        collection = db.collection(collection_name)
        existing_indexes = collection.indexes()
        # Check if a vector index already exists on the 'embedding' field
        index_exists = any(
            index['type'] == 'vector' and 'embedding' in index.get('fields', [])
            for index in existing_indexes
        )

        if not index_exists:
            # Get collection info to check if it's an edge collection
            collection_info = collection.properties()
            is_edge = collection_info.get('type') == 3  # type 3 indicates edge collection
            
            # Get collection size for nLists parameter
            collection_count = collection.count()
            nlist = min(collection_count, 100)
            
            index_definition = {
                "name": "vector_cosine",
                "type": "vector",
                "fields": ["embedding"],
                "params": {
                    "metric": "cosine",
                    "dimension": 768,
                    "nLists": nlist
                }
            }
            collection.add_index(index_definition)
            logger.info(f"Vector index created on 'embedding' field for {('edge' if is_edge else 'document')} collection '{collection_name}'.")
        else:
            logger.info(f"Vector index already exists on 'embedding' field for collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to create vector index for collection '{collection_name}': {e}")


def main() -> None:
    config = {
        "analyzer_config": {
            "analyzer_name": "text_analyzer",
            "properties": {
                "locale": "en",
                "case": "lower",
                "accent": False,
                "stemming": False,
                "stopwords": stopwords.words('english')
            },
            "features": ['position', 'frequency']
        },
        "arango_config": {
            "host": "http://localhost:8529",
            "username": "root",
            "password": "openSesame",
            "db_name": "verifaix",
            "unified_view_name": "microsoft_search_view",
            "collection_names": [
                "microsoft_glossary", 
                "microsoft_issues", 
                "microsoft_products",
                "microsoft_support"
            ],
            "analyzer_name": "text_analyzer"
        }
    }
    
    client = ArangoClient(hosts=config["arango_config"]["host"])
    db = client.db(
        config["arango_config"]["db_name"],
        username=config["arango_config"]["username"],
        password=config["arango_config"]["password"]
    )
    
    # Create custom text analyzer
    create_analyzer(db, config["analyzer_config"])
    
    # Create vector indexes
    for collection_name in config["arango_config"]["collection_names"]:
        create_vector_index(db, collection_name)
    
    # Create ArangoSearch view
    create_view(db, config["arango_config"])

if __name__ == "__main__":
    main()
