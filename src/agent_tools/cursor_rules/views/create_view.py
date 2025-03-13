import asyncio
from arango import ArangoClient
from loguru import logger
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


async def delete_analyzer_if_exists(db: ArangoClient, analyzer_name: str) -> None:
    """Delete an analyzer if it exists."""
    try:
        existing_analyzers = await asyncio.to_thread(db.analyzers)
        target_analyzer_name = f'{db.name}::{analyzer_name}'
        analyzer_exists = any(a.get('name') == target_analyzer_name for a in existing_analyzers)
        if analyzer_exists:
            await asyncio.to_thread(db.delete_analyzer, analyzer_name, force=True)
            logger.info(f"Analyzer '{analyzer_name}' deleted successfully.")     
    except Exception as e:
        logger.error(f"Failed to check existing analyzers: {e}")


async def create_analyzer(db: ArangoClient, analyzer_config: dict) -> None:
    """Create a custom text analyzer."""
    await delete_analyzer_if_exists(db, analyzer_config["analyzer_name"])

    try:
        await asyncio.to_thread(
            db.create_analyzer,
            name=analyzer_config["analyzer_name"],
            analyzer_type="text",
            properties=analyzer_config["properties"],
            features=analyzer_config["features"]
        )
        logger.info(f"Analyzer '{analyzer_config['analyzer_name']}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create analyzer: {e}")


async def create_view(db: ArangoClient, arango_config: dict) -> None:
    """Create a unified arangosearch view across all specified collections."""
    view_name = arango_config.get("unified_view_name", "microsoft_search_view")

    try:
        # Delete existing unified view if it exists
        existing_views = await asyncio.to_thread(db.views)
        view_exists = any(view['name'] == view_name for view in existing_views)

        if view_exists:
            await asyncio.to_thread(db.delete_view, view_name)
            logger.info(f"Existing unified view '{view_name}' deleted.")

        # Create unified view configuration
        links_config = {
            collection_name: {
                "includeAllFields": False,
                "fields": {
                    "cleanedTerm": {
                        "analyzers": [arango_config['analyzer_name']]
                    },
                    "embedding": {
                        "type": "vector",
                        "dimension": 768,
                        "similarity": "cosine"
                    }
                }
            }
            for collection_name in arango_config['collection_names']
        }

        # Create single view with all collections linked
        await asyncio.to_thread(
            db.create_view,
            name=view_name,
            view_type='arangosearch',
            properties={
                "links": links_config,
                "storedValues": [
                    {
                        "fields": ["embedding"],
                        "compression": "lz4"
                    }
                ],
                "optimizeTopK": ["COSINE_SIMILARITY"],
                "consolidationIntervalMsec": 1000,
                "commitIntervalMsec": 1000
            }
        )
        logger.info(f"Unified view '{view_name}' created successfully for collections: {arango_config['collection_names']}")
    except Exception as e:
        logger.error(f"Failed to create unified view: {e}")


async def create_vector_index(db: ArangoClient, collection_name: str) -> None:
    """Create a vector index on the 'embedding' field using the native vector type."""
    try:
        collection = db.collection(collection_name)
        existing_indexes = await asyncio.to_thread(collection.indexes)
        # Check if an inverted index already exists on the 'embedding' field
        index_exists = any(
            index['type'] == 'inverted' and 'embedding' in index.get('fields', [])
            for index in existing_indexes
        )

        if not index_exists:
            index_definition = {
                "type": "inverted",
                "fields": [
                    {
                        "name": "embedding",
                        "type": "vector",
                        "dimension": 768,
                        "similarity": "cosine"
                    }
                ],
                "includeAllFields": False
            }
            await asyncio.to_thread(collection.add_index, index_definition)
            logger.info(f"Vector index created on 'embedding' field for collection '{collection_name}'.")
        else:
            logger.info(f"Vector index already exists on 'embedding' field for collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to create vector index for collection '{collection_name}': {e}")


async def main() -> None:
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
            "unified_view_name": "microsoft_search_view", # the name of the view of all the collections
            "collection_names": [
                "microsoft_glossary", 
                "microsoft_issues", 
                "microsoft_products",
                "microsoft_support"
            ],  # List of collections
            "analyzer_name": "text_analyzer"
        }
    }
    
    client = ArangoClient(hosts=config["arango_config"]["host"])
    db = await asyncio.to_thread(
        client.db, 
        config["arango_config"]["db_name"],
        username=config["arango_config"]["username"],
        password=config["arango_config"]["password"]
    )
    
    # Create custom text analyzer for your text field
    await create_analyzer(db, config["analyzer_config"])
    
    # Create a native vector index on the 'embedding' field for each collection
    for collection_name in config["arango_config"]["collection_names"]:
        await create_vector_index(db, collection_name)
    
    # Create the ArangoSearch view
    await create_view(db, config["arango_config"])

if __name__ == "__main__":
    asyncio.run(main())
