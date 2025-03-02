import asyncio
from arango import ArangoClient
from loguru import logger
import nltk
from nltk.corpus import stopwords

# Ensure the stopwords corpus is downloaded
nltk.download('stopwords')


async def delete_analyzer_if_exists(db, analyzer_name, collection_name='verifaix'):
    try:
        existing_analyzers = await asyncio.to_thread(db.analyzers)
        target_analyzer_name = f'{db.name}::{analyzer_name}'
        analyzer_exists = any(a.get('name') == target_analyzer_name for a in existing_analyzers)
        if analyzer_exists:
            await asyncio.to_thread(db.delete_analyzer, analyzer_name, force=True)
            logger.info(f"Analyzer '{analyzer_name}' deleted successfully.")     
        return analyzer_exists

    except Exception as e:
        logger.error(f"Failed to check existing analyzers: {e}")
        return False



async def create_analyzer(db, config):
    # Extract properties from config
    analyzer_name = config["analyzer_name"]
    analyzer_properties = config["properties"]
    analyzer_features = config["features"]
    collection_name = config["collection_name"]

    await delete_analyzer_if_exists(db, analyzer_name, collection_name)

    # Create the new analyzer
    try:
        await asyncio.to_thread(
            db.create_analyzer,
            name=analyzer_name,
            analyzer_type="text",
            properties=analyzer_properties,
            features=analyzer_features
        )
        logger.info(f"Analyzer '{analyzer_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create analyzer: {e}")
        return False
        

async def main():
    client = ArangoClient(hosts='http://localhost:8529')
    db = await asyncio.to_thread(client.db, 'verifaix', username='root', password='openSesame')
    
    config = {
        "analyzer_name": "text_term_analyzer",
        "collection_name": "verifaix",
        "properties": {
            "locale": "en",
            "case": "lower",
            "accent": False,
            "stemming": False,
            # "edgeNgram": {
            #     "min": 3,
            #     "max": 6,
            #     "preserveOriginal": True
            # },
            "stopwords": stopwords.words('english')
        },
        "features": [] # ['position', 'frequency']
    }
    
    await create_analyzer(db, config)


if __name__ == "__main__":
    asyncio.run(main())
