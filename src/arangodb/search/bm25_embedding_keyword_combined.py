from pathlib import Path
from arango import ArangoClient
import arango
import regex as re
import json
from loguru import logger
from deepmerge import always_merger
from sentence_transformers import SentenceTransformer
from src.embedding.embedding_utils import get_sentence_transformer

from shared_utils.get_project_root import get_project_root

# Calculate project directory once
project_dir = get_project_root()

# Helper functions
def load_config(file_path: str | Path) -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise

def load_aql_query(file_path: str | Path) -> str:
    """Load AQL query from file."""
    try:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        with open(path, "r") as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Failed to load AQL query from {file_path}: {e}")
        raise

def normalize_text(text: str) -> str:
    """Normalize the input text by removing HTML, special characters, and extra spaces."""
    
    regex_patterns = [
        (r'<[^>]+>', ''),           # Remove HTML tags
        (r'[^a-zA-Z0-9 ]', ''),     # Remove special characters
        (r'\s+', ' ')               # Normalize whitespace
    ]
    
    normalized_text = text
    for pattern, replacement in regex_patterns:
        normalized_text = re.sub(pattern, replacement, normalized_text)
        
    return normalized_text.strip()

def initialize_arangodb_client(config: dict) -> tuple[ArangoClient, arango.database.StandardDatabase]:
    """Initialize and return the ArangoDB client and database."""
    client = ArangoClient(hosts=config["arango"]["hosts"])
    db = client.db(
        config["arango"]["db_name"],
        username=config["arango"]["username"],
        password=config["arango"]["password"]
    )
    return client, db

def generate_search_embedding(search_text: str, model: SentenceTransformer) -> list:
    """Normalize the search text and generate its embedding."""
    search_text = normalize_text(search_text)
    return model.encode(search_text).tolist()

def execute_bm25_embedding_keyword_query(config: dict, model: SentenceTransformer) -> list:
    """
    Execute the AQL query using the provided configuration and model.
    """
    try:
        # Initialize ArangoDB client
        logger.info("Connecting to ArangoDB...")
        client, db = initialize_arangodb_client(config)

        # Load AQL query file
        logger.info("Loading AQL query...")
        aql_query = load_aql_query(config["aql"]["query_path"])

        # Generate search embedding
        logger.info("Normalizing search text and generating embeddings...")
        search_embedding = generate_search_embedding(config["search"]["text"], model)

        # Bind variables for AQL query
        bind_vars = {
            "search_text": config["search"]["text"],
            "embedding_search": search_embedding,
            "embedding_similarity_threshold": config["aql"]["embedding_similarity_threshold"],
            "bm25_similarity_threshold": config["aql"]["bm25_similarity_threshold"],
            "k": config["aql"]["k"],
            "b": config["aql"]["b"],
            "top_n": config["aql"]["top_n"]
        }

        # Execute the AQL query
        logger.info("Executing AQL query...")
        results = db.aql.execute(aql_query, bind_vars=bind_vars)
        results = list(results)
        logger.debug(f"Query results: {json.dumps(results, indent=4)}")
        return results

    except arango.AQLQueryExecuteError as e:
        logger.error(f"AQL query execution failed: {e.error_message}")
        logger.debug(f"AQL query: {e.query}")
        logger.debug(f"Query parameters: {e.parameters}")
        raise
    except arango.ArangoServerError as e:
        logger.error(f"ArangoDB server error: {e.error_message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error executing AQL query: {e}")
        raise

def validate_config(config: dict) -> bool:
    """Validate the configuration dictionary."""
    required_keys = {
        "arango": ["hosts", "db_name", "username", "password"],
        "aql": ["query_path", "embedding_similarity_threshold", "bm25_similarity_threshold", "k", "b", "top_n"],
        "model": ["name"],
        "search": ["text"]
    }
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing section in config: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing key in config[{section}]: {key}")
    return True



def main():
    """Main function to execute the script."""
    try:
        # Load base configuration from file
        config_path = project_dir / "src/setttings/config.json" # hack for now
        config = load_config(config_path)

        # Validate the configuration
        validate_config(config)

        # Define the updates
        updates = {
            "search": {
                "text": "What is an Audit administrator where do I find an Auditor?!"
            },
            "aql": {
                "query_path": str(project_dir / "src/utils/aql/bm25_embedding_keyword_combined.aql"),
                "embedding_similarity_threshold": 0.6,
                "top_n": 5
            },
            "model": {
                "name": "sentence-transformers/all-mpnet-base-v2"
            }
        }

        # Perform a deep merge
        config = always_merger.merge(config, updates)

        # Load the SentenceTransformer model
        logger.info("Loading the SentenceTransformer model...")
        model = get_sentence_transformer(config["model"]["name"])

        # Execute the query using the configuration
        results = execute_bm25_embedding_keyword_query(config, model)
        logger.success("Query executed successfully!")

    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == "__main__":
    main()