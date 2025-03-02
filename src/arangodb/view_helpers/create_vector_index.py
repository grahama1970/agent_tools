from arango import ArangoClient
from loguru import logger

from aql_rag.embedding.embedding_utils import create_embedding_sync


def validate_embeddings(db, collection_name, dimension):
    """Validate embeddings with boolean AQL result."""
    try:
        # Collection name interpolation (must be sanitized)
        query = f"""
            RETURN COUNT(
                FOR doc IN {collection_name}
                FILTER 
                    NOT HAS(doc, 'embedding') OR
                    NOT IS_LIST(doc.embedding) OR
                    LENGTH(doc.embedding) != @dimension OR
                    LENGTH(
                        FOR e IN doc.embedding 
                        FILTER NOT IS_NUMBER(e) 
                        LIMIT 1 RETURN true
                    ) > 0
                RETURN 1
            ) == 0
        """
        result = db.aql.execute(query, bind_vars={"dimension": dimension})

        logger.info(f"{'✅' if result else '❌'} Validation result: {result}")
        return next(result, False)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def create_vector_index(db, collection_name, dimension=768):
    """Create a vector index on the 'embedding' field using the native vector type."""
    try:
        collection = db.collection(collection_name)
        existing_indexes = collection.indexes()

        # Check if a vector index already exists
        index_exists = any(
            index["type"] == "vector" and "embedding" in index.get("fields", [])
            for index in existing_indexes
        )
        collection_count = collection.count()
        nlist = min(collection_count, 100) 

        if not index_exists:
            if not validate_embeddings(db, collection_name, dimension):
                logger.error("❌ Aborting index creation due to invalid embeddings")
                return

            index_definition = {
                "name": "vector_cosine",
                "type": "vector",
                "fields": ["embedding"],
                "params": {"metric": "cosine", "dimension": dimension, "nLists": nlist},
            }
            collection.add_index(index_definition)

            logger.info(
                f"✅ Vector index successfully created on 'embedding' field in collection '{collection_name}'."
            )
        else:
            logger.info(
                f"⚠️ Vector index already exists on 'embedding' field in collection '{collection_name}'."
            )

    except Exception as e:
        logger.error(
            f"❌ Failed to create vector index for collection '{collection_name}': {e}"
        )

def query_embeddings_test(db, collection_name, query_text):
    """Query embeddings using the vector index."""

    # use embbeding utils to get the query vector
    query_vector = create_embedding_sync(query_text)

    try:
        aql_query = f"""
        FOR doc IN {collection_name}
        LET score = APPROX_NEAR_COSINE(doc.embedding, query_vector)
        SORT score DESC
        LIMIT 2
        RETURN {{doc, score}}
        """ 
        result = db.aql.execute(aql_query, bind_vars={"query_vector": query_text})
        return result

    except Exception as e:
        logger.error(f"❌ Failed to query embeddings: {e}")
        return None


if __name__ == "__main__":
    # Initialize the ArangoDB client
    client = ArangoClient()
    db = client.db("verifaix", username="root", password="openSesame")
    
    # create_vector_index(db, "microsoft_issues")
    test_collections = [
        "microsoft_issues","microsoft_products", 
        "glossary", "microsoft_glossary","microsoft_support"
    ]
    validated = create_vector_index(db, "microsoft_support", 768)
    print(validated)
