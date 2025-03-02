from arango import ArangoClient
import pyperclip

def get_first_embedding():
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("verifaix", username="root", password="openSesame")
    
    # python-arango is great for collection access
    glossary = db.collection("glossary")
    print(f"Total documents: {glossary.count()}")

    # But for complex queries, raw AQL is clearer
    aql = """
        FOR doc IN glossary
        FILTER HAS(doc, 'embedding') 
           AND IS_ARRAY(doc.embedding)  # Additional type safety
        LIMIT 1
        RETURN doc
    """
    
    # python-arango's AQL execution is still useful
    cursor = db.aql.execute(aql, ttl=10)
    try:
        doc = next(cursor)
        pyperclip.copy(str(doc["embedding"]))
        return f"Copied embedding from {doc['_key']}"
    except StopIteration:
        raise ValueError("No valid embeddings found")

if __name__ == "__main__":
    print(get_first_embedding())



"""
db.glossary.ensureIndex(
    {
            name: "vector_cosine",
            type: "vector",
            fields: ["embedding"],
            params: { metric: "cosine", dimension: 768, nLists: 100 }
    } 
)
"""