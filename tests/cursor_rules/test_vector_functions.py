from arango import ArangoClient

def test_vector_functions():
    # Connect to ArangoDB
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('_system', username='root', password='openSesame')
    
    print(f"ArangoDB version: {db.version()}")
    
    # Test different vector functions
    functions_to_test = [
        "APPROX_NEAR_COSINE",
        "VECTOR_SIMILARITY",
        "COSINE",
        "COSINE_SIMILARITY"
    ]
    
    for func_name in functions_to_test:
        try:
            # Try to execute a simple query with the function
            query = f"RETURN {func_name}([0.1, 0.2], [0.1, 0.2])"
            result = next(db.aql.execute(query), None)
            print(f"✅ {func_name} is available: {result}")
        except Exception as e:
            print(f"❌ {func_name} is not available: {e}")
    
    # Test COSINE_SIMILARITY with a more complex query similar to the gist
    try:
        test_query = """
        LET vec1 = [0.1, 0.2, 0.3]
        LET vec2 = [0.1, 0.2, 0.3]
        RETURN COSINE_SIMILARITY(vec1, vec2)
        """
        result = next(db.aql.execute(test_query), None)
        print(f"\nCOSINE_SIMILARITY test result: {result}")
    except Exception as e:
        print(f"\nCOSINE_SIMILARITY test failed: {e}")
    
    # Check vector index
    try:
        db_name = 'cursor_rules_test'
        test_db = client.db(db_name, username='root', password='openSesame')
        print(f"\nChecking vector indexes in {db_name}:")
        
        for idx in test_db.collection('rules').indexes():
            if idx.get('type') == 'vector':
                print(f"Found vector index: {idx}")
                
                # Check index parameters
                params = idx.get('params', {})
                if params:
                    print(f"Index parameters: {params}")
                else:
                    print("No parameters found in index")
                    
        # Check a sample document
        print("\nChecking sample document:")
        sample_doc = next(test_db.collection('rules').all(), None)
        if sample_doc:
            has_embedding = "embedding" in sample_doc
            embedding_length = len(sample_doc.get("embedding", [])) if has_embedding else 0
            print(f"Has embedding field: {has_embedding}")
            print(f"Embedding length: {embedding_length}")
    except Exception as e:
        print(f"Error checking vector index: {e}")

if __name__ == "__main__":
    test_vector_functions() 