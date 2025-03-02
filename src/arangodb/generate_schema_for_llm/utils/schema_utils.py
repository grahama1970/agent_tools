def generate_sparse_schema(schema: dict) -> dict:
    """
    Generates a sparse version of the schema by extracting only essential fields and relationships.

    Args:
        schema: The complete schema (as a dictionary).

    Returns:
        dict: The sparse schema.
    """
    sparse_schema = {
        "collections": [],
        "relationships": schema.get("relationships", [])
    }

    for collection in schema["collections"]:
        sparse_collection = {
            "name": collection["name"],
            "type": collection["type"],
            "fields": [
                {"name": field["name"], "type": field["type"]}
                for field in collection["fields"]
            ]
        }
        sparse_schema["collections"].append(sparse_collection)

    return sparse_schema


if __name__ == "__main__":
    import pyperclip
    print('loaded')