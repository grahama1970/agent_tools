from arango import ArangoClient

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("pdf_db", username="root", password="password")

def insert_into_arango(collection_name, document):
    if not db.has_collection(collection_name):
        db.create_collection(collection_name)
    db.collection(collection_name).insert(document)
