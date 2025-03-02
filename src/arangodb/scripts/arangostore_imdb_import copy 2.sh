#!/bin/bash

# Variables
CONTAINER_NAME="gallant_thompson"  # Replace with your actual container name
DESTINATION_PATH="/tmp/dump"  # Path to the extracted dump directory inside the container
DATABASE_NAME="IMDB"
SERVER_ENDPOINT="tcp://127.0.0.1:8529"
USERNAME="root"  # Replace with your database username
PASSWORD="openSesame"  # Replace with your database password

# Check if the container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container $CONTAINER_NAME is running."
else
    echo "Container $CONTAINER_NAME is not running."
    exit 1
fi

# Create the database if it doesn't exist
echo "Creating database '$DATABASE_NAME'..."
docker exec -it "$CONTAINER_NAME" arangosh --server.endpoint "$SERVER_ENDPOINT" --server.username "$USERNAME" --server.password "$PASSWORD" --server.database "_system" --javascript.execute-string "
    if (!db._databases().includes('$DATABASE_NAME')) {
        db._createDatabase('$DATABASE_NAME');
    }
"

# Function to create collections from structure files
create_collections() {
    echo "Creating collections from structure files..."
    structure_files=$(docker exec -it "$CONTAINER_NAME" /bin/sh -c "find $DESTINATION_PATH -type f -name '*.structure.json'")

    for file in $structure_files; do
        echo "Processing structure file: $file"
        docker exec -it "$CONTAINER_NAME" arangosh --server.endpoint "$SERVER_ENDPOINT" --server.username "$USERNAME" --server.password "$PASSWORD" --server.database "$DATABASE_NAME" --javascript.execute-string "
            const structure = JSON.parse(cat('$file'));
            const collectionName = structure.parameters.name;

            if (!db._collection(collectionName)) {
                db._create(collectionName, structure.parameters);
                console.log('Created collection:', collectionName);
            } else {
                console.log('Collection already exists:', collectionName);
            }
        "
    done
}

# Function to import data into collections
import_data() {
    echo "Importing data into collections..."
    data_files=$(docker exec -it "$CONTAINER_NAME" /bin/sh -c "find $DESTINATION_PATH -type f -name '*.data.json'")

    for file in $data_files; do
        collection_name=$(basename "$file" | cut -d'_' -f2)  # Extract collection name from file name
        echo "Importing data into collection: $collection_name from file: $file"
        docker exec -it "$CONTAINER_NAME" arangoimport \
            --server.endpoint "$SERVER_ENDPOINT" \
            --server.username "$USERNAME" \
            --server.password "$PASSWORD" \
            --server.database "$DATABASE_NAME" \
            --collection "$collection_name" \
            --file "$file" \
            --type jsonl
    done
}

# Execute functions
create_collections
import_data

echo "Database restoration completed."