#!/bin/bash

# Variables
CONTAINER_NAME="gallant_thompson"  # Replace with your container name
DUMP_URL="https://github.com/arangodb/example-datasets/releases/download/imdb-graph-dump-rev2/imdb_graph_dump_rev2.zip"
DESTINATION_PATH="/tmp/dump"
DATABASE_NAME="IMDB"
SERVER_ENDPOINT="tcp://127.0.0.1:8529"
USERNAME="root"  # Replace with your database username
PASSWORD="openSesame"  # Replace with your database password

# Step 1: Log into the container and download/extract the dump files
echo "Logging into the container and downloading/extracting dump files..."
docker exec -it "$CONTAINER_NAME" /bin/sh -c "
    set -e;  # Exit on error
    mkdir -p $DESTINATION_PATH;
    cd $DESTINATION_PATH;
    wget -q $DUMP_URL -O dump.zip;
    unzip -q dump.zip -d .;
    rm dump.zip;
    echo 'Dump files downloaded and extracted to $DESTINATION_PATH.';
"

# Step 2: Create the database if it doesn't exist
echo "Creating database '$DATABASE_NAME'..."
docker exec -it "$CONTAINER_NAME" arangosh --server.endpoint "$SERVER_ENDPOINT" --server.username "$USERNAME" --server.password "$PASSWORD" --server.database "_system" --javascript.execute-string "
    if (!db._databases().includes('$DATABASE_NAME')) {
        db._createDatabase('$DATABASE_NAME');
    }
"

# Step 3: Create collections from structure files
echo "Creating collections from structure files..."
docker exec -it "$CONTAINER_NAME" /bin/sh -c "
    set -e;  # Exit on error
    for structure_file in $DESTINATION_PATH/*.structure.json; do
        echo 'Processing structure file: \$structure_file';
        collection_name=\$(jq -r '.parameters.name' \$structure_file);
        echo 'Creating collection: \$collection_name';
        arangosh --server.endpoint '$SERVER_ENDPOINT' --server.username '$USERNAME' --server.password '$PASSWORD' --server.database '$DATABASE_NAME' --javascript.execute-string \"
            const structure = JSON.parse(cat('\$structure_file'));
            if (!db._collection('\$collection_name')) {
                db._create('\$collection_name', structure.parameters);
                console.log('Created collection:', '\$collection_name');
            } else {
                console.log('Collection already exists:', '\$collection_name');
            }
        \";
    done;
"

# Step 4: Import data into collections
echo "Importing data into collections..."
docker exec -it "$CONTAINER_NAME" /bin/sh -c "
    set -e;  # Exit on error
    for data_file in $DESTINATION_PATH/*.data.json; do
        collection_name=\$(basename \$data_file | cut -d'_' -f1,2);  # Extract collection name from file name
        echo 'Importing data into collection: \$collection_name from file: \$data_file';
        arangoimport \
            --server.endpoint '$SERVER_ENDPOINT' \
            --server.username '$USERNAME' \
            --server.password '$PASSWORD' \
            --server.database '$DATABASE_NAME' \
            --collection \$collection_name \
            --file \$data_file \
            --type jsonl;
    done;
"

echo "Database restoration completed."