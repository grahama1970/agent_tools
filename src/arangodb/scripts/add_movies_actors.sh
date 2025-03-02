#!/bin/bash

# Variables
CONTAINER_NAME="gallant_thompson"
DATABASE_NAME="test"
ARANGO_URL="http://localhost:8529/_db/${DATABASE_NAME}"
USERNAME="root"
PASSWORD="openSesame"
JAVASCRIPT_FILE="/Users/robert/Desktop/dev/projects/experiments/aql_rag/scripts/populate_movie_db.js"
JAVASCRIPT_COMMAND="db._collections();"

# Check if the Docker container is running
if ! docker ps --filter "name=${CONTAINER_NAME}" --format "{{.Names}}" | grep -q "${CONTAINER_NAME}"; then
    echo "Error: Docker container '${CONTAINER_NAME}' is not running."
    exit 1
fi

# Copy the JavaScript file into the Docker container
docker cp "${JAVASCRIPT_FILE}" "${CONTAINER_NAME}:/tmp/populate_movie_db.js"

# Log into the Docker container and execute the JavaScript file
docker exec -it "${CONTAINER_NAME}" arangosh \
    --server.endpoint tcp://127.0.0.1:8529 \
    --server.database "${DATABASE_NAME}" \
    --server.username "${USERNAME}" \
    --server.password "${PASSWORD}" \
    --javascript.execute "/tmp/populate_movie_db.js"

# Exit message
echo "JavaScript file executed successfully on ${ARANGO_URL}"
