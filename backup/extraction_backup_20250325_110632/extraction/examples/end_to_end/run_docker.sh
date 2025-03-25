#!/bin/bash
# Script to build and run the Docker validation server

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default port
PORT="${1:-8000}"

echo "Building Docker image..."
docker build -t dualipa-validation-server .

echo "Running Docker container on port $PORT..."
docker run --name dualipa-validation-container -d -p $PORT:8000 dualipa-validation-server

# Check if container started successfully
if [ $? -eq 0 ]; then
  # Get container IP
  CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dualipa-validation-container)
  
  echo "✅ Validation Server started successfully"
  echo ""
  echo "📊 Access the Validation Dashboard:"
  echo "  • Local access:  http://localhost:$PORT/"
  echo "  • Container IP:  http://$CONTAINER_IP:8000/"
  echo ""
  echo "📋 Available commands:"
  echo "  • View logs:     docker logs -f dualipa-validation-container"
  echo "  • Stop server:   docker stop dualipa-validation-container"
  echo "  • Remove:        docker rm dualipa-validation-container"
  echo ""
  echo "The server is running validation tests and generating reports."
  echo "This may take a few moments. Check the logs for progress."
  echo ""
  echo "To view the logs, run: docker logs -f dualipa-validation-container"
else
  echo "❌ Failed to start container"
fi