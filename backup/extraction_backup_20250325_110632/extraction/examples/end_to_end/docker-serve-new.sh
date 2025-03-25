#!/bin/bash
# New docker-serve script that uses docker-compose to run validation and serve results

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default port
PORT="${1:-8000}"

echo "Starting validation server on port $PORT..."

# Export port for docker-compose
export PORT=$PORT

# Stop any existing containers
docker-compose down >/dev/null 2>&1

# Start the container
docker-compose up -d

# Check if container started successfully
if [ $? -eq 0 ]; then
  # Get local IP
  LOCAL_IP=$(hostname -I | awk '{print $1}')
  
  echo "✅ Validation Server started successfully"
  echo ""
  echo "📊 Access the Validation Dashboard:"
  echo "  • Local access:  http://localhost:$PORT/"
  echo "  • Network:       http://$LOCAL_IP:$PORT/"
  echo ""
  echo "📋 Available commands:"
  echo "  • View logs:     docker-compose logs -f"
  echo "  • Stop server:   docker-compose down"
  echo ""
  echo "The server is running validation tests and generating reports."
  echo "This may take a few moments. Check the logs for progress."
  echo ""
  echo "To view the logs, run: docker-compose logs -f"
else
  echo "❌ Failed to start container"
fi