#!/bin/bash
# docker-serve.sh - Script to serve HTML test reports via Docker Compose

# Configuration
PORT=8765
HOST_DIR=""
SERVICE_NAME="test-report-server"
CONTAINER_NAME="dualipa-test-reports"
COMPOSE_FILE="docker-compose.yml"

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -d|--directory)
      HOST_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if directory is provided
if [ -z "$HOST_DIR" ]; then
  echo "Error: Directory is required"
  echo "Usage: $0 --directory PATH [--port PORT]"
  exit 1
fi

# Convert to absolute path if needed
if [[ ! "$HOST_DIR" = /* ]]; then
  HOST_DIR="$(pwd)/$HOST_DIR"
fi

# Check if directory exists
if [ ! -d "$HOST_DIR" ]; then
  echo "Error: Directory $HOST_DIR does not exist"
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
  echo "Docker Compose is not installed. Falling back to regular Docker."
  
  # Stop existing container if it exists
  if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Stopping existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME >/dev/null 2>&1
    docker rm $CONTAINER_NAME >/dev/null 2>&1
  fi

  # Run nginx container
  echo "Starting nginx container to serve $HOST_DIR on port $PORT"
  docker run --name $CONTAINER_NAME -d \
    -p $PORT:80 \
    -v "$HOST_DIR:/usr/share/nginx/html:ro" \
    -e NGINX_PORT=80 \
    nginx:alpine
else
  # Set environment variables for docker-compose
  export PORT=$PORT
  export RESULTS_DIR=$HOST_DIR

  # Stop any existing containers from this compose file
  echo "Stopping any existing containers..."
  docker-compose -f $COMPOSE_FILE down >/dev/null 2>&1

  # Start the container
  echo "Starting Docker Compose service to serve $HOST_DIR on port $PORT"
  docker-compose -f $COMPOSE_FILE up -d
fi

# Get container status
if [ $? -eq 0 ]; then
  echo -e "\n=========================================================="
  echo "✅ SERVER STARTED SUCCESSFULLY"
  echo "📁 Serving directory: $HOST_DIR"
  
  # Get local IP
  LOCAL_IP=$(hostname -I | awk '{print $1}')
  
  echo -e "\n🌐 ACCESS THE TEST REPORTS USING THESE URLS:"
  echo "  • Local access:  http://localhost:$PORT/"
  echo "  • Network:       http://$LOCAL_IP:$PORT/"

  # Check if running in WSL
  if grep -q Microsoft /proc/version; then
    echo -e "\n🪟 WINDOWS WSL ACCESS:"
    echo "  You can access the reports from your Windows browser using either URL above."
    echo "  Docker Desktop automatically forwards ports to your Windows host."
  fi
  
  echo -e "\n📋 COMMANDS:"
  if command -v docker-compose &> /dev/null; then
    echo "  • View logs:    docker-compose -f $COMPOSE_FILE logs"
    echo "  • Stop server:  docker-compose -f $COMPOSE_FILE down"
  else
    echo "  • View logs:    docker logs $CONTAINER_NAME"
    echo "  • Stop server:  docker stop $CONTAINER_NAME"
  fi
  echo "=========================================================="
else
  echo "❌ Failed to start container"
fi