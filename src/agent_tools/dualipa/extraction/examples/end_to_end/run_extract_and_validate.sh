#!/bin/bash
# run_extract_and_validate.sh - Run extraction and validation in one go

# Configuration
OUTPUT_DIR="test_results/$(date +%Y-%m-%d_%H%M%S)"
VALIDATION_DIR="$OUTPUT_DIR/validation"
DOCKER_PORT=8787

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      DOCKER_PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=================================="
echo "DuaLipa Extraction and Validation"
echo "=================================="

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VALIDATION_DIR"

# Step 1: Run extraction tests
echo "🔍 Running extraction tests..."
python run_transparent_tests.py --output-dir "$OUTPUT_DIR"

# Step 2: Validate extraction results
echo -e "\n🔍 Validating extraction results..."

# Find all JSON files in the output directory
JSON_FILES=$(find "$OUTPUT_DIR" -name "*blocks.json" | sort)

if [ -z "$JSON_FILES" ]; then
  echo "⚠️ No extraction JSON files found"
else
  echo "✅ Found $(echo "$JSON_FILES" | wc -l) extraction JSON files to validate"
  
  # Create validation reports
  for JSON_FILE in $JSON_FILES; do
    echo "  Processing $JSON_FILE"
    python validate_extraction_format.py "$JSON_FILE" --output-dir "$VALIDATION_DIR" --dashboard
  done
fi

# Step 3: Start Docker server
echo -e "\n🐳 Starting Docker server for dashboard..."

# Generate the Docker container name based on output dir to avoid conflicts
CONTAINER_NAME="dualipa-$(basename "$OUTPUT_DIR" | tr '.' '-' | tr '_' '-')"

# Stop existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
  echo "Stopping existing container: $CONTAINER_NAME"
  docker stop $CONTAINER_NAME >/dev/null 2>&1
  docker rm $CONTAINER_NAME >/dev/null 2>&1
fi

# Run nginx container
echo "Starting Docker container to serve $OUTPUT_DIR on port $DOCKER_PORT"
docker run --name $CONTAINER_NAME -d \
  -p $DOCKER_PORT:80 \
  -v "$OUTPUT_DIR:/usr/share/nginx/html:ro" \
  -e NGINX_PORT=80 \
  nginx:alpine

if [ $? -eq 0 ]; then
  # Get local IP for URLs
  LOCAL_IP=$(hostname -I | awk '{print $1}')
  
  echo -e "\n=========================================================="
  echo "✅ DASHBOARD SERVER STARTED SUCCESSFULLY"
  echo "📁 Serving directory: $OUTPUT_DIR"
  
  echo -e "\n🌐 ACCESS THE DASHBOARDS USING THESE URLS:"
  echo "  • Extraction Results: http://localhost:$DOCKER_PORT/summary.html"
  echo "  • Validation Results: http://localhost:$DOCKER_PORT/validation/extraction_dashboard.html"
  echo ""
  echo "  • Network - Extraction: http://$LOCAL_IP:$DOCKER_PORT/summary.html"
  echo "  • Network - Validation: http://$LOCAL_IP:$DOCKER_PORT/validation/extraction_dashboard.html"

  # Check if running in WSL
  if grep -q Microsoft /proc/version; then
    echo -e "\n🪟 WINDOWS WSL ACCESS:"
    echo "  You can access the reports from your Windows browser using either URL above."
    echo "  Docker Desktop automatically forwards ports to your Windows host."
  fi
  
  echo -e "\n📋 COMMANDS:"
  echo "  • View logs:    docker logs $CONTAINER_NAME"
  echo "  • Stop server:  docker stop $CONTAINER_NAME"
  echo "  • Re-run:       $0"
  echo "=========================================================="
  
  # Try to open in browser
  if [ -n "$DISPLAY" ]; then
    echo "Opening dashboard in browser..."
    xdg-open "http://localhost:$DOCKER_PORT/summary.html" >/dev/null 2>&1 || true
  elif [ "$(uname)" == "Darwin" ]; then
    echo "Opening dashboard in browser..."
    open "http://localhost:$DOCKER_PORT/summary.html" >/dev/null 2>&1 || true
  fi
else
  echo "❌ Failed to start Docker container"
  exit 1
fi