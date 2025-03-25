#!/bin/bash
# extraction_validation_dashboard.sh - Script to create extraction validation dashboard

# Configuration
ROOT_DIR=$(pwd)
OUTPUT_DIR="validation_reports"
DOCKER_PORT=8787
DOCKER_CONTAINER="dualipa-validation-dashboard"

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
echo "DuaLipa Extraction Validation Dashboard"
echo "=================================="

# Process all JSON files in test results
echo "🔍 Scanning for extraction JSON files..."

# Find all JSON files in test_results directories
JSON_FILES=$(find . -type f -name "*blocks.json" | sort)

if [ -z "$JSON_FILES" ]; then
  echo "❌ No extraction JSON files found"
  exit 1
fi

echo "✅ Found $(echo "$JSON_FILES" | wc -l) extraction JSON files"

# Create validation reports
echo "🛠️ Generating validation reports..."
for JSON_FILE in $JSON_FILES; do
  echo "  Processing $JSON_FILE"
  python validate_extraction_format.py "$JSON_FILE" --output-dir "$OUTPUT_DIR" --dashboard
done

# Create dashboard index if it doesn't exist
if [ ! -f "$OUTPUT_DIR/extraction_dashboard.html" ]; then
  echo "⚠️ Dashboard index not created. Creating simple dashboard..."
  
  # Create a simple dashboard HTML
  cat > "$OUTPUT_DIR/extraction_dashboard.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DuaLipa Extraction Validation Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #4a6fa5;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>DuaLipa Extraction Validation Dashboard</h1>
        <p>Last updated: $(date)</p>
        
        <h2>Validation Reports</h2>
        <ul>
$(find "$OUTPUT_DIR" -name "*.validation.html" | sort | while read report; do
  echo "            <li><a href=\"$(basename "$report")\">$(basename "$report")</a></li>"
done)
        </ul>
    </div>
</body>
</html>
EOF
fi

# Start Docker server
echo -e "\n🐳 Starting Docker server for dashboard..."

# Stop existing container if it exists
if docker ps -a | grep -q $DOCKER_CONTAINER; then
  echo "Stopping existing container: $DOCKER_CONTAINER"
  docker stop $DOCKER_CONTAINER >/dev/null 2>&1
  docker rm $DOCKER_CONTAINER >/dev/null 2>&1
fi

# Run nginx container
ABSOLUTE_PATH=$(realpath "$OUTPUT_DIR")
echo "Starting Docker container to serve $ABSOLUTE_PATH on port $DOCKER_PORT"
docker run --name $DOCKER_CONTAINER -d \
  -p $DOCKER_PORT:80 \
  -v "$ABSOLUTE_PATH:/usr/share/nginx/html:ro" \
  -e NGINX_PORT=80 \
  nginx:alpine

if [ $? -eq 0 ]; then
  # Get local IP for URLs
  LOCAL_IP=$(hostname -I | awk '{print $1}')
  
  echo -e "\n=========================================================="
  echo "✅ DASHBOARD SERVER STARTED SUCCESSFULLY"
  echo "📁 Serving directory: $ABSOLUTE_PATH"
  
  echo -e "\n🌐 ACCESS THE DASHBOARD USING THESE URLS:"
  echo "  • Local access:  http://localhost:$DOCKER_PORT/extraction_dashboard.html"
  echo "  • Network:       http://$LOCAL_IP:$DOCKER_PORT/extraction_dashboard.html"

  # Check if running in WSL
  if grep -q Microsoft /proc/version; then
    echo -e "\n🪟 WINDOWS WSL ACCESS:"
    echo "  You can access the reports from your Windows browser using either URL above."
    echo "  Docker Desktop automatically forwards ports to your Windows host."
  fi
  
  echo -e "\n📋 COMMANDS:"
  echo "  • View logs:    docker logs $DOCKER_CONTAINER"
  echo "  • Stop server:  docker stop $DOCKER_CONTAINER"
  echo "  • Update dashboard: $0"
  echo "=========================================================="
  
  # Try to open in browser
  if [ -n "$DISPLAY" ]; then
    echo "Opening dashboard in browser..."
    xdg-open "http://localhost:$DOCKER_PORT/extraction_dashboard.html" >/dev/null 2>&1 || true
  elif [ "$(uname)" == "Darwin" ]; then
    echo "Opening dashboard in browser..."
    open "http://localhost:$DOCKER_PORT/extraction_dashboard.html" >/dev/null 2>&1 || true
  fi
else
  echo "❌ Failed to start Docker container"
  exit 1
fi