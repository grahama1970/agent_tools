#!/bin/bash
# Run validation tests and start the server locally (without Docker)

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directories
mkdir -p ./test_results
mkdir -p ./test_results_dashboard

echo "Running validation tests and generating reports..."

# Run the validation script
bash ./docker-validation.sh

echo "Starting server on port 8000..."
echo "Open http://localhost:8000/ in your browser"

# Start the server
python ./simple_server.py