#!/bin/bash

# Configuration
DOCKERHUB_USERNAME="grahamaco"
IMAGE_NAME="${DOCKERHUB_USERNAME}/qwen2-72b-inference"
TAG="latest"
DIR="qwen2-runpod"
ENV_FILE="../.env"

# Load environment variables from .env
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE..."
    source "$ENV_FILE"
else
    echo "Error: $ENV_FILE not found. Please provide HF_TOKEN and RUNPOD_API_KEY."
    exit 1
fi

# Check required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set in $ENV_FILE."
    exit 1
fi
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY is not set in $ENV_FILE."
    exit 1
fi

# Step 1: Create directory and copy files
echo "Setting up directory and files..."
mkdir -p "$DIR"
cd "$DIR" || { echo "Failed to create/enter directory"; exit 1; }
cp ../inference.py .
cp ../deploy_to_runpod.py .

# Step 2: Build the Docker image
echo "Building Docker image: ${IMAGE_NAME}:${TAG}..."
docker build -t "${IMAGE_NAME}:${TAG}" \
    --build-arg HF_TOKEN="$HF_TOKEN" \
    --build-arg HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    -f ../Dockerfile . || {
    echo "Build failed. Check logs for details (e.g., HF_TOKEN invalid or network issues)."
    exit 1
}

# Step 3: Log in to Docker Hub
echo "Logging in to Docker Hub..."
docker login || {
    echo "Docker login failed. Check credentials."
    exit 1
}

# Step 4: Push the image to Docker Hub
echo "Pushing image to Docker Hub..."
docker push "${IMAGE_NAME}:${TAG}" || {
    echo "Push failed. Check Docker Hub permissions or network."
    exit 1
}

# Step 5: Deploy to RunPod
echo "Deploying to RunPod..."
python deploy_to_runpod.py

if [ $? -ne 0 ]; then
    echo "Deployment failed. Check the error message above."
    exit 1
fi

echo "Deployment complete! Pod ID and IP saved to pod_id.txt and pod_ip.txt."
