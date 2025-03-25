# Qwen2-72B Inference on RunPod 🚀

This project sets up a high-performance inference server for the `unsloth/Qwen2-72B-bnb-4bit` model on RunPod using SGLang, LiteLLM, Redis caching, and a FastAPI interface. It supports batch inference with progress tracking, optimized for an NVIDIA A40 GPU (48GB VRAM).

---

## 🌟 Overview

- **Model**: Qwen2-72B (4-bit quantized, ~40-44GB VRAM usage)
- **Server**: SGLang for efficient model serving
- **Inference**: LiteLLM with async `acompletion` for batch calls
- **Caching**: Redis for persistent caching (2-day TTL)
- **API**: FastAPI for single and batch inference endpoints
- **Dependencies**: Managed with `uv` for speed
- **Deployment**: RunPod Secure Cloud with A40 GPU

---

## 🛠️ Prerequisites

- **Docker**: Installed locally for building the image
- **Docker Hub Account**: For pushing the image (username: `grahamaco`)
- **RunPod Account**: With API key (get from [RunPod Console](https://runpod.io/console/keys))
- **Hugging Face Token**: For downloading the model (get from [Hugging Face](https://huggingface.co/settings/tokens))
- **Disk Space**: ~50GB free for model download during build
- **Python**: For deployment script (3.10+ recommended)

---

## 📋 Setup Instructions

### 1. 📂 Prepare the Environment

Clone or navigate to the project directory:

```bash
cd src/agent_tools/runpod_api/
```

Ensure the following files are present:
- `build_push_qwen2.sh`
- `inference.py`
- `deploy_to_runpod.py`
- `Dockerfile`
- `.env` (see below)

### 2. 🔑 Configure Environment Variables

Create or edit `.env` in `src/agent_tools/runpod_api/` with your credentials:

```plaintext
HF_TOKEN=your_hugging_face_token_here  # Your Hugging Face token
RUNPOD_API_KEY=your_runpod_api_key_here  # Your RunPod API key
HF_HUB_ENABLE_HF_TRANSFER=True  # Enable faster downloads (optional)
```

### 3. 🏗️ Build and Deploy

Run the setup script to build the Docker image, push it to Docker Hub, and deploy to RunPod:

```bash
chmod +x build_push_qwen2.sh
./build_push_qwen2.sh
```

- **Duration**: ~15-25 minutes (depends on internet speed for ~36GB model download)
- **Output**: 
  - Docker image: `grahamaco/qwen2-72b-inference:latest`
  - Pod ID and public IP (e.g., `123.45.67.89:8000`) saved to `qwen2-runpod/pod_id.txt` and `qwen2-runpod/pod_ip.txt`

The script:
1. Builds the image with `uv`, SGLang, Redis, and dependencies
2. Downloads the Qwen2-72B model during build
3. Pushes to Docker Hub
4. Deploys to RunPod with an A40 GPU

---

## 🔍 Files

- **`build_push_qwen2.sh`**:
  - Builds and deploys the container
  - Uses `uv` for dependency management
  - Starts Redis, SGLang (port 30000), and FastAPI (port 8000)

- **`inference.py`**:
  - FastAPI server for inference
  - Uses LiteLLM with SGLang backend (`localhost:30000/v1`)
  - Supports batch inference with `tqdm` progress
  - Initializes Redis caching

- **`deploy_to_runpod.py`**:
  - Handles RunPod deployment
  - Creates pod with specified configuration
  - Saves pod details to files

- **`initialize_litellm_cache.py`**:
  - Sets up LiteLLM cache with Redis
  - Includes fallback to in-memory cache

- **`qwen2_cli.py`**:
  - CLI tool for managing the deployment
  - Includes commands for building, deploying, and taking down pods

- **`.env`**:
  - Stores API tokens and configuration
  - Not tracked in git for security

- **`Dockerfile`**:
  - Base: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
  - Installs Redis, SGLang, and dependencies with `uv`
  - Pre-downloads the model
  - Includes health check

- **`docker-compose.yml`**:
  - Defines the service for local development and testing

---

## 🚀 Inference Instructions

Once deployed, the pod exposes two endpoints via FastAPI on port 8000. Use the public IP from the deployment output (e.g., `http://123.45.67.89:8000`).

### 1. 🎯 Single Inference

Test a single prompt:

```bash
curl "http://:8000/infer?prompt=What%20is%202%2B2%3F"
```

**Response**:
```json
{"response": "4"}
```

### 2. 📦 Batch Inference

Send multiple prompts with progress tracking (visible in pod logs):

```bash
curl -X POST "http://:8000/batch_infer" \
-H "Content-Type: application/json" \
-d '{"prompts": ["What is 2+2?", "Tell me a joke"]}'
```

**Response**:
```json
{
  "responses": [
    "4",
    "Why don't skeletons fight? They don't have guts."
  ]
}
```