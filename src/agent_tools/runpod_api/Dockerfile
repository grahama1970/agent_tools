FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /workspace

# Install system dependencies including Redis
RUN apt-get update && apt-get install -y \
    git \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv

# Install Python dependencies with uv
RUN uv pip install --system --no-cache-dir \
    torch \
    transformers \
    huggingface_hub \
    bitsandbytes>=0.43.0 \
    accelerate \
    "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git" \
    runpod \
    fastapi \
    uvicorn \
    litellm \
    redis \
    tqdm \
    loguru \
    aiohttp \
    "sglang[all] @ git+https://github.com/sgl-project/sglang.git"

# Pass build arguments
ARG HF_TOKEN
ARG HF_HUB_ENABLE_HF_TRANSFER=True
ENV HF_TOKEN=${HF_TOKEN}
ENV HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}

# Pre-download the model
RUN python -c "from unsloth import FastLanguageModel; model, tokenizer = FastLanguageModel.from_pretrained('unsloth/Qwen2-72B-bnb-4bit', load_in_4bit=True, token='${HF_TOKEN}'); model.save_pretrained('/workspace/model'); tokenizer.save_pretrained('/workspace/model')" || { echo "Model download failed"; exit 1; }

# Copy inference script
COPY inference.py /workspace/inference.py

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Expose ports: 8000 for FastAPI, 30000 for SGLang
EXPOSE 8000
EXPOSE 30000

# Start Redis, SGLang, and FastAPI
CMD redis-server --daemonize yes && \
    python -m sglang.launch_server --model-path /workspace/model --port 30000 --host 0.0.0.0 & \
    uvicorn inference:app --host 0.0.0.0 --port 8000
