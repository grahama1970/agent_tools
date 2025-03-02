from huggingface_hub import snapshot_download
import os
os.environ["HF_TOKEN"] = 
os.environ["HF_HOME"] = "/root/.cache/huggingface/hub"
os.environ["HF_HUB_CACHE_DIR"] = "/root/.cache/huggingface/hub"

snapshot_download(
    repo_id="grahamaco/Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter",
    cache_dir="/root/.cache/huggingface/adapters",
    local_dir="/root/.cache/huggingface/adapters/models--grahamaco--Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter"
)