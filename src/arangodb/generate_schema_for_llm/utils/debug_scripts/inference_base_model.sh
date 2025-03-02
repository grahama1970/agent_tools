VLLM_ALLOW_RUNTIME_LORA_UPDATING=True  \
curl http://192.168.86.49:30002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        "prompt": "What is the capital of France?",
        "max_tokens": 7,
        "verbose": true
    }' | jq