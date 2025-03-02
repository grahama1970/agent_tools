VLLM_ALLOW_RUNTIME_LORA_UPDATING=True  \
curl -vvv http://192.168.86.49:30002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "models--grahamaco--Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapterr",
        "prompt": "What is the capital of France?",
        "max_tokens": 7,
        "verbose": true
    }'