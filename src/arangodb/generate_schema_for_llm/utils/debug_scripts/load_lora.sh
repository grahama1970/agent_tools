VLLM_ALLOW_RUNTIME_LORA_UPDATING=True  \
curl -X POST http://192.168.86.49:30002/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
        "lora_name": "touch-rugby-rules_adapter",
        "lora_path": ""/root/.cache/huggingface/hub/models--grahamaco--Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter/snapshots/31cef1e9e3b7222eadb97e6a813e03223369b29a/"
      }'