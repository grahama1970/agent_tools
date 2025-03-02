# curl -X POST http://192.168.86.49:30002/v1/unload_lora_adapter \
# -H "Content-Type: application/json" \
# -d '{
#     "lora_name": "grahamaco/Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter"
#     }'

VLLM_ALLOW_RUNTIME_LORA_UPDATING=True  \
curl -vv -X POST http://192.168.86.49:30002/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
        "lora_name": "touch-rugby-rules_adapter"
      }' | jq