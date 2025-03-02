curl -vvv http://192.168.86.49:30002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "touch-rugby-rules-adapter",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 7
    }'