import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import litellm
from loguru import logger
from asyncio import as_completed
from tqdm.asyncio import tqdm
import redis
import os
import time
from .initialize_litellm_cache import initialize_litellm_cache

app = FastAPI()

# Complete BatchRequest class definition
class BatchRequest(BaseModel):
    prompts: List[str]
    max_tokens: Optional[int] = Field(default=100, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=0.2, description="Sampling temperature")

# SGLang server runs on localhost:30000 (started via CMD in Dockerfile)
SGLANG_API_BASE = "http://localhost:30000/v1"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# Add a global variable to track model readiness
MODEL_READY = False

async def check_sglang_health():
    """Check if SGLang server is running and responsive"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SGLANG_API_BASE}/health") as response:
                if response.status == 200:
                    return True
                logger.error(f"SGLang health check failed: {response.status}")
                return False
    except Exception as e:
        logger.error(f"SGLang connection error: {e}")
        return False

async def run_litellm_inference(prompt: str, max_tokens: int = 100, temperature: float = 0.2) -> str:
    global MODEL_READY
    try:
        # Check if SGLang is available and model is ready
        if not await check_sglang_health() or not MODEL_READY:
            raise HTTPException(status_code=503, detail="Model server unavailable or model not ready")
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        response = await litellm.acompletion(
            model="sglang/unsloth/Qwen2-72B-bnb-4bit",
            api_base=SGLANG_API_BASE,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            caching=True
        )
        elapsed = time.time() - start_time
        
        content = response.choices[0].message.content
        logger.debug(f"Inference complete in {elapsed:.2f}s for prompt: {prompt[:20]}...")
        return content
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# ... (rest of the code remains the same)

@app.get("/health")
async def health_check():
    """Health check endpoint for the API"""
    global MODEL_READY
    sglang_healthy = await check_sglang_health()
    
    # Check Redis
    redis_healthy = False
    try:
        test_redis = redis.Redis(host="localhost", port=6379, socket_timeout=2)
        redis_healthy = test_redis.ping()
    except:
        pass
    
    status = "healthy" if sglang_healthy and redis_healthy and MODEL_READY else "degraded"
    if not sglang_healthy:
        status = "unhealthy"  # Critical component
    elif not MODEL_READY:
        status = "initializing"
        
    return {
        "status": status,
        "components": {
            "api": "healthy",
            "sglang": "healthy" if sglang_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "degraded",
            "model": "ready" if MODEL_READY else "loading"
        },
        "version": "1.0.0"
    }

@app.on_event("startup")
async def startup_event():
    global MODEL_READY
    # Initialize cache
    initialize_litellm_cache()
    
    # Verify model loading
    try:
        # Check if SGLang is running
        if not await check_sglang_health():
            logger.error("SGLang server is not available. Ensure it's running on port 30000.")
        else:
            logger.info("SGLang server is running and healthy.")
            
        logger.info("Qwen2-72B inference server with SGLang and Redis started on RunPod.")
        
        # Simulate model loading time (replace this with actual model loading check)
        logger.info("Loading model... This may take a few minutes.")
        await asyncio.sleep(300)  # Simulating 5 minutes of loading time
        MODEL_READY = True
        logger.info("Model loaded and ready for inference.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
