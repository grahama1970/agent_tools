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
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# Complete BatchRequest class definition
class BatchRequest(BaseModel):
    prompts: List[str]
    max_tokens: Optional[int] = Field(default=100, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=0.2, description="Sampling temperature")

# SGLang server runs on localhost:30000 (started via CMD in Dockerfile)
SGLANG_API_BASE = "http://localhost:30000/v1"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

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
    try:
        # Check if SGLang is available
        if not await check_sglang_health():
            raise HTTPException(status_code=503, detail="Model server unavailable")
            
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

async def batch_inference(prompts: List[str], max_tokens: int = 100, temperature: float = 0.2) -> List[str]:
    tasks = [run_litellm_inference(prompt, max_tokens, temperature) for prompt in prompts]
    results = []
    
    # Rate limiting: process max 5 requests at a time
    semaphore = asyncio.Semaphore(5)
    
    async def limited_inference(prompt):
        async with semaphore:
            return await run_litellm_inference(prompt, max_tokens, temperature)
    
    tasks = [limited_inference(prompt) for prompt in prompts]
    
    with tqdm(total=len(tasks), desc="Processing batch inference") as pbar:
        for future in as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                results.append(f"Error: {str(e)}")
            pbar.update(1)
    return results

@app.post("/batch_infer")
async def batch_infer(request: BatchRequest):
    if not request.prompts:
        raise HTTPException(status_code=400, detail="No prompts provided")
    if len(request.prompts) > 50:
        raise HTTPException(status_code=400, detail="Maximum batch size is 50 prompts")
    
    results = await batch_inference(
        request.prompts, 
        max_tokens=request.max_tokens, 
        temperature=request.temperature
    )
    return {"responses": results}

@app.get("/infer")
async def single_infer(prompt: str, max_tokens: int = 100, temperature: float = 0.2):
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")
    
    result = await run_litellm_inference(prompt, max_tokens, temperature)
    return {"response": result}

@app.get("/health")
async def health_check():
    """Health check endpoint for the API"""
    sglang_healthy = await check_sglang_health()
    
    # Check Redis
    redis_healthy = False
    try:
        test_redis = redis.Redis(host="localhost", port=6379, socket_timeout=2)
        redis_healthy = test_redis.ping()
    except:
        pass
    
    status = "healthy" if sglang_healthy and redis_healthy else "degraded"
    if not sglang_healthy:
        status = "unhealthy"  # Critical component
        
    return {
        "status": status,
        "components": {
            "api": "healthy",
            "sglang": "healthy" if sglang_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "degraded"
        },
        "version": "1.0.0"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
