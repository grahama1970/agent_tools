import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
import aiohttp
import numpy as np
from loguru import logger

# GPU configurations with their hourly costs (March 2025 rates)
GPU_CONFIGS = {
    "1xA40": {"cost_per_hour": 0.79, "description": "Single NVIDIA A40 (48GB)"},
    "2xA40": {"cost_per_hour": 1.58, "description": "Dual NVIDIA A40 (96GB total)"},
    "1xH100": {"cost_per_hour": 3.50, "description": "Single NVIDIA H100 (80GB)"},
    "2xH100": {"cost_per_hour": 7.00, "description": "Dual NVIDIA H100 (160GB total)"}
}

# Standard test prompts of varying lengths
TEST_PROMPTS = [
    "What is 2+2?",  # Very short
    "Explain the concept of artificial intelligence in simple terms.",  # Medium
    "Write a detailed analysis of the economic impacts of climate change on global agriculture over the next 50 years. Include potential mitigation strategies and their costs.",  # Long
    "Provide a comprehensive explanation of quantum computing, including its principles, current state of development, and potential applications in cryptography, drug discovery, and optimization problems."  # Very long
]

async def measure_inference_speed(pod_ip: str, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
    """Measure inference speed for a single prompt"""
    url = f"http://{pod_ip}/infer"
    params = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Inference failed with status {response.status}: {error_text}")
            
            result = await response.json()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate tokens per second (input + output tokens)
    response_text = result["response"]
    input_tokens = len(prompt.split())  # Rough approximation
    output_tokens = len(response_text.split())
    total_tokens = input_tokens + output_tokens
    tokens_per_second = total_tokens / elapsed
    
    return {
        "prompt_length": len(prompt),
        "response_length": len(response_text),
        "input_tokens_approx": input_tokens,
        "output_tokens_approx": output_tokens,
        "total_tokens_approx": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second
    }

async def run_benchmark(pod_ip: str, num_iterations: int = 5, max_tokens: int = 100) -> Dict[str, Any]:
    """Run a comprehensive benchmark with multiple prompts and iterations"""
    all_results = []
    
    # First, check if the model is ready
    try:
        health_url = f"http://{pod_ip}/health"
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url) as response:
                health_data = await response.json()
                if health_data["status"] != "healthy":
                    return {"error": f"Model not ready. Status: {health_data['status']}"}
    except Exception as e:
        return {"error": f"Failed to check model health: {str(e)}"}
    
    # Run benchmarks for each prompt
    for prompt in TEST_PROMPTS:
        prompt_results = []
        
        # Run multiple iterations for statistical significance
        for i in range(num_iterations):
            try:
                result = await measure_inference_speed(pod_ip, prompt, max_tokens)
                prompt_results.append(result)
                # Small delay between requests
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in iteration {i} for prompt '{prompt[:20]}...': {e}")
        
        # Calculate statistics
        if prompt_results:
            tokens_per_second = [r["tokens_per_second"] for r in prompt_results]
            avg_tps = statistics.mean(tokens_per_second)
            median_tps = statistics.median(tokens_per_second)
            std_dev = statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0
            
            all_results.append({
                "prompt": prompt,
                "prompt_length": len(prompt),
                "iterations": len(prompt_results),
                "avg_tokens_per_second": avg_tps,
                "median_tokens_per_second": median_tps,
                "std_dev": std_dev,
                "raw_results": prompt_results
            })
    
    # Calculate overall statistics
    if all_results:
        all_tps = [r["avg_tokens_per_second"] for r in all_results]
        overall_avg_tps = statistics.mean(all_tps)
        
        # Calculate cost per million tokens
        cost_per_million_tokens = {}
        for config, details in GPU_CONFIGS.items():
            # Cost per token = hourly cost / (tokens per second * 3600)
            cost_per_token = details["cost_per_hour"] / (overall_avg_tps * 3600)
            cost_per_million = cost_per_token * 1_000_000
            cost_per_million_tokens[config] = cost_per_million
        
        return {
            "overall_avg_tokens_per_second": overall_avg_tps,
            "detailed_results": all_results,
            "cost_analysis": {
                "measured_config": "current",
                "tokens_per_second": overall_avg_tps,
                "tokens_per_hour": overall_avg_tps * 3600,
                "cost_per_million_tokens": cost_per_million_tokens
            }
        }
    else:
        return {"error": "No valid benchmark results collected"}

async def run_stress_test(pod_ip: str, concurrent_requests: int = 10, duration_seconds: int = 60) -> Dict[str, Any]:
    """Run a stress test to measure sustained throughput under load"""
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Use a medium-length prompt for stress testing
    prompt = TEST_PROMPTS[1]
    
    results = []
    total_tokens = 0
    total_requests = 0
    failed_requests = 0
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    async def single_request():
        nonlocal total_tokens, total_requests, failed_requests
        
        async with semaphore:
            try:
                result = await measure_inference_speed(pod_ip, prompt)
                results.append(result)
                total_tokens += result["total_tokens_approx"]
                total_requests += 1
            except Exception as e:
                logger.error(f"Request failed: {e}")
                failed_requests += 1
    
    # Keep sending requests until the duration is reached
    tasks = []
    current_time = time.time()
    
    while current_time < end_time:
        tasks.append(asyncio.create_task(single_request()))
        await asyncio.sleep(0.1)  # Small delay to prevent flooding
        current_time = time.time()
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    # Calculate statistics
    test_duration = time.time() - start_time
    requests_per_second = total_requests / test_duration
    tokens_per_second = total_tokens / test_duration
    
    # Calculate cost per million tokens for each GPU configuration
    cost_per_million_tokens = {}
    for config, details in GPU_CONFIGS.items():
        cost_per_token = details["cost_per_hour"] / (tokens_per_second * 3600)
        cost_per_million = cost_per_token * 1_000_000
        cost_per_million_tokens[config] = cost_per_million
    
    return {
        "stress_test_results": {
            "duration_seconds": test_duration,
            "concurrent_requests": concurrent_requests,
            "total_requests": total_requests,
            "failed_requests": failed_requests,
            "requests_per_second": requests_per_second,
            "tokens_per_second": tokens_per_second,
            "tokens_per_hour": tokens_per_second * 3600
        },
        "cost_analysis": {
            "measured_config": "current",
            "tokens_per_second": tokens_per_second,
            "tokens_per_hour": tokens_per_second * 3600,
            "cost_per_million_tokens": cost_per_million_tokens
        }
    }

def analyze_optimal_setup(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results to determine the optimal GPU setup"""
    if "cost_analysis" not in benchmark_results:
        return {"error": "No cost analysis data available"}
    
    cost_analysis = benchmark_results["cost_analysis"]
    tokens_per_second = cost_analysis["tokens_per_second"]
    
    # Calculate projected performance for different GPU setups
    # These are rough estimates based on typical scaling factors
    projected_performance = {
        "1xA40": tokens_per_second,  # Baseline (assuming benchmark was run on 1xA40)
        "2xA40": tokens_per_second * 1.8,  # Not quite 2x due to overhead
        "1xH100": tokens_per_second * 2.5,  # H100 is typically 2-3x faster than A40
        "2xH100": tokens_per_second * 4.5   # Not quite 2x 1xH100 due to overhead
    }
    
    # Calculate cost per million tokens for each setup
    cost_per_million = {}
    for config, tps in projected_performance.items():
        hourly_cost = GPU_CONFIGS[config]["cost_per_hour"]
        tokens_per_hour = tps * 3600
        cost_per_token = hourly_cost / tokens_per_hour
        cost_per_million[config] = cost_per_token * 1_000_000
    
    # Find the optimal setup (lowest cost per million tokens)
    optimal_setup = min(cost_per_million.items(), key=lambda x: x[1])
    
    # Calculate throughput comparison
    throughput_comparison = {}
    baseline_tps = projected_performance["1xA40"]
    for config, tps in projected_performance.items():
        throughput_comparison[config] = {
            "tokens_per_second": tps,
            "tokens_per_hour": tps * 3600,
            "relative_performance": tps / baseline_tps
        }
    
    return {
        "optimal_setup": {
            "configuration": optimal_setup[0],
            "cost_per_million_tokens": optimal_setup[1],
            "description": GPU_CONFIGS[optimal_setup[0]]["description"]
        },
        "all_configurations": {
            config: {
                "description": GPU_CONFIGS[config]["description"],
                "hourly_cost": GPU_CONFIGS[config]["cost_per_hour"],
                "projected_tokens_per_second": tps,
                "projected_tokens_per_hour": tps * 3600,
                "cost_per_million_tokens": cost_per_million[config]
            }
            for config, tps in projected_performance.items()
        },
        "throughput_comparison": throughput_comparison
    }

def save_benchmark_results(results: Dict[str, Any], filename: str = "benchmark_results.json"):
    """Save benchmark results to a JSON file"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark results saved to {filename}")


def get_gpu_configs():
    """Fetch current GPU pricing from RunPod API"""
    try:
        # Get all available GPU types and their pricing
        gpus = runpod.get_gpus()
        
        # Create a dictionary with configurations we're interested in
        gpu_configs = {}
        
        # Map RunPod GPU IDs to our config names
        gpu_mapping = {
            "NVIDIA A40": "1xA40",
            "NVIDIA H100": "1xH100"
        }
        
        # Fill in single GPU configurations with actual prices
        for gpu in gpus:
            if gpu["id"] in gpu_mapping:
                config_name = gpu_mapping[gpu["id"]]
                secure_price = next((price["price"] for price in gpu["prices"] if price["type"] == "SECURE"), None)
                if secure_price:
                    gpu_configs[config_name] = {
                        "cost_per_hour": secure_price,
                        "description": f"Single {gpu['id']} ({gpu['memoryInGb']}GB)"
                    }
        
        # Calculate multi-GPU configurations based on single GPU prices
        if "1xA40" in gpu_configs:
            gpu_configs["2xA40"] = {
                "cost_per_hour": gpu_configs["1xA40"]["cost_per_hour"] * 2,
                "description": "Dual NVIDIA A40 (96GB total)"
            }
        
        if "1xH100" in gpu_configs:
            gpu_configs["2xH100"] = {
                "cost_per_hour": gpu_configs["1xH100"]["cost_per_hour"] * 2,
                "description": "Dual NVIDIA H100 (160GB total)"
            }
        
        return gpu_configs
    
    except Exception as e:
        # Fallback to default values if API call fails
        logger.warning(f"Failed to fetch GPU pricing from RunPod API: {e}")
        logger.warning("Using default pricing values (may be outdated)")
        
        return {
            "1xA40": {"cost_per_hour": 0.79, "description": "Single NVIDIA A40 (48GB)"},
            "2xA40": {"cost_per_hour": 1.58, "description": "Dual NVIDIA A40 (96GB total)"},
            "1xH100": {"cost_per_hour": 3.50, "description": "Single NVIDIA H100 (80GB)"},
            "2xH100": {"cost_per_hour": 7.00, "description": "Dual NVIDIA H100 (160GB total)"}
        }