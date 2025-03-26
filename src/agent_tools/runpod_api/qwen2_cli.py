"""
Qwen2-72B Inference CLI 🚀

A command-line interface for managing Qwen2-72B inference pods on RunPod.
Designed for easy integration with AI agents through clear structured documentation.

Commands:
- deploy: Create a new inference pod
- infer-batch: Run batch inference on an existing pod
- status: Check the status and health of a pod
- takedown: Remove an existing inference pod
- ready-to-inference: Check if the model is ready for inference


Usage Examples:
1. Deploy a new inference pod:
   python qwen2_cli.py deploy --gpu "NVIDIA A40" --name "my-qwen-pod"

2. Run batch inference:
   python qwen2_cli.py infer-batch --pod-ip 123.45.67.89:8000 --prompts "What is AI?" "Explain quantum computing"

3. Check pod status:
   python qwen2_cli.py status --pod-id ABC123XYZ

4. Takedown pod:
   python qwen2_cli.py takedown --pod-id ABC123XYZ

5. Check if ready for inference:
   python qwen2_cli.py ready-to-inference --pod-ip 123.45.67.89:8000

Environment Setup:
1. Create .env file with:
   - RUNPOD_API_KEY: Your RunPod API key
   - HF_TOKEN: Your Hugging Face token

Full Documentation: https://github.com/sgl-project/sglang
"""

import click
import os
import runpod
import requests
import json
import time
from loguru import logger
from dotenv import load_dotenv

# Configure logger and environment
logger.add("qwen2_cli.log", rotation="10 MB")
load_dotenv()

def validate_gpu_type(ctx, param, value):
    valid_gpus = ["NVIDIA A40", "NVIDIA A100", "NVIDIA V100"]
    if value not in valid_gpus:
        raise click.BadParameter(f"Invalid GPU type. Choose from: {', '.join(valid_gpus)}")
    return value

def validate_pod_id(ctx, param, value):
    if not value or not value.strip():
        raise click.BadParameter("Pod ID cannot be empty")
    return value.strip()

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Main entry point for Qwen2-72B inference management"""
    if verbose:
        logger.configure(handlers=[{"sink": "qwen2_cli.log", "level": "DEBUG"}])
        click.echo("🔍 Verbose logging enabled")




@cli.command()
@click.option('--gpu', default='NVIDIA A40', show_default=True, callback=validate_gpu_type,
             help='GPU type to use (e.g., NVIDIA A40, NVIDIA A100)')
@click.option('--name', default='qwen2-72b-pod', help='Name for the inference pod')
def deploy(gpu, name):
    """Deploy a new Qwen2-72B inference pod"""
    try:
        click.echo(f"🚀 Deploying {name} on {gpu}")
        pod_config = {
            "name": name,
            "imageName": "grahamaco/qwen2-72b-inference:latest",
            "gpuTypeId": gpu,
            "cloudType": "SECURE",
            "supportPublicIp": True,
            "ports": "8000/http,30000/http",
            "volumeMountPath": "/workspace",
            "minVcpuCount": 4,
            "minMemoryInGb": 32,
            "containerDiskInGb": 100
        }
        pod = runpod.create_pod(**pod_config)
        click.echo(f"✅ Deployment successful")
        click.echo(f"Pod ID: {pod['id']}")
        click.echo(f"Pod IP: {pod['machine']['publicIp']}:8000")
        
        # Save pod details
        with open("pod_id.txt", "w") as f:
            f.write(pod["id"])
        with open("pod_ip.txt", "w") as f:
            f.write(f"{pod['machine']['publicIp']}:8000")
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        click.echo("❌ Deployment failed - check logs for details")

@cli.command()
@click.option('--pod-ip', required=True, help='Pod IP address')
def ready_to_inference(pod_ip):
    """Check if the model is ready for inference"""
    try:
        click.echo(f"🔍 Checking if model is ready for inference on pod {pod_ip}")
        health_url = f"http://{pod_ip}/health"
        response = requests.get(health_url)
        response.raise_for_status()
        health_data = response.json()
        
        if health_data['status'] == 'healthy' and health_data['components']['sglang'] == 'healthy':
            click.echo("✅ Model is ready for inference")
            return True
        else:
            click.echo("❌ Model is not ready for inference")
            click.echo(f"Status: {health_data['status']}")
            click.echo(f"SGLang status: {health_data['components']['sglang']}")
            return False
    except Exception as e:
        logger.error(f"Ready check failed: {str(e)}")
        click.echo("❌ Failed to check if model is ready - see logs for details")
        return False



@cli.command()
@click.option('--pod-ip', required=True, help='Pod IP address')
@click.option('--prompts', required=True, multiple=True, help='Prompts for inference (space-separated)')
def infer_batch(pod_ip, prompts):
    """Run batch inference on existing pod"""
    try:
        click.echo(f"📦 Processing {len(prompts)} prompts on pod {pod_ip}")
        url = f"http://{pod_ip}/batch_infer"
        payload = {"prompts": prompts}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        results = response.json()["responses"]
        click.echo("✨ Inference complete")
        click.echo("Results:")
        for prompt, result in zip(prompts, results):
            click.echo(f"  Prompt: {prompt}")
            click.echo(f"  Result: {result}")
            click.echo("---")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        click.echo("❌ Inference failed - check logs for details")

@cli.command()
@click.option('--pod-id', required=True, callback=validate_pod_id, help='Pod ID to check')
def status(pod_id):
    """Check pod status and health"""
    try:
        click.echo(f"🩺 Running health check for pod {pod_id}...")
        pod_status = runpod.get_pod(pod_id)
        click.echo(f"📊 Pod status: {pod_status['status']}")
        
        if pod_status['status'] == 'RUNNING':
            pod_ip = f"{pod_status['machine']['publicIp']}:8000"
            health_url = f"http://{pod_ip}/health"
            health_response = requests.get(health_url)
            health_data = health_response.json()
            click.echo(f"💓 Health: {health_data['status']}")
            click.echo("Component Status:")
            for component, status in health_data['components'].items():
                click.echo(f"  - {component}: {status}")
        else:
            click.echo("❗ Pod is not in RUNNING state, unable to check health")
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        click.echo("❌ Status check failed - check logs for details")

@cli.command()
@click.option('--pod-id', required=True, callback=validate_pod_id, help='Pod ID to remove')
def takedown(pod_id):
    """Remove an existing inference pod"""
    try:
        if click.confirm(f"❓ Are you sure you want to delete pod {pod_id}?"):
            click.echo(f"🗑️  Removing pod {pod_id}")
            runpod.stop_pod(pod_id)
            click.echo("Waiting for pod to stop...")
            while True:
                pod_status = runpod.get_pod(pod_id)
                if pod_status["status"] == "STOPPED":
                    break
                time.sleep(5)
            runpod.delete_pod(pod_id)
            click.echo("✅ Pod removed successfully")
        else:
            click.echo("Takedown cancelled")
    except Exception as e:
        logger.error(f"Takedown failed: {str(e)}")
        click.echo("❌ Takedown failed - check logs for details")


@cli.command()
@click.option('--pod-ip', required=True, help='Pod IP address')
@click.option('--mode', type=click.Choice(['standard', 'stress']), default='standard', 
              help='Benchmark mode: standard (single requests) or stress (concurrent load)')
@click.option('--iterations', default=5, help='Number of iterations for standard benchmark')
@click.option('--concurrent', default=10, help='Number of concurrent requests for stress test')
@click.option('--duration', default=60, help='Duration in seconds for stress test')
@click.option('--output', default='benchmark_results.json', help='Output file for results')
def benchmark(pod_ip, mode, iterations, concurrent, duration, output):
    """Run performance benchmark and cost analysis"""
    try:
        click.echo(f"🔍 Running {mode} benchmark on pod {pod_ip}...")
        
        # Import benchmark module
        from benchmark import run_benchmark, run_stress_test, analyze_optimal_setup, save_benchmark_results
        
        # Run the appropriate benchmark
        if mode == 'standard':
            click.echo(f"Running standard benchmark with {iterations} iterations per prompt...")
            benchmark_results = asyncio.run(run_benchmark(pod_ip, iterations))
        else:
            click.echo(f"Running stress test with {concurrent} concurrent requests for {duration} seconds...")
            benchmark_results = asyncio.run(run_stress_test(pod_ip, concurrent, duration))
        
        if "error" in benchmark_results:
            click.echo(f"❌ Benchmark failed: {benchmark_results['error']}")
            return
        
        # Analyze results to determine optimal setup
        analysis = analyze_optimal_setup(benchmark_results)
        benchmark_results["optimization_analysis"] = analysis
        
        # Save results to file
        save_benchmark_results(benchmark_results, output)
        
        # Display summary
        if mode == 'standard':
            tps = benchmark_results["overall_avg_tokens_per_second"]
            click.echo(f"✅ Average throughput: {tps:.2f} tokens/second ({tps*3600:.0f} tokens/hour)")
        else:
            tps = benchmark_results["stress_test_results"]["tokens_per_second"]
            rps = benchmark_results["stress_test_results"]["requests_per_second"]
            click.echo(f"✅ Sustained throughput: {tps:.2f} tokens/second ({tps*3600:.0f} tokens/hour)")
            click.echo(f"  Request rate: {rps:.2f} requests/second")
        
        # Display optimal setup
        optimal = analysis["optimal_setup"]
        click.echo(f"💰 Optimal GPU configuration: {optimal['configuration']} ({optimal['description']})")
        click.echo(f"  Cost per million tokens: ${optimal['cost_per_million_tokens']:.4f}")
        
        # Show comparison table
        click.echo("\n📊 Configuration Comparison:")
        click.echo("┌─────────┬────────────┬─────────────┬────────────────┐")
        click.echo("│ Config  │ Cost/hour  │ Tokens/hour │ Cost/M tokens  │")
        click.echo("├─────────┼────────────┼─────────────┼────────────────┤")
        
        for config, details in analysis["all_configurations"].items():
            cost = details["hourly_cost"]
            tokens = details["projected_tokens_per_hour"]
            cost_per_m = details["cost_per_million_tokens"]
            click.echo(f"│ {config:7} │ ${cost:8.2f} │ {tokens:9.0f} │ ${cost_per_m:12.4f} │")
        
        click.echo("└─────────┴────────────┴─────────────┴────────────────┘")
        click.echo(f"\nDetailed results saved to {output}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        click.echo("❌ Benchmark failed - check logs for details")

if __name__ == "__main__":
    cli()
