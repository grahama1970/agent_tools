import runpod
import os
import time
import sys

def deploy_to_runpod():
    try:
        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod.api_key:
            raise ValueError("RUNPOD_API_KEY environment variable is not set")
        
        pod_config = {
            "name": "qwen2-72b-inference-pod",
            "imageName": "grahamaco/qwen2-72b-inference:latest",
            "gpuTypeId": "NVIDIA A40",
            "cloudType": "SECURE",
            "supportPublicIp": True,
            "ports": "8000/http,30000/http",
            "volumeMountPath": "/workspace",
            "minVcpuCount": 4,
            "minMemoryInGb": 32,
            "containerDiskInGb": 100
        }

        print("Creating pod with configuration:", pod_config)
        pod = runpod.create_pod(**pod_config)
        print(f"Pod created: {pod['id']}")
        print(f"Public IP: {pod['machine']['publicIp']}:8000")
        
        # Save pod details
        with open("pod_id.txt", "w") as f:
            f.write(pod["id"])
        with open("pod_ip.txt", "w") as f:
            f.write(f"{pod['machine']['publicIp']}:8000")
        
        print("Waiting for pod to become ready...")
        # Poll the pod status until it's running
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            pod_status = runpod.get_pod(pod["id"])
            if pod_status["status"] == "RUNNING":
                print("Pod is now running!")
                break
            print(f"Pod status: {pod_status['status']}. Waiting...")
            time.sleep(15)
            attempts += 1
        
        if attempts >= max_attempts:
            print("Warning: Pod did not reach RUNNING state within the expected time.")
            print("Check the RunPod console for details.")
    
    except Exception as e:
        print(f"Error during deployment: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    deploy_to_runpod()
