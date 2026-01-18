#!/usr/bin/env python3
"""
RunPod CLI Automation Helpers

Utility functions for Claude Code agents to automate RunPod operations.
Requires: pip install runpod
"""

import os
import sys
import json
import time
import subprocess
from typing import Optional, Dict, List, Any

# Lazy import runpod to allow script inspection without installation
runpod = None

def _ensure_runpod():
    """Lazy load runpod SDK."""
    global runpod
    if runpod is None:
        try:
            import runpod as rp
            runpod = rp
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        except ImportError:
            print("Installing runpod SDK...")
            subprocess.run([sys.executable, "-m", "pip", "install", "runpod", "-q"])
            import runpod as rp
            runpod = rp
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
    return runpod


# =============================================================================
# GPU SELECTION
# =============================================================================

GPU_CATALOG = {
    # Budget tier
    "rtx4090": {"id": "NVIDIA GeForce RTX 4090", "vram": 24, "hourly": 0.34},
    "rtx3090": {"id": "NVIDIA GeForce RTX 3090", "vram": 24, "hourly": 0.22},
    "a4000": {"id": "NVIDIA RTX A4000", "vram": 16, "hourly": 0.19},
    # Mid tier
    "a40": {"id": "NVIDIA A40", "vram": 48, "hourly": 0.39},
    "l40": {"id": "NVIDIA L40", "vram": 48, "hourly": 0.59},
    "a6000": {"id": "NVIDIA RTX A6000", "vram": 48, "hourly": 0.50},
    # High tier
    "a100": {"id": "NVIDIA A100 80GB PCIe", "vram": 80, "hourly": 1.19},
    "a100sxm": {"id": "NVIDIA A100-SXM4-80GB", "vram": 80, "hourly": 1.69},
    # Enterprise tier
    "h100": {"id": "NVIDIA H100 80GB HBM3", "vram": 80, "hourly": 2.49},
    "h100nvl": {"id": "NVIDIA H100 NVL", "vram": 94, "hourly": 3.29},
    "h200": {"id": "NVIDIA H200", "vram": 141, "hourly": 4.00},
}


def select_gpu(
    min_vram: int = 24,
    prefer_cost: bool = True,
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Select appropriate GPU based on requirements.
    
    Args:
        min_vram: Minimum VRAM in GB
        prefer_cost: If True, select cheapest option meeting requirements
        keywords: List of keywords from user request (e.g., ["h100", "fast"])
    
    Returns:
        Dict with id, vram, hourly fields
    """
    # Check for explicit GPU request in keywords
    if keywords:
        keywords_lower = [k.lower() for k in keywords]
        for key, gpu in GPU_CATALOG.items():
            if key in keywords_lower:
                return gpu
    
    # Filter by VRAM requirement
    candidates = [
        (key, gpu) for key, gpu in GPU_CATALOG.items()
        if gpu["vram"] >= min_vram
    ]
    
    if not candidates:
        # Return largest available
        return GPU_CATALOG["h200"]
    
    # Sort by cost or performance
    if prefer_cost:
        candidates.sort(key=lambda x: x[1]["hourly"])
    else:
        candidates.sort(key=lambda x: -x[1]["vram"])
    
    return candidates[0][1]


def estimate_vram(model_params_b: float, precision: str = "fp16") -> int:
    """
    Estimate VRAM needed for a model.
    
    Args:
        model_params_b: Model parameters in billions (e.g., 7 for 7B)
        precision: "fp32", "fp16", "int8", "int4"
    
    Returns:
        Estimated VRAM in GB (with overhead buffer)
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    bpp = bytes_per_param.get(precision.lower(), 2)
    base_vram = model_params_b * bpp
    
    # Add 20% overhead for activations and framework
    return int(base_vram * 1.2) + 2


# =============================================================================
# POD MANAGEMENT
# =============================================================================

def create_pod(
    name: str,
    gpu_type: str,
    image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_count: int = 1,
    volume_gb: int = 50,
    container_disk_gb: int = 20,
    ports: str = "8888/http,22/tcp",
    env_vars: Optional[Dict[str, str]] = None,
    spot: bool = False,
    bid_price: float = 0.3
) -> Dict[str, Any]:
    """
    Create a new RunPod GPU pod.
    
    Returns:
        Pod info dict with 'id' field
    """
    rp = _ensure_runpod()
    
    pod = rp.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=gpu_count,
        volume_in_gb=volume_gb,
        container_disk_in_gb=container_disk_gb,
        ports=ports,
        env=env_vars or {}
    )
    
    if spot and pod.get("id"):
        # Start as spot instance
        subprocess.run([
            "runpodctl", "stop", "pod", pod["id"]
        ], capture_output=True)
        subprocess.run([
            "runpodctl", "start", "pod", pod["id"],
            f"--bid={bid_price}"
        ], capture_output=True)
    
    return pod


def list_pods() -> List[Dict[str, Any]]:
    """List all pods with status."""
    rp = _ensure_runpod()
    return rp.get_pods()


def get_pod(pod_id: str) -> Optional[Dict[str, Any]]:
    """Get specific pod details."""
    rp = _ensure_runpod()
    pods = rp.get_pods()
    for pod in pods:
        if pod.get("id") == pod_id:
            return pod
    return None


def stop_pod(pod_id: str) -> bool:
    """Stop a pod (preserves /workspace volume)."""
    rp = _ensure_runpod()
    try:
        rp.stop_pod(pod_id)
        return True
    except Exception as e:
        print(f"Error stopping pod: {e}")
        return False


def terminate_pod(pod_id: str) -> bool:
    """Terminate and delete a pod completely."""
    rp = _ensure_runpod()
    try:
        rp.terminate_pod(pod_id)
        return True
    except Exception as e:
        print(f"Error terminating pod: {e}")
        return False


def wait_for_pod_ready(pod_id: str, timeout: int = 300) -> bool:
    """
    Wait for pod to reach RUNNING state.
    
    Args:
        pod_id: Pod identifier
        timeout: Max seconds to wait
    
    Returns:
        True if pod is running, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        if pod and pod.get("desiredStatus") == "RUNNING":
            runtime = pod.get("runtime", {})
            if runtime.get("uptimeInSeconds", 0) > 10:
                return True
        time.sleep(5)
    return False


# =============================================================================
# SERVERLESS ENDPOINTS
# =============================================================================

def call_endpoint_sync(
    endpoint_id: str,
    input_data: Dict[str, Any],
    timeout: int = 120
) -> Dict[str, Any]:
    """
    Call serverless endpoint synchronously.
    
    Args:
        endpoint_id: RunPod endpoint ID
        input_data: Input payload
        timeout: Max seconds to wait
    
    Returns:
        Response data
    """
    rp = _ensure_runpod()
    endpoint = rp.Endpoint(endpoint_id)
    return endpoint.run_sync({"input": input_data}, timeout=timeout)


def call_endpoint_async(
    endpoint_id: str,
    input_data: Dict[str, Any]
) -> str:
    """
    Submit async job to serverless endpoint.
    
    Returns:
        Job ID for status polling
    """
    rp = _ensure_runpod()
    endpoint = rp.Endpoint(endpoint_id)
    job = endpoint.run({"input": input_data})
    return job.job_id


def get_job_status(endpoint_id: str, job_id: str) -> Dict[str, Any]:
    """Get status and output of async job."""
    rp = _ensure_runpod()
    endpoint = rp.Endpoint(endpoint_id)
    # Re-create job reference
    job = endpoint.run({"input": {}})  # Dummy
    job.job_id = job_id
    
    status = job.status()
    result = {"status": status}
    
    if status == "COMPLETED":
        result["output"] = job.output()
    elif status == "FAILED":
        result["error"] = str(job.error()) if hasattr(job, 'error') else "Unknown error"
    
    return result


# =============================================================================
# FILE TRANSFER
# =============================================================================

def send_file(path: str) -> Optional[str]:
    """
    Send file using runpodctl and return transfer code.
    
    Returns:
        Transfer code or None on failure
    """
    result = subprocess.run(
        ["runpodctl", "send", path],
        capture_output=True,
        text=True
    )
    
    # Parse transfer code from output
    # Expected: "Code is: 1234-word-word-word"
    for line in result.stdout.split("\n"):
        if "Code is:" in line:
            return line.split("Code is:")[1].strip()
    
    print(f"Send failed: {result.stderr}")
    return None


def receive_file(code: str, output_dir: str = ".") -> bool:
    """
    Receive file using transfer code.
    
    Returns:
        True on success
    """
    result = subprocess.run(
        ["runpodctl", "receive", code],
        cwd=output_dir,
        capture_output=True,
        text=True
    )
    return result.returncode == 0


# =============================================================================
# HANDLER GENERATION
# =============================================================================

def generate_handler(
    model_id: str,
    task: str = "text-generation",
    output_path: str = "handler.py"
) -> str:
    """
    Generate a serverless handler file for common tasks.
    
    Args:
        model_id: HuggingFace model ID
        task: text-generation, image-generation, embeddings
        output_path: Where to save handler
    
    Returns:
        Generated handler code
    """
    templates = {
        "text-generation": f'''import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")

def handler(event):
    input_data = event.get("input", {{}})
    prompt = input_data.get("prompt", "")
    max_tokens = input_data.get("max_tokens", 256)
    temperature = input_data.get("temperature", 0.7)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {{"generated_text": response}}

runpod.serverless.start({{"handler": handler}})
''',
        "embeddings": f'''import runpod
import torch
from sentence_transformers import SentenceTransformer

print("Loading model: {model_id}")
model = SentenceTransformer("{model_id}")
print("Model loaded!")

def handler(event):
    input_data = event.get("input", {{}})
    texts = input_data.get("texts", [])
    
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    return {{"embeddings": embeddings.cpu().tolist()}}

runpod.serverless.start({{"handler": handler}})
''',
        "image-generation": f'''import runpod
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionXLPipeline

print("Loading model: {model_id}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16
).to("cuda")
print("Model loaded!")

def handler(event):
    input_data = event.get("input", {{}})
    prompt = input_data.get("prompt", "")
    negative = input_data.get("negative_prompt", "")
    steps = input_data.get("steps", 30)
    guidance = input_data.get("guidance_scale", 7.5)
    
    image = pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {{"image_base64": b64}}

runpod.serverless.start({{"handler": handler}})
'''
    }
    
    code = templates.get(task, templates["text-generation"])
    
    with open(output_path, "w") as f:
        f.write(code)
    
    return code


def generate_dockerfile(
    base_image: str = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    requirements: Optional[List[str]] = None,
    output_path: str = "Dockerfile"
) -> str:
    """Generate Dockerfile for serverless worker."""
    
    reqs = requirements or ["runpod", "transformers", "torch", "accelerate"]
    reqs_str = " ".join(reqs)
    
    dockerfile = f'''FROM {base_image}

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir {reqs_str}

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
'''
    
    with open(output_path, "w") as f:
        f.write(dockerfile)
    
    return dockerfile


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod automation helpers")
    subparsers = parser.add_subparsers(dest="command")
    
    # List pods
    subparsers.add_parser("list", help="List all pods")
    
    # Create pod
    create_parser = subparsers.add_parser("create", help="Create pod")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--gpu", default="a40")
    create_parser.add_argument("--image", default="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04")
    create_parser.add_argument("--volume", type=int, default=50)
    create_parser.add_argument("--spot", action="store_true")
    
    # Terminate pod
    term_parser = subparsers.add_parser("terminate", help="Terminate pod")
    term_parser.add_argument("pod_id")
    
    # Generate handler
    gen_parser = subparsers.add_parser("generate-handler", help="Generate handler")
    gen_parser.add_argument("--model", required=True)
    gen_parser.add_argument("--task", default="text-generation")
    gen_parser.add_argument("--output", default="handler.py")
    
    args = parser.parse_args()
    
    if args.command == "list":
        pods = list_pods()
        for p in pods:
            print(f"{p['id']}: {p['name']} - {p.get('desiredStatus', 'unknown')}")
    
    elif args.command == "create":
        gpu = GPU_CATALOG.get(args.gpu.lower(), GPU_CATALOG["a40"])
        pod = create_pod(
            name=args.name,
            gpu_type=gpu["id"],
            image=args.image,
            volume_gb=args.volume,
            spot=args.spot
        )
        print(f"Created pod: {pod['id']}")
    
    elif args.command == "terminate":
        if terminate_pod(args.pod_id):
            print(f"Terminated: {args.pod_id}")
        else:
            print("Termination failed")
    
    elif args.command == "generate-handler":
        code = generate_handler(args.model, args.task, args.output)
        print(f"Generated {args.output}")
    
    else:
        parser.print_help()
