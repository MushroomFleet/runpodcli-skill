---
name: runpodcli
description: GPU cloud infrastructure automation via RunPod CLI and Python SDK. Use when user requests to create/manage GPU pods, deploy serverless endpoints, transfer data to/from RunPod, set up ML training environments, configure auto-scaling inference, or any RunPod infrastructure task. Triggers on phrases like "spin up a pod", "deploy to runpod", "create serverless endpoint", "GPU instance", "runpodctl", or ML deployment requests targeting RunPod.
---

# RunPod CLI Skill

Automate GPU cloud infrastructure through natural language commands. This skill enables unattended setup of pods, serverless endpoints, and ML workflows on RunPod.

## Prerequisites

Verify RunPod CLI is available:

```bash
which runpodctl || (wget -qO- cli.runpod.net | sudo bash)
runpodctl version  # Expect v1.14.x+
```

Authentication requires `RUNPOD_API_KEY` environment variable or prior config:

```bash
# Check existing config
cat ~/.runpod/config.toml 2>/dev/null || echo "Not configured"

# Configure if needed (requires user's API key)
runpodctl config --apiKey "$RUNPOD_API_KEY"
```

## Intent Recognition Patterns

Parse user requests into these operation categories:

| User Intent | Operation | Key Parameters |
|-------------|-----------|----------------|
| "spin up / create / launch pod" | `create_pod` | gpu_type, image, storage |
| "deploy model / inference endpoint" | `create_serverless` | handler, gpu, scaling |
| "stop / pause pod" | `stop_pod` | pod_id |
| "terminate / delete pod" | `remove_pod` | pod_id |
| "list my pods / instances" | `get_pods` | - |
| "send / upload files" | `send_files` | source_path |
| "download / receive files" | `receive_files` | transfer_code |
| "SSH into pod" | `connect_ssh` | pod_id |
| "check pod status" | `get_pod_status` | pod_id |

## GPU Selection Reference

Match user requirements to GPU identifiers:

| Request Pattern | GPU ID | VRAM | Use Case |
|-----------------|--------|------|----------|
| "cheap", "basic", "small model" | `NVIDIA GeForce RTX 4090` | 24GB | Inference, fine-tuning |
| "mid-range", "7B-13B model" | `NVIDIA A40` | 48GB | Training, medium models |
| "large model", "70B", "production" | `NVIDIA A100 80GB PCIe` | 80GB | Large training jobs |
| "fastest", "enterprise", "H100" | `NVIDIA H100 80GB HBM3` | 80GB | Maximum performance |
| "huge context", "200B+" | `NVIDIA H200` | 141GB | Largest models |

## Core Operations

### Create Pod

```bash
runpodctl create pods \
  --name "${POD_NAME}" \
  --gpuType "${GPU_TYPE}" \
  --gpuCount ${GPU_COUNT:-1} \
  --imageName "${IMAGE:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}" \
  --containerDiskSize ${CONTAINER_DISK:-20} \
  --volumeSize ${VOLUME_SIZE:-50} \
  --ports "${PORTS:-8888/http,22/tcp}"
```

### Pod Lifecycle

```bash
runpodctl get pod                    # List all
runpodctl get pod ${POD_ID}          # Status check
runpodctl stop pod ${POD_ID}         # Pause (keeps /workspace)
runpodctl start pod ${POD_ID}        # Resume on-demand
runpodctl start pod ${POD_ID} --bid=0.3  # Resume as spot
runpodctl remove pod ${POD_ID}       # Terminate completely
```

### File Transfer

```bash
# Send (outputs transfer code)
runpodctl send ${FILE_PATH}

# Receive (on target machine)
runpodctl receive ${TRANSFER_CODE}
```

### SSH Connection

Extract connection details from pod info, then connect:

```bash
POD_INFO=$(runpodctl get pod ${POD_ID} --format json)
# Parse SSH host/port from POD_INFO
ssh root@${HOST} -p ${PORT} -i ~/.runpod/ssh/id_ed25519
```

## Serverless Deployment Workflow

### 1. Create Handler File

Generate `handler.py` with model loading outside handler function:

```python
import runpod

# Load model at startup (CRITICAL: outside handler)
model = None

def load_model():
    global model
    if model is None:
        # User's model loading logic here
        model = ...
    return model

def handler(event):
    """Process inference requests."""
    model = load_model()
    input_data = event.get('input', {})
    
    # User's inference logic here
    result = model.process(input_data)
    
    return {"output": result}

runpod.serverless.start({"handler": handler})
```

### 2. Create Dockerfile

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]
```

### 3. Build and Push

```bash
docker build -t ${DOCKER_USER}/${IMAGE_NAME}:${TAG} .
docker push ${DOCKER_USER}/${IMAGE_NAME}:${TAG}
```

### 4. Deploy via API

```python
import runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]

endpoint = runpod.create_endpoint(
    name="${ENDPOINT_NAME}",
    template_id="${TEMPLATE_ID}",  # Or use docker image directly
    gpu_ids=["NVIDIA A40"],
    workers_min=0,
    workers_max=3,
    idle_timeout=60
)
```

## Python SDK Operations

For programmatic control, use the `runpod` package:

```python
import runpod
import os

runpod.api_key = os.environ.get("RUNPOD_API_KEY")

# Create pod
pod = runpod.create_pod(
    name="training-pod",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA A40",
    gpu_count=1,
    volume_in_gb=100
)
print(f"Pod ID: {pod['id']}")

# List pods
pods = runpod.get_pods()
for p in pods:
    print(f"{p['id']}: {p['name']} - {p['desiredStatus']}")

# Terminate
runpod.terminate_pod(pod_id)
```

## Cost Optimization Rules

Apply automatically based on workload:

| Workload Type | Instance Type | Rationale |
|---------------|---------------|-----------|
| Development/testing | Spot (`--bid=0.3`) | 50-70% savings, interruptible OK |
| Training with checkpoints | Spot | Save state, resume on interrupt |
| Production inference | On-demand | Reliability required |
| Variable traffic | Serverless (min_workers=0) | Pay only when processing |

## Error Handling

| Error Pattern | Resolution |
|---------------|------------|
| "context deadline exceeded" | Network issue; retry with `--graphql-timeout 30s` |
| "unauthorized" | Re-run `runpodctl config --apiKey` |
| "GPU type not available" | List alternatives: `runpodctl get gpu` |
| "OCI runtime create failed" | CUDA version mismatch; use compatible image |
| "out of memory" | Select larger GPU or reduce batch size |

## Execution Checklist

Before executing any RunPod operation:

1. ☐ Confirm API key is configured (`runpodctl config` or `$RUNPOD_API_KEY`)
2. ☐ Validate GPU selection matches user's VRAM requirements
3. ☐ Confirm storage sizes (container disk resets on stop; volume persists)
4. ☐ For serverless: model loads outside handler function
5. ☐ For spot instances: checkpoint strategy in place
6. ☐ User has confirmed cost implications for on-demand workloads

## Reference Files

- `references/commands.md` - Complete CLI command reference with all flags
- `references/serverless.md` - Detailed serverless deployment patterns
- `scripts/pod_manager.py` - Python automation helpers
