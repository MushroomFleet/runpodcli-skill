# Serverless Deployment Patterns

Detailed patterns for deploying inference endpoints on RunPod Serverless.

## Handler Architecture

### Basic Handler Template

```python
import runpod

def handler(event):
    """
    event structure:
    {
        "id": "job-id",
        "input": { ... user payload ... },
        "webhook": "optional-callback-url"
    }
    """
    input_data = event.get("input", {})
    
    # Process request
    result = process(input_data)
    
    # Return formats:
    # Simple: return result
    # Structured: return {"output": result, "status": "success"}
    return result

runpod.serverless.start({"handler": handler})
```

### Model Preloading Pattern (Critical)

Always load models OUTSIDE the handler for cold start optimization:

```python
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== LOAD AT STARTUP =====
print("Loading model...")
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded!")
# ============================

def handler(event):
    prompt = event["input"].get("prompt", "")
    max_tokens = event["input"].get("max_tokens", 256)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": response}

runpod.serverless.start({"handler": handler})
```

### Streaming Handler

For long-running generations, stream results:

```python
import runpod

def handler(event):
    prompt = event["input"]["prompt"]
    
    for token in model.stream_generate(prompt):
        yield {"token": token, "finished": False}
    
    yield {"token": "", "finished": True}

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True
})
```

### Generator Pattern (Progress Updates)

```python
import runpod

def handler(event):
    total_steps = event["input"].get("steps", 100)
    
    for step in range(total_steps):
        # Process step
        result = process_step(step)
        
        # Yield progress
        yield {
            "progress": (step + 1) / total_steps,
            "step": step,
            "partial_result": result
        }
    
    # Final yield is the complete result
    return {"final_result": complete_result}

runpod.serverless.start({"handler": handler})
```

## Dockerfile Patterns

### PyTorch + Transformers

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time (reduces cold start)
ARG HF_TOKEN
RUN python -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', \
    token='${HF_TOKEN}')"

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
```

### vLLM Inference

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install vllm runpod

COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "-u", "handler.py"]
```

vLLM handler:
```python
import runpod
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

def handler(event):
    prompts = event["input"].get("prompts", [event["input"].get("prompt")])
    params = SamplingParams(
        temperature=event["input"].get("temperature", 0.7),
        max_tokens=event["input"].get("max_tokens", 256)
    )
    outputs = llm.generate(prompts, params)
    return [{"text": o.outputs[0].text} for o in outputs]

runpod.serverless.start({"handler": handler})
```

### Diffusion Models

```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN pip install diffusers transformers accelerate runpod

COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "-u", "handler.py"]
```

Diffusion handler:
```python
import runpod
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

def handler(event):
    prompt = event["input"]["prompt"]
    negative = event["input"].get("negative_prompt", "")
    steps = event["input"].get("steps", 30)
    
    image = pipe(prompt, negative_prompt=negative, num_inference_steps=steps).images[0]
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode()
    
    return {"image_base64": b64_image}

runpod.serverless.start({"handler": handler})
```

## API Interaction Patterns

### Synchronous Call (Python)

```python
import runpod

runpod.api_key = "your_key"
endpoint = runpod.Endpoint("endpoint_id")

result = endpoint.run_sync({
    "input": {"prompt": "Hello, world!"}
}, timeout=60)

print(result)
```

### Asynchronous Call (Python)

```python
import runpod
import time

endpoint = runpod.Endpoint("endpoint_id")

# Submit job
job = endpoint.run({"input": {"prompt": "Generate a story"}})
print(f"Job ID: {job.job_id}")

# Poll for completion
while job.status() not in ["COMPLETED", "FAILED"]:
    time.sleep(1)

# Get result
if job.status() == "COMPLETED":
    print(job.output())
else:
    print(f"Failed: {job.error()}")
```

### cURL Examples

Synchronous:
```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello"}}'
```

Asynchronous:
```bash
# Submit
JOB_ID=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello"}}' | jq -r '.id')

# Check status
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## Scaling Configuration

### Worker Types

**Active Workers** (always on):
- No cold start
- 20-30% billing discount
- Best for consistent traffic

**Flex Workers** (scale to zero):
- Cold start on first request
- Pay only when processing
- Best for variable/low traffic

### Configuration Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `min_workers` | Always-on workers | 0 for cost savings, 1+ for latency |
| `max_workers` | Concurrent limit | Based on expected peak |
| `idle_timeout` | Seconds before scale down | 60-300 based on traffic patterns |
| `execution_timeout` | Max job duration | 300-600 for inference, 3600 for training |
| `flashboot` | Fast cold starts | Always enable if available |

### Example Configurations

**Development/Testing:**
```python
{
    "min_workers": 0,
    "max_workers": 1,
    "idle_timeout": 30,
    "execution_timeout": 300
}
```

**Production - Latency Sensitive:**
```python
{
    "min_workers": 2,
    "max_workers": 10,
    "idle_timeout": 120,
    "execution_timeout": 60
}
```

**Production - Cost Optimized:**
```python
{
    "min_workers": 0,
    "max_workers": 5,
    "idle_timeout": 60,
    "execution_timeout": 300,
    "flashboot": True
}
```

## Error Handling

### Handler Error Pattern

```python
import runpod

def handler(event):
    try:
        # Validate input
        if "prompt" not in event.get("input", {}):
            return {"error": "Missing required field: prompt"}
        
        result = process(event["input"])
        return {"output": result}
        
    except torch.cuda.OutOfMemoryError:
        return {"error": "GPU out of memory", "code": "OOM"}
    except Exception as e:
        return {"error": str(e), "code": "INTERNAL_ERROR"}

runpod.serverless.start({"handler": handler})
```

### Client-Side Retry

```python
import runpod
import time

def call_with_retry(endpoint, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = endpoint.run_sync(input_data, timeout=120)
            if "error" not in result:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return result
```

## Local Development

### Test Handler Locally

```bash
# Install dev dependencies
pip install runpod

# Create test input
echo '{"input": {"prompt": "Hello"}}' > test_input.json

# Run with debug server
python handler.py --rp_serve_api --rp_log_level DEBUG --rp_api_port 8080

# Test in another terminal
curl -X POST http://localhost:8080/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Debugger Mode

```python
# At end of handler.py
if __name__ == "__main__":
    # Local testing
    test_event = {"input": {"prompt": "Test prompt"}}
    result = handler(test_event)
    print(result)
```
