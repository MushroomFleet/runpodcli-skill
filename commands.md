# RunPod CLI Command Reference

Complete flag reference for `runpodctl` commands.

## Pod Commands

### runpodctl create pods

Create a new GPU pod.

```
Flags:
  --name string              Pod name (required)
  --gpuType string           GPU type identifier (required)
  --gpuCount int             Number of GPUs (default: 1)
  --imageName string         Docker image (required)
  --containerDiskSize int    Container disk GB (default: 20, wiped on stop)
  --volumeSize int           Persistent volume GB (default: 0)
  --volumeMountPath string   Volume mount point (default: /workspace)
  --ports string             Port mappings: "8888/http,22/tcp,5000/http"
  --env string               Environment vars: "KEY1=val1,KEY2=val2"
  --networkVolumeId string   Attach existing network volume
  --startSsh                 Enable SSH access (default: true)
  --supportPublicIp          Request public IP
  --dataCenterId string      Specific datacenter
  --minDownload int          Min download Mbps
  --minUpload int            Min upload Mbps
```

Example:
```bash
runpodctl create pods \
  --name "llm-training" \
  --gpuType "NVIDIA A100 80GB PCIe" \
  --gpuCount 4 \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 500 \
  --ports "8888/http,22/tcp,6006/http" \
  --env "WANDB_API_KEY=xxx,HF_TOKEN=xxx"
```

### runpodctl get pod

List or inspect pods.

```
Usage:
  runpodctl get pod              # List all pods
  runpodctl get pod {podId}      # Get specific pod
  
Flags:
  --format string    Output format: table, json (default: table)
```

### runpodctl start pod

Start a stopped pod.

```
Usage:
  runpodctl start pod {podId}
  
Flags:
  --bid float    Bid price for spot instance (omit for on-demand)
```

### runpodctl stop pod

Stop a running pod. Container disk is wiped; volume disk persists.

```
Usage:
  runpodctl stop pod {podId}
```

### runpodctl remove pod

Terminate and delete a pod completely.

```
Usage:
  runpodctl remove pod {podId}
```

## Data Transfer Commands

### runpodctl send

Send files using peer-to-peer encrypted transfer.

```
Usage:
  runpodctl send {path}
  
Output:
  Code is: 1234-word-word-word
  On the other computer run: runpodctl receive 1234-word-word-word
```

### runpodctl receive

Receive files using transfer code.

```
Usage:
  runpodctl receive {code}
```

## SSH Commands

### runpodctl ssh

Generate or manage SSH keys.

```
Usage:
  runpodctl ssh keygen           # Generate new SSH keypair
  runpodctl ssh add              # Add public key to account
```

SSH keys are stored in `~/.runpod/ssh/`.

## Project Commands

### runpodctl project

Manage serverless projects locally.

```
Usage:
  runpodctl project create       # Initialize new project
  runpodctl project deploy       # Deploy to RunPod
  runpodctl project dev          # Start local dev server
  runpodctl project build        # Build Docker image
```

## Configuration Commands

### runpodctl config

Configure CLI authentication.

```
Usage:
  runpodctl config --apiKey {key}
  
Stores config in: ~/.runpod/config.toml
```

## GPU Listing

### runpodctl get gpu

List available GPU types and pricing.

```
Usage:
  runpodctl get gpu
```

## Common GPU Type Identifiers

```
NVIDIA GeForce RTX 3090        24GB   $0.22/hr community
NVIDIA GeForce RTX 4090        24GB   $0.34/hr community
NVIDIA RTX A4000               16GB   $0.19/hr community
NVIDIA RTX A5000               24GB   $0.22/hr community
NVIDIA RTX A6000               48GB   $0.50/hr community
NVIDIA A40                     48GB   $0.39/hr secure
NVIDIA L40                     48GB   $0.59/hr secure
NVIDIA A100 80GB PCIe          80GB   $1.19/hr secure
NVIDIA A100-SXM4-80GB          80GB   $1.69/hr secure
NVIDIA H100 80GB HBM3          80GB   $2.49/hr secure
NVIDIA H100 NVL                94GB   $3.29/hr secure
NVIDIA H200                   141GB   Contact
```

Prices are approximate and vary by datacenter and availability.

## Environment Variables

Auto-set inside RunPod pods:

```
RUNPOD_POD_ID          Unique pod identifier
RUNPOD_API_KEY         Pod-scoped API key
RUNPOD_PUBLIC_IP       Public IP (if available)
RUNPOD_TCP_PORT_22     External SSH port
RUNPOD_DC_ID           Datacenter identifier
RUNPOD_GPU_COUNT       Number of GPUs attached
```

## Exit Codes

```
0    Success
1    General error
2    Authentication failure
3    Resource not found
4    Invalid parameters
5    Network timeout
```

## Timeout Configuration

For slow networks:

```bash
runpodctl get pod --graphql-timeout 30s
```
