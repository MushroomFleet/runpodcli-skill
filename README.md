# 🚀 RunPod CLI Skill for Claude Code

A comprehensive skill package that enables Claude Code agents to automate GPU cloud infrastructure through natural language commands. Deploy pods, manage serverless endpoints, transfer data, and orchestrate ML workflows on RunPod — all through conversational instructions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RunPod](https://img.shields.io/badge/RunPod-Compatible-purple)](https://runpod.io)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blue)](https://claude.ai)

---

## ✨ What This Skill Does

The RunPod CLI skill transforms Claude Code into a GPU infrastructure automation agent capable of:

| Capability | Example Command |
|------------|-----------------|
| **Pod Management** | "Spin up an A100 pod for training my 70B model" |
| **Serverless Deployment** | "Deploy a serverless endpoint for Llama-2-7b inference" |
| **File Transfer** | "Send my dataset to the running pod" |
| **Resource Optimization** | "List my pods and terminate the idle ones" |
| **Cost Management** | "Switch my training pod to a spot instance" |

The skill handles intent recognition, GPU selection based on model requirements, cost optimization, and error recovery — enabling unattended execution of complex infrastructure tasks.

---

## 📦 Package Contents

```
runpodcli-skill/
├── SKILL.md              # Core skill definition (triggers, workflows, commands)
├── CLAUDE.md             # Full agentic system prompt for autonomous operation
├── references/
│   ├── commands.md       # Complete CLI flag reference
│   └── serverless.md     # Handler patterns & deployment templates
├── scripts/
│   └── pod_manager.py    # Python automation helpers
└── README.md             # This file
```

### Quick-Start Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `runpodcli.skill` | Packaged skill bundle | Drop into Claude Code's skill directory |
| `SKILL.md` | Skill triggers & core logic | Standalone skill integration |
| `CLAUDE.md` | Full agent instructions | Complete autonomous agent setup |

---

## 🤖 CLAUDE.md — Full Agentic System Prompt

The [`CLAUDE.md`](./CLAUDE.md) file contains comprehensive agent instructions for fully autonomous operation. It includes:

- **Intent Recognition Patterns** — Maps natural language to specific operations
- **GPU Selection Logic** — Automatic GPU matching based on model parameters
- **Standard Workflows** — Pod creation, serverless deployment, file transfer
- **Error Recovery** — Automated handling of common failure modes
- **Safety Rules** — Cost awareness, confirmation prompts, API key protection

Use `CLAUDE.md` when you want Claude Code to operate as a dedicated RunPod infrastructure agent with minimal supervision.

---

## ⚡ Quick Use with .skill Package

The `runpodcli.skill` file is a ready-to-use package for Claude Code:

### Installation

1. Download `runpodcli.skill` from this repository
2. Place it in your Claude Code skills directory:
   - **macOS/Linux**: `~/.claude/skills/`
   - **Windows**: `%USERPROFILE%\.claude\skills\`
3. Restart Claude Code or reload skills

### Usage

Simply invoke the skill in your Claude Code session:

```
use the runpodcli skill
```

Then issue natural language commands:

```
Create a pod with 2x A40 GPUs for distributed training
```

The skill works with or without the full `CLAUDE.md` agent prompt — it provides all necessary context for RunPod operations.

---

## 🔧 Integration Guide for Developers

### Prerequisites

1. **RunPod Account** — Sign up at [runpod.io](https://runpod.io)
2. **API Key** — Generate at Settings → API Keys
3. **runpodctl CLI** — Install via:
   ```bash
   wget -qO- cli.runpod.net | sudo bash
   ```

### Environment Setup

```bash
# Configure authentication
export RUNPOD_API_KEY="your_api_key_here"
runpodctl config --apiKey $RUNPOD_API_KEY

# Verify installation
runpodctl version
runpodctl get pod
```

### Using the Python Helpers

The `scripts/pod_manager.py` provides programmatic access:

```python
from pod_manager import create_pod, select_gpu, estimate_vram

# Auto-select GPU for a 13B model
gpu = select_gpu(min_vram=estimate_vram(13))
print(f"Selected: {gpu['id']} ({gpu['vram']}GB) @ ${gpu['hourly']}/hr")

# Create pod
pod = create_pod(
    name="training-job",
    gpu_type=gpu["id"],
    volume_gb=100,
    spot=True
)
print(f"Pod ID: {pod['id']}")
```

### Extending the Skill

To add custom workflows, edit `SKILL.md`:

```markdown
## Custom Workflow: Fine-Tune LLM

1. Create pod with appropriate GPU
2. Clone training repository
3. Download base model to /workspace
4. Launch training with checkpointing
5. Monitor via WandB integration
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy to RunPod
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install runpodctl
        run: wget -qO- cli.runpod.net | sudo bash
      
      - name: Deploy
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
        run: |
          runpodctl config --apiKey $RUNPOD_API_KEY
          python scripts/pod_manager.py create \
            --name "ci-${{ github.sha }}" \
            --gpu a40 \
            --spot
```

---

## 📋 Supported Operations

### Pod Management
- `create pods` — Launch GPU instances with custom configurations
- `get pod` — List pods or inspect specific instances
- `start/stop pod` — Control pod lifecycle
- `remove pod` — Terminate and delete pods

### Serverless Endpoints
- Handler generation for LLMs, embeddings, image generation
- Dockerfile templates for common ML frameworks
- Scaling configuration (active vs flex workers)
- Local testing and debugging

### Data Transfer
- `send/receive` — Peer-to-peer encrypted file transfer
- rsync integration for large datasets
- SSH/SCP for interactive workflows

### Cost Optimization
- Automatic spot instance recommendations
- VRAM-based GPU selection
- Storage cost warnings
- Idle resource detection

---

## 🔗 Resources

- [RunPod Documentation](https://docs.runpod.io)
- [runpodctl GitHub](https://github.com/runpod/runpodctl)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [Claude Code Documentation](https://docs.anthropic.com)

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{runpodcli_skill,
  title = {RunPod CLI Skill: GPU Cloud Infrastructure Automation for Claude Code Agents},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/runpodcli-skill},
  version = {1.0.0}
}
```

### Donate

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
