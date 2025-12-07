# Tool Decathlon Setup Guide

## Implementation Complete ✅

**Total: 780 lines of glue code**

| File | Lines | Status |
|------|-------|--------|
| `docker/task_api.py` | 158 | ✅ Complete |
| `tool_decathlon.py` | 317 | ✅ Complete |
| `create_hf_dataset.py` | 194 | ✅ Complete |
| `build_toolathlon_image.sh` | 111 | ✅ Complete |
| **Dataset** | 30 tasks | ✅ Created |

## What We Built

**A meta-environment for SkyRL training:**
- Each Toolathlon task = sub-environment in Docker container
- Toolathlon's MCPs/eval scripts = tools/rewards
- task_api.py = step-by-step control wrapper
- ToolDecathlonEnv = communication layer

## Setup Steps

### 1. Install Docker

**On Mac:**
```bash
# Install Docker Desktop
# https://www.docker.com/products/docker-desktop/
```

**On GCP (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
```

### 2. Build Toolathlon Image

```bash
cd /Users/skrachur/Desktop/rl-environments/tool-decathlon
./scripts/build_toolathlon_image.sh
```

**This creates:** `toolathlon:latest` (Toolathlon + task_api.py)

### 3. Test Locally (Optional)

```bash
# Requires Docker + OpenAI API key
export OPENAI_API_KEY=sk-...

vf-eval tool-decathlon -m gpt-4o-mini -n 1
```

### 4. Use with SkyRL on GCP

```python
from skyrl import Trainer
from tool_decathlon import load_environment

# Load environment
env = load_environment(
    dataset_path="/path/to/tool_decathlon_dataset",
    toolathlon_image="toolathlon:latest",
    max_turns=100
)

# Train with SkyRL
trainer = Trainer(
    env=env,
    model="your-model",
    algorithm="ppo",
    # ... SkyRL config
)

trainer.train()
```

## Architecture for SkyRL

```
SkyRL Components:
├── Trainer (on Blackwell GPU)
│   └── PPO/GRPO optimizer
├── Generator 
│   ├── InferenceEngine (vLLM)
│   └── Environment ← ToolDecathlonEnv
│       └── Docker containers (Toolathlon tasks)
└── Controller
```

**Each training step:**
1. SkyRL samples 1024 tasks from dataset
2. ToolDecathlonEnv spins up 1024 Docker containers
3. Each container runs one Toolathlon task
4. Agent interacts step-by-step (controlled by SkyRL)
5. Eval scripts run → rewards extracted
6. Containers destroyed
7. SkyRL updates model weights

## Key Files

**Environment:**
- `environments/tool_decathlon/tool_decathlon.py` - Main environment class

**Docker:**
- `docker/task_api.py` - API wrapper for Toolathlon
- `scripts/build_toolathlon_image.sh` - Image builder

**Data:**
- `scripts/create_hf_dataset.py` - Dataset creator
- `data/tool_decathlon_dataset/` - 30 tasks from GitHub

**Docs:**
- `ARCHITECTURE.md` - Detailed architecture
- `README.md` - Overview

## Future: Dense Rewards

**To add trajectory-based rewards:**

1. Fork Toolathlon:
   ```bash
   git clone https://github.com/hkust-nlp/Toolathlon your-toolathlon
   cd your-toolathlon
   ```

2. Modify eval scripts:
   ```python
   # tasks/finalpool/find-alita-paper/evaluator.py
   def evaluate(workspace):
       return {
           "task_complete": check_completion(),
           "efficiency": 1.0 - turn_count/20,
           "tool_quality": used_good_tools(),
       }
   ```

3. Rebuild image:
   ```bash
   # Point build script to your fork
   git clone https://github.com/your-team/toolathlon
   ./build_toolathlon_image.sh
   ```

4. Update rubric weights in `tool_decathlon.py`

**No architecture changes needed!**

## Troubleshooting

**"Docker not found":**
- Install Docker Desktop (Mac) or docker.io (Linux)

**"Image not found: toolathlon:latest":**
- Run: `./scripts/build_toolathlon_image.sh`

**"Task setup failed":**
- Check task_api.py is in the image
- Verify Toolathlon dependencies installed

## Resources

- Toolathlon: https://github.com/hkust-nlp/Toolathlon
- SkyRL: https://skyrl.readthedocs.io
- Verifiers: https://verifiers.readthedocs.io
- Docker SDK: https://docker-py.readthedocs.io

