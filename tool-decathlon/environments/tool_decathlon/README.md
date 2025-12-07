# tool-decathlon

### Overview
- **Environment ID**: `tool-decathlon`
- **Short description**: Meta-environment wrapping Toolathlon's 108 tasks via prime-sandboxes
- **Tags**: tool-use, agents, multi-turn, meta-learning, train, eval

### What This Is

**A thin wrapper (~200 lines of glue code)** that connects:

```
Toolathlon          prime-sandboxes         verifiers
(has everything)    (isolation)             (training)
     ↓                    ↓                      ↓
  108 tasks          Docker containers      ToolEnv API
  MCP servers        Resource mgmt          Rubric/rewards
  Eval scripts       Parallelization        Dataset format
     ↓                    ↓                      ↓
          ToolDecathlonEnv (glue ~200 lines)
```

**We DON'T reimplement anything.** Toolathlon's infrastructure (MCPs, eval scripts, tools) runs inside sandboxes. We just:
1. Load task metadata as a verifiers dataset
2. Spin up sandboxes with Toolathlon pre-configured
3. Delegate tool execution to Toolathlon's MCPs
4. Extract rewards from Toolathlon's eval scripts

### Architecture

**Meta-Environment Pattern:**

Each Toolathlon task = sub-environment with its own:
- Agent harness (prompts)
- Tool set (task-specific MCPs)
- Reward function (evaluation script)
- Isolated workspace (sandbox)

```python
# Sample task from dataset
task = dataset[idx]  # e.g., "find-alita-paper"

# Create sub-environment in sandbox
state = await env.setup_state(task)
# → Starts Toolathlon in Docker
# → Initializes task-specific MCPs
# → Creates isolated workspace

# Agent interacts
tools = state["info"]["oai_tools"]  # From Toolathlon's MCPs
responses = await env.env_response(tool_calls, state)

# Get reward
reward = await rubric.score(state)  # From Toolathlon's eval script

# Cleanup
await env.cleanup_state(state)  # Destroys sandbox
```

### Datasets

- **Primary dataset**: Toolathlon (108 tasks, 600+ tools, 32+ applications)
- **Source**: [https://github.com/hkust-nlp/Toolathlon](https://github.com/hkust-nlp/Toolathlon)
- **Paper**: [https://arxiv.org/abs/2510.25726](https://arxiv.org/abs/2510.25726)
- **Tasks**: Research, office, code, data, cloud, media, other domains

### Task Types

**Research:**
- Find academic papers
- Generate bibliographies
- Analyze research trends

**Office:**
- Manage spreadsheets
- Schedule meetings
- Process emails

**Code:**
- Git operations
- Deploy applications
- Manage repositories

**Cloud:**
- Configure K8s
- Manage databases
- Monitor services

### Installation

```bash
# 1. Install environment
uv pip install tool-decathlon

# 2. Build Toolathlon Docker image
./scripts/build_toolathlon_image.sh

# 3. Create dataset
python scripts/create_hf_dataset.py --subset 30

# 4. Ready to train!
```

### Quickstart

```python
from tool_decathlon import load_environment

# Load meta-environment
env = load_environment(
    toolathlon_image="toolathlon:latest",
    max_turns=100
)

# Each rollout uses isolated sandbox
for batch in dataloader:
    states = await env.setup_state(batch)  # Create sandboxes
    # ... training loop ...
    await env.cleanup_state(states)  # Destroy sandboxes
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | auto | Path to Toolathlon dataset |
| `toolathlon_image` | str | `"toolathlon:latest"` | Docker image with Toolathlon |
| `max_turns` | int | `100` | Maximum turns per task |
| `domains` | list | `None` | Filter tasks by domain |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary task success (from Toolathlon eval) |

**Future (extensible):**
- Partial credit for progress
- Efficiency rewards (fewer turns)
- Tool usage quality
- Trajectory-based rewards

### Reward Shaping (Future)

The architecture is designed for easy reward shaping:

```python
# Current: Binary (sparse)
def evaluate(workspace):
    return True  # or False

# Future: Dense rewards
def evaluate(workspace):
    return {
        "task_complete": True,      # 1.0 weight
        "partial_progress": 0.7,    # 0.3 weight
        "efficiency": 0.9,          # 0.1 weight
        "tool_quality": 0.8,        # 0.1 weight
    }
```

Just modify Toolathlon's `evaluator.py` files and rebuild the image.

### Parallelization

Each sandbox is fully isolated:

```python
# Run 1000s of tasks in parallel
tasks = dataset.shuffle().select(range(1000))
rewards = await asyncio.gather(*[
    run_task_in_sandbox(task) 
    for task in tasks
])
```

No interference, no rate limits, perfect scaling.

### Why Sandboxes?

| Approach | Complexity | Fidelity | Setup |
|----------|-----------|----------|-------|
| Replicate MCPs locally | High | Medium | Days |
| Simple mock tools | Low | Low | Hours |
| **Toolathlon in sandboxes** | **Low** | **High** | **1 script** |

Sandboxes let us use Toolathlon's exact infrastructure without reimplementing anything.

### Training Workflow

```
1. Build Toolathlon image (once)
   └─ Contains all MCPs, eval scripts, setup

2. Create dataset (once)  
   └─ Just task metadata (prompts, IDs)

3. Train with verifiers
   └─ Each rollout = fresh Toolathlon sandbox
   └─ Unlimited parallelism
   └─ Real eval scripts for accurate rewards

4. Validate periodically
   └─ Use Toolathlon's public eval service
   └─ Get official benchmark scores
```

### Files

```
tool-decathlon/
├── tool_decathlon.py          # ~200 lines of glue code
├── pyproject.toml             # Dependencies
├── README.md                  # This file
└── scripts/
    ├── build_toolathlon_image.sh    # Build Docker image
    └── create_hf_dataset.py         # Create dataset
```

### References

- Toolathlon Paper: https://arxiv.org/abs/2510.25726
- Toolathlon GitHub: https://github.com/hkust-nlp/Toolathlon
- Prime Sandboxes: https://docs.primeintellect.ai/sandboxes
- Verifiers: https://verifiers.readthedocs.io
