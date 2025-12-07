# Tool Decathlon Environment

**Thin wrapper (~650 lines) connecting Toolathlon benchmark to verifiers for large-scale RL training.**

## Architecture

```
                    Your Training Code
                           ↓
                    verifiers (RL framework)
                           ↓
              ToolDecathlonEnv (glue layer ~300 lines)
                           ↓
              Docker SDK (container isolation)
                           ↓
           ┌──────────────────────────────────┐
           │  Toolathlon (in Docker container) │
           ├──────────────────────────────────┤
           │  • 108 tasks                      │
           │  • 30+ MCP servers (tools)        │
           │  • Eval scripts (rewards)         │
           │  • task_api.py (our wrapper ~160) │
           └──────────────────────────────────┘
```

## What Each Component Does

| Component | Lines | Purpose |
|-----------|-------|---------|
| `task_api.py` | 158 | Wraps Toolathlon's MCPs/eval for step-by-step control |
| `tool_decathlon.py` | 299 | Glue between verifiers ↔ sandboxes |
| `create_hf_dataset.py` | 194 | Fetch task prompts from Toolathlon GitHub |
| **Total** | **651** | **Complete training environment** |

## How It Works

### 1. One-Time Setup

```bash
# Build Docker image with Toolathlon + our API wrapper
./scripts/build_toolathlon_image.sh

# Create dataset (just task prompts)
python scripts/create_hf_dataset.py --subset 30
```

### 2. Per-Episode Execution

```python
# Load environment
env = load_environment()

# Sample task from dataset
task = dataset["find-alita-paper"]

# Setup (in sandbox)
state = await env.setup_state(task)
```

**What happens:**
```
1. Create Docker container from toolathlon:latest
2. Inside container, run:
   python task_api.py setup find-alita-paper
   
3. task_api.py does:
   - Reads task_config.json
   - Starts arxiv_local MCP
   - Starts filesystem MCP
   - Returns list of tools
   
4. Returns to trainer:
   state["info"]["oai_tools"] = [
       {"name": "arxiv_local__search", ...},
       {"name": "filesystem__write_file", ...},
       {"name": "claim_done", ...}
   ]
```

### 3. Agent Interaction

```python
# Agent generates tool call
response = model.generate(messages)  
# {"tool_calls": [{"name": "arxiv_local__search", "args": {...}}]}

# Environment executes
results, state = await env.env_response(response, state)
```

**What happens:**
```
Execute in sandbox:
  python task_api.py execute "arxiv_local__search" '{"query": "Alita"}'

task_api.py:
  - Calls Toolathlon's MCP manager
  - MCP hits real ArXiv API
  - Returns search results
  
Returns to trainer:
  {"role": "tool", "content": "Found 3 papers..."}
```

### 4. Completion & Reward

```python
# Agent calls claim_done
response = model.generate()
# {"tool_calls": [{"name": "claim_done"}]}

# Environment evaluates
results, state = await env.env_response(response, state)
```

**What happens:**
```
1. Mark task_done = True

2. Run evaluation in sandbox:
   python task_api.py evaluate
   
3. task_api.py:
   - Imports tasks/finalpool/find-alita-paper/evaluator.py
   - Runs evaluate(workspace)
   - Checks: "Did they download the right paper?"
   - Returns: {"success": True/False}

4. Store eval_result in state

5. Rubric calculates reward:
   reward = 1.0 if eval_result else 0.0
```

### 5. Cleanup

```python
await env.cleanup_state(state)
# Destroys Docker container (MCPs auto-cleanup)
```

## The Communication Layer

**ToolDecathlonEnv translates between two systems:**

| Verifiers API | → Translation → | Sandbox API |
|---------------|-----------------|-------------|
| `setup_state(task)` | → | `sandbox.create()` + `task_api.py setup` |
| `env_response(tool_calls)` | → | `task_api.py execute` |
| `rubric.score(state)` | → | `task_api.py evaluate` |
| `cleanup_state(state)` | → | `sandbox.delete()` |

## Why This Architecture

**Meta-Learning:**
- 108 different tasks = distribution of environments
- Train on task A, B, C → generalize to task D

**Scalability:**
- Each episode = isolated container
- Run 1000s in parallel on Blackwell
- No resource conflicts

**Correctness:**
- Use Toolathlon's exact eval scripts
- Real MCP servers (not mocks)
- Accurate rewards

**Simplicity:**
- 651 lines total
- Reuse Toolathlon's infrastructure
- Just glue code

## Extensibility (Future Dense Rewards)

**Current:** Binary task success
```python
def evaluate(workspace):
    return True  # or False
```

**Future:** Modify Toolathlon's eval scripts
```python
def evaluate(workspace):
    return {
        "task_complete": check_completion(workspace),  # 1.0 weight
        "partial_progress": check_progress(workspace),  # 0.3 weight
        "efficiency": count_files(workspace) < 10,     # 0.1 weight
    }
```

Then update rubric weights - no code changes to environment needed.

## Files

```
tool-decathlon/
├── docker/
│   └── task_api.py              # API wrapper for Toolathlon (158 lines)
├── environments/tool_decathlon/
│   ├── tool_decathlon.py        # Verifiers ↔ Sandbox glue (299 lines)
│   ├── pyproject.toml           # Dependencies
│   └── README.md                # Docs
└── scripts/
    ├── build_toolathlon_image.sh     # Docker image builder
    └── create_hf_dataset.py          # Dataset creator (194 lines)
```

## Setup Instructions

See [scripts/build_toolathlon_image.sh](scripts/build_toolathlon_image.sh) for building the Docker image with Toolathlon + task_api.py.

## References

- Toolathlon: https://github.com/hkust-nlp/Toolathlon
- Paper: https://arxiv.org/abs/2510.25726
- Verifiers: https://verifiers.readthedocs.io
- Docker SDK: https://docker-py.readthedocs.io
