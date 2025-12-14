# Tool Decathlon Environment

Wrapper connecting Toolathlon benchmark (108 tasks, 600+ tools) to verifiers for RL training.

⚠️ **ACTIVE DEVELOPMENT**: 
- Some tasks may fail due to buggy MCP server configs
- Actively debugging and updating the repo
- Working on modifying eval to provide denser intermediate rewards

## Setup

```bash
# Build Docker image with Toolathlon
./scripts/build_toolathlon_image.sh

# Create dataset (task metadata only)
python scripts/create_hf_dataset.py --subset 30
```

## Usage

```python
from tool_decathlon import load_environment

env = load_environment(toolathlon_image="toolathlon:latest")

# Sample task
task = dataset["find-alita-paper"]

# Setup (creates Docker container with MCPs)
state = await env.setup_state(task)

# Agent uses tools
tools = state["info"]["oai_tools"]
results = await env.env_response(tool_calls, state)

# Evaluate
reward = await rubric.score(state)  # 1.0 or 0.0

# Cleanup
await env.cleanup_state(state)
```

## Architecture

```
Your Code → verifiers → ToolDecathlonEnv → Docker → Toolathlon (MCPs + eval)
```

Each task runs in isolated container:
- Task-specific MCP servers (arxiv, git, k8s, etc.)
- Real API calls (not mocked)
- Toolathlon's eval scripts for rewards
- Full workspace isolation

## Key Files

- `docker/task_api.py` - API wrapper for Toolathlon in containers
- `environments/tool_decathlon/tool_decathlon.py` - verifiers ↔ sandbox glue
- `scripts/create_hf_dataset.py` - Dataset creator

## Why Sandboxes

- **Correctness**: Use Toolathlon's exact infrastructure
- **Scalability**: Run 1000s of tasks in parallel
- **Simplicity**: Don't reimplement 600+ tools

## References

- [Toolathlon Paper](https://arxiv.org/abs/2510.25726)
- [Toolathlon GitHub](https://github.com/hkust-nlp/Toolathlon)
- [Verifiers](https://verifiers.readthedocs.io)
