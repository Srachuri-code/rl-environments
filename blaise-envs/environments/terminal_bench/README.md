# terminal-bench

### Overview
- **Environment ID**: `terminal-bench`
- **Short description**: Evaluates LLM agents on terminal/command-line tasks from Terminal-Bench Pro across 8 domains
- **Tags**: terminal, command-line, bash, tool-use, multi-turn, eval

### Datasets
- **Primary dataset(s)**: [Terminal-Bench Pro](https://huggingface.co/datasets/alibabagroup/terminal-bench-pro) - 200 public tasks for terminal/CLI agent evaluation
- **Source links**:
  - HuggingFace: https://huggingface.co/datasets/alibabagroup/terminal-bench-pro
  - GitHub: https://github.com/alibaba/terminal-bench-pro
- **Split sizes**: 200 public tasks (train split)

### Task
- **Type**: Multi-turn with tool use (bash command execution in Docker)
- **Parser**: None (tool-based interaction)
- **Rubric overview**:
  - Binary reward (1.0 for pass, 0.0 for fail) based on pytest test execution
  - Tests from task archives verify task completion
  - `task_completed` metric tracks explicit completion signals

### Prerequisites

**Docker must be installed and running.** Uses crun runtime by default for faster container startup.

```bash
# Verify Docker is running
docker info

# Install crun (optional, falls back to runc)
# Ubuntu/Debian: sudo apt-get install crun
# Or build from source: https://github.com/containers/crun
```

### Domains
Terminal-Bench Pro covers 8 domains:
- **data-processing**: Data manipulation, parsing, transformation tasks
- **games**: Game-related programming challenges
- **debugging**: Finding and fixing bugs in code
- **system-administration**: System config, file management, scripting
- **scientific-computing**: Numerical computation, data analysis
- **software-engineering**: Building, testing, deploying software
- **machine-learning**: ML model training, evaluation, data prep
- **security**: Security analysis, vulnerability assessment

### Eval

Requires Docker running.

```bash
# Run eval and save results as JSONL
uv run vf-eval terminal-bench -s

# With custom model
uv run vf-eval terminal-bench -s -m gpt-4.1-mini

# Filter by difficulty/category
uv run vf-eval terminal-bench -s -a '{"difficulty": "easy", "category": "data-processing"}'
```

Results saved to `./outputs/` as `results.jsonl`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | str | `None` | Filter tasks by difficulty: 'easy', 'medium', 'hard' |
| `category` | str | `None` | Filter tasks by category (see Domains above) |
| `docker_image` | str | `"python:3.11-slim"` | Docker image to use for containers |
| `cpu_cores` | int | `1` | CPU cores per container |
| `memory_gb` | int | `2` | Memory limit in GB per container |
| `command_timeout` | int | `60` | Timeout in seconds for individual bash commands |
| `task_timeout` | int | `600` | Overall timeout for test verification |
| `max_turns` | int | `100` | Maximum conversation turns per task |
| `max_command_timeouts` | int | `10` | Max command timeouts before aborting |
| `rollout_timeout_seconds` | float | `3600.0` | Wall-clock timeout for rollout |
| `runtime` | str | `"crun"` | OCI runtime (crun for speed, runc as fallback) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary pass/fail (1.0 or 0.0) based on test execution |
| `task_completed` | Whether agent explicitly signaled task completion |
| `command_timeout_count` | Number of command timeouts during rollout |
| `rollout_duration_seconds` | Total duration of the rollout |
| `container_error` | Whether a container error occurred |

### How It Works

1. **Container Setup**: Creates a Docker container with the configured image for each task
2. **Archive Extraction**: Uploads and extracts the task archive to `/workspace` in the container
3. **Execution**: Agent executes bash commands via tool calls inside the sandboxed container
4. **Verification**: After completion (max turns or agent signals done), test scripts run in the container
5. **Scoring**: Tests write reward (1 or 0) to `/workspace/logs/verifier/reward.txt`
6. **Cleanup**: Container is removed after rollout

### Implementation Notes

This implementation references two existing Terminal-Bench environments on the Prime Intellect hub:

**[popfido/terminalbench](https://app.primeintellect.ai/dashboard/environments/popfido/terminalbench)**
- Used as the primary reference for verifiers environment structure
- Followed the `load_environment()` entry point pattern and dataset conversion approach

**[primeintellect/terminal-bench-env](https://app.primeintellect.ai/dashboard/environments/primeintellect/terminal-bench-env)**
- Adapted the harness patterns: stop condition decorators (`@vf.stop`), cleanup handlers (`@vf.cleanup`)
- Used the same reward verification approach (test scripts write to `/logs/verifier/reward.txt`)

**Key differences from reference implementations:**
- **Local Docker execution** instead of Prime sandboxes
- Uses `docker-py` library for container lifecycle management
- Adapted Docker execution patterns from the `swe_bench` environment in this repository
- No cloud infrastructure required - runs entirely locally

