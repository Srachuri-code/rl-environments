# swe-bench

### Overview
- **Environment ID**: `swe-bench`
- **Short description**: Evaluates LLM agents on software engineering tasks using minisweagent Docker execution
- **Tags**: swe, multi-turn, tool-use, docker, coding

### Datasets
- **Primary dataset(s)**:
  - [SWE-Bench](https://huggingface.co/datasets/princeton-nlp/SWE-Bench) - Full benchmark
  - [SWE-Bench-Lite](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Lite) - Lighter evaluation set
  - [SWE-Bench-Verified](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified) - Human-verified test set
  - [SWE-Bench-Multilingual](https://huggingface.co/datasets/swe-bench/SWE-Bench_Multilingual) - Multilingual tasks
  - [SWE-Bench-Multimodal](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Multimodal) - Multimodal tasks
- **Source links**:
  - https://huggingface.co/datasets/princeton-nlp/SWE-Bench

### Task
- **Type**: Multi-turn with bash command execution
- **Parser**: Bash code block extraction (regex-based)
- **Rubric**: Binary reward based on swebench harness evaluation

### Prerequisites

**Docker must be installed and running.** Uses pre-built swebench Docker images from Docker Hub.

```bash
# Verify Docker is running
docker info
```

### Eval

```bash
# Run eval and save results as JSONL
uv run vf-eval swe-bench -s

# With custom model
uv run vf-eval swe-bench -s -m gpt-4.1-mini

# With different subset
uv run vf-eval swe-bench -s -a '{"subset": "verified"}'
```

Results saved to `./outputs/` as `results.jsonl`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `yaml_path` | str | `"swebench.yaml"` | Path to YAML configuration file |
| `subset` | str | `"lite"` | Dataset subset: full, lite, verified, multilingual, multimodal |
| `split` | str | `"dev"` | Dataset split to use |
| `timeout` | int | `60` | Timeout for individual commands (seconds) |
| `step_limit` | int | `250` | Maximum number of steps/turns |
| `output_dir` | str | `None` | Directory for prediction outputs |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `completed_instances` | Number of instances correctly solved (0 or 1) |
| `num_turns` | Number of conversation turns taken |

### How It Works

1. **Container Setup**: Creates a Docker container using the appropriate swebench image for each task instance
2. **Interaction**: Agent issues bash commands in code blocks, environment executes them via minisweagent
3. **Completion**: Agent signals completion with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`
4. **Verification**: Runs swebench harness evaluation and calculates binary reward
5. **Cleanup**: Container is removed after rollout

### Agent Response Format

The agent must respond with exactly ONE bash code block:

```
THOUGHT: Your reasoning here

```bash
your_command_here
```
```

### Submission

When work is complete, submit with:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```

### Implementation Notes

This implementation adapts [mini_swe_agent_bench](https://github.com/PrimeIntellect-ai/prime-environments/tree/main/environments/mini_swe_agent_bench) from the Prime Intellect prime-environments repository.

**Key components:**
- Uses `minisweagent` library for Docker container management
- Uses `swebench.harness.run_evaluation` for test execution and grading
- YAML-based configuration for prompts and environment settings
- Bash code block parsing (regex-based) instead of OpenAI tool calling

