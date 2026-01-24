# multi-swe

### Overview
- **Environment ID**: `multi-swe`
- **Short description**: Multi-SWE RL environment combining ByteDance's Multi-SWE-RL dataset with OpenHands agent harness for multilingual software engineering tasks.
- **Tags**: multi-swe, openhands, train, eval, multilingual, software-engineering

### Datasets
- **Primary dataset(s)**: ByteDance-Seed/Multi-SWE-RL - 4,723 multilingual training samples with reproducible Docker environments
- **Source links**: 
  - [Multi-SWE-bench GitHub](https://github.com/multi-swe-bench)
  - [HuggingFace Dataset](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-RL)
- **Split sizes**: train (4,723 instances after filtering C/C++)
- **Languages**: Python, Java, TypeScript, JavaScript, Go, Rust

### Task
- **Type**: multi-turn tool use
- **Parser**: Default Parser (vf.Parser)
- **Rubric overview**: 
  - `solved` (weight: 1.0): Binary reward based on test results validation using Multi-SWE-bench harness
  - `has_error` (weight: 0.0): Infrastructure error detection for masking failed rollouts

### Architecture

This environment combines:
1. **Dataset**: Multi-SWE-RL from ByteDance (same as mini_swe_agent_plus)
2. **Reward Functions**: Same as mini_swe_agent_plus using `validate_report_against_dataset`
3. **Agent Harness**: OpenHands-style tools instead of minimal 2-tool harness

### OpenHands Tools

| Tool | Description |
| ---- | ----------- |
| `execute_bash` | Execute bash commands with interactive input support (`is_input` flag) |
| `execute_ipython_cell` | Run Python code in IPython-like environment |
| `str_replace_editor` | File editing with commands: `view`, `create`, `str_replace`, `insert`, `undo_edit` |
| `file_read` | Read files with optional line range |
| `submit` | Submit changes for evaluation |

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval multi-swe
```

Configure model and sampling:

```bash
uv run vf-eval multi-swe \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"max_turns": 100, "test_timeout": 1800}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"ByteDance-Seed/Multi-SWE-RL"` | Dataset to load |
| `max_turns` | int | `200` | Maximum turns per episode |
| `total_timeout_minutes` | int | `120` | Total timeout for episode in minutes |
| `test_timeout` | int | `1800` | Timeout for running tests in seconds |

### Sandbox Configuration

| Setting | Value |
| ------- | ----- |
| CPU Cores | 8 |
| Memory | 8 GB |
| Disk Size | 10 GB |
| Turn Timeout | 90 seconds |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Binary (0 or 1) - 1 if all tests pass (f2p, p2p, s2p, n2p validated) |
| `has_error` | 1 if infrastructure failure, 0 otherwise (used for masking) |
| `instance_completed` | Whether agent explicitly submitted changes |

### Reward Function Details

The reward function validates test results against expected transitions:
- **f2p_tests**: Fail-to-pass tests (must all be present in report)
- **p2p_tests**: Pass-to-pass tests (must all be present)
- **s2p_tests**: Skip-to-pass tests (must all be present)
- **n2p_tests**: None-to-pass tests (must all be present)

Success requires ALL expected test transitions to be present in the evaluation report.

### Comparison with mini_swe_agent_plus

| Aspect | mini_swe_agent_plus | multi-swe (this env) |
| ------ | ------------------- | -------------------- |
| Tools | 2 (execute_bash, str_replace) | 5 (OpenHands-style) |
| Tool constraint | 1 tool/turn only | Flexible |
| IPython | Blocked | Available |
| File viewing | Via bash commands | Dedicated `file_read` + `str_replace_editor view` |
| Undo support | No | Yes (`undo_edit` command) |
| Interactive processes | Limited | `is_input` flag support |
| Dataset | Same (Multi-SWE-RL) | Same (Multi-SWE-RL) |
| Reward function | Same | Same |
