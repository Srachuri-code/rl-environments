# DeepDive

### Overview

- **Environment ID**: `deepdive`
- **Short description**: Complex QA with Google search, click, and open tools using Gemini as judge
- **Tags**: `qa`, `multi-turn`, `search`, `tool-use`

### Datasets

- **Primary dataset(s)**: [DeepDive](https://arxiv.org/pdf/2509.10446) - Multi-turn QA with web search
- **Source Link(s)**: [zai-org/DeepDive](https://huggingface.co/datasets/zai-org/DeepDive)
- **Split sizes**: 2k train, 0.2k eval

### Task

- **Type**: Multi-turn + tool use
- **Parser**: MaybeThinkParser with boxed answer extraction
- **Rubric overview**: Judge-based gold answer matching using Gemini 2.5 Flash

### Tools

The environment provides four tools for web search and content retrieval:

| Tool | Description |
| ---- | ----------- |
| `search(query, num_results)` | Search Google via Serper API, returns formatted results |
| `open(urls)` | Fetch and extract readable content from a list of URLs |
| `click(result_indices)` | Open specific results from previous search by index |
| `finish(final_answer)` | Submit final answer and stop execution |

### Setup and Install

```bash
uv run vf-install deepdive
```

**Required API Keys:**
- **Serper API Key**: Get from [serper.dev](https://serper.dev/) for Google search
- **Gemini API Key**: Get from [aistudio.google.com](https://aistudio.google.com/app/apikey) for the judge model

### Eval

```bash
export SERPER_API_KEY="your-serper-key"
export GEMINI_API_KEY="your-gemini-key"

# Run eval and save results as JSONL
uv run vf-eval deepdive -s

# With custom model
uv run vf-eval deepdive -s -m gpt-4.1-mini
```

Results saved to `./outputs/` as `results.jsonl`.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `32` | Max number of turns |
| `serper_api_key_var` | str | `"SERPER_API_KEY"` | Env var with Serper API key |
| `judge_api_key_var` | str | `"GEMINI_API_KEY"` | Env var with Gemini API key |
| `judge_model` | str | `"gemini-2.5-flash"` | Judge model for evaluation |
| `judge_base_url` | str | Gemini URL | Base URL for judge API |
| `max_search_results` | int | `10` | Maximum search results from Serper |
| `max_response_chars` | int | `20000` | Truncate outputs to this length |
| `serper_timeout` | float | `15` | Timeout for search requests |
| `debug` | bool | `False` | Print tool-call debug info |
| `finish_with_tool` | bool | `True` | Finish via `finish` tool; else use `\boxed{}` |
| `open_max_workers` | int | `64` | Threads for URL fetching |
| `cache_dir` | str | `None` | Disk cache directory (default: `/tmp/deepdive_cache`) |
| `cache_size_limit_gb` | int | `10` | Cache size limit in GB |
| `cache_ttl_seconds` | int | `604800` | Cache TTL (default: 1 week) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Accuracy (1.0 if judge says "yes", 0.0 otherwise) |

