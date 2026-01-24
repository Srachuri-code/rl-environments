# Tool Decathlon RL Environment

Production-ready [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment for [Toolathlon](https://github.com/hkust-nlp/Toolathlon) - a benchmark with 108 diverse, realistic, and long-horizon tasks for language agents.

## Overview

This environment wraps Toolathlon's benchmark for reinforcement learning training:

- **108 tasks** spanning web browsing, file manipulation, API calls, email, K8s, and more
- **34 MCP servers** providing real-world tool interfaces
- **Task isolation** via Docker containers
- **Binary rewards** from Toolathlon's evaluation scripts

## Quick Start

### 1. Build Docker Image

```bash
cd tool-decathlon
bash scripts/build_toolathlon_image.sh
```

This pulls Toolathlon's pre-built image (`lockon0927/toolathlon-task-image:1016beta`) which includes all 34 MCP servers pre-installed.

### 2. Install Environment

```bash
cd environments/tool_decathlon
uv sync
source .venv/bin/activate
vf-install
```

### 3. Run Evaluation

```bash
# Quick test with 1 task
vf-eval tool_decathlon -m gpt-4.1-mini -n 1

# Full evaluation
vf-eval tool_decathlon -m gpt-4.1-mini -n 108
```

## Task Categories

Toolathlon tasks use various MCP servers. Here's what works out of the box:

### Works Without External Credentials

These tasks only need the Docker image:

| MCP Server | Description |
|------------|-------------|
| `filesystem` | File operations |
| `terminal` | Shell commands |
| `pdf-tools` | PDF reading/manipulation |
| `excel` | Excel file operations |
| `howtocook` | Recipe database |
| `memory` | Key-value storage |
| `arxiv_local` | arXiv paper search |
| `scholarly` | Academic search |

**Dataset**: `data/tool_decathlon_dataset_no_creds` (8 tasks)

### Requires External API Credentials

| MCP Server | Required Credentials |
|------------|---------------------|
| `google-cloud` | Google Cloud API key, OAuth, service account |
| `google_sheet` | Google OAuth credentials |
| `google_forms` | Google OAuth credentials |
| `google_calendar` | Google OAuth credentials |
| `github` | GitHub personal access token |
| `notion` | Notion integration key |
| `huggingface` | HuggingFace API token |
| `wandb` | Weights & Biases API key |
| `snowflake` | Snowflake database credentials |

### Requires Local Services

| MCP Server | Local Service | Ports |
|------------|--------------|-------|
| `canvas` | Canvas LMS | 10001, 20001 |
| `woocommerce` | WooCommerce store | 10003 |
| `emails` | Poste.io email server | 10005, 2525, 1143, 2587 |
| `k8s` | Kind cluster | various |

## Full Setup (All 108 Tasks)

### Step 1: Clone Toolathlon

```bash
cd tool-decathlon
git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git toolathlon-server
```

### Step 2: Configure Credentials

```bash
# Copy example configs
cd toolathlon-server/configs
cp global_configs_example.py global_configs.py
cp token_key_session_example.py token_key_session.py

# Edit token_key_session.py with your API keys
```

See [Toolathlon's credential guide](https://github.com/hkust-nlp/Toolathlon/blob/main/global_preparation/how2register_accounts.md) for detailed instructions.

### Step 3: Deploy Local Services (Optional)

For tasks using Canvas, WooCommerce, or email:

```bash
cd tool-decathlon
bash scripts/deploy_services.sh
```

This starts Docker containers for:
- Canvas LMS (ports 10001, 20001)
- WooCommerce (port 10003)
- Poste.io email server (ports 10005, 2525, 1143, 2587)

### Step 4: Set Up Google OAuth (Optional)

For Google-related tasks:

1. Create project at [Google Cloud Console](https://console.cloud.google.com)
2. Enable APIs: Sheets, Forms, Calendar, Cloud Storage, BigQuery
3. Create OAuth credentials (Desktop app)
4. Download and save as `toolathlon-server/configs/google_credentials.json`
5. Run: `cd toolathlon-server && uv run python global_preparation/simple_google_auth.py`

### Step 5: Build Image and Run

```bash
bash scripts/build_toolathlon_image.sh
vf-eval tool_decathlon -m gpt-4.1-mini -n 108
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    verifiers Framework                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ToolDecathlonEnv                           │ │
│  │  - Load Toolathlon dataset                              │ │
│  │  - Create isolated Docker containers                    │ │
│  │  - Route tool calls via HTTP API                        │ │
│  │  - Extract rewards from Toolathlon evaluators           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Docker Container (per task)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              task_api.py (HTTP Server)                  │ │
│  │  POST /setup    - Initialize MCP servers for task       │ │
│  │  POST /execute  - Execute tool calls                    │ │
│  │  POST /evaluate - Run Toolathlon evaluator              │ │
│  │  POST /cleanup  - Disconnect MCP servers                │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           MCPServerManager (Toolathlon)                 │ │
│  │  - Manages connections to MCP servers                   │ │
│  │  - Keeps servers alive across tool calls                │ │
│  │  - Routes calls to appropriate server                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │filesystem│ │terminal │ │pdf-tools│ │  ...    │           │
│  │   MCP   │ │   MCP   │ │   MCP   │ │34 total │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (host network)
┌─────────────────────────────────────────────────────────────┐
│                    Local Services (Optional)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Canvas LMS │  │ WooCommerce │  │  Poste.io   │         │
│  │ :10001,:20001│  │   :10003   │  │ :10005,...  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Options

```python
from tool_decathlon import load_environment

env = load_environment(
    dataset_path="data/tool_decathlon_dataset",  # Or _no_creds for subset
    toolathlon_image="toolathlon:latest",
    max_turns=100,
    use_host_network=True,      # Access local services
    mount_docker_socket=True,    # For K8s tasks
    configs_dir="toolathlon-server/configs",
    setup_timeout_s=600.0,       # MCP server init timeout
    tool_timeout_s=300.0,        # Tool call timeout
)
```

## Datasets

| Dataset | Tasks | Description |
|---------|-------|-------------|
| `tool_decathlon_dataset` | 108 | Full benchmark |
| `tool_decathlon_dataset_no_creds` | 8 | No external credentials needed |
| `tool_decathlon_dataset_minimal` | 30 | Subset for testing |

## Scripts

| Script | Description |
|--------|-------------|
| `build_toolathlon_image.sh` | Build Docker image from Toolathlon's pre-built image |
| `deploy_services.sh` | Deploy Canvas, WooCommerce, Poste.io containers |
| `setup_credentials.sh` | Guide for configuring API credentials |
| `create_hf_dataset.py` | Create dataset from Toolathlon tasks |

## Using Toolathlon's Public Eval Service

Toolathlon provides a public evaluation service with all credentials pre-configured:

```bash
python eval_client.py run \
  --mode public \
  --base-url https://api.openai.com/v1 \
  --model-name gpt-4o \
  --server-host 47.253.6.47 \
  --api-key $OPENAI_API_KEY \
  --workers 10
```

See [EVAL_SERVICE_README.md](https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md) for details.

## Troubleshooting

### "No such image: toolathlon:latest"

Build the Docker image:
```bash
bash scripts/build_toolathlon_image.sh
```

### "httpx.ReadTimeout" during setup

Increase the setup timeout in your environment config:
```python
env = load_environment(setup_timeout_s=900.0)  # 15 min
```

### MCP server connection failures

1. Check if local services are running: `docker ps`
2. Verify credentials in `toolathlon-server/configs/token_key_session.py`
3. For Google APIs, ensure OAuth is set up correctly

### Container runs but tools fail

Check container logs:
```bash
docker logs toolathlon-<task_id>-<uuid>
```

## References

- [Toolathlon Paper](https://arxiv.org/abs/2510.25726)
- [Toolathlon GitHub](https://github.com/hkust-nlp/Toolathlon)
- [verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers)

## Citation

```bibtex
@article{li2025toolathlon,
  title={The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution},
  author={Junlong Li and Wenshuo Zhao and Jian Zhao and others},
  year={2025},
  eprint={2510.25726},
  archivePrefix={arXiv},
}
```
