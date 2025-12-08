# Credential-Free Tasks for Testing

## Quick Start (No Credentials Needed!)

Test the environment without setting up any credentials:

```bash
vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n 3 \
  -a '{"dataset_path": "data/tool_decathlon_dataset_no_creds"}'
```

## Available Tasks (8 total)

These tasks only use local MCP servers and don't require API credentials:

### 1. **arrange-workspace**
- **Tools:** filesystem, terminal, pdf-tools, excel
- **Description:** Organize files and PDFs in a workspace

### 2. **cooking-guidance**  
- **Tools:** filesystem, howtocook
- **Description:** Provide cooking recipes and instructions

### 3. **course-schedule**
- **Tools:** filesystem, memory, excel, pdf-tools, fetch
- **Description:** Manage course schedules and materials

### 4. **courses-ta-hws**
- **Tools:** terminal, excel, filesystem
- **Description:** Grade and manage homework assignments

### 5. **detect-revised-terms**
- **Tools:** filesystem, pdf-tools
- **Description:** Compare PDFs to detect changes in terms

### 6. **dietary-health**
- **Tools:** filesystem, howtocook, excel, terminal
- **Description:** Analyze dietary information and health data

### 7. **excel-data-transformation**
- **Tools:** excel, filesystem, terminal
- **Description:** Transform and process Excel data

### 8. **excel-market-research**
- **Tools:** excel, filesystem, terminal
- **Description:** Analyze market research data in Excel

## Why These Work Without Credentials

These tasks only use MCP servers that operate locally:
- **arxiv_local** - Public ArXiv API (no auth needed)
- **filesystem** - Local file operations
- **terminal** - Shell commands
- **excel** - Local Excel file manipulation
- **pdf-tools** - Local PDF processing
- **howtocook** - Local recipe database
- **fetch** - Public HTTP requests
- **memory** - In-memory storage
- **time** - Time/date operations

## Tasks That NEED Credentials (22 in your dataset)

These require account setup:
- **Google Cloud** (9 tasks) - Needs GCP project, service account
- **Canvas LMS** (8 tasks) - Needs Canvas instance and credentials
- **GitHub** (2 tasks) - Needs GitHub token
- **Scholarly/Playwright** (5+ tasks) - Need special setup
- **Others** - Emails, Notion, Snowflake, etc.

## Full Credential Setup

If you want to run ALL tasks, follow Toolathlon's setup:

```bash
# Inside a running Toolathlon container:
docker run -it toolathlon:latest bash

# Then run:
bash global_preparation/automated_google_setup.sh
# Follow the prompts (~30 minutes for full setup)
```

Or use their public evaluation service (see their README).

## Dataset Locations

- **Full dataset (30 tasks):** `data/tool_decathlon_dataset/`
- **No-cred dataset (8 tasks):** `data/tool_decathlon_dataset_no_creds/`

