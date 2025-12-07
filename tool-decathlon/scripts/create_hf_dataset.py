#!/usr/bin/env python3
"""
Create minimal HuggingFace dataset from Toolathlon tasks.

This script only fetches task prompts - everything else (MCPs, eval scripts,
tools) is handled by Toolathlon infrastructure in prime-sandboxes.

Usage:
    python create_hf_dataset.py --subset 30
    python create_hf_dataset.py --all
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset

# GitHub configuration
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
REPO_OWNER = "hkust-nlp"
REPO_NAME = "Toolathlon"
BRANCH = "main"

# All 108 Toolathlon tasks (cached to avoid API calls)
ALL_TASKS = [
    "ab-testing", "academic-pdf-report", "academic-warning", "add-bibtex",
    "apply-phd-email", "arrange-workspace", "canvas-arrange-exam", "canvas-art-manager",
    "canvas-art-quiz", "canvas-do-quiz", "canvas-homework-grader-python", "canvas-list-test",
    "canvas-new-students-notification", "canvas-submit-late-work", "cooking-guidance",
    "course-assistant", "course-schedule", "courses-ta-hws", "cvpr-research",
    "dataset-license-issue", "detect-revised-terms", "dietary-health", "email-paper-homepage",
    "excel-data-transformation", "excel-market-research", "experiments-recordings",
    "fillout-online-forms", "filter-low-selling-products", "find-alita-paper",
    "flagged-transactions", "game-statistics", "gdp-cr5-analysis", "git-bug-hunt",
    "git-milestone", "git-repo", "hk-top-conf", "huggingface-upload", "identify-all-songs",
    "imagenet", "inter-final-performance-analysis", "interview-report", "inventory-sync",
    "investment-decision-analysis", "invoice-org", "ipad-edu-price", "k8s-deployment-cleanup",
    "k8s-mysql", "k8s-pr-preview-testing", "k8s-redis-helm-upgrade", "k8s-safety-audit",
    "landing-task-reminder", "language-school", "latex-prompt-box", "live-transactions",
    "llm-training-dataset", "logical-datasets-collection", "machine-operating", "meeting-assign",
    "merge-hf-datasets", "mrbeast-analysis", "music-analysis", "nhl-b2b-analysis",
    "notion-find-job", "notion-hr", "notion-movies", "notion-personal-website", "nvidia-market",
    "nvidia-stock-analysis", "oil-price", "paper-checker", "payable-invoice-checker",
    "personal-website-construct", "ppt-analysis", "price-comparison", "privacy-desensitization",
    "profile-update-online", "quantitative-financial-analysis", "reimbursement-form-filler",
    "sales-accounting", "search-ca-school", "set-conf-cr-ddl", "shopping-helper",
    "sla-timeout-monitor", "stock-build-position", "student-interview", "subway-planning",
    "sync-todo-to-readme", "task-tracker", "train-ticket-plan", "travel-exchange",
    "travel-expense-reimbursement", "trip-adviser", "trip-itinerary-generator",
    "university-course-selection", "update-material-inventory", "upenn-campus-route",
    "verl-dataset", "vlm-history-completer", "wandb-best-score", "wandb-shortest-length",
    "woocommerce-customer-survey", "woocommerce-new-product", "woocommerce-new-welcome",
    "woocommerce-product-recall", "woocommerce-stock-alert", "woocommerce-update-cover",
    "yahoo-analysis", "youtube-repo",
]


async def fetch_raw(client: httpx.AsyncClient, path: str, retries: int = 3) -> str | None:
    """Fetch raw file content from GitHub."""
    url = f"{GITHUB_RAW_BASE}/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/{path}"
    for attempt in range(retries):
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403 and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(5)
    return None


async def fetch_task(client: httpx.AsyncClient, task_id: str) -> dict[str, Any] | None:
    """Fetch task definition including tools needed."""
    base = f"tasks/finalpool/{task_id}"
    
    # Get prompts
    system_prompt = await fetch_raw(client, f"{base}/docs/agent_system_prompt.md")
    task_desc = await fetch_raw(client, f"{base}/docs/task.md")
    
    if not task_desc:
        print(f"  Warning: No task.md for {task_id}")
        return None
    
    # Get task config for tools
    config_raw = await fetch_raw(client, f"{base}/task_config.json")
    config = json.loads(config_raw) if config_raw else {}
    
    return {
        "task_id": task_id,
        "system_prompt": system_prompt or "",
        "task_description": task_desc,
        "mcp_servers": config.get("needed_mcp_servers", []),
        "local_tools": config.get("needed_local_tools", []),
    }


def build_row(task: dict[str, Any]) -> dict[str, Any]:
    """Build dataset row in verifiers format."""
    messages = []
    
    if task["system_prompt"]:
        messages.append({"role": "system", "content": task["system_prompt"]})
    
    messages.append({"role": "user", "content": task["task_description"]})
    
    return {
        "task_id": task["task_id"],
        "prompt": messages,
        "info": {
            "task_id": task["task_id"],
            "mcp_servers": task.get("mcp_servers", []),
            "local_tools": task.get("local_tools", []),
        },
    }


def select_subset(all_tasks: list[str], n: int) -> list[str]:
    """Select first n tasks (simple)."""
    return all_tasks[:n]


async def main(args):
    """Extract Toolathlon tasks and create dataset."""
    print("="*60)
    print("Toolathlon Dataset Extraction")
    print("="*60)
    
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {}
    if args.github_token:
        headers["Authorization"] = f"token {args.github_token}"
    
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        # Select tasks
        if args.all:
            selected = ALL_TASKS
        else:
            selected = select_subset(ALL_TASKS, args.subset)
        
        print(f"\nFetching {len(selected)} tasks...")
        
        # Fetch task prompts
        tasks = []
        for i, task_id in enumerate(selected):
            print(f"[{i+1}/{len(selected)}] {task_id}")
            task = await fetch_task(client, task_id)
            if task:
                tasks.append(task)
            
            if (i + 1) % 10 == 0:
                await asyncio.sleep(1)
        
        print(f"\n✓ Fetched {len(tasks)}/{len(selected)} tasks")
    
    # Build dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = [build_row(t) for t in tasks]
    dataset = Dataset.from_list(rows)
    
    # Save
    dataset_path = output_dir / "tool_decathlon_dataset"
    dataset.save_to_disk(str(dataset_path))
    print(f"✓ Saved dataset to {dataset_path}")
    
    json_path = output_dir / "tool_decathlon_tasks.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"✓ Saved JSON to {json_path}")
    
    # Summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total tasks: {len(rows)}")
    print(f"\nSample tasks:")
    for row in rows[:5]:
        print(f"  - {row['task_id']}")
    
    print("\nNOTE: Everything else (MCPs, eval, tools) handled by Toolathlon in sandboxes")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Toolathlon dataset")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--subset", type=int, default=30, help="Number of tasks (default: 30)")
    parser.add_argument("--all", action="store_true", help="Include all 108 tasks")
    parser.add_argument("--github-token", help="GitHub token for rate limits")
    
    args = parser.parse_args()
    asyncio.run(main(args))

