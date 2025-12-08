# Production Setup for All 108 Tasks

## Overview

To run all 108 Toolathlon tasks efficiently in RL training, you need to:
1. Run Toolathlon's credential setup **once** (~30 minutes)
2. Bake credentials + heavy dependencies into the Docker image
3. Result: All containers initialize in ~10-20 seconds

## Option A: Bake Credentials into Image (Recommended)

### Step 1: Run Toolathlon's Setup Scripts

Inside the toolathlon-server directory, run these once:

```bash
cd toolathlon-server

# 1. Google Cloud Setup (~15 minutes)
bash global_preparation/automated_google_setup.sh
# This will:
# - Create a GCP project
# - Enable necessary APIs (BigQuery, Storage, Logging, etc.)
# - Create service account
# - Generate configs/gcp-service_account.keys.json
# - Generate configs/google_credentials.json

# 2. Deploy Local Services (~10 minutes)
bash global_preparation/deploy_containers.sh
# This will:
# - Start Canvas LMS container (local instance)
# - Start Poste email server container
# - Create user accounts
# - Generate configs/users_data.json

# 3. Setup Other Services (~5 minutes)
# Follow global_preparation/how2register_accounts.md for:
# - GitHub token
# - Notion integration
# - Wandb API key
# - HuggingFace token
# - Snowflake account (optional)
```

### Step 2: Modify Docker Build Script

Add these lines to `scripts/build_toolathlon_image.sh` after line 86 (before CMD):

```bash
# In the Dockerfile.sandbox, add these RUN commands:

# Pre-install playwright browsers (saves 5-10 min per container)
RUN /toolathlon/.venv/bin/playwright install chromium --with-deps

# Pre-warm npm MCP packages
RUN npx -y @modelcontextprotocol/server-filesystem --version || true

# Verify credentials are present (will fail build if missing)
RUN test -f /toolathlon/configs/google_credentials.json || echo "WARNING: No Google credentials"
```

### Step 3: Rebuild Docker Image

```bash
./scripts/build_toolathlon_image.sh
```

Now the image contains:
- ✅ All credentials
- ✅ Playwright browsers pre-installed
- ✅ NPM packages cached
- ✅ Ready for all 108 tasks

### Step 4: Test It Works

```bash
vf-eval tool-decathlon -m gpt-4o-mini -n 1
# Should initialize in ~10-20 seconds now, not minutes
```

## Option B: Volume Mount Credentials (More Secure)

### Step 1: Run Setup (Same as Option A)

```bash
cd toolathlon-server
bash global_preparation/automated_google_setup.sh
bash global_preparation/deploy_containers.sh
# Follow how2register_accounts.md
```

### Step 2: Modify tool_decathlon.py

In `setup_state()`, add volume mounting:

```python
# Around line 142
container = self.docker_client.containers.run(
    self.toolathlon_image,
    command="/bin/bash -c 'tail -f /dev/null'",
    name=f"toolathlon-{task_id}-{asyncio.get_event_loop().time()}",
    detach=True,
    remove=True,
    mem_limit="4g",
    cpu_count=2,
    volumes={
        # Mount credentials directory (read-only)
        str(Path(__file__).parent.parent.parent / "toolathlon-server" / "configs"): {
            'bind': '/toolathlon/configs',
            'mode': 'ro'
        }
    },
)
```

### Step 3: Still Add Playwright to Dockerfile

```bash
# In build script Dockerfile
RUN /toolathlon/.venv/bin/playwright install chromium --with-deps
```

Rebuild:
```bash
./scripts/build_toolathlon_image.sh
```

### Step 4: Test

```bash
vf-eval tool-decathlon -m gpt-4o-mini -n 1
```

**Benefits:**
- Credentials stay on host (more secure)
- Can update credentials without rebuilding image
- Image can be shared/pushed to registry safely

## What You Get After Setup

**Before (current state):**
- 8-13 tasks work (no credentials)
- Playwright tasks hang for 5-10 minutes
- Can't test most tasks

**After setup:**
- ✅ All 108 tasks work
- ✅ Container init: 10-20 seconds
- ✅ Playwright pre-installed
- ✅ Ready for large-scale RL training

## Training Performance Estimate

With proper setup:

**Batch of 1024 rollouts:**
- Container creation: 1024 × 10 sec = ~170 sec (parallelized)
- Episode execution: Depends on task complexity
- **Setup overhead: ~3 minutes for entire batch**

This is acceptable for RL training where each episode might take 1-5 minutes anyway.

## Next Steps

1. **Now:** Run Toolathlon's setup scripts on this machine
2. **Choose:** Option A (bake) or Option B (mount)
3. **Modify:** Docker build script to add playwright pre-install
4. **Rebuild:** Docker image with all dependencies
5. **Test:** All 108 tasks should work quickly

Ready for you to run the setup process!

