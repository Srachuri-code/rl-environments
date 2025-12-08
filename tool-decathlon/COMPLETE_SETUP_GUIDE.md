# Complete Toolathlon Setup Guide

## Overview

**Total time:** ~30-45 minutes one-time setup  
**What you'll get:** All 108 tasks working with credentials configured

## Account Requirements

### **Remote Accounts (Need to Register)**
1. **Google Account** (primary) - Use for multiple services
2. **GitHub Account**
3. **Weights & Biases (wandb) Account**
4. **Notion Account**
5. **Snowflake Account** (optional - only for 1-2 tasks)
6. **HuggingFace Account**
7. **Serper API Key** (for search)

### **Local Accounts (Auto-Created)**
- Canvas LMS users (created by deploy script)
- Email accounts (created by deploy script)
- WooCommerce accounts (created by deploy script)

## Step-by-Step Setup Process

### **Phase 1: Prerequisites (~5 minutes)**

#### 1. Install gcloud CLI

**Mac:**
```bash
# Download and install
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-darwin-arm.tar.gz
tar -xf google-cloud-cli-darwin-arm.tar.gz
./google-cloud-sdk/install.sh

# Restart terminal or:
source ~/.zshrc
```

**Linux:**
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-456.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-sdk-*-x86_64.tar.gz
./google-cloud-sdk/install.sh
source ~/.bashrc
```

#### 2. Create/Use Google Account

- Create a new Gmail account specifically for Toolathlon (recommended)
- Or use existing account
- Go to https://console.cloud.google.com/ and accept Terms of Service

### **Phase 2: Google Cloud Setup (~15 minutes)**

Navigate to toolathlon-server directory:
```bash
cd toolathlon-server
```

#### Run Automated Google Setup:
```bash
bash global_preparation/automated_google_setup.sh
```

**This script will:**
1. ✅ Create a new GCP project (or use existing)
2. ✅ Enable billing (requires credit card)
3. ✅ Enable required APIs (BigQuery, Storage, Logging, etc.)
4. ✅ Create service account
5. ✅ Generate service account keys → `configs/gcp-service_account.keys.json`
6. ✅ Create API key → saved to `token_key_session.py`
7. ⚠️ **Manual step:** Create OAuth client (Web application)
   - The script will open browser and guide you
   - Download `client_secret_*.json`
   - Rename to `gcp-oauth.keys.json`
   - Place in `configs/` directory
8. ✅ Run OAuth flow → generates `configs/google_credentials.json`

**What you'll need:**
- Credit card (for GCP billing - free tier available)
- Browser access for OAuth setup

**Output files:**
- `configs/gcp-service_account.keys.json`
- `configs/gcp-oauth.keys.json`
- `configs/google_credentials.json`
- Updated `configs/token_key_session.py` with GCP values

### **Phase 3: Deploy Local Services (~10 minutes)**

```bash
bash global_preparation/deploy_containers.sh
```

**This script will:**
1. ✅ Start Canvas LMS container (local instance)
2. ✅ Start Poste email server container
3. ✅ Create 100 user accounts
4. ✅ Register accounts to Canvas
5. ✅ Register accounts to Poste (email)
6. ✅ Create WooCommerce subsites for accounts #81-100
7. ✅ Generate `configs/users_data.json`

**Requirements:**
- Docker running (you have this with Colima)
- Ports 80, 443, 3000 available

**What you get:**
- Local Canvas LMS at http://localhost (or configured URL)
- Local email server
- 100 pre-configured test accounts

### **Phase 4: Other Services (~10 minutes)**

```bash
bash global_preparation/automated_additional_services.sh
```

**This script handles:**

#### GitHub:
- Prompts you to create a GitHub Personal Access Token
- Guide: https://github.com/settings/tokens/new
- Permissions needed: `repo`, `workflow`, `admin:org`
- Saves to `token_key_session.py`

#### Weights & Biases:
- Prompts you to get wandb API key
- Guide: https://wandb.ai/authorize
- Saves to `token_key_session.py`

#### HuggingFace:
- Prompts for HF token
- Guide: https://huggingface.co/settings/tokens
- Permissions: Write access
- Saves to `token_key_session.py`

#### Serper (Google Search API):
- Register at https://serper.dev/
- Get API key (free tier: 2500 searches/month)
- Saves to `token_key_session.py`

### **Phase 5: Notion Setup (~5 minutes, Optional)**

Only needed for 1 Notion task.

```bash
uv run -m global_preparation.special_setup_notion_official
```

**Manual steps:**
1. Create Notion workspace (https://www.notion.so/)
2. Duplicate this public page to your workspace:
   https://amazing-wave-b38.notion.site/Notion-Source-Page-27ad10a48436805b9179fdaff2f65be2
3. Create a new page called "Notion Eval Page"
4. Create two Notion integrations:
   - One with access to both pages → `notion_integration_key`
   - One with access to eval page only → `notion_integration_key_eval`
5. Fill in `token_key_session.py`:
   - `source_notion_page_url`
   - `eval_notion_page_url`
   - `notion_integration_key`
   - `notion_integration_key_eval`

### **Phase 6: Snowflake Setup (~5 minutes, Optional)**

Only needed for 1-2 Snowflake tasks.

1. Register at https://signup.snowflake.com/
2. Activate your account
3. Find account details in console
4. Fill in `token_key_session.py`:
   - `snowflake_account`
   - `snowflake_user`
   - `snowflake_password`
   - `snowflake_role`

## What Gets Created

After full setup, your `toolathlon-server/configs/` directory will have:

**Credential Files:**
- `gcp-service_account.keys.json` - GCP service account
- `gcp-oauth.keys.json` - OAuth client secrets
- `google_credentials.json` - OAuth tokens
- `users_data.json` - Canvas/email/WooCommerce accounts

**Configuration Files:**
- `token_key_session.py` - All API keys and tokens filled in
- `global_configs.py` - General configuration

## After Setup: Build Production Image

Once all credentials are configured:

```bash
cd /Users/skrachur/Desktop/rl-environments/tool-decathlon

# Copy credentials to build context
cp -r toolathlon-server/configs ./docker/configs_backup

# Modify build script to include credentials
# (or use volume mounting approach)

# Rebuild with pre-installed packages
./scripts/build_toolathlon_image.sh
```

## Verification

Test that all services work:

```bash
cd toolathlon-server
uv run python global_preparation/check_installation.py
```

This validates:
- ✅ All credentials are valid
- ✅ APIs are accessible
- ✅ Local services are running
- ✅ Ready for evaluation

## Tasks by Credential Requirement

### **No Credentials (7 tasks):**
- arrange-workspace, cooking-guidance, courses-ta-hws, detect-revised-terms, dietary-health, excel-data-transformation, excel-market-research

### **Google Cloud Only (3 tasks):**
- ab-testing, academic-warning, flagged-transactions

### **Canvas Only (8 tasks):**
- canvas-arrange-exam, canvas-art-manager, canvas-art-quiz, canvas-do-quiz, canvas-homework-grader-python, canvas-list-test, canvas-new-students-notification, canvas-submit-late-work

### **GitHub (2 tasks):**
- dataset-license-issue, email-paper-homepage

### **Multiple Services (10+ tasks):**
- Various tasks using combinations of services

## Cost Breakdown

### **Free Tier Available:**
- ✅ Google Cloud (free tier: $300 credit)
- ✅ GitHub (free for public repos)
- ✅ Wandb (free tier: unlimited projects)
- ✅ HuggingFace (free tier)
- ✅ Notion (free tier)
- ✅ Serper (free tier: 2500 searches/month)

### **Paid (Optional):**
- Snowflake (free trial, then ~$25/month minimum)

**Total cost for basic setup: $0** (using free tiers)

## Troubleshooting

**gcloud command not found:**
- Make sure you sourced ~/.zshrc or ~/.bashrc after installation

**GCP billing required:**
- Need credit card even for free tier
- Won't be charged unless you exceed free tier limits

**Canvas container won't start:**
- Check port 80/443 not in use
- May need to adjust port mappings

**OAuth flow fails:**
- Make sure you published the OAuth consent screen
- Check redirect URI is exactly: http://localhost:3000/oauth2callback

## Quick Start Path (Minimal Setup)

If you want to test quickly without full setup:

**Option 1: Just Google Cloud (unlocks 16 tasks)**
```bash
bash global_preparation/automated_google_setup.sh
# Skip other services
```

**Option 2: Google + Canvas (unlocks 24 tasks)**  
```bash
bash global_preparation/automated_google_setup.sh
bash global_preparation/deploy_containers.sh
```

**Option 3: Full setup (all 108 tasks)**
- Follow all phases above

