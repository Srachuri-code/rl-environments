#!/bin/bash
# =============================================================================
# Setup Credentials for Toolathlon MCP Servers
# =============================================================================
# This script guides you through configuring credentials for Toolathlon's
# MCP servers that require external API access.
#
# Required for tasks using these MCP servers:
#   - google-cloud: Google Cloud API (BigQuery, Storage, etc.)
#   - github: GitHub API
#   - google_sheet: Google Sheets API
#   - google_forms: Google Forms API
#   - google_calendar: Google Calendar API
#   - notion: Notion API
#   - snowflake: Snowflake database
#   - huggingface: HuggingFace Hub
#   - wandb: Weights & Biases
#   - emails: Email (via Poste.io local server)
#   - woocommerce: WooCommerce (local server)
#   - canvas: Canvas LMS (local server)
#
# Usage:
#   ./setup_credentials.sh
#
# Output:
#   - toolathlon-server/configs/token_key_session.py
#   - toolathlon-server/configs/google_credentials.json
#   - ~/.mcp-auth/ (MCP authentication data)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLATHLON_DIR="$PROJECT_ROOT/toolathlon-server"
CONFIGS_DIR="$TOOLATHLON_DIR/configs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}\n"; }

echo "============================================================"
echo "Toolathlon Credentials Setup"
echo "============================================================"
echo ""
echo "This script will guide you through setting up credentials"
echo "for Toolathlon's MCP servers."
echo ""

# Check if toolathlon-server exists
if [[ ! -d "$TOOLATHLON_DIR" ]]; then
    log_info "Cloning Toolathlon repository..."
    git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git "$TOOLATHLON_DIR"
fi

# Ensure configs directory exists
mkdir -p "$CONFIGS_DIR"

# Copy example configs if not exists
if [[ ! -f "$CONFIGS_DIR/global_configs.py" ]]; then
    log_info "Creating global_configs.py from example..."
    cp "$CONFIGS_DIR/global_configs_example.py" "$CONFIGS_DIR/global_configs.py"
fi

if [[ ! -f "$CONFIGS_DIR/token_key_session.py" ]]; then
    log_info "Creating token_key_session.py from example..."
    cp "$CONFIGS_DIR/token_key_session_example.py" "$CONFIGS_DIR/token_key_session.py"
fi

log_section "Credential Configuration Options"

echo "Toolathlon tasks require different credentials depending on which"
echo "MCP servers they use. Here are your options:"
echo ""
echo "1. RUN WITHOUT EXTERNAL CREDENTIALS"
echo "   - Use only tasks that don't require external APIs"
echo "   - Works with: filesystem, terminal, pdf-tools, excel, howtocook, etc."
echo "   - Dataset: tool_decathlon_dataset_no_creds (8 tasks)"
echo ""
echo "2. CONFIGURE GOOGLE CLOUD (most tasks)"
echo "   - Required for: google-cloud, google_sheet, google_forms, google_calendar"
echo "   - See: https://github.com/hkust-nlp/Toolathlon/blob/main/global_preparation/how2register_accounts.md"
echo ""
echo "3. CONFIGURE GITHUB"
echo "   - Required for: github MCP server"
echo "   - Create token at: https://github.com/settings/tokens"
echo ""
echo "4. CONFIGURE OTHER SERVICES"
echo "   - Notion: https://www.notion.so/my-integrations"
echo "   - HuggingFace: https://huggingface.co/settings/tokens"
echo "   - Wandb: https://wandb.ai/authorize"
echo "   - Snowflake: Your Snowflake account credentials"
echo ""

log_section "Quick Setup (Minimal Credentials)"

echo "For quick testing, you can set environment variables:"
echo ""
echo "  export GITHUB_TOKEN='your-github-token'"
echo "  export SERPER_API_KEY='your-serper-key'  # For web search"
echo ""

log_section "Full Setup (Edit Config File)"

echo "For full access to all tasks, edit:"
echo "  $CONFIGS_DIR/token_key_session.py"
echo ""
echo "Fill in the API keys and credentials as documented in:"
echo "  https://github.com/hkust-nlp/Toolathlon/blob/main/global_preparation/how2register_accounts.md"
echo ""

log_section "Google OAuth Setup (Required for Google APIs)"

echo "Many tasks use Google services. To set up Google OAuth:"
echo ""
echo "1. Go to Google Cloud Console: https://console.cloud.google.com"
echo "2. Create a new project or select existing"
echo "3. Enable APIs: Sheets, Forms, Calendar, Cloud Storage, BigQuery"
echo "4. Create OAuth credentials (Desktop application)"
echo "5. Download credentials.json"
echo "6. Copy to: $CONFIGS_DIR/google_credentials.json"
echo ""
echo "Then run Toolathlon's Google setup:"
echo "  cd $TOOLATHLON_DIR"
echo "  uv run python global_preparation/simple_google_auth.py"
echo ""

log_section "Next Steps"

echo "1. Edit credentials in: $CONFIGS_DIR/token_key_session.py"
echo ""
echo "2. (Optional) Set up Google OAuth credentials"
echo ""
echo "3. (Optional) Deploy local services for canvas/woocommerce/email tasks:"
echo "   bash scripts/deploy_services.sh"
echo ""
echo "4. Build Docker image:"
echo "   bash scripts/build_toolathlon_image.sh"
echo ""
echo "5. Run evaluation:"
echo "   vf-eval tool_decathlon -m gpt-4.1-mini -n 1"
echo ""
echo "============================================================"

