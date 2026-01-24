#!/bin/bash
# =============================================================================
# Deploy Local Services for Toolathlon Tasks
# =============================================================================
# Many Toolathlon tasks require local services (Canvas LMS, WooCommerce, 
# Poste.io email server, K8s clusters). This script deploys them.
#
# Services and ports:
#   - Canvas LMS:     10001, 20001
#   - Poste.io:       10005, 2525, 1143, 2587
#   - WooCommerce:    10003
#   - K8s (Kind):     various
#
# Usage:
#   ./deploy_services.sh [--with-sudo]
#
# Note: Some tasks work without these services. Check task_config.json
# for each task to see which MCP servers it needs.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLATHLON_DIR="$PROJECT_ROOT/toolathlon-server"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse args
WITH_SUDO=false
if [[ "$1" == "--with-sudo" ]]; then
    WITH_SUDO=true
fi

echo "============================================================"
echo "Deploying Toolathlon Local Services"
echo "============================================================"
echo ""
echo "This will deploy:"
echo "  - Canvas LMS (for canvas-* tasks)"
echo "  - WooCommerce (for woocommerce-* tasks)"
echo "  - Poste.io email server (for email tasks)"
echo "  - K8s test cluster (for k8s-* tasks)"
echo ""

# Check if toolathlon-server exists
if [[ ! -d "$TOOLATHLON_DIR" ]]; then
    log_warn "Toolathlon server directory not found at $TOOLATHLON_DIR"
    log_info "Cloning Toolathlon repository..."
    git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git "$TOOLATHLON_DIR"
fi

cd "$TOOLATHLON_DIR"

# Required ports
REQUIRED_PORTS=(10001 20001 10005 2525 1143 2587 10003 30123 30124 30137)

log_info "Checking and clearing required ports..."
for port in "${REQUIRED_PORTS[@]}"; do
    if lsof -i :$port -t >/dev/null 2>&1; then
        log_warn "Port $port is in use. Killing process..."
        pids=$(lsof -i :$port -t)
        for pid in $pids; do
            kill -9 $pid 2>/dev/null || true
        done
        sleep 1
    else
        echo "  Port $port is free"
    fi
done

echo ""
log_info "Deploying services..."

# Deploy K8s test cluster (optional, for k8s tasks)
if [[ -f "deployment/k8s/scripts/setup.sh" ]]; then
    log_info "Setting up K8s test cluster..."
    bash deployment/k8s/scripts/setup.sh || log_warn "K8s setup failed (non-fatal)"
fi

# Deploy Canvas LMS
if [[ -f "deployment/canvas/scripts/setup.sh" ]]; then
    log_info "Deploying Canvas LMS (ports 10001, 20001)..."
    bash deployment/canvas/scripts/setup.sh || log_warn "Canvas setup failed (non-fatal)"
fi

# Deploy Poste.io email server
if [[ -f "deployment/poste/scripts/setup.sh" ]]; then
    log_info "Deploying Poste.io email server (ports 10005, 2525, 1143, 2587)..."
    # Note: First arg is whether to configure dovecot for plaintext auth
    bash deployment/poste/scripts/setup.sh start true || log_warn "Poste setup failed (non-fatal)"
fi

# Deploy WooCommerce
if [[ -f "deployment/woocommerce/scripts/setup.sh" ]]; then
    log_info "Deploying WooCommerce (port 10003)..."
    bash deployment/woocommerce/scripts/setup.sh start 81 20 || log_warn "WooCommerce setup failed (non-fatal)"
fi

echo ""
echo "============================================================"
echo "Local Services Deployment Complete"
echo "============================================================"
echo ""
echo "Services should now be running at:"
echo "  - Canvas LMS:     http://localhost:20001"
echo "  - WooCommerce:    http://localhost:10003"
echo "  - Poste.io:       ports 10005, 2525, 1143, 2587"
echo ""
echo "To check status:"
echo "  docker ps"
echo ""
echo "Tasks that can run WITHOUT these services:"
echo "  - Tasks using only: filesystem, terminal, pdf-tools, excel,"
echo "    arxiv_local, scholarly, playwright, fetch, howtocook, etc."
echo ""
echo "============================================================"

