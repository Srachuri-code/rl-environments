#!/bin/bash
# =============================================================================
# Build Toolathlon Docker Image for Verifiers RL Environment
# =============================================================================
# Creates a Docker image based on official Toolathlon pre-built image
# with task_api.py added for verifiers integration.
#
# The base image (lockon0927/toolathlon-task-image:1016beta) includes:
#   - All 34 MCP servers pre-installed
#   - Playwright with Chromium browser
#   - kubectl, kind, helm for K8s tasks
#   - All Python/Node.js dependencies
#
# Usage:
#   ./build_toolathlon_image.sh
#
# Output:
#   Docker image: toolathlon:latest
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/.docker-build"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Base image from Toolathlon (includes all MCP servers)
BASE_IMAGE="lockon0927/toolathlon-task-image:1016beta"

echo "============================================================"
echo "Building Toolathlon Docker Image for Verifiers"
echo "============================================================"
echo ""
echo "Base image: $BASE_IMAGE"
echo ""

# Pull the base image first
log_info "Pulling official Toolathlon image..."
docker pull "$BASE_IMAGE"

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy task_api.py to build context
log_info "Preparing build context..."
cp "$PROJECT_ROOT/docker/task_api.py" "$BUILD_DIR/task_api.py"

# Create Dockerfile that extends the official image
log_info "Creating Dockerfile..."
cat > "$BUILD_DIR/Dockerfile" << 'DOCKERFILE'
# Use official Toolathlon image as base (includes all MCP servers)
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Clone Toolathlon source code to get utils/, configs/, tasks/
# The base image has MCP tools but not the source code
RUN git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git /tmp/toolathlon && \
    cp -r /tmp/toolathlon/utils /workspace/utils && \
    cp -r /tmp/toolathlon/configs /workspace/configs && \
    cp -r /tmp/toolathlon/tasks /workspace/tasks && \
    cp -r /tmp/toolathlon/local_binary /workspace/local_binary && \
    rm -rf /tmp/toolathlon

# Create config files from examples (required for imports)
# These will be overwritten at runtime with real credentials if provided
RUN cp /workspace/configs/global_configs_example.py /workspace/configs/global_configs.py && \
    cp /workspace/configs/token_key_session_example.py /workspace/configs/token_key_session.py

# Add __init__.py to make configs a proper Python package
RUN touch /workspace/configs/__init__.py

# Add task_api.py for verifiers integration
COPY task_api.py /workspace/task_api.py

# Ensure task_api dependencies are available in the venv
RUN . .venv/bin/activate && pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    httpx==0.27.0

# Create agent workspace directory
RUN mkdir -p /workspace/agent_workspace

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
DOCKERFILE

# Build the image
log_info "Building Docker image..."
cd "$BUILD_DIR"
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t toolathlon:latest \
    -t toolathlon:verifiers \
    .

log_info "✓ Image built: toolathlon:latest"

# Test the image
log_info "Testing image..."
docker run --rm toolathlon:latest bash -c "
echo '=== Testing Toolathlon Image ==='

# Activate venv
source .venv/bin/activate

# Test core imports
python3 -c '
import sys
sys.path.insert(0, \"/workspace\")

# Test task_api
try:
    from task_api import TaskAPI
    print(\"✓ task_api.py OK\")
except ImportError as e:
    print(f\"✗ task_api import failed: {e}\")
    sys.exit(1)

# Test MCP imports
try:
    from utils.mcp.tool_servers import MCPServerManager
    print(\"✓ MCPServerManager OK\")
except ImportError as e:
    print(f\"⚠ MCPServerManager: {e}\")

# Test configs
try:
    from configs.global_configs import global_configs
    print(\"✓ Toolathlon configs OK\")
except ImportError as e:
    print(f\"⚠ Config import: {e}\")

# Test FastAPI
try:
    import fastapi
    import uvicorn
    print(\"✓ FastAPI/Uvicorn OK\")
except ImportError as e:
    print(f\"✗ FastAPI import failed: {e}\")
    sys.exit(1)

print()
print(\"✓ All critical checks passed!\")
'

# Test MCP server tools are installed
echo ''
echo '=== Checking MCP Tools ==='
which pdf-tools-mcp >/dev/null 2>&1 && echo '✓ pdf-tools-mcp' || echo '⚠ pdf-tools-mcp not in PATH'
which emails-mcp >/dev/null 2>&1 && echo '✓ emails-mcp' || echo '⚠ emails-mcp not in PATH'

# Test Node.js MCP servers
echo ''
echo '=== Checking Node.js ==='
node --version
npm --version

# Test Playwright
echo ''
echo '=== Checking Playwright ==='
python3 -c 'from playwright.sync_api import sync_playwright; print(\"✓ Playwright OK\")'

# Test kubectl/kind
echo ''
echo '=== Checking K8s Tools ==='
which kubectl >/dev/null 2>&1 && echo '✓ kubectl' || echo '⚠ kubectl not found'
which kind >/dev/null 2>&1 && echo '✓ kind' || echo '⚠ kind not found'
which helm >/dev/null 2>&1 && echo '✓ helm' || echo '⚠ helm not found'

echo ''
echo '=== Image Test Complete ==='
" || {
    log_warn "Some tests failed - check output above"
}

# Cleanup build directory
cd "$PROJECT_ROOT"
rm -rf "$BUILD_DIR"

echo ""
echo "============================================================"
echo "Toolathlon Docker Image Ready"
echo "============================================================"
echo ""
echo "Image: toolathlon:latest"
echo "Base:  $BASE_IMAGE"
echo ""
echo "IMPORTANT: Before running evaluations, you must:"
echo ""
echo "1. Configure credentials (see tool-decathlon/configs/README.md)"
echo "   - Google Cloud OAuth credentials"
echo "   - GitHub personal access token"
echo "   - Other API keys as needed"
echo ""
echo "2. Deploy local services (for tasks requiring them):"
echo "   cd tool-decathlon && bash scripts/deploy_services.sh"
echo ""
echo "3. Run evaluation:"
echo "   vf-eval tool_decathlon -m gpt-4.1-mini -n 1"
echo ""
echo "============================================================"
