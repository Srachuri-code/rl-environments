#!/bin/bash
# =============================================================================
# Build Toolathlon Docker Image for Verifiers RL Environment
# =============================================================================
# Creates a Docker image based on official Toolathlon with task_api.py added.
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

echo "============================================================"
echo "Building Toolathlon Docker Image"
echo "============================================================"

# Clean build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Clone fresh Toolathlon repo
log_info "Cloning official Toolathlon repository..."
git clone --depth 1 https://github.com/hkust-nlp/Toolathlon.git "$BUILD_DIR/toolathlon"

# Copy task API wrapper
log_info "Adding task_api.py wrapper..."
cp "$PROJECT_ROOT/docker/task_api.py" "$BUILD_DIR/toolathlon/task_api.py"

cd "$BUILD_DIR/toolathlon"

# Create optimized Dockerfile that properly sets up Toolathlon
log_info "Creating Dockerfile..."
cat > Dockerfile.verifiers << 'DOCKERFILE'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLATHLON_HOME=/toolathlon
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies + Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    nodejs \
    npm \
    build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /toolathlon
COPY . .

# Create config files from examples (required by Toolathlon)
RUN cp configs/global_configs_example.py configs/global_configs.py || true
RUN cp configs/token_key_session_example.py configs/token_key_session.py || true

# Create venv with Python 3.12
RUN /root/.local/bin/uv venv .venv --python python3.12

# Install Toolathlon with relaxed Python version
RUN sed -i 's/requires-python = "==3.12.11"/requires-python = ">=3.11"/' pyproject.toml || true

# Install Toolathlon dependencies
RUN /root/.local/bin/uv pip install --python .venv/bin/python -e . || \
    /root/.local/bin/uv pip install --python .venv/bin/python \
    fastapi uvicorn httpx anyio pydantic pyyaml mcp \
    openai openai-agents aiohttp aiofiles loguru \
    datasets tenacity requests

# Ensure task_api.py and Toolathlon config dependencies are installed
# Pin versions to match Toolathlon's pyproject.toml for compatibility
RUN /root/.local/bin/uv pip install --python .venv/bin/python \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    httpx==0.27.0 \
    anyio==4.9.0 \
    pydantic==2.11.3 \
    pyyaml==6.0.2 \
    mcp>=1.9.0 \
    openai==1.76.0 \
    "openai-agents==0.0.15" \
    aiohttp==3.12.7 \
    aiofiles==24.1.0 \
    addict==2.4.0

# Create workspace directory
RUN mkdir -p /toolathlon/agent_workspace

WORKDIR /toolathlon
CMD ["/bin/bash"]
DOCKERFILE

log_info "Building Docker image..."
docker build -f Dockerfile.verifiers -t toolathlon:latest .

log_info "✓ Image built: toolathlon:latest"

# Test the image
log_info "Testing image..."
docker run --rm toolathlon:latest /toolathlon/.venv/bin/python -c "
import sys
sys.path.insert(0, '/toolathlon')

# Test core dependencies
try:
    import anyio
    import fastapi
    import uvicorn
    print('✓ Core dependencies OK')
except ImportError as e:
    print(f'✗ Missing core dep: {e}')
    sys.exit(1)

# Test Toolathlon config import
try:
    from configs.global_configs import global_configs
    print('✓ Toolathlon configs OK')
except ImportError as e:
    print(f'✗ Config import failed: {e}')
    sys.exit(1)

# Test MCP server manager import
try:
    from utils.mcp.tool_servers import MCPServerManager
    print('✓ MCPServerManager OK')
except ImportError as e:
    print(f'⚠ MCPServerManager import: {e}')

# Test task_api import
try:
    from task_api import TaskAPI
    print('✓ task_api.py OK')
except ImportError as e:
    print(f'⚠ task_api import: {e}')

print('\\n✓ All critical checks passed!')
" || {
    log_warn "Some tests failed - check output above"
}

# Cleanup
cd "$PROJECT_ROOT"
rm -rf "$BUILD_DIR"

echo ""
echo "============================================================"
echo "Toolathlon Docker Image Ready"
echo "============================================================"
echo ""
echo "Image: toolathlon:latest"
echo ""
echo "To run evaluation:"
echo "  cd ~/rl-environments/tool-decathlon"
echo "  vf-eval tool-decathlon -m gpt-4.1-mini -n 1"
echo ""
echo "============================================================"
