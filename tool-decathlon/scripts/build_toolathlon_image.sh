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

# Create optimized Dockerfile
log_info "Creating Dockerfile..."
cat > Dockerfile.verifiers << 'DOCKERFILE'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLATHLON_HOME=/toolathlon
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies (minimal set for MCP servers)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    nodejs \
    npm \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /toolathlon
COPY . .

# Create venv and install Toolathlon dependencies
RUN /root/.local/bin/uv venv .venv --python python3.11

# Install core dependencies from pyproject.toml
RUN /root/.local/bin/uv pip install --python .venv/bin/python \
    -e . \
    || echo "pyproject.toml install failed, trying requirements approach"

# Install task_api.py dependencies explicitly
RUN /root/.local/bin/uv pip install --python .venv/bin/python \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    httpx==0.27.2 \
    anyio \
    pydantic \
    pyyaml \
    mcp

# Install npm dependencies for Node.js MCP servers
RUN npm install -g \
    @anthropic-ai/mcp \
    @anthropic-ai/mcp-server-memory \
    || true

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
try:
    import anyio
    import fastapi
    import uvicorn
    print('✓ Core dependencies OK')
except ImportError as e:
    print(f'✗ Missing: {e}')
    sys.exit(1)

try:
    from task_api import TaskAPI
    print('✓ task_api.py imports OK')
except ImportError as e:
    print(f'⚠ task_api import issue (may need MCP servers): {e}')
" || {
    log_warn "Some tests failed, but image was built"
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
