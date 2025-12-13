#!/bin/bash
# =============================================================================
# Build Toolathlon Docker Image for Prime Sandboxes
# =============================================================================
# Creates a Docker image with Toolathlon fully configured.
# This image is used by prime-sandboxes for isolated task execution.
#
# Usage:
#   ./build_toolathlon_image.sh
#
# Output:
#   Docker image: toolathlon:latest
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLATHLON_DIR="${SCRIPT_DIR}/../toolathlon-server"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "============================================================"
echo "Building Toolathlon Docker Image"
echo "============================================================"

# Clone Toolathlon if not exists
if [ ! -d "$TOOLATHLON_DIR" ]; then
    log_info "Cloning Toolathlon..."
    git clone https://github.com/hkust-nlp/Toolathlon.git "$TOOLATHLON_DIR"
fi

# Copy task API wrapper BEFORE cd
log_info "Adding task API wrapper..."
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cp "$PROJECT_ROOT/docker/task_api.py" "$TOOLATHLON_DIR/task_api.py"

cd "$TOOLATHLON_DIR"

# Create Dockerfile for sandboxes
log_info "Creating Dockerfile..."

cat > Dockerfile.sandbox << 'EOF'
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLATHLON_HOME=/toolathlon

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nodejs \
    npm \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone Toolathlon
WORKDIR /toolathlon
COPY . .

# Install dependencies
RUN bash global_preparation/install_env_minimal.sh false || true

# Install lightweight runtime server deps (for step-by-step RL tool execution)
# This avoids spawning a new Python process per tool call via `docker exec`.
RUN /root/.local/bin/uv pip install --python /toolathlon/.venv/bin/python fastapi==0.115.5 uvicorn==0.32.1 httpx==0.27.2 || true

# Pull Toolathlon Docker image (for task containers)
RUN bash global_preparation/pull_toolathlon_image.sh || true

# Create workspace directory
RUN mkdir -p /toolathlon/agent_workspace

WORKDIR /toolathlon
CMD ["/bin/bash"]
EOF

log_info "Building Docker image..."
docker build -f Dockerfile.sandbox -t toolathlon:latest .

log_info "✓ Image built: toolathlon:latest"

# Test the image
log_info "Testing image..."
docker run --rm toolathlon:latest python3 -c "print('Toolathlon image ready')" || {
    log_warn "Image test failed, but image was built"
}

echo ""
echo "============================================================"
echo "Toolathlon Docker Image Ready"
echo "============================================================"
echo ""
echo "Image: toolathlon:latest"
echo ""
echo "Next steps:"
echo "  1. Push to registry: docker push your-registry/toolathlon:latest"
echo "  2. Use in environment: load_environment(toolathlon_image='toolathlon:latest')"
echo ""
echo "============================================================"

