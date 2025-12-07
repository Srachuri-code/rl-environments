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

# Pull Toolathlon Docker image (for task containers)
RUN bash global_preparation/pull_toolathlon_image.sh || true

# Create workspace directory
RUN mkdir -p /toolathlon/agent_workspace

# Expose API for task management
COPY <<'PYTHON' /toolathlon/api.py
"""
Simple API to run Toolathlon tasks.
This is the interface that prime-sandboxes will call.
"""
import json
import sys
from pathlib import Path

def setup_task(task_id: str):
    """Setup a Toolathlon task (start MCPs, create workspace, etc)."""
    # Implementation will use Toolathlon's main.py
    from main import TaskRunner
    runner = TaskRunner(task_id)
    runner.setup()
    return {"status": "ready", "task_id": task_id}

def get_task_tools(task_id: str):
    """Get tools available for this task."""
    from main import TaskRunner
    runner = TaskRunner(task_id)
    return runner.get_tools()

def execute_tool(tool_name: str, args: dict):
    """Execute a tool call."""
    from main import TaskRunner
    runner = TaskRunner.current()  # Get active runner
    return runner.execute_tool(tool_name, args)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "setup":
        print(json.dumps(setup_task(sys.argv[2])))
    elif cmd == "tools":
        print(json.dumps(get_task_tools(sys.argv[2])))
    elif cmd == "execute":
        print(json.dumps(execute_tool(sys.argv[2], json.loads(sys.argv[3]))))
PYTHON

WORKDIR /toolathlon

# Default command
CMD ["python3", "api.py"]
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

