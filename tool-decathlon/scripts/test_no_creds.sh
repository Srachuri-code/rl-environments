#!/bin/bash
# =============================================================================
# Quick Test Script - No Credentials Required
# =============================================================================
# Tests the Tool Decathlon environment on the 7 tasks that don't need credentials
#
# Usage:
#   ./scripts/test_no_creds.sh          # Quick test (1 task)
#   ./scripts/test_no_creds.sh full     # All 7 tasks
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================================"
echo "Tool Decathlon - Quick Test (No Credentials)"
echo "============================================================"
echo ""

# Check if Docker is running
log_info "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running"
    echo ""
    echo "Please start Docker first:"
    echo "  colima start --cpu 4 --memory 8"
    echo ""
    echo "Or start Docker Desktop if you're using that."
    exit 1
fi
log_info "✓ Docker is running"

# Check if image exists
log_info "Checking for toolathlon:latest image..."
if ! docker image inspect toolathlon:latest > /dev/null 2>&1; then
    log_warn "Image not found. Building it now..."
    cd "$PROJECT_ROOT"
    ./scripts/build_toolathlon_image.sh
else
    log_info "✓ Image found"
fi

# Check if environment is installed
log_info "Checking tool-decathlon package..."
if ! python3 -c "import tool_decathlon" 2>/dev/null; then
    log_warn "Package not installed. Installing now..."
    cd "$PROJECT_ROOT"
    uv pip install -e environments/tool_decathlon
fi
log_info "✓ Package installed"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    log_error "OPENAI_API_KEY not set"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY=sk-..."
    echo ""
    exit 1
fi
log_info "✓ API key found"

# Determine test mode
TEST_MODE="${1:-quick}"
if [ "$TEST_MODE" = "full" ]; then
    NUM_TASKS=7
    NUM_ROLLOUTS=2
    log_info "Running FULL test (7 tasks × 2 rollouts)"
else
    NUM_TASKS=1
    NUM_ROLLOUTS=1
    log_info "Running QUICK test (1 task × 1 rollout)"
fi

echo ""
echo "============================================================"
echo "Starting Evaluation"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

# Run evaluation
vf-eval tool-decathlon \
  -m gpt-4o-mini \
  -n $NUM_TASKS \
  -r $NUM_ROLLOUTS \
  --save \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}' \
  || {
    log_error "Evaluation failed"
    echo ""
    echo "Check logs above for errors. Common issues:"
    echo "  - Container initialization timeout"
    echo "  - Runtime server failed to start"
    echo "  - Tool execution errors"
    echo ""
    exit 1
  }

echo ""
echo "============================================================"
echo "Test Complete!"
echo "============================================================"
echo ""
log_info "Results saved to: environments/tool_decathlon/outputs/"
echo ""
echo "Next steps:"
echo "  - Check outputs/ for full trajectories"
echo "  - If successful, try: ./scripts/test_no_creds.sh full"
echo "  - For all 108 tasks, follow COMPLETE_SETUP_GUIDE.md"
echo ""
