#!/bin/bash
# Diagnose which MCP servers work in the Toolathlon image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "============================================================"
echo "MCP Server Diagnostics"
echo "============================================================"
echo ""

# Check image exists
if ! docker image inspect toolathlon:latest > /dev/null 2>&1; then
    log_error "Image toolathlon:latest not found"
    echo "Run: ./scripts/build_toolathlon_image.sh"
    exit 1
fi

log_info "Starting test container..."
CONTAINER_ID=$(docker run -d --rm toolathlon:latest tail -f /dev/null)
CONTAINER_NAME=$(docker inspect --format='{{.Name}}' $CONTAINER_ID | sed 's/\///')

log_info "Container: $CONTAINER_NAME"

# Copy test script into container
log_info "Copying test script..."
docker cp "$SCRIPT_DIR/test_mcp_servers.py" $CONTAINER_ID:/toolathlon/test_mcp_servers.py

# Run the test
log_info "Running MCP server tests (this may take 5-10 minutes)..."
echo ""

docker exec $CONTAINER_ID /bin/bash -c "cd /toolathlon && /root/.local/bin/uv run python test_mcp_servers.py"

EXIT_CODE=$?

# Get results
echo ""
log_info "Retrieving results..."
docker cp $CONTAINER_ID:/tmp/mcp_test_results.json "$PROJECT_ROOT/mcp_test_results.json" 2>/dev/null || {
    log_warn "Could not retrieve results file"
}

# Cleanup
log_info "Cleaning up container..."
docker stop $CONTAINER_ID > /dev/null

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    log_info "Test complete!"
    if [ -f "$PROJECT_ROOT/mcp_test_results.json" ]; then
        echo ""
        echo "Results saved to: mcp_test_results.json"
        echo ""
        echo "Summary:"
        python3 -c "
import json
with open('$PROJECT_ROOT/mcp_test_results.json') as f:
    results = json.load(f)
working = [r for r in results if r['status'] == 'working']
print(f'  ✅ {len(working)} servers working')
print(f'  ❌ {len(results) - len(working)} servers broken')
"
    fi
else
    log_error "Test failed"
fi
echo "============================================================"
