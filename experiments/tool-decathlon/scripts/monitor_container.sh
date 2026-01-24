#!/bin/bash
# Monitor what's happening inside a Toolathlon container

if [ -z "$1" ]; then
    # Find first toolathlon container
    CONTAINER=$(docker ps --format "{{.Names}}" | grep toolathlon | head -1)
else
    CONTAINER=$1
fi

if [ -z "$CONTAINER" ]; then
    echo "No toolathlon containers found"
    exit 1
fi

echo "=== Monitoring Container: $CONTAINER ==="
echo ""

echo "📊 Running Processes:"
docker exec $CONTAINER ps aux | grep -E "python|node|npx|uv" | grep -v grep
echo ""

echo "📦 npm Cache Size (indicates downloads):"
docker exec $CONTAINER bash -c "du -sh /root/.npm 2>/dev/null || echo 'No npm cache yet'"
echo ""

echo "💾 UV Cache Size:"
docker exec $CONTAINER bash -c "du -sh /root/.cache/uv 2>/dev/null || echo 'No uv cache yet'"
echo ""

echo "📁 Workspace Contents:"
docker exec $CONTAINER bash -c "ls -lah /toolathlon/agent_workspace/*/ 2>/dev/null | head -20 || echo 'No workspace yet'"
echo ""

echo "🔄 task_api.py status:"
if docker exec $CONTAINER ps aux | grep -q "task_api.py setup"; then
    echo "  ⏳ Still running setup..."
else
    echo "  ✅ Setup complete (or not started)"
fi
echo ""

echo "📝 Container Logs (last 10 lines):"
docker logs $CONTAINER 2>&1 | tail -10
echo ""

echo "====================================="
echo "Run: watch -n 2 $0 $CONTAINER"
echo "For continuous monitoring"



