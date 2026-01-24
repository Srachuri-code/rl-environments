#!/bin/bash
# Test with automatic log viewing on failure

set +e  # Don't exit on error

cd /Users/skrachur/Desktop/rl-environments/tool-decathlon

# Check if image exists
if ! docker image inspect toolathlon:latest > /dev/null 2>&1; then
    echo "⚠️  Image toolathlon:latest not found. Building it now..."
    ./scripts/build_toolathlon_image.sh
    if [ $? -ne 0 ]; then
        echo "❌ Image build failed"
        exit 1
    fi
fi

echo "✓ Docker image ready"
echo ""
echo "Starting vf-eval..."
echo ""

vf-eval tool-decathlon \
  -m gpt-4.1-mini \
  -n 1 \
  -r 1 \
  -a '{"dataset_path": "data/tool_decathlon_dataset_minimal"}'

EXIT_CODE=$?

echo ""
echo "==================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Evaluation failed (exit code: $EXIT_CODE)"
    echo ""
    echo "Fetching runtime server logs from container..."
    echo ""
    
    CONTAINER=$(docker ps -a --format "{{.Names}}" | grep toolathlon | head -1)
    if [ -n "$CONTAINER" ]; then
        echo "Container: $CONTAINER"
        echo ""
        echo "=== Runtime Server Logs ===" 
        docker exec $CONTAINER cat /tmp/runtime.log 2>/dev/null || echo "No runtime.log found"
        echo ""
        echo "=== Container Stdout/Stderr ==="
        docker logs $CONTAINER 2>&1 | tail -50
    else
        echo "No toolathlon container found"
    fi
else
    echo "✅ Evaluation completed successfully!"
fi

exit $EXIT_CODE
