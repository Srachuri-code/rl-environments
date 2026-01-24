#!/bin/bash
# Debug script to test runtime server startup manually

set -e

echo "=== Testing Runtime Server Manually ==="
echo ""

# Start container
echo "1. Starting test container..."
CONTAINER_ID=$(docker run -d -p 8000:8000 --name toolathlon-debug toolathlon:latest tail -f /dev/null)
echo "Container ID: $CONTAINER_ID"

# Test imports
echo ""
echo "2. Testing task_api.py imports..."
docker exec $CONTAINER_ID /bin/bash -c "cd /toolathlon && /root/.local/bin/uv run python -c 'import task_api; print(\"✓ Imports OK\")'"

# Try to start server
echo ""
echo "3. Starting runtime server..."
docker exec -d $CONTAINER_ID /bin/bash -c "cd /toolathlon && /root/.local/bin/uv run python task_api.py serve --host 0.0.0.0 --port 8000"

# Wait and check
echo ""
echo "4. Waiting for server to start (5 seconds)..."
sleep 5

# Check if server is responding
echo ""
echo "5. Testing /health endpoint..."
if curl -s http://localhost:8000/health; then
    echo "✓ Server is healthy!"
else
    echo "✗ Server not responding"
    echo ""
    echo "6. Checking server logs..."
    docker exec $CONTAINER_ID cat /tmp/runtime.log || echo "No log file"
    docker logs $CONTAINER_ID
fi

# Cleanup
echo ""
echo "7. Cleaning up..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo ""
echo "=== Test Complete ==="
