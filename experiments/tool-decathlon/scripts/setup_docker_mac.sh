#!/bin/bash
# =============================================================================
# Setup Docker on Mac using Colima (Free Alternative to Docker Desktop)
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "============================================================"
echo "Setting Up Docker on Mac (via Colima)"
echo "============================================================"
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

log_info "Installing Docker CLI and Colima..."
brew install docker colima

log_info "Starting Colima (Docker runtime)..."
colima start --cpu 4 --memory 8 --disk 50

log_info "Verifying Docker works..."
docker ps

echo ""
echo "============================================================"
echo "Docker Setup Complete!"
echo "============================================================"
echo ""
echo "Docker is now running via Colima."
echo ""
echo "Next steps:"
echo "  1. Build Toolathlon image:"
echo "     cd /Users/skrachur/Desktop/rl-environments/tool-decathlon"
echo "     ./scripts/build_toolathlon_image.sh"
echo ""
echo "  2. Test the environment:"
echo "     vf-eval tool-decathlon -m gpt-4o-mini -n 1"
echo ""
echo "Manage Colima:"
echo "  - Stop: colima stop"
echo "  - Start: colima start"
echo "  - Status: colima status"
echo "  - Delete: colima delete"
echo ""
echo "============================================================"
