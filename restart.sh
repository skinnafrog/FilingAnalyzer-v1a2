#!/bin/bash

# Financial Intelligence Platform - Restart Script
# This script restarts all services (stop then start)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Restarting Financial Intelligence Platform${NC}"
echo -e "${CYAN}========================================${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Stop all services
echo -e "\n${YELLOW}Phase 1: Stopping services...${NC}"
echo -e "${YELLOW}----------------------------------------${NC}"

# Use 'yes n' to automatically answer 'n' to the Docker services prompt
yes n | "$SCRIPT_DIR/stop.sh"

# Wait a moment for everything to stop cleanly
echo -e "\n${YELLOW}Waiting for services to stop completely...${NC}"
sleep 3

# Start all services
echo -e "\n${YELLOW}Phase 2: Starting services...${NC}"
echo -e "${YELLOW}----------------------------------------${NC}"
"$SCRIPT_DIR/start.sh"

echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}Restart complete!${NC}"
echo -e "${CYAN}========================================${NC}"