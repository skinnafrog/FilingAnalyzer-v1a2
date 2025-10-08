#!/bin/bash

# Clean restart script - stops all services and starts fresh
# This ensures no port conflicts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Clean Restart - Financial Intelligence Platform${NC}"
echo -e "${CYAN}========================================${NC}"

# Kill any processes on the key ports
echo -e "\n${YELLOW}Cleaning up ports...${NC}"

# Port 8080 (old frontend)
if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null; then
    echo "Stopping process on port 8080..."
    kill -9 $(lsof -Pi :8080 -sTCP:LISTEN -t) 2>/dev/null
fi

# Port 3000 (frontend)
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null; then
    echo "Stopping process on port 3000..."
    kill -9 $(lsof -Pi :3000 -sTCP:LISTEN -t) 2>/dev/null
fi

# Port 8000 (backend API)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
    echo "Stopping process on port 8000..."
    kill -9 $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null
fi

echo -e "${GREEN}✓ Ports cleaned${NC}"

# Stop any running Python servers
echo -e "\n${YELLOW}Stopping any running Python servers...${NC}"
pkill -f "python.*http.server" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -f "celery" 2>/dev/null

# Stop Docker containers if running
echo -e "\n${YELLOW}Checking Docker containers...${NC}"
if docker ps | grep -q financial_intel; then
    echo "Stopping Financial Intelligence Docker containers..."
    docker-compose stop 2>/dev/null
fi

# Wait for everything to stop
echo -e "\n${YELLOW}Waiting for services to stop completely...${NC}"
sleep 3

# Now start fresh
echo -e "\n${GREEN}Starting services fresh...${NC}"
echo -e "${GREEN}========================================${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the start script
"$SCRIPT_DIR/start.sh"

echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}Clean restart complete!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${GREEN}Frontend is now available at: http://localhost:3000${NC}"
echo -e "${GREEN}Backend API is available at: http://localhost:8000${NC}"
echo -e "${GREEN}API Documentation at: http://localhost:8000/docs${NC}"