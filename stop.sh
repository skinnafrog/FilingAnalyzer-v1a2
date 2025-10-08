#!/bin/bash

# Financial Intelligence Platform - Stop Script
# This script stops all services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Stopping Financial Intelligence Platform${NC}"
echo -e "${YELLOW}========================================${NC}"

# Stop all Docker Compose services
echo -e "\n${YELLOW}Stopping Docker Compose services...${NC}"
docker-compose down

echo -e "\n${GREEN}All services stopped successfully!${NC}"
echo ""
echo -e "${YELLOW}To start services again, run: ./start.sh${NC}"