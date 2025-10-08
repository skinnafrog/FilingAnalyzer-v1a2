#!/bin/bash

# Financial Intelligence Platform - Stop Script
# This script stops all running services for the application

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Stopping Financial Intelligence Platform${NC}"
echo -e "${YELLOW}========================================${NC}"

# Function to check if a process is running
check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to stop a process gracefully
stop_process() {
    local process_name=$1
    local process_pattern=$2

    if check_process "$process_pattern"; then
        echo -e "${YELLOW}Stopping $process_name...${NC}"
        pkill -f "$process_pattern"

        # Wait for process to stop (max 5 seconds)
        local count=0
        while check_process "$process_pattern" && [ $count -lt 5 ]; do
            sleep 1
            count=$((count + 1))
        done

        # Force kill if still running
        if check_process "$process_pattern"; then
            echo -e "${YELLOW}Force stopping $process_name...${NC}"
            pkill -9 -f "$process_pattern"
            sleep 1
        fi

        echo -e "${GREEN}✓ $process_name stopped${NC}"
    else
        echo -e "${YELLOW}$process_name is not running${NC}"
    fi
}

# 1. Stop Frontend server
echo -e "\n${YELLOW}1. Stopping Frontend server...${NC}"

# Check if frontend is running in Docker
if docker ps | grep -q financial_intel_frontend; then
    echo -e "${YELLOW}Stopping Frontend Docker container...${NC}"
    docker-compose stop frontend 2>/dev/null
    echo -e "${GREEN}✓ Frontend container stopped${NC}"
else
    # Stop standalone Python server
    stop_process "Frontend server" "python3.*http.server.*3000"
fi

# 2. Stop Backend API server
echo -e "\n${YELLOW}2. Stopping Backend API server...${NC}"
stop_process "Backend API" "uvicorn.*src.api.main:app"

# 3. Stop Celery worker
echo -e "\n${YELLOW}3. Stopping Celery worker...${NC}"
stop_process "Celery worker" "celery.*worker"

# 4. Stop Celery beat
echo -e "\n${YELLOW}4. Stopping Celery beat scheduler...${NC}"
stop_process "Celery beat" "celery.*beat"

# 5. Remove celerybeat-schedule file if it exists
if [ -f "backend/celerybeat-schedule" ]; then
    echo -e "${YELLOW}Removing Celery beat schedule file...${NC}"
    rm backend/celerybeat-schedule
fi

# 6. Stop Docker services (optional - commented out by default)
echo -e "\n${YELLOW}5. Docker services...${NC}"
read -p "Do you want to stop Docker services (PostgreSQL, Redis, Neo4j)? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "docker-compose.yml" ]; then
        echo -e "${YELLOW}Stopping Docker services...${NC}"
        docker-compose down
        echo -e "${GREEN}✓ Docker services stopped${NC}"
    else
        echo -e "${YELLOW}No docker-compose.yml found${NC}"
    fi
else
    echo -e "${YELLOW}Docker services will continue running${NC}"
    echo -e "${YELLOW}To stop them manually, run: docker-compose down${NC}"
fi

# 7. Clean up PID file
if [ -f ".pids" ]; then
    echo -e "\n${YELLOW}Cleaning up PID file...${NC}"
    rm .pids
    echo -e "${GREEN}✓ PID file removed${NC}"
fi

# 8. Display status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services stopped successfully!${NC}"
echo -e "${GREEN}========================================${NC}"

# 9. Check for any remaining processes
echo -e "\n${YELLOW}Checking for remaining processes...${NC}"

remaining=false
if check_process "uvicorn"; then
    echo -e "${RED}⚠ Some uvicorn processes are still running${NC}"
    remaining=true
fi

if check_process "celery"; then
    echo -e "${RED}⚠ Some celery processes are still running${NC}"
    remaining=true
fi

if [ "$remaining" = false ]; then
    echo -e "${GREEN}✓ All application processes have been stopped${NC}"
fi

echo -e "\n${YELLOW}To restart the application, run: ./start.sh${NC}"