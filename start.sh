#!/bin/bash

# Financial Intelligence Platform - Start Script
# This script starts all required services for the application

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Financial Intelligence Platform${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if a process is running
check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $service_name to be ready...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        if eval $check_command 2>/dev/null; then
            echo -e "${GREEN}✓ $service_name is ready${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done

    echo -e "${RED}✗ $service_name failed to start${NC}"
    return 1
}

# 1. Check if Docker is running
echo -e "\n${YELLOW}1. Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"

# 2. Start Docker services (databases)
echo -e "\n${YELLOW}2. Starting Docker services...${NC}"

# Check if docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    echo "Starting PostgreSQL, Redis, and Neo4j..."
    docker-compose up -d postgres redis neo4j 2>/dev/null || {
        echo -e "${YELLOW}Note: Some Docker services may not be configured yet${NC}"
    }

    # Wait for PostgreSQL if it's configured
    if docker-compose ps | grep -q postgres; then
        wait_for_service "PostgreSQL" "docker exec $(docker-compose ps -q postgres) pg_isready -U postgres"
    fi

    # Wait for Redis if it's configured
    if docker-compose ps | grep -q redis; then
        wait_for_service "Redis" "docker exec $(docker-compose ps -q redis) redis-cli ping"
    fi
else
    echo -e "${YELLOW}No docker-compose.yml found. Skipping Docker services.${NC}"
fi

# 3. Start Backend API server
echo -e "\n${YELLOW}3. Starting Backend API server...${NC}"

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate

    # Install requirements if they exist
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies..."
        pip install -r requirements.txt
    fi

    cd ..
fi

# Kill any existing backend process
if check_process "uvicorn.*src.api.main:app"; then
    echo "Stopping existing backend server..."
    pkill -f "uvicorn.*src.api.main:app"
    sleep 2
fi

# Start the backend server
cd backend
source venv/bin/activate

echo "Starting FastAPI server..."
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
wait_for_service "Backend API" "curl -s http://localhost:8000/health"

# 4. Start Celery worker (if configured)
echo -e "\n${YELLOW}4. Starting Celery worker...${NC}"

if [ -f "backend/src/tasks.py" ]; then
    # Kill any existing Celery worker
    if check_process "celery.*worker"; then
        echo "Stopping existing Celery worker..."
        pkill -f "celery.*worker"
        sleep 2
    fi

    cd backend
    source venv/bin/activate

    echo "Starting Celery worker..."
    nohup celery -A src.tasks worker --loglevel=info --concurrency=2 > ../logs/celery.log 2>&1 &
    CELERY_PID=$!
    cd ..

    echo -e "${GREEN}✓ Celery worker started${NC}"
else
    echo -e "${YELLOW}Celery tasks not configured. Skipping.${NC}"
fi

# 5. Start Celery beat scheduler (if configured)
echo -e "\n${YELLOW}5. Starting Celery beat scheduler...${NC}"

if [ -f "backend/src/tasks.py" ]; then
    # Kill any existing Celery beat
    if check_process "celery.*beat"; then
        echo "Stopping existing Celery beat..."
        pkill -f "celery.*beat"
        sleep 2
    fi

    cd backend
    source venv/bin/activate

    echo "Starting Celery beat..."
    nohup celery -A src.tasks beat --loglevel=info > ../logs/celery-beat.log 2>&1 &
    BEAT_PID=$!
    cd ..

    echo -e "${GREEN}✓ Celery beat started${NC}"
else
    echo -e "${YELLOW}Celery beat not configured. Skipping.${NC}"
fi

# 6. Start Frontend server
echo -e "\n${YELLOW}6. Starting Frontend server...${NC}"

# Check if we should use Docker or standalone
if [ -f "docker-compose.yml" ] && docker-compose ps | grep -q frontend; then
    echo "Frontend is configured in Docker Compose. Restarting frontend container..."
    docker-compose restart frontend 2>/dev/null || {
        echo "Starting frontend container..."
        docker-compose up -d frontend
    }
    FRONTEND_MODE="docker"
    FRONTEND_PORT=3000
else
    # Fallback to Python's built-in server for development
    # Kill any existing frontend server
    if check_process "python3.*http.server.*3000"; then
        echo "Stopping existing frontend server..."
        pkill -f "python3.*http.server.*3000"
        sleep 2
    fi

    cd frontend
    echo "Starting frontend development server on port 3000..."
    nohup python3 -m http.server 3000 > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    FRONTEND_MODE="standalone"
    FRONTEND_PORT=3000

    # Wait a moment for the frontend to start
    sleep 2
fi

# 7. Create logs directory if it doesn't exist
mkdir -p logs

# 8. Save PIDs for stop script
echo "BACKEND_PID=$BACKEND_PID" > .pids
echo "CELERY_PID=$CELERY_PID" >> .pids
echo "BEAT_PID=$BEAT_PID" >> .pids
echo "FRONTEND_PID=$FRONTEND_PID" >> .pids

# 9. Display status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Service URLs:"
echo -e "  ${GREEN}• Frontend:${NC} http://localhost:${FRONTEND_PORT}"
echo -e "  ${GREEN}• Backend API:${NC} http://localhost:8000"
echo -e "  ${GREEN}• API Docs:${NC} http://localhost:8000/docs"
echo ""
echo "Logs location:"
echo -e "  ${GREEN}• Backend:${NC} logs/backend.log"
echo -e "  ${GREEN}• Celery:${NC} logs/celery.log"
echo -e "  ${GREEN}• Frontend:${NC} logs/frontend.log"
echo ""
echo -e "${YELLOW}To stop all services, run: ./stop.sh${NC}"
echo ""
echo -e "${GREEN}Open http://localhost:${FRONTEND_PORT} in your browser to access the application${NC}"

# 10. Optional: Open browser automatically
if command -v open &> /dev/null; then
    echo -e "\n${YELLOW}Opening browser...${NC}"
    sleep 2
    open "http://localhost:${FRONTEND_PORT}"
elif command -v xdg-open &> /dev/null; then
    echo -e "\n${YELLOW}Opening browser...${NC}"
    sleep 2
    xdg-open "http://localhost:${FRONTEND_PORT}"
fi