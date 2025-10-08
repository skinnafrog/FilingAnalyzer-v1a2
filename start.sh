#!/bin/bash

# Financial Intelligence Platform - Start Script
# This script starts all required services using Docker Compose

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Financial Intelligence Platform${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=60
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

# 2. Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}docker-compose.yml not found!${NC}"
    exit 1
fi

# 3. Stop any existing containers
echo -e "\n${YELLOW}2. Stopping any existing containers...${NC}"
docker-compose down 2>/dev/null || true

# 4. Start all services with Docker Compose
echo -e "\n${YELLOW}3. Starting all services with Docker Compose...${NC}"
docker-compose up -d

# 5. Wait for services to be ready
echo -e "\n${YELLOW}4. Waiting for services to be ready...${NC}"

# Wait for PostgreSQL
wait_for_service "PostgreSQL" "docker exec financial_intel_postgres pg_isready -U postgres"

# Wait for Redis
wait_for_service "Redis" "docker exec financial_intel_redis redis-cli ping | grep -q PONG"

# Wait for Neo4j
wait_for_service "Neo4j" "curl -s http://localhost:7475 > /dev/null"

# Wait for Qdrant
wait_for_service "Qdrant" "curl -s http://localhost:6333/readyz | grep -q 'all shards are ready'"

# Wait for Backend API
wait_for_service "Backend API" "curl -s http://localhost:8000/docs > /dev/null"

# Wait for Frontend
wait_for_service "Frontend" "curl -s http://localhost:3000 > /dev/null"

# Wait for Flower (optional)
if docker ps | grep -q financial_intel_flower; then
    wait_for_service "Flower" "curl -s http://localhost:5555 > /dev/null"
fi

# 6. Display status
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Service URLs:"
echo -e "  ${GREEN}• Frontend:${NC} http://localhost:3000"
echo -e "  ${GREEN}• Backend API:${NC} http://localhost:8000"
echo -e "  ${GREEN}• API Docs:${NC} http://localhost:8000/docs"
echo -e "  ${GREEN}• Flower (Celery):${NC} http://localhost:5555"
echo -e "  ${GREEN}• Neo4j Browser:${NC} http://localhost:7475"
echo -e "  ${GREEN}• Qdrant Dashboard:${NC} http://localhost:6333/dashboard"
echo ""
echo "Database Credentials:"
echo -e "  ${GREEN}• Neo4j:${NC} user=neo4j, password=password"
echo -e "  ${GREEN}• PostgreSQL:${NC} user=postgres, password=postgres"
echo ""
echo "Container Status:"
docker-compose ps
echo ""
echo -e "${YELLOW}To stop all services, run: ./stop.sh${NC}"
echo -e "${YELLOW}To restart services, run: ./restart.sh${NC}"
echo -e "${YELLOW}To view logs, run: docker-compose logs -f [service-name]${NC}"
echo ""
echo -e "${GREEN}Open http://localhost:3000 in your browser to access the application${NC}"

# 7. Optional: Open browser automatically
if command -v open &> /dev/null; then
    echo -e "\n${YELLOW}Opening browser...${NC}"
    sleep 2
    open "http://localhost:3000"
elif command -v xdg-open &> /dev/null; then
    echo -e "\n${YELLOW}Opening browser...${NC}"
    sleep 2
    xdg-open "http://localhost:3000"
fi