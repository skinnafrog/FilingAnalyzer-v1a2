# Port Configuration - Financial Intelligence Platform

## Standard Port Assignment (Per PRP Architecture)

| Service | Port | Description | Status |
|---------|------|-------------|---------|
| **Frontend UI** | 3000 | Web interface (HTML/JS) | Primary |
| **Backend API** | 8000 | FastAPI REST endpoints | Active |
| **PostgreSQL** | 5433 | Main database (mapped from 5432) | Docker |
| **Redis** | 6380 | Cache & Celery broker (mapped from 6379) | Docker |
| **Neo4j Browser** | 7475 | Graph DB web UI (mapped from 7474) | Docker |
| **Neo4j Bolt** | 7688 | Graph DB connection (mapped from 7687) | Docker |
| **Qdrant** | 6333 | Vector database | Docker |
| **Flower** | 5555 | Celery monitoring UI | Optional |

## Port Cleanup and Management

### Issue Resolution
- **Port 8080**: Previously used for frontend, now deprecated
- **Port 3000**: Standard frontend port per docker-compose.yml
- **Backend**: Remains on port 8000 as designed

### Management Scripts

1. **`./clean-restart.sh`** - Recommended for applying changes
   - Kills all processes on relevant ports
   - Stops Docker containers
   - Performs clean restart
   - Ensures no port conflicts

2. **`./start.sh`** - Standard start
   - Checks for running services
   - Starts all components
   - Opens browser to frontend

3. **`./stop.sh`** - Graceful shutdown
   - Stops all services
   - Optionally stops Docker services

4. **`./restart.sh`** - Quick restart
   - Runs stop then start
   - Maintains Docker services

## Frontend Access

After running any start script:
- **Main Application**: http://localhost:3000
  - Chat Interface: http://localhost:3000/index.html
  - Filings Explorer: http://localhost:3000/filings.html
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Docker vs Standalone Mode

The scripts automatically detect whether to use Docker or standalone mode:

### Docker Mode (if docker-compose.yml exists and containers running):
- Frontend runs in nginx container on port 3000
- Backend runs in container on port 8000
- All databases run in containers

### Standalone Mode (development):
- Frontend served via Python http.server on port 3000
- Backend runs via uvicorn on port 8000
- Databases need to be started separately

## Troubleshooting Ports

If you encounter port conflicts:

```bash
# Check what's using a port
lsof -i :3000
lsof -i :8000
lsof -i :8080

# Kill process on specific port
kill -9 $(lsof -Pi :3000 -sTCP:LISTEN -t)

# Or use the clean restart script
./clean-restart.sh
```

## Environment Variables

Ensure your `backend/.env` file contains:
```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend expects API at
BACKEND_URL=http://localhost:8000
```

## Architecture Compliance

This configuration aligns with:
- **PRP Phase 1**: Data Ingestion Pipeline specification
- **Docker Compose**: Container orchestration setup
- **Frontend/Backend Separation**: Clear service boundaries
- **Scalability**: Ready for production deployment