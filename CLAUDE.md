@AGENTS.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Financial Intelligence Platform** - An AI-powered system for analyzing SEC filings of US-listed public companies with watchlist functionality and intelligent chat interface.

## Core Architecture

### System Components

```
backend/src/
├── ingestion/          # Data pipeline for SEC filings
│   ├── rss_fetcher     # RSS feed monitoring (configurable URL in .env)
│   ├── filing_downloader # SEC filing downloads with rate limiting
│   └── docling_processor # Document processing using Docling
├── knowledge/          # RAG and Knowledge Graph systems
│   ├── rag_pipeline    # Document chunking, embedding, retrieval
│   ├── graphiti_builder # Knowledge graph construction
│   ├── vector_store    # Vector database operations (TBD)
│   ├── shareholding_classifier    # Entity classification for shareholding data
│   ├── shareholding_neo4j_store  # Enhanced knowledge graph schema
│   ├── shareholding_pipeline     # Integrated shareholding processing
│   └── migrate_shareholding_data # Data migration and cleanup
├── api/               # API and protocol servers
│   ├── auth           # JWT-based authentication
│   ├── watchlist      # Watchlist CRUD operations
│   ├── chat           # AI chat interface endpoints
│   └── mcp_server     # MCP protocol server for future integrations
└── models/            # Data models
    ├── user           # User and authentication models
    ├── company        # Company and watchlist models
    └── filing         # SEC filing models
```

### Technology Stack

- **Backend**: FastAPI (async), PostgreSQL, Neo4j, Redis, Celery/Bull
- **Frontend**: Next.js 14+, TailwindCSS, NextAuth.js, React Query
- **AI/ML**: OpenAI API (configurable), Docling, Graphiti, LangChain
- **Infrastructure**: Docker (self-hosted), environment-based config

### Key Integrations

- **Docling**: Document structure extraction from SEC filings
- **Graphiti**: Knowledge graph construction and querying
- **RAG Pipeline**: Retrieval-augmented generation for intelligent responses
- **Shareholding Knowledge Graph**: Specialized entity classification and optimization for shareholding queries
- **MCP Protocol**: Future API access and tool integration
- **Archon MCP Server**: Project management and documentation

## Development Workflow

### PRP Methodology

This project uses the PRP (Product Requirement Prompt) framework from `planning-docs/`:
- Each component has a corresponding GitHub repository for reference
- PRPs are created using templates from `planning-docs/PRPs/templates/`
- Validation loops are mandatory for all implementations

### Implementation Phases

1. **Phase 1**: Data Ingestion Pipeline ✅ (Implemented)
   - RSS feed parsing and monitoring
   - SEC filing download and processing
   - Docling integration for document extraction
   - Knowledge graph and RAG ingestion
   - **NEW**: Shareholding-optimized knowledge graph system with enhanced entity classification
   - **NEW**: Form 3/4/5 issuer identification and reporting owner disambiguation
   - See [INGESTION_PROCESS.md](./INGESTION_PROCESS.md) for detailed documentation

2. **Phase 2**: Backend Infrastructure (In Progress)
   - Authentication system
   - Watchlist management
   - Database schemas
   - API endpoints

3. **Phase 3**: AI Chat Interface
   - RAG retrieval system
   - LLM integration (configurable provider)
   - Context management
   - Response streaming

4. **Phase 4**: Frontend Web UI
   - Authentication flow
   - Watchlist interface
   - Chat interface with citations
   - Filing viewer

## Ingestion Pipeline Details

The SEC filing ingestion pipeline is fully implemented and operational. Key components:

### File Locations
- **Orchestrator**: `backend/src/main.py:IngestionPipeline`
- **Celery Tasks**: `backend/src/tasks.py`
- **RSS Monitor**: `backend/src/ingestion/rss_monitor.py`
- **Downloader**: `backend/src/ingestion/filing_downloader.py`
- **Processor**: `backend/src/ingestion/docling_processor.py`
- **RAG Pipeline**: `backend/src/knowledge/rag_pipeline.py`

### Processing Flow
1. Celery Beat triggers `ingest_sec_filings` task every 10 minutes during market hours
2. RSS Monitor discovers new filings from SEC feed
3. Downloader retrieves documents with rate limiting (10 req/sec max)
4. Docling Processor extracts structured content
5. **NEW**: Shareholding Pipeline performs enhanced entity classification and knowledge graph optimization
6. RAG Pipeline generates embeddings and updates knowledge graph
7. Data persisted to PostgreSQL, Neo4j, and Qdrant with shareholding-optimized schema

### Monitoring
- Flower UI: http://localhost:5555 (Celery task monitoring)
- Logs: `backend/ingestion.log`
- Status endpoint: `GET /api/ingestion/status`

## Configuration

### Environment Variables (.env)

```bash
# RSS Feed Configuration
RSS_FEED_URL=           # SEC RSS feed URL
RSS_POLL_INTERVAL=3600  # Polling interval in seconds

# LLM Configuration (Configurable)
LLM_PROVIDER=openai     # Default: openai
OPENAI_API_KEY=
LLM_MODEL=gpt-4

# Databases
DATABASE_URL=postgresql://...
NEO4J_URL=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# Vector Store (TBD during planning)
VECTOR_DB_TYPE=
VECTOR_DB_API_KEY=

# Authentication
JWT_SECRET=
JWT_EXPIRY=24h
```

## Development Commands

### Setup (To be implemented)

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install

# Docker services
docker-compose up -d postgres redis neo4j
```

### Testing (To be implemented)

```bash
# Backend tests
pytest backend/tests/ -v

# Frontend tests
npm test

# Integration tests
docker-compose -f docker-compose.test.yml up
```

## Key Design Decisions

1. **Modular Architecture**: Each src/ component maps to a reference GitHub repository
2. **Async-First**: FastAPI for handling concurrent filing processing
3. **Configurable LLM**: Provider-agnostic design via .env configuration
4. **Self-Hosted**: Full control over data and processing
5. **MCP-Ready**: Built for future protocol integrations

## Reference Repositories

Component-specific GitHub repositories will be provided during implementation for:
- Each backend/src/ module
- Docling integration patterns
- Graphiti usage examples
- MCP server implementation
- Authentication patterns

## Project Management

- **Archon MCP Server**: Used for task tracking and technical documentation
- **PRP Framework**: All features implemented using PRP methodology
- **Validation Gates**: 4-level validation for all implementations

## Change Management & Tracking

### 📋 ERROR_LOG.md Requirements

**CRITICAL**: All significant changes, bug fixes, and enhancements MUST be documented in [ERROR_LOG.md](./ERROR_LOG.md) using the standardized format:

```markdown
### [YYYY-MM-DD HH:MM] - [TYPE] - [TITLE]
**Issue:** Description of the problem or enhancement request
**Root Cause:** Technical analysis of underlying cause
**Solution:** Detailed implementation approach
**Files Modified:** List of changed files
**Impact:** System impact and restart requirements
**Status:** ✅ Resolved | ⚠️ Partial | ❌ Open
**Restart Required:** Yes/No - Use ./stop.sh && ./start.sh
```

### 🔄 Automated Change Management

1. **Document First**: Always update ERROR_LOG.md before implementing changes
2. **Assess Impact**: Determine if system restart is required (see restart guidelines)
3. **Implement Changes**: Make code modifications with clear commit messages
4. **Test Thoroughly**: Verify changes don't break existing functionality
5. **Auto-Push**: Automatically push productive changes to GitHub with proper commit messages including:
   - Clear description of changes
   - Reference to ERROR_LOG.md entry
   - Impact assessment
   - Co-authored by Claude tag

### 🚨 System Restart Assessment

**Always restart after changes to:**
- Database models (`backend/src/database/models.py`)
- Core ingestion pipeline (`backend/src/ingestion/*.py`)
- API endpoints (`backend/src/api/*.py`)
- Docker configuration (`docker-compose.yml`, `Dockerfile`)
- Environment variables (`.env` files)
- Dependencies (`requirements.txt`, `package.json`)

**Use appropriate restart method:**
```bash
# Full system restart (recommended for major changes)
./stop.sh && ./start.sh

# Backend-only restart (for Python code changes)
docker-compose restart backend

# Specific service restart
docker-compose restart [service-name]
```

### 📝 Commit Message Standards

```bash
git commit -m "[TYPE]: Brief description

- Detailed change 1
- Detailed change 2
- Reference: ERROR_LOG.md entry [YYYY-MM-DD HH:MM]

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Types:** BUG FIX, ENHANCEMENT, OPTIMIZATION, SECURITY, MIGRATION, DOCUMENTATION, CONFIGURATION

---

Note: This file will be continuously updated as the project evolves and new patterns emerge.