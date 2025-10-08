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
│   └── vector_store    # Vector database operations (TBD)
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
- **MCP Protocol**: Future API access and tool integration
- **Archon MCP Server**: Project management and documentation

## Development Workflow

### PRP Methodology

This project uses the PRP (Product Requirement Prompt) framework from `planning-docs/`:
- Each component has a corresponding GitHub repository for reference
- PRPs are created using templates from `planning-docs/PRPs/templates/`
- Validation loops are mandatory for all implementations

### Implementation Phases

1. **Phase 1**: Data Ingestion Pipeline (Priority)
   - RSS feed parsing and monitoring
   - SEC filing download and processing
   - Docling integration for document extraction
   - Knowledge graph and RAG ingestion

2. **Phase 2**: Backend Infrastructure
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

---

Note: This file will be continuously updated as the project evolves and new patterns emerge.