# Financial Intelligence Platform

AI-powered system for analyzing SEC filings of US-listed public companies with watchlist functionality and intelligent chat interface.

## Overview

This platform allows authenticated users to:
- Create and manage watchlists of US-listed public companies
- Automatically ingest SEC filings via RSS feeds
- Query filing contents through an AI chat interface
- Access filing data via RAG and knowledge graph systems

## Architecture

- **Backend**: FastAPI with async support
- **Frontend**: Next.js 14+ with TailwindCSS
- **Databases**: PostgreSQL (primary), Neo4j (knowledge graph), Redis (caching)
- **AI/ML**: Docling (document processing), Graphiti (knowledge graphs), Configurable LLM
- **Infrastructure**: Docker-based, self-hosted

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 20+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd v1a2
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services**
   ```bash
   cd docker
   docker-compose up -d
   ```

4. **Install backend dependencies**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

6. **Run development servers**
   ```bash
   # Backend (from backend/)
   uvicorn src.main:app --reload

   # Frontend (from frontend/)
   npm run dev
   ```

## Project Structure

```
v1a2/
├── backend/           # FastAPI backend
│   ├── src/
│   │   ├── ingestion/    # Data pipeline
│   │   ├── knowledge/    # RAG & knowledge graph
│   │   ├── api/          # API endpoints
│   │   └── models/       # Data models
│   └── tests/
├── frontend/          # Next.js frontend
│   ├── app/
│   └── components/
├── docker/           # Docker configuration
├── planning-docs/    # PRP methodology docs
└── PRPs/            # Product requirement prompts
```

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate

# Run tests
pytest tests/ -v

# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/
```

### Frontend Development

```bash
cd frontend

# Run development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Documentation

- Project specifications: See `CLAUDE.md`
- PRP methodology: See `planning-docs/`
- API documentation: Available at `http://localhost:8000/docs` when running

## License

[Your License]

## Contact

[Your Contact Information]