# Financial Intelligence Platform

AI-powered SEC filing analysis platform with automated ingestion, knowledge graph construction, and intelligent chat interface.

## 📚 Documentation

- **[INGESTION_PROCESS.md](./INGESTION_PROCESS.md)** - Detailed SEC filing ingestion pipeline documentation
- **[CLAUDE.md](./CLAUDE.md)** - Development guidelines for Claude Code
- **[AGENTS.md](./AGENTS.md)** - Agent configuration and response formatting
- **[PORT_CONFIGURATION.md](./PORT_CONFIGURATION.md)** - Service port mappings

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- OpenAI API key
- Valid SEC User Agent email (required for SEC API compliance)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd v1a2
   ```

2. **Configure environment**
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your configuration:
   # - Set SEC_USER_AGENT with your email (REQUIRED)
   # - Add OPENAI_API_KEY
   # - Verify database passwords match docker-compose.yml
   ```

3. **Start all services**
   ```bash
   ./start.sh
   # OR manually: docker-compose up -d
   ```

4. **Stop all services**
   ```bash
   ./stop.sh
   # OR manually: docker-compose down
   ```

5. **Verify services are running**
   ```bash
   docker-compose ps
   # All containers should show as "running" or "healthy"
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs
   - Flower (Celery): http://localhost:5555
   - Neo4j Browser: http://localhost:7475 (user: neo4j, pass: password)
   - Qdrant Dashboard: http://localhost:6333/dashboard

## 🏗️ Architecture

### System Overview

The platform consists of multiple microservices working together:

- **Ingestion Pipeline**: Automated SEC filing discovery and processing
- **Knowledge Base**: Vector embeddings and knowledge graph
- **API Server**: FastAPI backend with WebSocket support
- **Task Queue**: Celery workers for background processing
- **Web UI**: Next.js frontend with real-time updates

### Project Structure

```
v1a2/
├── backend/
│   ├── src/
│   │   ├── api/           # FastAPI endpoints
│   │   ├── ingestion/     # SEC filing pipeline
│   │   │   ├── rss_monitor.py       # RSS feed discovery
│   │   │   ├── filing_downloader.py # Document retrieval
│   │   │   ├── docling_processor.py # Content extraction with shareholding pipeline
│   │   │   └── issuer_extractor.py  # Form 3/4/5 issuer identification
│   │   ├── knowledge/     # RAG and knowledge graph
│   │   │   ├── rag_pipeline.py              # Embeddings & indexing
│   │   │   ├── vector_store.py              # Qdrant integration
│   │   │   ├── hybrid_search.py             # Semantic + keyword search
│   │   │   ├── shareholding_classifier.py   # Entity classification system
│   │   │   ├── shareholding_neo4j_store.py  # Enhanced knowledge graph schema
│   │   │   ├── shareholding_pipeline.py     # Integrated shareholding processing
│   │   │   └── migrate_shareholding_data.py # Data migration script
│   │   ├── database/      # Database models & connections
│   │   ├── models/        # Data models
│   │   ├── tasks.py       # Celery tasks
│   │   └── main.py        # Pipeline orchestrator
│   ├── data/             # Local storage
│   │   ├── filings/      # Downloaded documents
│   │   ├── processed/    # Extracted content
│   │   └── temp/         # Temporary files
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── INGESTION_PROCESS.md  # Detailed pipeline documentation
├── PORT_CONFIGURATION.md
├── CLAUDE.md
└── README.md
```

## 🔧 Core Components

### 1. RSS Monitor (`ingestion/rss_monitor.py`)
- Polls SEC RSS feed every 10 minutes
- Discovers new filings (10-K, 10-Q, 8-K, etc.)
- Respects SEC rate limits

### 2. Filing Downloader (`ingestion/filing_downloader.py`)
- Downloads filing documents with retry logic
- Rate limiting (10 req/sec max)
- Handles multiple document formats

### 3. Docling Processor (`ingestion/docling_processor.py`)
- Extracts text, tables, and images from documents
- Identifies financial statements
- Extracts shareholder information
- **NEW**: Integrated shareholding pipeline for enhanced entity classification
- **NEW**: Automatic processing of filings with shareholding-optimized knowledge graph storage

### 4. RAG Pipeline (`knowledge/rag_pipeline.py`)
- Chunks documents intelligently
- Generates OpenAI embeddings (text-embedding-ada-002)
- Stores vectors in Qdrant with metadata
- Hybrid search combining semantic and keyword matching

### 5. **NEW**: Shareholding Knowledge Graph (`knowledge/shareholding_*.py`)
- **Enhanced Entity Classification**: Prevents misclassification of financial data as people
- **Optimized Neo4j Schema**: Specialized nodes for shareholders, positions, and ownership events
- **Temporal Tracking**: Captures ownership evolution and transaction history
- **Query Optimization**: Maximized for shareholding, percentage, and ownership queries
- **Data Migration**: Automated cleanup of existing incorrect classifications

## 🔍 Monitoring

### View Logs
```bash
tail -f ingestion.log
```

### Check Services
```bash
docker-compose ps
```

### Celery Monitoring (if using task queue)
```bash
# Access Flower UI at http://localhost:5555
docker-compose up flower
```

## 📊 Pipeline Statistics

The pipeline tracks:
- Total filings discovered/processed
- Success/failure rates
- Processing times
- Token usage
- Embeddings generated

## 🔬 Hybrid RAG Search

The platform implements a sophisticated hybrid search system combining:
- **Vector Search**: Semantic similarity using OpenAI embeddings stored in Qdrant
- **Keyword Search**: BM25 ranking and PostgreSQL full-text search
- **Weighted Scoring**: Configurable balance between semantic and keyword relevance

### Reprocess Embeddings
```bash
# Generate embeddings for existing chunks
docker-compose exec backend python reprocess_embeddings.py --limit 50

# Test hybrid search
docker-compose exec backend python test_hybrid_search.py
```

## 🧪 Testing

### Run Basic Test
```bash
python -m src.main --mode test
```

### Verify Components
```python
# Test RSS connection
from src.ingestion.rss_monitor import RSSMonitor
monitor = RSSMonitor()
await monitor.test_connection()

# Process single filing
await pipeline.run_once(limit=1)
```

## ⚙️ Configuration

### Key Settings (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `SEC_USER_AGENT` | Your contact email for SEC | ✅ |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | ✅ |
| `RSS_POLL_INTERVAL` | Seconds between RSS polls (default: 600) | |
| `VECTOR_DB_TYPE` | Vector store type (qdrant/pinecone/chroma) | |
| `DOCLING_EXTRACT_TABLES` | Extract tables from documents | |

## 🐳 Docker Services

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres redis neo4j qdrant

# View logs
docker-compose logs -f [service-name]

# Stop services
docker-compose down
```

## 📈 Performance

- **RSS Polling**: Every 10 minutes
- **Processing Time**: <2 minutes per filing
- **Chunk Size**: 1000 tokens (configurable)
- **Embedding Batch**: 100 chunks at a time
- **Error Rate Target**: <5%

## 🚨 Troubleshooting

### Common Issues

1. **SEC Rate Limiting**
   - Ensure `SEC_USER_AGENT` includes valid email
   - Respect 0.1s delay between requests

2. **OpenAI Rate Limits**
   - Batch embeddings in groups of 100
   - Implement exponential backoff

3. **Memory Issues**
   - Process documents in chunks
   - Limit concurrent processing

4. **Docker Services Not Starting**
   ```bash
   docker-compose down -v  # Remove volumes
   docker-compose up -d    # Restart
   ```

### Recent Improvements

1. **✅ Form 3/4/5 Issuer Identification - RESOLVED**
   - **Issue**: Insider ownership forms previously showed filer's name instead of issuer company
   - **Solution**: Implemented dedicated issuer extraction for Form 3/4/5 filings
   - **Features**:
     - Automatic detection of Form 3/4/5 filings during ingestion
     - Advanced regex patterns for issuer company extraction (15+ patterns)
     - Separate tracking of reporting owner vs. issuer company
     - Database schema enhancement with new issuer-specific fields
     - Confidence scoring for extraction quality
   - **Impact**: Chat queries now correctly identify issuer companies for insider filings

2. **✅ Shareholding-Optimized Knowledge Graph System - COMPLETED**
   - **Issue**: Knowledge graph incorrectly classified financial numbers and text fragments as "Person" entities
   - **Solution**: Comprehensive shareholding-optimized entity classification and storage system
   - **Features**:
     - **Enhanced Entity Classification**: Specialized classifier prevents misclassification of numbers/fragments as people
     - **Shareholding-Optimized Neo4j Schema**: New node types (ShareholderPerson, ShareholderEntity, SharePosition, IssuanceEvent)
     - **Temporal Ownership Tracking**: Captures ownership evolution and transaction history
     - **Integrated Processing Pipeline**: Automatic processing through enhanced Docling processor
     - **Query Optimization**: Maximized retrieval success for shareholding, percentage, and temporal queries
     - **Data Migration**: Comprehensive cleanup of existing incorrect classifications
     - **Hybrid Search Enhancement**: Optimized for ownership-related semantic and keyword searches
   - **Key Components**:
     - `shareholding_classifier.py`: Advanced entity validation and classification
     - `shareholding_neo4j_store.py`: Enhanced knowledge graph schema and operations
     - `shareholding_pipeline.py`: Integrated extraction and storage pipeline
     - `migrate_shareholding_data.py`: Legacy data cleanup and migration
     - `test_shareholding_system.py`: Comprehensive testing suite
   - **Impact**: Dramatically improved accuracy for shareholder queries, beneficial ownership tracking, and ownership percentage retrieval

## 📝 Next Steps

### Phase 2: Backend Infrastructure
- [ ] Authentication system
- [ ] Watchlist management
- [ ] Database schemas
- [ ] API endpoints

### Phase 3: AI Chat Interface
- [x] RAG retrieval system (✅ Hybrid search implemented)
- [x] Vector storage with Qdrant
- [ ] LLM integration for chat
- [ ] Context management
- [ ] Response streaming

### Phase 4: Frontend Web UI
- [ ] Authentication flow
- [ ] Watchlist interface
- [ ] Chat interface with citations
- [ ] Filing viewer

## 📄 License

[Your License]

## 👥 Contributors

[Your Team]

---

**Note**: This is Phase 1 of the Financial Intelligence Platform focusing on the data ingestion pipeline. The system is designed to be modular and extensible for future phases.