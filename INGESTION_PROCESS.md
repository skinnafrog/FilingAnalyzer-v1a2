# SEC Filing Ingestion Process

## Overview

This document describes the complete data ingestion pipeline for SEC filings in the Financial Intelligence Platform. The system automatically discovers, downloads, processes, and indexes SEC filings during market hours.

## Architecture Diagram

```
RSS Feed (every 10 min during market hours)
    ↓
RSSMonitor.poll_feed() [discovers new filings]
    ↓
FilingDownloader.download_filing() [rate-limited downloads]
    ↓
DoclingProcessor.process_filing() [extract structured data]
    ↓
RAGPipeline.ingest_filing() [embeddings & knowledge graph]
    ↓
PostgreSQL + Neo4j + Qdrant [persistent storage]
```

## Component Details

### 1. Entry Point & Orchestration

- **File**: `backend/src/tasks.py` (lines 117-171)
- **Celery Task**: `ingest_sec_filings`
- **Schedule**: Every 10 minutes during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
- **Orchestrator**: `backend/src/main.py:IngestionPipeline` class (lines 35-165)

### 2. RSS Feed Discovery

**File**: `backend/src/ingestion/rss_monitor.py`

**Key Methods**:
- `RSSMonitor.poll_feed()` (lines 41-95): Fetches SEC RSS feed from configured URL
- `_parse_feed_entries()` (lines 97-143): Parses RSS entries into `SECFiling` objects
- `_extract_edgar_fields()` (lines 145-201): Extracts EDGAR-specific metadata
- `_enrich_filing_urls()` (lines 203-252): Builds document URLs from accession numbers

**Data Extracted**:
- Company information (name, CIK number)
- Filing metadata (form type, accession number, file number)
- Document URLs (HTML, XML, XBRL formats)
- Timestamps (filing date, acceptance datetime, period)
- Additional metadata (SIC codes, fiscal year end)

### 3. Document Download

**File**: `backend/src/ingestion/filing_downloader.py`

**Key Features**:
- **Rate Limiting**: `RateLimiter` class (lines 23-49) ensures SEC compliance
  - Maximum 10 requests/second
  - Configurable via `SEC_RATE_LIMIT_DELAY` in `.env`
- **Download Logic**: `FilingDownloader.download_filing()` (lines 82-146)
  - Intelligent URL selection based on document type
  - Retry logic with exponential backoff
  - Progress tracking and statistics
- **Storage**: Downloads saved to `data/filings/` directory
- **File Size Limit**: Configurable via `MAX_FILE_SIZE_MB` (default: 100MB)

### 4. Document Processing

**File**: `backend/src/ingestion/docling_processor.py`

**Processing Pipeline**:
- `DoclingProcessor.process_filing()` (lines 91-150): Main processing function
- Uses IBM Docling library for structured extraction
- Parallel processing with configurable workers

**Extracted Content**:
- **Text Content**: Full document text with structure preservation
- **Tables**: Financial statements, metrics tables
- **Images**: Charts, graphs, diagrams
- **Metadata**: Document properties, headers, sections
- **Financial Data**:
  - Income statements
  - Balance sheets
  - Cash flow statements
  - Key financial metrics
- **Shareholder Information**:
  - Major shareholders
  - Equity issuances
  - Ownership changes

**Configuration Options**:
- `DOCLING_ENABLE_OCR`: Process scanned documents
- `DOCLING_EXTRACT_TABLES`: Extract structured tables
- `DOCLING_EXTRACT_IMAGES`: Process embedded images
- `DOCLING_MAX_WORKERS`: Parallel processing threads

### 5. Knowledge Base Ingestion

**File**: `backend/src/knowledge/rag_pipeline.py`

**RAG Pipeline Components**:
- **Text Chunking**:
  - Chunk size: 1000 characters
  - Overlap: 200 characters
  - Smart sentence boundary detection
- **Embedding Generation**:
  - Model: OpenAI text-embedding-3-small
  - Dimension: 1536
  - Batch processing for efficiency
- **Vector Storage** (Qdrant):
  - Semantic search index
  - Metadata filtering capabilities
  - Similarity search optimization
- **Knowledge Graph** (Neo4j):
  - Company relationships
  - Filing timelines
  - Cross-references between documents

### 6. Data Storage Architecture

#### PostgreSQL (Relational Data)
**Tables** (`backend/src/database/models.py`):
- `Filing`: Core filing metadata and status
- `Company`: Company information and identifiers
- `FilingDocument`: Document references and paths
- `FilingChunk`: Text chunks for RAG retrieval
- `ProcessingLog`: Audit trail of processing steps

#### Neo4j (Knowledge Graph)
**Node Types**:
- Company nodes (with properties: name, CIK, SIC)
- Filing nodes (with properties: type, date, accession)
- Person nodes (executives, directors)
- Financial metric nodes

**Relationships**:
- FILED_BY (Filing → Company)
- MENTIONS (Filing → Company)
- SUCCEEDED_BY (Filing → Filing)
- EXECUTIVE_OF (Person → Company)

#### Qdrant (Vector Database)
**Collections**:
- `sec_filings`: Document embeddings
- Metadata fields for filtering:
  - company_name
  - form_type
  - filing_date
  - cik_number

### 7. Processing States

**Filing Status Flow**:
1. `DISCOVERED`: Found in RSS feed
2. `DOWNLOADING`: Retrieving from SEC
3. `PROCESSING`: Extracting with Docling
4. `EMBEDDING`: Generating vectors
5. `INDEXED`: Available for search
6. `FAILED`: Processing error (with retry count)

**Processing Stages**:
- `RSS_DISCOVERY`
- `DOCUMENT_DOWNLOAD`
- `DOCLING_EXTRACTION`
- `TEXT_CHUNKING`
- `EMBEDDING_GENERATION`
- `VECTOR_INDEXING`
- `GRAPH_CONSTRUCTION`
- `COMPLETED`

### 8. Configuration Reference

#### Environment Variables (`backend/.env`)

**RSS Feed Settings**:
```bash
RSS_FEED_URL="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom"
RSS_POLL_INTERVAL=600  # 10 minutes
RSS_FEED_LIMIT=100     # Max entries per poll
```

**SEC Compliance**:
```bash
SEC_USER_AGENT="YourApp/1.0 (your.email@example.com)"  # REQUIRED
SEC_RATE_LIMIT_DELAY=0.1  # 10 requests/second max
SEC_MAX_RETRIES=3
SEC_RETRY_DELAY=1.0
```

**Document Processing**:
```bash
DOCLING_MAX_WORKERS=4
DOCLING_ENABLE_OCR=true
DOCLING_EXTRACT_TABLES=true
DOCLING_EXTRACT_IMAGES=true
DOCLING_CHUNK_SIZE=1000
DOCLING_CHUNK_OVERLAP=200
```

**LLM & Embeddings**:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=100
```

### 9. Market Hours Logic

**Implementation**: `backend/src/tasks.py:80-102`

**Schedule**:
- **Active Hours**: Monday-Friday, 9:30 AM - 4:00 PM ET
- **Timezone**: America/New_York (handles DST automatically)
- **Holiday Handling**: Basic implementation (extend for full holiday calendar)
- **Manual Override**: `manual_ingest` task for off-hours processing

**Celery Beat Schedule**:
```python
beat_schedule={
    'ingest-sec-filings-market-hours': {
        'task': 'src.tasks.ingest_sec_filings',
        'schedule': timedelta(seconds=600),  # Every 10 minutes
        'options': {'expires': 300}  # Expire if not executed within 5 minutes
    },
    'market-open-check': {
        'task': 'src.tasks.check_market_status',
        'schedule': crontab(hour=9, minute=25)  # 9:25 AM ET daily
    },
    'market-close-check': {
        'task': 'src.tasks.check_market_status',
        'schedule': crontab(hour=16, minute=5)  # 4:05 PM ET daily
    }
}
```

### 10. Error Handling & Recovery

**Retry Strategy**:
- Automatic retry with exponential backoff
- Maximum 3 retries per filing
- Failed filings queued for manual review
- Detailed error logging with stack traces

**Monitoring**:
- `get_ingestion_status()`: Real-time pipeline status
- Flower UI (http://localhost:5555): Celery task monitoring
- Comprehensive logging to `ingestion.log`
- Statistics tracking (success/failure rates, processing times)

**Health Checks**:
- RSS feed connectivity test
- Database connection validation
- Vector store availability
- API rate limit status

### 11. Performance Optimization

**Concurrency**:
- Celery workers: 4 concurrent tasks
- Docling processors: 4 parallel threads
- Async I/O for network operations
- Connection pooling for databases

**Resource Management**:
- Memory-efficient streaming for large documents
- Temporary file cleanup after processing
- Database connection pooling
- Rate-limited API calls

**Caching**:
- Redis cache for frequently accessed data
- Processed accession number tracking
- Embedding cache for duplicate content

### 12. API Endpoints

**Status & Control** (`backend/src/api/main.py`):
- `GET /api/ingestion/status`: Current pipeline status
- `POST /api/ingestion/trigger`: Manual ingestion trigger
- `GET /api/ingestion/statistics`: Processing statistics
- `GET /api/ingestion/queue`: View pending filings

### 13. Troubleshooting

**Common Issues**:

1. **Rate Limiting Errors**:
   - Check `SEC_RATE_LIMIT_DELAY` setting
   - Verify `SEC_USER_AGENT` is properly configured

2. **Download Failures**:
   - Verify network connectivity
   - Check disk space in `data/filings/`
   - Review proxy/firewall settings

3. **Processing Errors**:
   - Check Docling dependencies
   - Verify OCR libraries if enabled
   - Review memory allocation for workers

4. **Database Connection Issues**:
   - Verify container health: `docker-compose ps`
   - Check connection strings in `.env`
   - Review database logs

**Log Files**:
- `ingestion.log`: Main pipeline logs
- `backend/celerybeat-schedule`: Beat scheduler state
- Docker logs: `docker logs financial_intel_celery_worker`

### 14. Testing

**Unit Tests**:
```bash
pytest backend/tests/test_rss_monitor.py
pytest backend/tests/test_filing_downloader.py
pytest backend/tests/test_docling_processor.py
```

**Integration Tests**:
```bash
pytest backend/tests/integration/test_ingestion_pipeline.py
```

**Manual Testing**:
```python
# Trigger manual ingestion
from src.tasks import manual_ingest
result = manual_ingest.delay(limit=5)
print(result.get())
```

## Monitoring Dashboard

Access monitoring tools:
- **Flower** (Celery): http://localhost:5555
- **Neo4j Browser**: http://localhost:7475
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Application API**: http://localhost:8000/docs

## Security Considerations

1. **API Keys**: Store securely in `.env`, never commit to repository
2. **Rate Limiting**: Respect SEC's usage guidelines
3. **User Agent**: Must include valid email for SEC compliance
4. **Data Privacy**: Handle insider trading data appropriately
5. **Access Control**: Implement authentication for production deployment

## Future Enhancements

1. **Advanced NLP**: Entity extraction, sentiment analysis
2. **Real-time Streaming**: WebSocket for live updates
3. **ML Models**: Anomaly detection, trend prediction
4. **Distributed Processing**: Kubernetes scaling
5. **Data Quality**: Automated validation and cleaning
6. **Custom Extractors**: Industry-specific parsers