name: "Phase 1: SEC Filing Data Ingestion Pipeline"
description: |
  Comprehensive implementation of RSS feed monitoring, SEC filing downloads, and document processing with Docling, RAG, and Graphiti knowledge graph integration.

---

## Goal

**Feature Goal**: Implement an MVP-quality data ingestion pipeline that monitors SEC RSS feeds, locates and downloads filings (leveraging Crawl4ai as needed), processes documents with Docling, and ingests into both RAG vector store and Graphiti knowledge graph, in addition to storing in the file system for subsequent reference and retrieval as needed. A GUI will be created that allows for interrogation of such filings via an AI chat interface (based on those available in the database).

**Deliverable**: Async ingestion service with RSS monitoring, filing download queue, Docling processing, embeddings generation, and dual storage in vector DB and Neo4j knowledge graph. GUI for interrogation of such filings via an AI chat interface (based on those available in the database).

**Success Definition**: System successfully polls RSS feed every 10 minutes, locates and downloads new filings, extracts structured content with Docling, generates embeddings, and stores in both vector and graph databases with <5% error rate.

## User Persona

**Target User**: System administrators and backend services

**Use Case**: Automated ingestion of SEC filings for downstream AI chat queries

**User Journey**:
1. System polls RSS feed automatically
2. New filings detected and queued
3. Documents downloaded and processed
4. Content indexed for retrieval
5. Available for user queries via chat

## Why

- Enables real-time monitoring of SEC filings
- Provides structured data extraction from complex financial documents
- Builds searchable knowledge base for AI-powered queries
- Creates entity relationships for enhanced context understanding
- Foundation for all downstream features

## What

Automated ingestion pipeline that:
- Polls SEC RSS feed every 10 minutes during market hours
- Locates, downloads and processes new filings (10-K, 10-Q, 8-K, etc.), once triggered for download (manually or automatically)
- Extracts structured content using Docling
- Generates embeddings for RAG retrieval
- Builds knowledge graph with Graphiti
- Handles failures with retry logic
- Stores processed filings in file system for subsequent reference and retrieval as needed

### Success Criteria

- [ ] RSS feed polled successfully every 10 minutes
- [ ] New filings detected and queued within 1 minute
- [ ] Documents processed with Docling including tables/charts/images
- [ ] Embeddings generated and stored in vector DB
- [ ] Knowledge graph updated with entities and relationships, with a particular focus on shareholders/equityholders, and their respective issuance date(s) of attributable equity
- [ ] Error rate below 5% with proper retry logic (ensure logging is in place, preferably with AI-tracing enabled such as via LangFuse)
- [ ] Processing time < 2 minutes per filing

## All Needed Context

### Context Completeness Check

_This PRP hopes that it contains, but very unulikely containes, all RSS feed patterns, Docling processing examples, RAG pipeline code, and Graphiti integration needed for implementation without prior knowledge, however it is the best root document available at this time (further/future user interactions may better elucidate whichever other resources may be/come requisite for implementation)_

### Documentation & References

IMPORTANT: Always use the Archon MCP server for both task & project management, as well as for technical documentation and code examples for each and every component of the tech stack being implemented as described herein.

```yaml
# MUST READ - Include these in your context window

- file: reference-repos/ottomator-agents/docling-rag-agent/ingestion/ingest.py
  why: Complete Docling document processing pattern with multi-format support
  pattern: DocumentProcessor class with format handlers
  gotcha: Must handle large PDFs in chunks to avoid memory issues

- file: reference-repos/ottomator-agents/docling-rag-agent/ingestion/chunker.py
  why: Document chunking strategy for optimal retrieval
  pattern: Recursive text splitting with overlap
  gotcha: Balance chunk size (1000-2000 tokens) with context preservation

- file: reference-repos/ottomator-agents/docling-rag-agent/ingestion/embedder.py
  why: Embedding generation with caching to avoid redundant API calls
  pattern: Batch processing with retry logic
  gotcha: OpenAI rate limits - batch embeddings in groups of 100

- file: reference-repos/ai-agent-mastery/6_Agent_Deployment/PRPs/examples/ingestion/graph_builder.py
  why: Graphiti knowledge graph construction from documents
  pattern: Entity extraction and relationship mapping, with a focus on shareholders/equityholders, and their respective issuance date(s) of attributable equity
  gotcha: Token limits for Graphiti - process in 8000 token chunks

- file: reference-repos/mcp-crawl4ai-rag/src/crawl4ai_mcp.py
  why: MCP server patterns for future API exposure and to provide internal functionality for the ingestion pipeline sub-steps as needed, including RSS feed polling, filing download, and document processing, and as to identify the chain or URL's and their respective artifacts such as to ascertain the underlying source URL of the actual filing document(s) referenced initially in the RSS feed, given that HTML-element codification of the resultant filing document(s) is not always available directly from the infromation contained in the RSS feed and nor in its resultant URL's contents, and nor that of such derivative URL's contents, and therefore chaining of URL's may be required to satisfy the ultimate objective of ascertaining the underlying source URL of the actual filing document(s) referenced initially in the RSS feed
  pattern: Tool registration and async handlers
  gotcha: MCP requires specific response formats

- url: https://github.com/DS4SD/docling
  why: Official Docling documentation for table/chart extraction
  critical: Use DocumentConverter with PdfPipeline for financial documents
  gotcha: Best if documentation for Docling is referenced via Archon MCP server, as with other reference documentation and code examples

- docfile: PRPs/ai_docs/sec_filing_structure.md
  why: SEC filing formats and XBRL structure (to be created)
  section: Filing types and data extraction, as well as XBRL structure, chain of URL's and their respective artifacts necessary to locate and download the actual filing document(s) referenced initially in the RSS feed
```

### Current Codebase tree

```bash
backend/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   └── __init__.py
│   ├── knowledge/
│   │   └── __init__.py
│   ├── api/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── config/
│       └── __init__.py
└── tests/
```

### Desired Codebase tree with files to be added

```bash
backend/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── rss_monitor.py         # RSS feed polling service
│   │   ├── filing_downloader.py   # SEC filing download with rate limiting
│   │   ├── docling_processor.py   # Document processing with Docling
│   │   └── ingestion_queue.py     # Celery task queue management
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py        # Chunking, embedding, vector storage
│   │   ├── graphiti_builder.py    # Knowledge graph construction
│   │   ├── vector_store.py        # Vector DB operations (Pinecone/Qdrant)
│   │   └── graph_client.py        # Neo4j connection wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── filing.py               # SEC filing data models
│   │   └── ingestion.py           # Ingestion status models
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py            # Environment configuration
│   └── tasks.py                    # Celery task definitions
└── tests/
    ├── test_rss_monitor.py
    ├── test_filing_downloader.py
    ├── test_docling_processor.py
    └── test_rag_pipeline.py
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: SEC requires specific User-Agent header (try Crawl4ai first but have a backup plan configurable via .env as regards user agent etc. )
# Example: "CompanyName/1.0 (contact@email.com)"
# Rate limit: 1 request per 10 seconds

# CRITICAL: Docling requires proper initialization
# Must use DocumentConverter with PdfPipeline for tables
from docling.document_converter import DocumentConverter, PdfPipeline
converter = DocumentConverter(pipeline=PdfPipeline())

# CRITICAL: Graphiti token limits
# Maximum 8000 tokens per processing batch
# Must chunk large documents before sending to Graphiti

# CRITICAL: OpenAI embedding rate limits
# Batch embeddings in groups of 100 max
# Implement exponential backoff for rate limit errors

# CRITICAL: Neo4j connection pooling
# Use async driver with connection pool for performance
# Close connections properly to avoid leaks
```

## Implementation Blueprint

### Data models and structure

```python
# Filing model for database storage
# CRITICAL: for class FilingType, the actual potential file types should be extensible, as the below examples are not exhaustive
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class FilingType(str, Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_DEF14A = "DEF 14A"
    OTHER = "OTHER"

class FilingStatus(str, Enum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SECFiling(BaseModel):
    # From RSS feed
    company_name: str
    cik_number: str
    form_type: FilingType
    filing_date: datetime
    accession_number: str
    file_number: Optional[str]
    acceptance_datetime: datetime
    period: Optional[datetime]

    # URLs
    filing_url: HttpUrl
    xbrl_zip_url: Optional[HttpUrl]
    document_urls: List[HttpUrl] = Field(default_factory=list)

    # Processing status
    status: FilingStatus = FilingStatus.DISCOVERED
    download_attempts: int = 0
    processing_attempts: int = 0
    error_message: Optional[str]

    # Extracted data
    extracted_text: Optional[str]
    extracted_tables: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Vector storage
    embedding_ids: List[str] = Field(default_factory=list)
    chunk_count: int = 0

    # Knowledge graph
    graph_node_ids: List[str] = Field(default_factory=list)
    entities_extracted: List[Dict[str, str]] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime]

class IngestionJob(BaseModel):
    job_id: str
    filing_accession: str
    task_type: str  # "download", "process", "embed", "graph"
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    retry_count: int = 0
    max_retries: int = 3
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE backend/src/config/settings.py
  - IMPLEMENT: Pydantic Settings with environment variables
  - FOLLOW pattern: reference-repos/ottomator-agents/docling-rag-agent/config.py
  - NAMING: Settings class with typed fields
  - PLACEMENT: Central configuration in src/config/

Task 2: CREATE backend/src/models/filing.py
  - IMPLEMENT: SECFiling, FilingType, FilingStatus, IngestionJob models
  - FOLLOW pattern: Pydantic BaseModel with validators
  - NAMING: CamelCase for classes, snake_case for fields
  - PLACEMENT: Domain models in src/models/

Task 3: CREATE backend/src/ingestion/rss_monitor.py
  - IMPLEMENT: RSSMonitor class with async feed parsing
  - FOLLOW pattern: Async service with scheduled polling
  - NAMING: RSSMonitor class, poll_feed(), parse_entries() methods
  - DEPENDENCIES: feedparser, httpx with proper User-Agent
  - PLACEMENT: Ingestion services in src/ingestion/

Task 4: CREATE backend/src/ingestion/filing_downloader.py
  - IMPLEMENT: FilingDownloader with rate limiting and retry logic
  - FOLLOW pattern: reference-repos/mcp-crawl4ai-rag async patterns
  - NAMING: FilingDownloader class, download_filing() method
  - DEPENDENCIES: httpx with rate limiting, exponential backoff
  - PLACEMENT: Download service in src/ingestion/

Task 5: CREATE backend/src/ingestion/docling_processor.py
  - IMPLEMENT: DoclingProcessor for document extraction
  - FOLLOW pattern: reference-repos/ottomator-agents/docling-rag-agent/ingestion/ingest.py
  - NAMING: DoclingProcessor class, process_document() method
  - DEPENDENCIES: docling with PdfPipeline for tables
  - PLACEMENT: Processing service in src/ingestion/

Task 6: CREATE backend/src/knowledge/rag_pipeline.py
  - IMPLEMENT: RAGPipeline with chunking and embedding
  - FOLLOW pattern: reference-repos/ottomator-agents/docling-rag-agent/ingestion/chunker.py
  - NAMING: RAGPipeline class, chunk_document(), generate_embeddings() methods
  - DEPENDENCIES: langchain, openai, vector store client
  - PLACEMENT: Knowledge processing in src/knowledge/

Task 7: CREATE backend/src/knowledge/graphiti_builder.py
  - IMPLEMENT: GraphitiBuilder for knowledge graph construction
  - FOLLOW pattern: reference-repos/ai-agent-mastery graph_builder.py
  - NAMING: GraphitiBuilder class, build_graph(), extract_entities() methods
  - DEPENDENCIES: graphiti-core, neo4j async driver
  - PLACEMENT: Graph builder in src/knowledge/

Task 8: CREATE backend/src/knowledge/vector_store.py
  - IMPLEMENT: VectorStore abstraction for multiple backends
  - FOLLOW pattern: Factory pattern for Pinecone/Qdrant/Chroma
  - NAMING: VectorStore abstract class, PineconeStore, QdrantStore implementations
  - DEPENDENCIES: Vector DB clients based on config
  - PLACEMENT: Storage abstraction in src/knowledge/

Task 9: CREATE backend/src/tasks.py
  - IMPLEMENT: Celery tasks for async processing
  - FOLLOW pattern: Celery best practices with error handling
  - NAMING: process_filing_task, update_knowledge_graph_task
  - DEPENDENCIES: celery, redis
  - PLACEMENT: Task definitions in src/

Task 10: CREATE backend/src/ingestion/ingestion_queue.py
  - IMPLEMENT: Queue manager for orchestrating pipeline
  - FOLLOW pattern: Producer-consumer with priority queue
  - NAMING: IngestionQueue class, enqueue(), process_next() methods
  - DEPENDENCIES: celery, redis
  - PLACEMENT: Queue management in src/ingestion/

Task 11: CREATE tests for all components
  - IMPLEMENT: Pytest async tests with mocking
  - FOLLOW pattern: reference-repos fixtures and mocking patterns
  - NAMING: test_{component}_{scenario} naming
  - COVERAGE: Happy path, error cases, edge cases
  - PLACEMENT: tests/ directory with mirrors of src/
```

### Implementation Patterns & Key Details

```python
# RSS Monitor Pattern
import feedparser
import httpx
from datetime import datetime, timedelta
from typing import List

class RSSMonitor:
    def __init__(self, settings: Settings):
        self.feed_url = settings.RSS_FEED_URL
        self.poll_interval = settings.RSS_POLL_INTERVAL
        self.user_agent = settings.SEC_USER_AGENT
        self.last_poll = None

    async def poll_feed(self) -> List[SECFiling]:
        # PATTERN: Respect SEC rate limits
        headers = {"User-Agent": self.user_agent}

        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(self.feed_url)

        feed = feedparser.parse(response.text)

        # CRITICAL: Extract all XBRL namespace fields
        filings = []
        for entry in feed.entries:
            edgar_ns = entry.get("edgar_xbrlfiling", {})
            filing = SECFiling(
                company_name=edgar_ns.get("edgar_companyname"),
                cik_number=edgar_ns.get("edgar_ciknumber"),
                form_type=edgar_ns.get("edgar_formtype"),
                # ... extract all fields
            )
            filings.append(filing)

        return filings

# Docling Processing Pattern
from docling.document_converter import DocumentConverter, PdfPipeline
from docling.datamodel.base_models import DocumentStream

class DoclingProcessor:
    def __init__(self):
        # PATTERN: Initialize with table extraction pipeline
        self.converter = DocumentConverter(
            pdf_pipeline=PdfPipeline(
                do_table_structure=True,
                do_ocr=True
            )
        )

    async def process_document(self, file_path: str) -> Dict:
        # PATTERN: Handle large documents with streaming
        with open(file_path, 'rb') as f:
            stream = DocumentStream(name=file_path, stream=f)
            result = self.converter.convert(stream)

        # CRITICAL: Extract tables separately for financial data
        tables = []
        for element in result.elements:
            if element.type == "table":
                tables.append(element.to_dict())

        return {
            "text": result.text,
            "tables": tables,
            "metadata": result.metadata
        }

# Graphiti Integration Pattern
from graphiti_core import Graphiti
from neo4j import AsyncGraphDatabase

class GraphitiBuilder:
    def __init__(self, settings: Settings):
        self.graphiti = Graphiti(
            neo4j_uri=settings.NEO4J_URL,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )

    async def build_graph(self, filing: SECFiling, content: str):
        # PATTERN: Chunk for token limits
        chunks = self.chunk_for_graphiti(content, max_tokens=8000)

        for chunk in chunks:
            # CRITICAL: Extract entities and relationships
            await self.graphiti.add_episode(
                name=f"{filing.company_name}_{filing.form_type}",
                episode_body=chunk,
                reference_time=filing.filing_date
            )
```

### Integration Points

```yaml
DATABASE:
  - migration: "CREATE TABLE filings with all SECFiling fields"
  - index: "CREATE INDEX idx_filings_accession ON filings(accession_number)"
  - index: "CREATE INDEX idx_filings_cik ON filings(cik_number)"
  - index: "CREATE INDEX idx_filings_status ON filings(status)"

CONFIG:
  - add to: backend/src/config/settings.py
  - pattern: |
      RSS_FEED_URL = str
      RSS_POLL_INTERVAL = int  # seconds
      SEC_USER_AGENT = str  # "Company email@example.com"
      DOCLING_MAX_WORKERS = int
      VECTOR_DB_TYPE = str  # "pinecone" | "qdrant" | "chroma"
      GRAPHITI_MAX_TOKENS = int

CELERY:
  - add to: backend/src/tasks.py
  - pattern: |
      @celery_app.task(bind=True, max_retries=3)
      def process_filing_task(self, accession_number: str):
          # Download, process, embed, graph
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation
cd backend
ruff check src/ingestion/ --fix
mypy src/ingestion/
ruff format src/ingestion/

# Full project validation
ruff check src/ --fix
mypy src/ --strict
ruff format src/

# Expected: Zero errors, all type hints valid
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test RSS monitor
pytest tests/test_rss_monitor.py -v

# Test filing downloader with mocked requests
pytest tests/test_filing_downloader.py -v

# Test Docling processor with sample documents
pytest tests/test_docling_processor.py -v

# Test RAG pipeline
pytest tests/test_rag_pipeline.py -v

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Expected: 80%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Start required services
docker-compose up -d postgres redis neo4j

# Test RSS feed connection
python -m src.ingestion.rss_monitor --test

# Test SEC download with rate limiting
curl -H "User-Agent: Test test@example.com" \
  "https://www.sec.gov/Archives/edgar/data/1007019/000149315225017225/form8-k.htm"

# Test Celery workers
celery -A src.tasks worker --loglevel=info &
celery -A src.tasks flower  # Monitor at http://localhost:5555

# Test end-to-end pipeline with single filing
python -c "
from src.tasks import process_filing_task
result = process_filing_task.delay('0001493152-25-017225')
print(result.get(timeout=60))
"

# Verify vector storage
python -c "
from src.knowledge.vector_store import get_store
store = get_store()
results = store.search('coffee holding', k=5)
print(results)
"

# Verify knowledge graph
cypher-shell -u neo4j -p password \
  "MATCH (n:Company)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10"

# Expected: All services connected, filing processed, data stored
```

### Level 4: Performance & Reliability Testing

```bash
# Load test RSS polling
python -c "
import asyncio
from src.ingestion.rss_monitor import RSSMonitor
monitor = RSSMonitor()
# Simulate 1 hour of polling
for _ in range(6):
    filings = asyncio.run(monitor.poll_feed())
    print(f'Found {len(filings)} filings')
    asyncio.sleep(600)
"

# Test parallel processing
python -c "
from src.tasks import process_filing_task
# Queue 10 filings for parallel processing
tasks = []
for filing in recent_filings[:10]:
    task = process_filing_task.delay(filing.accession_number)
    tasks.append(task)
# Wait for completion
for task in tasks:
    result = task.get(timeout=300)
    print(f'Processed: {result}')
"

# Monitor system resources
docker stats

# Check error rates
grep ERROR /var/log/app/*.log | wc -l

# Verify retry logic
python -c "
# Simulate failure and retry
from src.ingestion.filing_downloader import FilingDownloader
downloader = FilingDownloader()
# Test with invalid URL to trigger retry
result = downloader.download_with_retry('invalid-url')
"

# Expected: <2min per filing, <5% error rate, successful retries
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] RSS feed polling works with proper User-Agent
- [ ] SEC rate limits respected (10 req/sec)
- [ ] Docling extracts tables from financial documents
- [ ] Embeddings generated and stored in vector DB
- [ ] Knowledge graph populated with entities
- [ ] Celery tasks process asynchronously
- [ ] Error handling and retry logic functional

### Feature Validation

- [ ] RSS feed monitored every 10 minutes
- [ ] New filings detected within 1 minute
- [ ] Documents downloaded successfully
- [ ] Tables and text extracted accurately
- [ ] RAG retrieval returns relevant results
- [ ] Knowledge graph shows entity relationships
- [ ] Processing time <2 minutes per filing
- [ ] Error rate below 5%

### Code Quality Validation

- [ ] Follows async patterns from reference repos
- [ ] Proper error handling and logging
- [ ] Rate limiting implemented correctly
- [ ] Database transactions handled properly
- [ ] Memory efficient for large documents
- [ ] Connection pooling for databases
- [ ] Comprehensive test coverage (80%+)

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] API endpoints documented if exposed
- [ ] Deployment instructions clear
- [ ] Monitoring and alerting configured
- [ ] Backup strategy for data
- [ ] Security considerations addressed

---

## Anti-Patterns to Avoid

- ❌ Don't poll RSS feed too frequently (respect 10-minute interval)
- ❌ Don't skip User-Agent header (SEC will block requests)
- ❌ Don't process entire documents in memory (use streaming)
- ❌ Don't ignore rate limits (implement exponential backoff)
- ❌ Don't store sensitive data unencrypted
- ❌ Don't skip error handling in async code
- ❌ Don't forget to close database connections
- ❌ Don't process large documents in single Graphiti call