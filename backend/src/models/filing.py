"""
Data models for SEC filings and ingestion jobs.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl, validator
import uuid


class FilingType(str, Enum):
    """SEC filing form types."""
    FORM_10K = "10-K"
    FORM_10K_A = "10-K/A"
    FORM_10Q = "10-Q"
    FORM_10Q_A = "10-Q/A"
    FORM_8K = "8-K"
    FORM_8K_A = "8-K/A"
    FORM_20F = "20-F"
    FORM_20F_A = "20-F/A"
    FORM_11K = "11-K"
    FORM_DEF14A = "DEF 14A"
    FORM_DEFM14A = "DEFM14A"
    FORM_DEFA14A = "DEFA14A"
    FORM_PRE14A = "PRE 14A"
    FORM_S1 = "S-1"
    FORM_S1_A = "S-1/A"
    FORM_S3 = "S-3"
    FORM_S3_A = "S-3/A"
    FORM_S4 = "S-4"
    FORM_S4_A = "S-4/A"
    FORM_S8 = "S-8"
    FORM_424B1 = "424B1"
    FORM_424B2 = "424B2"
    FORM_424B3 = "424B3"
    FORM_424B4 = "424B4"
    FORM_424B5 = "424B5"
    FORM_SC13D = "SC 13D"
    FORM_SC13D_A = "SC 13D/A"
    FORM_SC13G = "SC 13G"
    FORM_SC13G_A = "SC 13G/A"
    FORM_13F_HR = "13F-HR"
    FORM_13F_HR_A = "13F-HR/A"
    FORM_4 = "4"
    FORM_5 = "5"
    FORM_3 = "3"
    FORM_144 = "144"
    OTHER = "OTHER"

    @classmethod
    def from_string(cls, value: str) -> "FilingType":
        """Convert string to FilingType, handling unknown types."""
        normalized = value.upper().replace("/", "_").replace(" ", "")
        try:
            return cls[f"FORM_{normalized}"]
        except (KeyError, AttributeError):
            # Try to find a partial match
            for filing_type in cls:
                if filing_type.value == value:
                    return filing_type
            return cls.OTHER


class FilingStatus(str, Enum):
    """Filing processing status."""
    DISCOVERED = "discovered"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    GRAPHING = "graphing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class ProcessingStage(str, Enum):
    """Detailed processing stages for tracking."""
    RSS_DISCOVERY = "rss_discovery"
    URL_RESOLUTION = "url_resolution"
    DOCUMENT_DOWNLOAD = "document_download"
    DOCLING_EXTRACTION = "docling_extraction"
    TEXT_CHUNKING = "text_chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_STORAGE = "vector_storage"
    ENTITY_EXTRACTION = "entity_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    COMPLETED = "completed"


class SECFiling(BaseModel):
    """SEC filing document model."""

    # Unique identifiers
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    accession_number: str = Field(..., description="SEC accession number")

    # Company information
    company_name: str
    cik_number: str = Field(..., description="Central Index Key")
    ticker_symbol: Optional[str] = None

    # Filing information
    form_type: FilingType
    filing_date: datetime
    acceptance_datetime: datetime
    period_date: Optional[datetime] = None
    file_number: Optional[str] = None
    film_number: Optional[str] = None

    # URLs and paths
    filing_url: HttpUrl = Field(..., description="Main filing URL")
    filing_html_url: Optional[HttpUrl] = None
    filing_xml_url: Optional[HttpUrl] = None
    xbrl_zip_url: Optional[HttpUrl] = None
    document_urls: List[HttpUrl] = Field(default_factory=list)
    exhibits_urls: List[HttpUrl] = Field(default_factory=list)

    # Local storage paths
    raw_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None

    # Processing status
    status: FilingStatus = FilingStatus.DISCOVERED
    current_stage: ProcessingStage = ProcessingStage.RSS_DISCOVERY
    download_attempts: int = 0
    processing_attempts: int = 0
    error_message: Optional[str] = None
    error_trace: Optional[str] = None

    # Extracted content
    extracted_text: Optional[str] = None
    extracted_tables: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_images: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Financial data extraction
    financial_statements: Dict[str, Any] = Field(default_factory=dict)
    key_metrics: Dict[str, Any] = Field(default_factory=dict)

    # RAG storage
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = 0
    embedding_ids: List[str] = Field(default_factory=list)
    vector_store_id: Optional[str] = None

    # Knowledge graph
    graph_node_ids: List[str] = Field(default_factory=list)
    entities_extracted: List[Dict[str, str]] = Field(default_factory=list)
    relationships_extracted: List[Dict[str, Any]] = Field(default_factory=list)

    # Shareholders/equity holders focus
    shareholders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted shareholder information with equity details"
    )
    equity_issuances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Equity issuance dates and attributions"
    )

    # Processing metrics
    download_time_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @validator("form_type", pre=True)
    def normalize_form_type(cls, v):
        """Normalize form type string to enum."""
        if isinstance(v, str):
            return FilingType.from_string(v)
        return v

    @validator("updated_at", always=True)
    def update_timestamp(cls, v):
        """Always update the timestamp when model is modified."""
        return datetime.utcnow()

    def mark_failed(self, error: str, trace: Optional[str] = None):
        """Mark filing as failed with error details."""
        self.status = FilingStatus.FAILED
        self.error_message = error
        self.error_trace = trace
        self.updated_at = datetime.utcnow()

    def advance_stage(self, stage: ProcessingStage):
        """Advance to next processing stage."""
        self.current_stage = stage
        self.updated_at = datetime.utcnow()

    def should_retry(self, max_retries: int = 3) -> bool:
        """Check if filing should be retried."""
        return (
            self.status in [FilingStatus.FAILED, FilingStatus.RETRY]
            and self.processing_attempts < max_retries
        )

    class Config:
        """Pydantic config."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IngestionJob(BaseModel):
    """Job tracking for async processing."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filing_id: str = Field(..., description="Related filing ID")
    accession_number: str

    # Job type and status
    task_type: str = Field(..., description="Task type: download, process, embed, graph")
    status: str = Field(default="pending", description="Job status")
    priority: int = Field(default=5, description="Job priority (1-10, higher = more priority)")

    # Execution details
    worker_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    queue_name: str = Field(default="default")

    # Retry management
    retry_count: int = 0
    max_retries: int = 3
    retry_after: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None
    error_trace: Optional[str] = None
    error_count: int = 0

    # Performance metrics
    queue_time_seconds: Optional[float] = None
    execution_time_seconds: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Job configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def mark_started(self, worker_id: str, task_id: Optional[str] = None):
        """Mark job as started."""
        self.status = "running"
        self.worker_id = worker_id
        self.celery_task_id = task_id
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        if self.created_at:
            self.queue_time_seconds = (self.started_at - self.created_at).total_seconds()

    def mark_completed(self, result: Optional[Dict[str, Any]] = None):
        """Mark job as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.result = result

        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def mark_failed(self, error: str, trace: Optional[str] = None):
        """Mark job as failed."""
        self.status = "failed"
        self.error_message = error
        self.error_trace = trace
        self.error_count += 1
        self.updated_at = datetime.utcnow()

        if self.retry_count < self.max_retries:
            self.status = "retry"
            self.retry_count += 1
            # Exponential backoff for retry
            import math
            delay_minutes = math.pow(2, self.retry_count)
            self.retry_after = datetime.utcnow() + timedelta(minutes=delay_minutes)

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RSSFeedEntry(BaseModel):
    """Parsed RSS feed entry."""

    title: str
    link: HttpUrl
    published: datetime
    updated: Optional[datetime] = None
    summary: Optional[str] = None

    # EDGAR-specific fields (from edgar namespace in RSS)
    company_name: Optional[str] = None
    cik_number: Optional[str] = None
    form_type: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    file_number: Optional[str] = None
    acceptance_datetime: Optional[str] = None
    period: Optional[str] = None
    assistant_director: Optional[str] = None
    assigned_sic: Optional[str] = None
    fiscal_year_end: Optional[str] = None

    # XBRL fields if available
    xbrl_files: List[Dict[str, Any]] = Field(default_factory=list)

    def to_sec_filing(self) -> SECFiling:
        """Convert RSS entry to SECFiling model."""
        # Parse dates
        filing_date_parsed = None
        if self.filing_date:
            try:
                filing_date_parsed = datetime.strptime(self.filing_date, "%m/%d/%Y")
            except:
                filing_date_parsed = datetime.utcnow()

        acceptance_dt_parsed = datetime.utcnow()
        if self.acceptance_datetime:
            try:
                acceptance_dt_parsed = datetime.fromisoformat(self.acceptance_datetime.replace("Z", "+00:00"))
            except:
                pass

        period_date_parsed = None
        if self.period:
            try:
                period_date_parsed = datetime.strptime(self.period, "%m/%d/%Y")
            except:
                pass

        return SECFiling(
            accession_number=self.accession_number or "",
            company_name=self.company_name or self.title,
            cik_number=self.cik_number or "",
            form_type=FilingType.from_string(self.form_type or "OTHER"),
            filing_date=filing_date_parsed or self.published,
            acceptance_datetime=acceptance_dt_parsed,
            period_date=period_date_parsed,
            file_number=self.file_number,
            filing_url=self.link,
            status=FilingStatus.DISCOVERED,
            current_stage=ProcessingStage.RSS_DISCOVERY
        )


class ProcessingMetrics(BaseModel):
    """Metrics for monitoring processing performance."""

    total_filings_discovered: int = 0
    total_filings_queued: int = 0
    total_filings_processing: int = 0
    total_filings_completed: int = 0
    total_filings_failed: int = 0

    total_jobs_created: int = 0
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0
    total_jobs_retried: int = 0

    average_download_time: float = 0.0
    average_processing_time: float = 0.0
    average_queue_time: float = 0.0

    error_rate: float = 0.0
    success_rate: float = 0.0

    last_rss_poll: Optional[datetime] = None
    last_filing_processed: Optional[datetime] = None

    current_queue_size: int = 0
    current_processing_count: int = 0

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def calculate_rates(self):
        """Calculate success and error rates."""
        total_processed = self.total_filings_completed + self.total_filings_failed
        if total_processed > 0:
            self.error_rate = self.total_filings_failed / total_processed
            self.success_rate = self.total_filings_completed / total_processed


from datetime import timedelta