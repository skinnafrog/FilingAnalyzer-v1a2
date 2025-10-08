"""
Configuration settings for the Financial Intelligence Platform.
Uses pydantic-settings for environment variable management.
"""
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Settings
    APP_NAME: str = Field(default="Financial Intelligence Platform")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    # RSS Feed Configuration
    RSS_FEED_URL: str = Field(
        default="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=40&output=atom",
        description="SEC RSS feed URL for monitoring filings"
    )
    RSS_POLL_INTERVAL: int = Field(
        default=600,  # 10 minutes
        description="RSS feed polling interval in seconds"
    )
    RSS_FEED_LIMIT: int = Field(
        default=100,
        description="Maximum number of entries to process from RSS feed"
    )

    # SEC API Configuration
    SEC_USER_AGENT: str = Field(
        default="FinancialIntelligencePlatform/1.0 (admin@example.com)",
        description="User-Agent header for SEC API requests"
    )
    SEC_RATE_LIMIT_DELAY: float = Field(
        default=0.1,  # 10 requests per second max
        description="Delay between SEC API requests in seconds"
    )
    SEC_MAX_RETRIES: int = Field(default=3)
    SEC_RETRY_DELAY: float = Field(default=1.0)

    # LLM Configuration
    LLM_PROVIDER: Literal["openai", "anthropic", "azure"] = Field(default="openai")
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None)
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None)
    LLM_MODEL: str = Field(default="gpt-4-turbo-preview")
    LLM_TEMPERATURE: float = Field(default=0.7)
    LLM_MAX_TOKENS: int = Field(default=4000)

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    EMBEDDING_DIMENSION: int = Field(default=1536)
    EMBEDDING_BATCH_SIZE: int = Field(default=100)

    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/financial_intel",
        description="PostgreSQL database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=20)

    # Neo4j Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")
    NEO4J_DATABASE: str = Field(default="neo4j")

    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_POOL_SIZE: int = Field(default=10)

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    CELERY_TASK_SERIALIZER: str = Field(default="json")
    CELERY_RESULT_SERIALIZER: str = Field(default="json")
    CELERY_ACCEPT_CONTENT: list = Field(default=["json"])
    CELERY_TIMEZONE: str = Field(default="UTC")
    CELERY_ENABLE_UTC: bool = Field(default=True)

    # Vector Store Configuration
    VECTOR_DB_TYPE: Literal["pinecone", "qdrant", "chroma", "weaviate"] = Field(
        default="qdrant"
    )
    PINECONE_API_KEY: Optional[str] = Field(default=None)
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None)
    PINECONE_INDEX_NAME: str = Field(default="sec-filings")

    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_COLLECTION_NAME: str = Field(default="sec_filings")

    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8000)
    CHROMA_COLLECTION_NAME: str = Field(default="sec_filings")

    WEAVIATE_URL: str = Field(default="http://localhost:8080")
    WEAVIATE_API_KEY: Optional[str] = Field(default=None)

    # Document Processing Configuration
    DOCLING_MAX_WORKERS: int = Field(default=4)
    DOCLING_ENABLE_OCR: bool = Field(default=True)
    DOCLING_EXTRACT_TABLES: bool = Field(default=True)
    DOCLING_EXTRACT_IMAGES: bool = Field(default=True)
    DOCLING_CHUNK_SIZE: int = Field(default=1000)
    DOCLING_CHUNK_OVERLAP: int = Field(default=200)

    # Graphiti Configuration
    GRAPHITI_MAX_TOKENS: int = Field(default=8000)
    GRAPHITI_ENTITY_TYPES: list = Field(
        default=[
            "Company", "Person", "Product", "Location",
            "Date", "Money", "Percentage", "Shareholder"
        ]
    )
    GRAPHITI_RELATIONSHIP_TYPES: list = Field(
        default=[
            "owns", "works_for", "located_in", "filed_on",
            "invested_in", "acquired", "partnered_with"
        ]
    )

    # File Storage Configuration
    FILING_STORAGE_PATH: str = Field(default="./data/filings")
    PROCESSED_STORAGE_PATH: str = Field(default="./data/processed")
    TEMP_STORAGE_PATH: str = Field(default="./data/temp")
    MAX_FILE_SIZE_MB: int = Field(default=100)

    # Processing Configuration
    MAX_PROCESSING_TIME_SECONDS: int = Field(default=120)  # 2 minutes per filing
    MAX_CONCURRENT_DOWNLOADS: int = Field(default=5)
    MAX_CONCURRENT_PROCESSING: int = Field(default=3)
    ERROR_RATE_THRESHOLD: float = Field(default=0.05)  # 5% error rate threshold

    # Authentication (for future API)
    JWT_SECRET_KEY: str = Field(default="your-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRY_HOURS: int = Field(default=24)

    # Monitoring & Observability
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=8001)
    ENABLE_TRACING: bool = Field(default=False)
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None)
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None)
    LANGFUSE_HOST: Optional[str] = Field(default="https://cloud.langfuse.com")

    @field_validator("SEC_USER_AGENT")
    @classmethod
    def validate_user_agent(cls, v):
        """Ensure User-Agent follows SEC requirements."""
        if not v or "@" not in v:
            raise ValueError(
                "SEC_USER_AGENT must include contact email (e.g., 'AppName/1.0 (email@example.com)')"
            )
        return v

    @model_validator(mode='after')
    def validate_llm_provider(self):
        """Ensure appropriate API key is set for the selected provider."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI provider")
        elif self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set when using Anthropic provider")
        elif self.LLM_PROVIDER == "azure" and not self.AZURE_OPENAI_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY must be set when using Azure provider")
        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG

    @property
    def database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy."""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return self.DATABASE_URL

    @property
    def redis_url_decoded(self) -> dict:
        """Parse Redis URL into connection parameters."""
        from urllib.parse import urlparse
        parsed = urlparse(self.REDIS_URL)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "db": int(parsed.path.lstrip("/")) if parsed.path else 0,
            "password": parsed.password
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()