"""SQLAlchemy database models."""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, DateTime, Text, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship

from .connection import Base


class Company(Base):
    """Company model."""
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True)
    cik = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    ticker = Column(String(10), index=True)
    sic_code = Column(String(10))
    industry = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    filings = relationship("Filing", back_populates="company", cascade="all, delete-orphan")


class Filing(Base):
    """SEC Filing model."""
    __tablename__ = "filings"

    id = Column(Integer, primary_key=True)
    accession_number = Column(String(50), unique=True, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    form_type = Column(String(20), nullable=False, index=True)
    filing_date = Column(DateTime, nullable=False, index=True)
    document_count = Column(Integer, default=1)

    # Processing status
    status = Column(String(20), default="pending", index=True)  # pending, processing, completed, failed
    processed_at = Column(DateTime)
    error_message = Column(Text)

    # Metadata
    file_url = Column(String(500))
    file_size = Column(Integer)
    download_path = Column(String(500))

    # Extracted data summary
    summary = Column(Text)
    key_metrics = Column(JSON)  # Store extracted financial metrics

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="filings")
    documents = relationship("FilingDocument", back_populates="filing", cascade="all, delete-orphan")
    chunks = relationship("FilingChunk", back_populates="filing", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('ix_filing_date_form', filing_date, form_type),
        Index('ix_company_filing_date', company_id, filing_date),
    )


class FilingDocument(Base):
    """Individual document within a filing."""
    __tablename__ = "filing_documents"

    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey("filings.id"), nullable=False)
    document_type = Column(String(50))  # e.g., "10-K", "EX-99.1"
    sequence = Column(Integer)
    filename = Column(String(255))
    description = Column(Text)

    # Extracted content
    text_content = Column(Text)
    tables_json = Column(JSON)  # Store extracted tables
    metadata_json = Column(JSON)  # Store document metadata

    # Processing info
    processed_at = Column(DateTime)
    processing_time = Column(Float)
    char_count = Column(Integer)
    table_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    filing = relationship("Filing", back_populates="documents")


class FilingChunk(Base):
    """Chunks for RAG retrieval."""
    __tablename__ = "filing_chunks"

    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey("filings.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("filing_documents.id"))

    # Chunk content
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    start_char = Column(Integer)
    end_char = Column(Integer)

    # Metadata
    section = Column(String(100))  # e.g., "Risk Factors", "MD&A"
    metadata_json = Column(JSON)

    # Vector embedding (stored separately in vector DB, this is just reference)
    vector_id = Column(String(100), unique=True)
    embedding_model = Column(String(50))
    token_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    filing = relationship("Filing", back_populates="chunks")

    # Index for retrieval
    __table_args__ = (
        Index('ix_chunk_filing_section', filing_id, section),
    )