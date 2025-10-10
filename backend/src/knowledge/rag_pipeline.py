"""
RAG pipeline for document chunking, embedding generation, and vector storage.
Handles the creation and management of embeddings for retrieval-augmented generation.
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
from openai import AsyncOpenAI
import numpy as np

from ..config.settings import Settings, get_settings
from ..config.prompts import (
    get_system_prompt,
    format_context_for_llm,
    format_user_query
)
from ..models.filing import SECFiling, ProcessingStage
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embedding: Optional[List[float]] = None
    token_count: int = 0


class RAGPipeline:
    """Pipeline for RAG processing: chunking, embedding, and storage."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize RAG pipeline."""
        self.settings = settings or get_settings()

        # OpenAI client for embeddings
        self.openai_client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        self.embedding_model = self.settings.EMBEDDING_MODEL
        self.embedding_dimension = self.settings.EMBEDDING_DIMENSION
        self.embedding_batch_size = self.settings.EMBEDDING_BATCH_SIZE

        # Chunking configuration
        self.chunk_size = self.settings.DOCLING_CHUNK_SIZE
        self.chunk_overlap = self.settings.DOCLING_CHUNK_OVERLAP

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Token encoder for counting
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # Cache for embeddings
        self.embedding_cache: Dict[str, List[float]] = {}

        # Statistics
        self.total_chunks_processed = 0
        self.total_embeddings_generated = 0
        self.total_tokens_processed = 0

        # Initialize vector store
        try:
            self.vector_store = VectorStore()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None

    def _token_length(self, text: str) -> int:
        """
        Calculate token length of text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))

    async def process_filing(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process filing through RAG pipeline.

        Args:
            filing: SECFiling object
            extracted_data: Extracted document data from Docling

        Returns:
            Tuple of (success, rag_data)
        """
        logger.info(f"Processing {filing.accession_number} through RAG pipeline")

        # Update filing status
        filing.current_stage = ProcessingStage.TEXT_CHUNKING

        try:
            # Create chunks from extracted text and tables
            chunks = await self._create_chunks(filing, extracted_data)

            if not chunks:
                raise ValueError("No chunks created from document")

            filing.chunk_count = len(chunks)
            filing.chunks = [self._chunk_to_dict(chunk) for chunk in chunks]

            logger.info(f"Created {len(chunks)} chunks for {filing.accession_number}")

            # Update stage
            filing.current_stage = ProcessingStage.EMBEDDING_GENERATION

            # Generate embeddings
            chunks_with_embeddings = await self._generate_embeddings(chunks)

            # Count tokens
            total_tokens = sum(chunk.token_count for chunk in chunks_with_embeddings)
            self.total_tokens_processed += total_tokens

            logger.info(
                f"Generated embeddings for {len(chunks_with_embeddings)} chunks "
                f"({total_tokens} tokens)"
            )

            # Prepare for storage
            filing.current_stage = ProcessingStage.VECTOR_STORAGE

            # Store in vector database and get vector IDs
            chunks_data = []
            vector_stored = False

            if self.vector_store and chunks_with_embeddings:
                try:
                    # Prepare chunks for vector storage
                    chunks_for_storage = []
                    for chunk in chunks_with_embeddings:
                        chunk_dict = self._chunk_to_dict(chunk)
                        # Add filing metadata
                        chunk_dict['accession_number'] = filing.accession_number
                        chunk_dict['company_name'] = filing.company_name
                        chunk_dict['form_type'] = filing.form_type
                        chunk_dict['filing_date'] = filing.filing_date.isoformat() if filing.filing_date else None
                        chunks_for_storage.append(chunk_dict)

                    # Store in Qdrant and get vector IDs
                    vector_ids = await self.vector_store.store_chunks(
                        chunks_for_storage,
                        filing.accession_number
                    )

                    if vector_ids:
                        vector_stored = True
                        logger.info(f"Successfully stored {len(vector_ids)} vectors for {filing.accession_number}")

                        # Add vector IDs to chunks data
                        for chunk_dict, vector_id in zip(chunks_for_storage, vector_ids):
                            chunk_dict['vector_id'] = vector_id
                            chunks_data.append(chunk_dict)
                    else:
                        logger.warning(f"Failed to store vectors for {filing.accession_number}")
                        chunks_data = chunks_for_storage
                except Exception as e:
                    logger.error(f"Error storing vectors: {e}")
                    chunks_data = [self._chunk_to_dict(chunk) for chunk in chunks_with_embeddings]
            else:
                chunks_data = [self._chunk_to_dict(chunk) for chunk in chunks_with_embeddings]

            storage_data = {
                "chunks": chunks_data,
                "total_chunks": len(chunks_with_embeddings),
                "total_tokens": total_tokens,
                "metadata": self._create_filing_metadata(filing),
                "vectors_stored": vector_stored
            }

            # Update filing
            filing.embedding_ids = [chunk.chunk_id for chunk in chunks_with_embeddings]

            self.total_chunks_processed += len(chunks)
            self.total_embeddings_generated += len(chunks_with_embeddings)

            return True, storage_data

        except Exception as e:
            error_msg = f"RAG processing failed: {str(e)}"
            logger.error(f"Failed to process {filing.accession_number}: {error_msg}", exc_info=True)
            return False, {"error": error_msg}

    async def _create_chunks(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Create chunks from extracted document data.

        Args:
            filing: SECFiling object
            extracted_data: Extracted document data

        Returns:
            List of Chunk objects
        """
        chunks = []

        # Process main text
        main_text = extracted_data.get("text", "")
        if main_text:
            text_chunks = await self._chunk_text(
                main_text,
                filing,
                source_type="main_text"
            )
            chunks.extend(text_chunks)

        # Process sections separately for better context
        sections = extracted_data.get("sections", [])
        for section in sections:
            section_text = section.get("text", "")
            if section_text:
                section_chunks = await self._chunk_text(
                    section_text,
                    filing,
                    source_type="section",
                    additional_metadata={"section_title": section.get("title", "")}
                )
                chunks.extend(section_chunks)

        # Process tables as separate chunks
        tables = extracted_data.get("tables", [])
        for idx, table in enumerate(tables):
            table_chunk = await self._create_table_chunk(
                table,
                filing,
                table_index=idx
            )
            if table_chunk:
                chunks.append(table_chunk)

        # Process financial statements specially
        financial_statements = filing.financial_statements
        if financial_statements:
            for statement_type, statement_data in financial_statements.items():
                statement_chunk = await self._create_financial_chunk(
                    statement_type,
                    statement_data,
                    filing
                )
                if statement_chunk:
                    chunks.append(statement_chunk)

        # Add shareholder information as chunks
        if filing.shareholders:
            shareholder_chunk = await self._create_shareholder_chunk(
                filing.shareholders,
                filing
            )
            if shareholder_chunk:
                chunks.append(shareholder_chunk)

        return chunks

    async def _chunk_text(
        self,
        text: str,
        filing: SECFiling,
        source_type: str,
        additional_metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Chunk text with metadata.

        Args:
            text: Text to chunk
            filing: SECFiling object
            source_type: Type of source
            additional_metadata: Additional metadata

        Returns:
            List of chunks
        """
        # Split text
        documents = self.text_splitter.create_documents(
            [text],
            metadatas=[{
                "source": source_type,
                "filing_id": filing.id,
                "accession_number": filing.accession_number,
                "company_name": filing.company_name,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                **(additional_metadata or {})
            }]
        )

        # Convert to Chunk objects
        chunks = []
        for idx, doc in enumerate(documents):
            chunk_id = self._generate_chunk_id(
                filing.accession_number,
                source_type,
                idx,
                doc.page_content
            )

            chunk = Chunk(
                content=doc.page_content,
                metadata=doc.metadata,
                chunk_id=chunk_id,
                token_count=self._token_length(doc.page_content)
            )
            chunks.append(chunk)

        return chunks

    async def _create_table_chunk(
        self,
        table: Dict[str, Any],
        filing: SECFiling,
        table_index: int
    ) -> Optional[Chunk]:
        """
        Create a chunk from a table.

        Args:
            table: Table data
            filing: SECFiling object
            table_index: Table index

        Returns:
            Chunk object or None
        """
        try:
            # Format table as text
            table_text = f"Table: {table.get('title', f'Table {table_index + 1}')}\n\n"

            # Add column headers
            columns = table.get("metadata", {}).get("columns", [])
            if columns:
                table_text += " | ".join(str(col) for col in columns) + "\n"
                table_text += "-" * 50 + "\n"

            # Add rows
            rows = table.get("rows", [])
            for row in rows[:20]:  # Limit rows to avoid huge chunks
                if isinstance(row, dict):
                    row_values = [str(row.get(col, "")) for col in columns] if columns else list(row.values())
                else:
                    row_values = [str(val) for val in row]
                table_text += " | ".join(row_values) + "\n"

            if len(rows) > 20:
                table_text += f"\n... ({len(rows) - 20} more rows)\n"

            # Create chunk
            chunk_id = self._generate_chunk_id(
                filing.accession_number,
                "table",
                table_index,
                table_text
            )

            return Chunk(
                content=table_text,
                metadata={
                    "source": "table",
                    "table_index": table_index,
                    "table_title": table.get("title", ""),
                    "is_financial": table.get("is_financial", False),
                    "row_count": len(rows),
                    "filing_id": filing.id,
                    "accession_number": filing.accession_number,
                    "company_name": filing.company_name,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date.isoformat()
                },
                chunk_id=chunk_id,
                token_count=self._token_length(table_text)
            )

        except Exception as e:
            logger.error(f"Error creating table chunk: {e}")
            return None

    async def _create_financial_chunk(
        self,
        statement_type: str,
        statement_data: Dict[str, Any],
        filing: SECFiling
    ) -> Optional[Chunk]:
        """
        Create a chunk from financial statement.

        Args:
            statement_type: Type of financial statement
            statement_data: Statement data
            filing: SECFiling object

        Returns:
            Chunk object or None
        """
        try:
            # Format financial statement as text
            statement_text = f"Financial Statement: {statement_type.replace('_', ' ').title()}\n\n"

            # Add statement data
            if isinstance(statement_data, dict):
                # If it's a table structure
                if "rows" in statement_data:
                    return await self._create_table_chunk(statement_data, filing, 0)

                # Otherwise format as key-value pairs
                for key, value in statement_data.items():
                    statement_text += f"{key}: {value}\n"

            chunk_id = self._generate_chunk_id(
                filing.accession_number,
                "financial",
                0,
                statement_text
            )

            return Chunk(
                content=statement_text,
                metadata={
                    "source": "financial_statement",
                    "statement_type": statement_type,
                    "filing_id": filing.id,
                    "accession_number": filing.accession_number,
                    "company_name": filing.company_name,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date.isoformat()
                },
                chunk_id=chunk_id,
                token_count=self._token_length(statement_text)
            )

        except Exception as e:
            logger.error(f"Error creating financial chunk: {e}")
            return None

    async def _create_shareholder_chunk(
        self,
        shareholders: List[Dict[str, Any]],
        filing: SECFiling
    ) -> Optional[Chunk]:
        """
        Create a chunk from shareholder information.

        Args:
            shareholders: List of shareholder data
            filing: SECFiling object

        Returns:
            Chunk object or None
        """
        try:
            # Format shareholder information
            shareholder_text = "Shareholder Information\n\n"

            for shareholder in shareholders[:20]:  # Limit to avoid huge chunks
                name = shareholder.get("name", "Unknown")
                percentage = shareholder.get("percentage", "")
                shares = shareholder.get("shares", "")

                shareholder_text += f"- {name}"
                if percentage:
                    shareholder_text += f": {percentage}%"
                if shares:
                    shareholder_text += f" ({shares} shares)"
                shareholder_text += "\n"

            if len(shareholders) > 20:
                shareholder_text += f"\n... ({len(shareholders) - 20} more shareholders)\n"

            chunk_id = self._generate_chunk_id(
                filing.accession_number,
                "shareholders",
                0,
                shareholder_text
            )

            return Chunk(
                content=shareholder_text,
                metadata={
                    "source": "shareholders",
                    "shareholder_count": len(shareholders),
                    "filing_id": filing.id,
                    "accession_number": filing.accession_number,
                    "company_name": filing.company_name,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date.isoformat()
                },
                chunk_id=chunk_id,
                token_count=self._token_length(shareholder_text)
            )

        except Exception as e:
            logger.error(f"Error creating shareholder chunk: {e}")
            return None

    async def _generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with embeddings
        """
        chunks_with_embeddings = []
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process in batches
        for i in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[i:i + self.embedding_batch_size]
            logger.debug(f"Processing batch {i // self.embedding_batch_size + 1}, size: {len(batch)}")

            # Check cache first
            uncached_chunks = []
            for chunk in batch:
                if chunk.chunk_id in self.embedding_cache:
                    chunk.embedding = self.embedding_cache[chunk.chunk_id]
                    chunks_with_embeddings.append(chunk)
                else:
                    uncached_chunks.append(chunk)

            if uncached_chunks:
                logger.info(f"Generating embeddings for {len(uncached_chunks)} uncached chunks")
                # Generate embeddings for uncached chunks
                texts = [chunk.content for chunk in uncached_chunks]

                try:
                    # Call OpenAI API with retry logic
                    embeddings = await self._generate_embeddings_with_retry(texts)
                    logger.info(f"Successfully generated {len(embeddings)} embeddings")

                    # Assign embeddings to chunks
                    for chunk, embedding in zip(uncached_chunks, embeddings):
                        if embedding and len(embedding) > 0:
                            chunk.embedding = embedding
                            # Cache the embedding
                            self.embedding_cache[chunk.chunk_id] = embedding
                            chunks_with_embeddings.append(chunk)
                            logger.debug(f"Added embedding for chunk {chunk.chunk_id}, dim: {len(embedding)}")
                        else:
                            logger.warning(f"Empty embedding for chunk {chunk.chunk_id}")
                            chunks_with_embeddings.append(chunk)

                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}", exc_info=True)
                    # Continue without embeddings for failed batch
                    chunks_with_embeddings.extend(uncached_chunks)

        return chunks_with_embeddings

    async def _generate_embeddings_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Generate embeddings with retry logic.

        Args:
            texts: List of texts to embed
            max_retries: Maximum retry attempts

        Returns:
            List of embeddings
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Calling OpenAI API for {len(texts)} texts with model {self.embedding_model}")
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )

                embeddings = [item.embedding for item in response.data]
                logger.info(f"OpenAI API returned {len(embeddings)} embeddings")
                return embeddings

            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise

        raise Exception("Failed to generate embeddings after retries")

    def _generate_chunk_id(
        self,
        accession_number: str,
        source_type: str,
        index: int,
        content: str
    ) -> str:
        """
        Generate unique chunk ID.

        Args:
            accession_number: Filing accession number
            source_type: Type of source
            index: Chunk index
            content: Chunk content

        Returns:
            Unique chunk ID
        """
        # Create hash from content for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{accession_number}_{source_type}_{index}_{content_hash}"

    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Convert Chunk to dictionary.

        Args:
            chunk: Chunk object

        Returns:
            Dictionary representation
        """
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "token_count": chunk.token_count
        }

    def _create_filing_metadata(self, filing: SECFiling) -> Dict[str, Any]:
        """
        Create filing metadata for storage.

        Args:
            filing: SECFiling object

        Returns:
            Metadata dictionary
        """
        return {
            "filing_id": filing.id,
            "accession_number": filing.accession_number,
            "company_name": filing.company_name,
            "cik_number": filing.cik_number,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date.isoformat(),
            "filing_url": str(filing.filing_url),
            "chunk_count": filing.chunk_count,
            "shareholder_count": len(filing.shareholders),
            "has_financial_data": bool(filing.financial_statements)
        }

    async def search_similar(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks (placeholder for vector search).

        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters

        Returns:
            List of similar chunks
        """
        # Generate query embedding
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            query_embedding = response.data[0].embedding

            # This would normally search the vector database
            # Placeholder implementation
            logger.info(f"Searching for: {query} (k={k})")

            return []

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _retrieve_from_database(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents directly from database.

        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum number of documents

        Returns:
            List of document chunks with metadata
        """
        from ..database import get_db_context
        from ..database.models import Filing, Company, FilingDocument, FilingChunk
        from sqlalchemy import or_, and_, func

        try:
            with get_db_context() as db:
                # Build base query joining tables
                query_obj = db.query(
                    FilingChunk,
                    Filing,
                    Company,
                    FilingDocument
                ).join(
                    Filing, FilingChunk.filing_id == Filing.id
                ).join(
                    Company, Filing.company_id == Company.id
                ).join(
                    FilingDocument, FilingChunk.document_id == FilingDocument.id
                )

                # Apply filters if provided
                if filters:
                    logger.info(f"Applying filters: {filters}")
                    if filters.get("company"):
                        query_obj = query_obj.filter(
                            or_(
                                Company.name.ilike(f"%{filters['company']}%"),
                                Company.ticker == filters['company'].upper()
                            )
                        )
                    if filters.get("form_type"):
                        query_obj = query_obj.filter(Filing.form_type == filters["form_type"])
                    if filters.get("accession_number"):
                        query_obj = query_obj.filter(Filing.accession_number == filters["accession_number"])

                # Search in chunk text and document content
                # But skip text search if we have an accession_number filter
                if filters and filters.get("accession_number"):
                    logger.info("Using accession number filter for chunks, skipping text search")
                    # Just use the accession filter, no text search needed
                elif query:
                    search_pattern = f"%{query}%"
                    query_obj = query_obj.filter(
                        or_(
                            FilingChunk.text.ilike(search_pattern),
                            Company.name.ilike(search_pattern),
                            Filing.summary.ilike(search_pattern) if hasattr(Filing, 'summary') else False
                        )
                    )

                # Order by relevance (simple text matching for now)
                # In production, use proper text search or vector similarity
                query_obj = query_obj.order_by(Filing.filing_date.desc())

                # Limit results
                results = query_obj.limit(limit).all()

                # Format results
                formatted_results = []
                for chunk, filing, company, document in results:
                    formatted_results.append({
                        "text": chunk.text[:1000] if chunk.text else "",  # Limit text length
                        "accession_number": filing.accession_number,
                        "company_name": company.name,
                        "ticker_symbol": company.ticker if hasattr(company, 'ticker') else None,
                        "form_type": filing.form_type,
                        "filing_date": filing.filing_date.isoformat() if filing.filing_date else "",
                        "chunk_index": chunk.chunk_index if hasattr(chunk, 'chunk_index') else 0,
                        "document_type": document.document_type if hasattr(document, 'document_type') else filing.form_type,
                        "score": 0.85,  # Default relevance score
                        "metadata": chunk.metadata_json if hasattr(chunk, 'metadata_json') else {}
                    })

                # If no chunks found, try to get filing documents directly
                if not formatted_results:
                    logger.info("No chunks found, trying filing documents directly")

                    # Query filing documents directly
                    doc_query = db.query(
                        FilingDocument,
                        Filing,
                        Company
                    ).join(
                        Filing, FilingDocument.filing_id == Filing.id
                    ).join(
                        Company, Filing.company_id == Company.id
                    )

                    # Apply same filters
                    if filters:
                        logger.info(f"Applying filters to documents query: {filters}")
                        if filters.get("company"):
                            doc_query = doc_query.filter(
                                or_(
                                    Company.name.ilike(f"%{filters['company']}%"),
                                    Company.ticker == filters['company'].upper()
                                )
                            )
                        if filters.get("form_type"):
                            doc_query = doc_query.filter(Filing.form_type == filters["form_type"])
                        if filters.get("accession_number"):
                            doc_query = doc_query.filter(Filing.accession_number == filters["accession_number"])

                    # If we have an accession_number filter, don't do text search
                    # as we're looking for a specific filing
                    if filters and filters.get("accession_number"):
                        logger.info("Using accession number filter, skipping text search")
                        # Just use the accession filter, no text search needed
                    elif not query and filters:
                        logger.info("No query text, using filters only")
                        # Just filter by company/form, don't search text
                    elif query:
                        # Only search text if there's an actual query and no accession filter
                        search_pattern = f"%{query}%"
                        logger.info(f"Searching documents for pattern: {search_pattern[:50]}...")
                        doc_query = doc_query.filter(
                            or_(
                                FilingDocument.text_content.ilike(search_pattern),
                                Company.name.ilike(search_pattern),
                                Filing.summary.ilike(search_pattern) if Filing.summary else False
                            )
                        )

                    doc_query = doc_query.order_by(Filing.filing_date.desc())
                    doc_results = doc_query.limit(limit).all()

                    for document, filing, company in doc_results:
                        # Create chunks from document text on the fly
                        text_content = document.text_content if document.text_content else filing.summary if hasattr(filing, 'summary') else ""
                        if text_content:
                            # Take first 1000 chars as a chunk
                            formatted_results.append({
                                "text": text_content[:1000],
                                "accession_number": filing.accession_number,
                                "company_name": company.name,
                                "ticker_symbol": company.ticker if hasattr(company, 'ticker') else None,
                                "form_type": filing.form_type,
                                "filing_date": filing.filing_date.isoformat() if filing.filing_date else "",
                                "chunk_index": 0,
                                "document_type": document.document_type if hasattr(document, 'document_type') else filing.form_type,
                                "score": 0.75,  # Lower score for non-chunk results
                                "metadata": {}
                            })

                logger.info(f"Retrieved {len(formatted_results)} documents from database")
                return formatted_results

        except Exception as e:
            logger.error(f"Database retrieval error: {e}", exc_info=True)
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get RAG pipeline statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_tokens_processed": self.total_tokens_processed,
            "embedding_cache_size": len(self.embedding_cache),
            "average_tokens_per_chunk": (
                self.total_tokens_processed / self.total_chunks_processed
                if self.total_chunks_processed > 0 else 0
            )
        }

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            filters: Optional filters for document retrieval
            limit: Maximum number of documents to return

        Returns:
            List of relevant document chunks
        """
        try:
            # First try vector search if available
            results = await self.search_similar(query, k=limit, filters=filters)

            # If no results from vector store, query database directly
            if not results:
                logger.info("Falling back to database search")
                results = await self._retrieve_from_database(query, filters, limit)

            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            # Return empty list on error to avoid breaking the chat
            return []

    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chat_history: Optional[List] = None
    ) -> str:
        """
        Generate a response using retrieved context.

        Args:
            query: User query
            context: Retrieved document chunks
            chat_history: Optional chat history

        Returns:
            Generated response
        """
        try:
            # Format context using our prompt utilities
            context_str = format_context_for_llm(context, max_docs=5)

            # Determine appropriate prompt type based on context
            if not context or len(context) == 0:
                prompt_type = "no_context"
            else:
                prompt_type = "analysis"

            # Build messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": get_system_prompt(prompt_type=prompt_type)
                }
            ]

            # Add chat history if provided
            if chat_history:
                for msg in chat_history[-5:]:  # Last 5 messages
                    # Handle both dict and object formats
                    if isinstance(msg, dict):
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    else:
                        messages.append({
                            "role": msg.role if hasattr(msg, 'role') else 'user',
                            "content": msg.content if hasattr(msg, 'content') else str(msg)
                        })

            # Extract metadata about specific filing if mentioned in query
            query_metadata = {}
            # Check if query contains an accession number
            import re
            accession_pattern = r'\d{10}-\d{2}-\d{6}'
            accession_match = re.search(accession_pattern, query)
            if accession_match:
                query_metadata['accession_number'] = accession_match.group()

            # Add current query with context
            messages.append({
                "role": "user",
                "content": format_user_query(query, context_str, query_metadata)
            })

            # Generate response
            response = await self.openai_client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response. Please try again. Error: {str(e)}"

    async def generate_response_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chat_history: Optional[List] = None
    ):
        """
        Generate a streaming response using retrieved context.

        Args:
            query: User query
            context: Retrieved document chunks
            chat_history: Optional chat history

        Yields:
            Response chunks
        """
        try:
            # Format context using our prompt utilities
            context_str = format_context_for_llm(context, max_docs=5)

            # Use streaming prompt type
            prompt_type = "streaming" if context and len(context) > 0 else "no_context"

            # Build messages for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": get_system_prompt(prompt_type=prompt_type)
                }
            ]

            # Add chat history if provided
            if chat_history:
                for msg in chat_history[-5:]:
                    # Handle both dict and object formats
                    if isinstance(msg, dict):
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    else:
                        messages.append({
                            "role": msg.role if hasattr(msg, 'role') else 'user',
                            "content": msg.content if hasattr(msg, 'content') else str(msg)
                        })

            # Extract metadata about specific filing if mentioned in query
            query_metadata = {}
            import re
            accession_pattern = r'\d{10}-\d{2}-\d{6}'
            accession_match = re.search(accession_pattern, query)
            if accession_match:
                query_metadata['accession_number'] = accession_match.group()

            # Add current query with context
            messages.append({
                "role": "user",
                "content": format_user_query(query, context_str, query_metadata)
            })

            # Generate streaming response
            stream = await self.openai_client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
