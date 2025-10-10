#!/usr/bin/env python
"""
Script to reprocess existing chunks to generate embeddings and store in Qdrant.
"""
import asyncio
import os
import sys
import logging
from typing import List, Dict, Any
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Add src to path
sys.path.insert(0, '/app')

from src.database.models import Filing, FilingChunk, Company
from src.knowledge.rag_pipeline import RAGPipeline, Chunk
from openai import AsyncOpenAI
from src.knowledge.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingReprocessor:
    """Reprocesses existing chunks to generate and store embeddings."""

    def __init__(self):
        self.vector_store = VectorStore(host="qdrant", port=6333)
        # Initialize OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.openai_client = AsyncOpenAI(api_key=api_key)

        # Convert regular DATABASE_URL to async version
        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/financial_intel")
        self.database_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def get_filings_without_vectors(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get filings that have chunks but no vector IDs."""
        async with self.async_session() as session:
            # Query for filings with chunks that have no vector_id
            query = (
                select(Filing, Company)
                .join(FilingChunk, Filing.id == FilingChunk.filing_id)
                .outerjoin(Company, Filing.company_id == Company.id)
                .where(FilingChunk.vector_id.is_(None))
                .group_by(Filing.id, Company.id)
                .order_by(Filing.filing_date.desc())
            )

            if limit:
                query = query.limit(limit)

            result = await session.execute(query)
            rows = result.unique().all()

            # Extract filing and company info
            filings_data = []
            for filing, company in rows:
                filing.company_name = company.name if company else "Unknown"
                filing.cik = company.cik if company else None
                filings_data.append(filing)

            logger.info(f"Found {len(filings_data)} filings with chunks missing vectors")
            return filings_data

    async def get_chunks_for_filing(self, filing_id: str) -> List[FilingChunk]:
        """Get all chunks for a filing that don't have vector IDs."""
        async with self.async_session() as session:
            query = (
                select(FilingChunk)
                .where(
                    and_(
                        FilingChunk.filing_id == filing_id,
                        FilingChunk.vector_id.is_(None)
                    )
                )
                .order_by(FilingChunk.chunk_index)
            )

            result = await session.execute(query)
            chunks = result.scalars().all()
            return chunks

    async def process_filing(self, filing: Filing) -> bool:
        """Process a single filing to generate embeddings."""
        try:
            logger.info(f"Processing filing {filing.id}: {filing.company_name} - {filing.accession_number}")

            # Get chunks without vectors
            chunks = await self.get_chunks_for_filing(filing.id)
            if not chunks:
                logger.info(f"No chunks without vectors for filing {filing.id}")
                return True

            logger.info(f"Found {len(chunks)} chunks to process")

            # Prepare chunk data for embedding generation
            chunk_data = []
            for chunk in chunks:
                chunk_dict = {
                    'chunk_id': chunk.id,
                    'chunk_index': chunk.chunk_index or 0,
                    'content': chunk.text,  # FilingChunk uses 'text' field
                    'source_type': 'text',  # Default to text
                    'section': chunk.section or '',
                    'accession_number': filing.accession_number,
                    'company_name': getattr(filing, 'company_name', 'Unknown'),
                    'form_type': filing.form_type,
                    'filing_date': filing.filing_date.isoformat() if filing.filing_date else '',
                    'token_count': chunk.token_count or 0
                }
                chunk_data.append(chunk_dict)

            # Generate embeddings using OpenAI directly
            logger.info(f"Generating embeddings for {len(chunk_data)} chunks...")
            chunks_with_embeddings = []

            for chunk in chunk_data:
                try:
                    # Generate embedding for chunk content
                    response = await self.openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=chunk['content'][:8000]  # Limit to 8000 chars to avoid token limit
                    )

                    # Add embedding to chunk
                    chunk['embedding'] = response.data[0].embedding
                    chunks_with_embeddings.append(chunk)

                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {chunk['chunk_id']}: {e}")
                    chunks_with_embeddings.append(chunk)  # Include even without embedding

            # Count chunks with embeddings
            chunks_with_valid_embeddings = [c for c in chunks_with_embeddings if c.get('embedding')]
            logger.info(f"Generated {len(chunks_with_valid_embeddings)} valid embeddings")

            if not chunks_with_valid_embeddings:
                logger.warning(f"No valid embeddings generated for filing {filing.id}")
                return False

            # Store in Qdrant
            logger.info(f"Storing {len(chunks_with_valid_embeddings)} vectors in Qdrant...")
            vector_ids = await self.vector_store.store_chunks(chunks_with_valid_embeddings, filing.id)

            # Update database with vector IDs
            if vector_ids:
                async with self.async_session() as session:
                    for chunk, vector_id in zip(chunks, vector_ids):
                        if vector_id:  # Only update if we got a valid vector_id
                            chunk_obj = await session.get(FilingChunk, chunk.id)
                            if chunk_obj:
                                chunk_obj.vector_id = vector_id

                    await session.commit()
                    logger.info(f"Updated {len([v for v in vector_ids if v])} chunks with vector IDs")

            return True

        except Exception as e:
            logger.error(f"Failed to process filing {filing.id}: {e}", exc_info=True)
            return False

    async def run(self, limit: int = None):
        """Run the reprocessing for all filings without vectors."""
        try:
            # Get filings to process
            filings = await self.get_filings_without_vectors(limit)

            if not filings:
                logger.info("No filings found that need vector processing")
                return

            logger.info(f"Starting reprocessing for {len(filings)} filings")

            # Process each filing
            success_count = 0
            for i, filing in enumerate(filings, 1):
                logger.info(f"Processing filing {i}/{len(filings)}")
                if await self.process_filing(filing):
                    success_count += 1

                # Add small delay to avoid rate limiting
                if i < len(filings):
                    await asyncio.sleep(1)

            logger.info(f"Reprocessing complete: {success_count}/{len(filings)} filings processed successfully")

            # Check final statistics
            await self.check_statistics()

        except Exception as e:
            logger.error(f"Reprocessing failed: {e}", exc_info=True)
        finally:
            await self.engine.dispose()

    async def check_statistics(self):
        """Check and report statistics after processing."""
        try:
            async with self.async_session() as session:
                # Count chunks with vectors
                from sqlalchemy import text
                query = text("SELECT COUNT(*) FROM filing_chunks WHERE vector_id IS NOT NULL")
                result = await session.execute(query)
                chunks_with_vectors = result.scalar()

                query = text("SELECT COUNT(*) FROM filing_chunks")
                result = await session.execute(query)
                total_chunks = result.scalar()

                logger.info(f"Final statistics: {chunks_with_vectors}/{total_chunks} chunks have vectors")

                # Check Qdrant statistics
                collection_info = await self.vector_store.get_collection_info()
                logger.info(f"Qdrant collection info: {collection_info}")

        except Exception as e:
            logger.error(f"Failed to check statistics: {e}")

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Reprocess chunks to generate embeddings")
    parser.add_argument("--limit", type=int, help="Limit number of filings to process")
    args = parser.parse_args()

    reprocessor = EmbeddingReprocessor()
    await reprocessor.run(limit=args.limit)

if __name__ == "__main__":
    asyncio.run(main())