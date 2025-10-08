"""
Main entry point for the Financial Intelligence Platform ingestion pipeline.
Orchestrates RSS monitoring, document processing, and knowledge base updates.
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import signal

from src.config.settings import get_settings
from src.models.filing import SECFiling, FilingStatus
from src.ingestion.rss_monitor import RSSMonitor
from src.ingestion.filing_downloader import FilingDownloader
from src.ingestion.docling_processor import DoclingProcessor
from src.knowledge.rag_pipeline import RAGPipeline
from src.knowledge.neo4j_store import Neo4jStore
from src.database import init_db, get_db_context
from src.database.models import Filing, Company, FilingDocument, FilingChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingestion.log')
    ]
)

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main ingestion pipeline orchestrator."""

    def __init__(self):
        """Initialize the ingestion pipeline."""
        self.settings = get_settings()
        self.running = False

        # Initialize database
        try:
            init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

        # Initialize components
        logger.info("Initializing ingestion pipeline components...")

        self.rss_monitor = RSSMonitor(self.settings)
        self.downloader = FilingDownloader(self.settings)
        try:
            self.processor = DoclingProcessor(self.settings)
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            self.processor = None
        self.rag_pipeline = RAGPipeline(self.settings)

        # Initialize Neo4j store
        try:
            self.neo4j_store = Neo4jStore(self.settings)
            logger.info("Neo4j knowledge graph initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j store: {e}")
            self.neo4j_store = None

        # Track processing
        self.processed_filings: List[str] = []
        self.failed_filings: List[str] = []

    async def process_filing_batch(self, filings: List[SECFiling]) -> None:
        """
        Process a batch of discovered filings.

        Args:
            filings: List of SEC filings to process
        """
        logger.info(f"Processing batch of {len(filings)} filings")

        for filing in filings:
            try:
                # Skip if already processed
                if filing.accession_number in self.processed_filings:
                    logger.debug(f"Skipping already processed: {filing.accession_number}")
                    continue

                logger.info(
                    f"Processing: {filing.company_name} - "
                    f"{filing.form_type} ({filing.accession_number})"
                )

                # Step 1: Download filing
                success, result = await self.downloader.download_filing(filing)
                if not success:
                    logger.error(f"Download failed for {filing.accession_number}: {result}")
                    self.failed_filings.append(filing.accession_number)
                    continue

                file_path = result
                logger.info(f"Downloaded to: {file_path}")

                # Step 2: Process with Docling
                success, extracted_data = await self.processor.process_filing(filing, file_path)
                if not success:
                    logger.error(f"Processing failed for {filing.accession_number}")
                    self.failed_filings.append(filing.accession_number)
                    continue

                logger.info(
                    f"Extracted: {len(extracted_data.get('text', ''))} chars, "
                    f"{len(extracted_data.get('tables', []))} tables"
                )

                # Step 3: Generate embeddings and prepare for storage
                success, rag_data = await self.rag_pipeline.process_filing(filing, extracted_data)
                if not success:
                    logger.error(f"RAG processing failed for {filing.accession_number}")
                    self.failed_filings.append(filing.accession_number)
                    continue

                logger.info(
                    f"Created {rag_data['total_chunks']} chunks with embeddings "
                    f"({rag_data['total_tokens']} tokens)"
                )

                # Step 4: Save to database
                try:
                    with get_db_context() as db:
                        # Get or create company
                        company = db.query(Company).filter_by(cik=filing.cik_number).first()
                        if not company:
                            company = Company(
                                cik=filing.cik_number,
                                name=filing.company_name,
                                ticker=filing.ticker_symbol
                            )
                            db.add(company)
                            db.flush()  # Get company ID

                        # Create filing record
                        db_filing = Filing(
                            accession_number=filing.accession_number,
                            company_id=company.id,
                            form_type=filing.form_type,
                            filing_date=filing.filing_date,
                            status="completed",
                            processed_at=datetime.utcnow(),
                            file_url=str(filing.filing_url),
                            download_path=str(file_path),
                            summary=extracted_data.get('text', '')[:500] if extracted_data.get('text') else '',
                            key_metrics=extracted_data.get('metadata', {})
                        )
                        db.add(db_filing)
                        db.flush()  # Get filing ID

                        # Save document
                        doc = FilingDocument(
                            filing_id=db_filing.id,
                            document_type=filing.form_type,
                            text_content=extracted_data.get('text', ''),
                            tables_json=extracted_data.get('tables', []),
                            char_count=len(extracted_data.get('text', '')),
                            table_count=len(extracted_data.get('tables', [])),
                            processed_at=datetime.utcnow()
                        )
                        db.add(doc)
                        db.flush()

                        # Save chunks if available
                        if rag_data and 'chunks' in rag_data:
                            for i, chunk_data in enumerate(rag_data.get('chunks', [])):
                                chunk = FilingChunk(
                                    filing_id=db_filing.id,
                                    document_id=doc.id,
                                    text=chunk_data.get('text', ''),
                                    chunk_index=i,
                                    vector_id=chunk_data.get('vector_id'),
                                    embedding_model=self.settings.EMBEDDING_MODEL,
                                    token_count=chunk_data.get('tokens', 0),
                                    metadata_json=chunk_data.get('metadata', {})
                                )
                                db.add(chunk)

                        db.commit()
                        logger.info(f"Saved filing {filing.accession_number} to database")

                        # Store in Neo4j knowledge graph
                        if self.neo4j_store:
                            try:
                                # Gather comprehensive company data
                                company_data = {
                                    "ticker": company.ticker if hasattr(company, 'ticker') else filing.ticker_symbol,
                                    "exchange": company.exchange if hasattr(company, 'exchange') else None,
                                    "sic_code": company.sic_code if hasattr(company, 'sic_code') else None,
                                    "industry": company.industry if hasattr(company, 'industry') else None,
                                    "sector": company.sector if hasattr(company, 'sector') else None,
                                    "state": company.state if hasattr(company, 'state') else None,
                                    "state_of_incorporation": company.state_of_incorporation if hasattr(company, 'state_of_incorporation') else None,
                                    "business_address": company.business_address if hasattr(company, 'business_address') else None,
                                    "mailing_address": company.mailing_address if hasattr(company, 'mailing_address') else None,
                                    "phone": company.phone if hasattr(company, 'phone') else None,
                                    "website": company.website if hasattr(company, 'website') else None,
                                    "fiscal_year_end": company.fiscal_year_end if hasattr(company, 'fiscal_year_end') else None,
                                    "irs_number": company.irs_number if hasattr(company, 'irs_number') else None
                                }
                                self.neo4j_store.store_filing(filing, company_data)
                                logger.info(f"Stored filing {filing.accession_number} in Neo4j")
                            except Exception as neo4j_error:
                                logger.error(f"Failed to store in Neo4j: {neo4j_error}")
                except Exception as e:
                    logger.error(f"Failed to save filing to database: {e}")

                # Mark as completed
                filing.status = FilingStatus.COMPLETED
                self.processed_filings.append(filing.accession_number)

                logger.info(f"✓ Successfully processed: {filing.accession_number}")

            except Exception as e:
                logger.error(
                    f"Unexpected error processing {filing.accession_number}: {str(e)}",
                    exc_info=True
                )
                self.failed_filings.append(filing.accession_number)

        # Log batch statistics
        logger.info(
            f"Batch complete: {len(self.processed_filings)} processed, "
            f"{len(self.failed_filings)} failed"
        )

    async def run_continuous(self) -> None:
        """Run continuous RSS monitoring and processing."""
        self.running = True
        logger.info("Starting continuous ingestion pipeline...")

        # Test connections first
        logger.info("Testing RSS feed connection...")
        if not await self.rss_monitor.test_connection():
            logger.error("Failed to connect to RSS feed. Please check your configuration.")
            return

        logger.info("RSS feed connection successful. Starting monitoring...")

        try:
            # Start continuous monitoring with callback
            await self.rss_monitor.continuous_monitor(
                callback=self.process_filing_batch
            )
        except KeyboardInterrupt:
            logger.info("Ingestion pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        finally:
            self.running = False
            logger.info("Ingestion pipeline stopped")

    async def run_once(self, limit: Optional[int] = None) -> None:
        """
        Run a single poll and process cycle.

        Args:
            limit: Maximum number of filings to process
        """
        logger.info("Running single ingestion cycle...")

        try:
            # Poll RSS feed once
            filings = await self.rss_monitor.poll_feed()
            logger.info(f"Discovered {len(filings)} filings")

            if limit:
                filings = filings[:limit]
                logger.info(f"Limited to {limit} filings")

            if filings:
                await self.process_filing_batch(filings)
            else:
                logger.info("No new filings to process")

        except Exception as e:
            logger.error(f"Error in single cycle: {str(e)}", exc_info=True)

    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            "rss_monitor": self.rss_monitor.get_status(),
            "downloader": self.downloader.get_statistics(),
            "processor": self.processor.get_statistics(),
            "rag_pipeline": self.rag_pipeline.get_statistics(),
            "processed_count": len(self.processed_filings),
            "failed_count": len(self.failed_filings),
            "success_rate": (
                len(self.processed_filings) /
                (len(self.processed_filings) + len(self.failed_filings))
                if (self.processed_filings or self.failed_filings) else 0
            )
        }

        # Add Neo4j statistics if available
        if self.neo4j_store:
            try:
                stats["neo4j"] = self.neo4j_store.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get Neo4j statistics: {e}")
                stats["neo4j"] = {}

        return stats


async def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("Financial Intelligence Platform - Ingestion Pipeline")
    logger.info("=" * 60)

    # Load settings
    settings = get_settings()
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"RSS Feed: {settings.RSS_FEED_URL}")
    logger.info(f"Poll Interval: {settings.RSS_POLL_INTERVAL}s")

    # Create pipeline
    pipeline = IngestionPipeline()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal...")
        pipeline.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="SEC Filing Ingestion Pipeline")
    parser.add_argument(
        "--mode",
        choices=["continuous", "once", "test"],
        default="continuous",
        help="Run mode: continuous monitoring, single run, or test"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of filings to process (for 'once' mode)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "continuous":
            logger.info("Starting continuous monitoring mode...")
            await pipeline.run_continuous()

        elif args.mode == "once":
            logger.info("Running single cycle mode...")
            await pipeline.run_once(limit=args.limit)

        elif args.mode == "test":
            logger.info("Running test mode...")
            # Test RSS connection
            if await pipeline.rss_monitor.test_connection():
                logger.info("✓ RSS feed connection successful")

                # Try to process one filing
                filings = await pipeline.rss_monitor.poll_feed()
                if filings:
                    logger.info(f"✓ Found {len(filings)} filings")
                    logger.info(f"Sample: {filings[0].company_name} - {filings[0].form_type}")

                    # Process just one
                    await pipeline.process_filing_batch(filings[:1])
                else:
                    logger.warning("No filings found in RSS feed")
            else:
                logger.error("✗ RSS feed connection failed")

        # Print final statistics
        stats = pipeline.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Processed: {stats['processed_count']}")
        logger.info(f"  Failed: {stats['failed_count']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"  Downloads: {stats['downloader']['total_downloads']}")
        logger.info(f"  Documents Processed: {stats['processor']['processed_count']}")
        logger.info(f"  Chunks Created: {stats['rag_pipeline']['total_chunks_processed']}")
        logger.info(f"  Embeddings Generated: {stats['rag_pipeline']['total_embeddings_generated']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)