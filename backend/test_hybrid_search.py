#!/usr/bin/env python
"""
Test script to verify hybrid search functionality.
"""
import asyncio
import sys
import logging
from typing import List, Optional

# Add src to path
sys.path.insert(0, '/app')

from src.knowledge.hybrid_search import HybridSearch
from src.knowledge.vector_store import VectorStore
from src.database import get_db_context
from openai import AsyncOpenAI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_hybrid_search():
    """Test the hybrid search functionality."""
    try:
        # Initialize dependencies
        logger.info("Initializing dependencies...")
        vector_store = VectorStore(host="qdrant", port=6333)

        # Initialize OpenAI for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            return

        openai_client = AsyncOpenAI(api_key=api_key)

        # Test queries
        test_queries = [
            "revenue growth",
            "risk factors",
            "executive compensation",
            "financial performance 2024"
        ]

        # Use database context
        with get_db_context() as db:
            # Initialize hybrid search
            logger.info("Initializing hybrid search...")
            hybrid_search = HybridSearch(vector_store=vector_store, db_session=db)

            for query in test_queries:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing query: '{query}'")
                logger.info(f"{'='*60}")

                # Generate query embedding
                logger.info("Generating query embedding...")
                response = await openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query
                )
                query_embedding = response.data[0].embedding

                # Test hybrid search
                logger.info("\n🔍 HYBRID SEARCH Results:")
                hybrid_results = await hybrid_search.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    limit=5,
                    alpha=0.5  # Equal weight to semantic and keyword search
                )

                if not hybrid_results:
                    logger.info("   No results found")
                else:
                    for i, result in enumerate(hybrid_results, 1):
                        logger.info(f"   Result {i}:")
                        logger.info(f"      Score: {result['score']:.4f}")
                        logger.info(f"      Company: {result.get('metadata', {}).get('company_name', 'Unknown')}")
                        logger.info(f"      Filing: {result.get('metadata', {}).get('accession_number', 'Unknown')}")
                        logger.info(f"      Section: {result.get('metadata', {}).get('section', 'Unknown')}")
                        logger.info(f"      Content: {result['content'][:150]}...")

                # Add delay between queries
                await asyncio.sleep(1)

        logger.info("\n✅ Hybrid search tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Hybrid search test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_hybrid_search())