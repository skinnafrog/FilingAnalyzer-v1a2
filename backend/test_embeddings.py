#!/usr/bin/env python
"""
Test script to verify OpenAI embeddings generation.
"""
import asyncio
import os
from openai import AsyncOpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_embeddings():
    """Test OpenAI embeddings generation."""
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set")
            return

        logger.info(f"API key present: {api_key[:10]}...")

        # Initialize client
        client = AsyncOpenAI(api_key=api_key)

        # Test text
        test_texts = [
            "This is a test document for embedding generation.",
            "SEC filings contain important financial information."
        ]

        # Generate embeddings
        logger.info("Calling OpenAI API...")
        response = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=test_texts
        )

        # Check results
        logger.info(f"Response received: {len(response.data)} embeddings")
        for i, item in enumerate(response.data):
            embedding = item.embedding
            logger.info(f"Embedding {i+1}: dimension={len(embedding)}, first_5_values={embedding[:5]}")

        logger.info("✅ Embeddings generated successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_embeddings())