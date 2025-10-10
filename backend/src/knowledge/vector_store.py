"""
Vector store management for filing embeddings using Qdrant.
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams
)
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using Qdrant.
    """

    def __init__(self, host: str = "qdrant", port: int = 6333):
        """
        Initialize vector store connection.

        Args:
            host: Qdrant host
            port: Qdrant port
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "filings"
        self.embedding_dimension = 1536  # OpenAI embeddings dimension

        # Initialize collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    async def store_chunks(self, chunks: List[Dict[str, Any]], filing_id: str) -> List[str]:
        """
        Store filing chunks with embeddings in Qdrant.

        Args:
            chunks: List of chunk dictionaries with embeddings
            filing_id: Filing identifier

        Returns:
            List of vector IDs for stored chunks
        """
        try:
            logger.info(f"Storing {len(chunks)} chunks for filing {filing_id}")
            points = []
            vector_ids = []
            skipped = 0

            for idx, chunk in enumerate(chunks):
                # Skip chunks without embeddings
                embedding = chunk.get('embedding')
                if not embedding:
                    logger.debug(f"Chunk {idx} has no embedding, skipping")
                    vector_ids.append(None)
                    skipped += 1
                    continue

                if not isinstance(embedding, list) or len(embedding) == 0:
                    logger.warning(f"Chunk {idx} has invalid embedding format: {type(embedding)}, len: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
                    vector_ids.append(None)
                    skipped += 1
                    continue

                # Generate unique ID for the vector
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)

                # Prepare metadata
                metadata = {
                    "filing_id": filing_id,
                    "chunk_id": chunk.get('chunk_id', ''),
                    "chunk_index": chunk.get('chunk_index', 0),
                    "content": chunk.get('content', ''),
                    "source_type": chunk.get('source_type', 'text'),
                    "section": chunk.get('section', ''),
                    "accession_number": chunk.get('accession_number', ''),
                    "company_name": chunk.get('company_name', ''),
                    "form_type": chunk.get('form_type', ''),
                    "filing_date": chunk.get('filing_date', ''),
                    "token_count": chunk.get('token_count', 0),
                    "created_at": datetime.utcnow().isoformat()
                }

                # Create point
                point = PointStruct(
                    id=vector_id,
                    vector=chunk['embedding'],
                    payload=metadata
                )
                points.append(point)

            if points:
                logger.info(f"Uploading {len(points)} vectors to Qdrant (skipped {skipped} chunks without embeddings)")
                # Batch upload to Qdrant
                operation_result = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Qdrant upload result: {operation_result}")
                logger.info(f"Successfully stored {len(points)} vectors for filing {filing_id}")
                return vector_ids
            else:
                logger.warning(f"No vectors to store for filing {filing_id} (all {len(chunks)} chunks lacked embeddings)")
                return []

        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return []

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding.

        Args:
            query_embedding: Query vector
            limit: Number of results
            filters: Optional filters

        Returns:
            List of matched chunks with scores
        """
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    filter_conditions = Filter(must=conditions)

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.

        Returns:
            Collection statistics
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
                "config": {
                    "size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def delete_filing_vectors(self, filing_id: str) -> bool:
        """
        Delete all vectors for a specific filing.

        Args:
            filing_id: Filing identifier

        Returns:
            Success status
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="filing_id",
                            match=MatchValue(value=filing_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted vectors for filing {filing_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False