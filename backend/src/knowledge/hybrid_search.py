"""
Hybrid search combining vector similarity and keyword matching.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from ..database.models import FilingChunk
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Implements hybrid search combining:
    - Vector similarity search (semantic)
    - BM25 keyword search
    - Optional reranking
    """

    def __init__(self, vector_store: VectorStore, db_session: Session):
        """
        Initialize hybrid search.

        Args:
            vector_store: Vector store instance
            db_session: Database session for keyword search
        """
        self.vector_store = vector_store
        self.db_session = db_session
        self.bm25_index = None
        self.doc_ids = []

    def build_bm25_index(self, limit: int = 10000):
        """
        Build BM25 index from filing chunks.

        Args:
            limit: Maximum number of documents to index
        """
        try:
            # Fetch chunks from database
            chunks = self.db_session.query(FilingChunk).limit(limit).all()

            if not chunks:
                logger.warning("No chunks found to build BM25 index")
                return

            # Tokenize documents for BM25
            tokenized_docs = []
            self.doc_ids = []

            for chunk in chunks:
                # Simple tokenization (can be improved with proper NLP)
                tokens = chunk.text.lower().split()
                tokenized_docs.append(tokens)
                self.doc_ids.append(chunk.id)

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")

    async def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text
            query_embedding: Pre-computed query embedding (optional)
            limit: Number of results to return
            alpha: Weight for vector search (1-alpha for keyword search)
            filters: Optional filters for search

        Returns:
            Combined and ranked search results
        """
        try:
            results = []

            # 1. Vector search (semantic)
            vector_results = []
            if query_embedding and self.vector_store:
                vector_results = await self.vector_store.search(
                    query_embedding=query_embedding,
                    limit=limit * 2,  # Get more for merging
                    filters=filters
                )
                logger.info(f"Vector search returned {len(vector_results)} results")

            # 2. Keyword search using PostgreSQL full-text search
            keyword_results = await self._keyword_search_postgres(
                query=query,
                limit=limit * 2,
                filters=filters
            )
            logger.info(f"Keyword search returned {len(keyword_results)} results")

            # 3. BM25 search if index is available
            bm25_results = []
            if self.bm25_index:
                bm25_results = self._bm25_search(query, limit * 2)
                logger.info(f"BM25 search returned {len(bm25_results)} results")

            # 4. Combine and rerank results
            combined_results = self._combine_results(
                vector_results=vector_results,
                keyword_results=keyword_results,
                bm25_results=bm25_results,
                alpha=alpha,
                limit=limit
            )

            return combined_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def _keyword_search_postgres(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword search using PostgreSQL full-text search.

        Args:
            query: Search query
            limit: Number of results
            filters: Optional filters

        Returns:
            Keyword search results
        """
        try:
            # Build the SQL query with full-text search
            sql = text("""
                SELECT
                    fc.id,
                    fc.text,
                    fc.metadata_json,
                    fc.filing_id,
                    f.accession_number,
                    c.name as company_name,
                    f.form_type,
                    ts_rank(to_tsvector('english', fc.text),
                           plainto_tsquery('english', :query)) as score
                FROM filing_chunks fc
                JOIN filings f ON fc.filing_id = f.id
                JOIN companies c ON f.company_id = c.id
                WHERE to_tsvector('english', fc.text) @@
                      plainto_tsquery('english', :query)
                ORDER BY score DESC
                LIMIT :limit
            """)

            result = self.db_session.execute(sql, {"query": query, "limit": limit})

            results = []
            for row in result:
                results.append({
                    "id": row.id,
                    "content": row.text,
                    "score": float(row.score),
                    "metadata": {
                        "filing_id": row.filing_id,
                        "accession_number": row.accession_number,
                        "company_name": row.company_name,
                        "form_type": row.form_type,
                        **(row.metadata_json or {})
                    }
                })

            return results

        except Exception as e:
            logger.error(f"PostgreSQL keyword search failed: {e}")
            return []

    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            limit: Number of results

        Returns:
            BM25 search results
        """
        if not self.bm25_index:
            return []

        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top K indices
            top_indices = np.argsort(scores)[-limit:][::-1]

            # Fetch corresponding chunks
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include positive scores
                    chunk_id = self.doc_ids[idx]
                    chunk = self.db_session.query(FilingChunk).filter_by(id=chunk_id).first()

                    if chunk:
                        results.append({
                            "id": chunk.id,
                            "content": chunk.text,
                            "score": float(scores[idx]),
                            "metadata": chunk.metadata_json or {}
                        })

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        alpha: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank results from different search methods.

        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            bm25_results: BM25 search results
            alpha: Weight for vector search
            limit: Final number of results

        Returns:
            Combined and ranked results
        """
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return []
            max_score = max(r['score'] for r in results)
            if max_score > 0:
                for r in results:
                    r['normalized_score'] = r['score'] / max_score
            else:
                for r in results:
                    r['normalized_score'] = 0
            return results

        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)
        bm25_results = normalize_scores(bm25_results)

        # Combine results with weighted scores
        combined = {}

        # Process vector results
        for result in vector_results:
            content_id = result.get('metadata', {}).get('chunk_id', result.get('content', '')[:50])
            if content_id not in combined:
                combined[content_id] = {
                    **result,
                    'final_score': alpha * result['normalized_score'],
                    'sources': ['vector']
                }
            else:
                combined[content_id]['final_score'] += alpha * result['normalized_score']
                combined[content_id]['sources'].append('vector')

        # Process keyword results
        for result in keyword_results:
            content_id = result.get('content', '')[:50]
            if content_id not in combined:
                combined[content_id] = {
                    **result,
                    'final_score': (1 - alpha) * 0.5 * result['normalized_score'],
                    'sources': ['keyword']
                }
            else:
                combined[content_id]['final_score'] += (1 - alpha) * 0.5 * result['normalized_score']
                if 'keyword' not in combined[content_id]['sources']:
                    combined[content_id]['sources'].append('keyword')

        # Process BM25 results
        for result in bm25_results:
            content_id = result.get('content', '')[:50]
            if content_id not in combined:
                combined[content_id] = {
                    **result,
                    'final_score': (1 - alpha) * 0.5 * result['normalized_score'],
                    'sources': ['bm25']
                }
            else:
                combined[content_id]['final_score'] += (1 - alpha) * 0.5 * result['normalized_score']
                if 'bm25' not in combined[content_id]['sources']:
                    combined[content_id]['sources'].append('bm25')

        # Boost scores for results from multiple sources
        for content_id, result in combined.items():
            if len(result['sources']) > 1:
                result['final_score'] *= (1 + 0.2 * (len(result['sources']) - 1))

        # Sort by final score and return top results
        sorted_results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)

        return sorted_results[:limit]