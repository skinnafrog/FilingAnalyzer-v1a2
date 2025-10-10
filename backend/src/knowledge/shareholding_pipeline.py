"""
Enhanced pipeline for processing SEC filings with shareholding-optimized extraction.
Integrates with existing Docling processor and RAG pipeline to provide
specialized shareholding data extraction and storage.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..models.filing import SECFiling
from .shareholding_classifier import ShareholdingClassifier, ShareholdingEntity
from .shareholding_neo4j_store import ShareholdingNeo4jStore
from .rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ShareholdingPipeline:
    """Pipeline for processing filings with enhanced shareholding extraction."""

    def __init__(self, settings=None):
        """Initialize shareholding pipeline."""
        self.shareholding_classifier = ShareholdingClassifier()
        self.neo4j_store = ShareholdingNeo4jStore(settings)
        self.rag_pipeline = RAGPipeline(settings)

        # Pipeline statistics
        self.processed_filings = 0
        self.extracted_shareholdings = 0
        self.validation_failures = 0

    async def process_filing_with_shareholding_extraction(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any],
        company_data: Dict[str, Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process filing through shareholding-optimized pipeline.

        Args:
            filing: SECFiling object
            extracted_data: Extracted document data from Docling
            company_data: Additional company metadata

        Returns:
            Tuple of (success, processing_results)
        """
        logger.info(f"Processing {filing.accession_number} through shareholding pipeline")

        try:
            processing_results = {
                "shareholding_entities": [],
                "rag_chunks": [],
                "neo4j_stored": False,
                "validation_report": {},
                "processing_time": 0
            }

            start_time = datetime.utcnow()

            # Step 1: Extract shareholding relationships
            shareholding_entities = await self._extract_shareholding_relationships(filing, extracted_data)
            processing_results["shareholding_entities"] = shareholding_entities

            # Step 2: Validate extracted entities
            validation_report = self._validate_shareholding_entities(shareholding_entities)
            processing_results["validation_report"] = validation_report

            # Step 3: Store in Neo4j knowledge graph
            if shareholding_entities:
                neo4j_success = self.neo4j_store.store_shareholding_filing(
                    filing, company_data or {}, extracted_data
                )
                processing_results["neo4j_stored"] = neo4j_success

            # Step 4: Create shareholding-optimized RAG chunks
            rag_chunks = await self._create_shareholding_rag_chunks(filing, shareholding_entities, extracted_data)
            processing_results["rag_chunks"] = rag_chunks

            # Step 5: Process through standard RAG pipeline with enhanced chunks
            if rag_chunks:
                await self._process_enhanced_rag_chunks(filing, rag_chunks)

            # Update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            processing_results["processing_time"] = processing_time

            self.processed_filings += 1
            self.extracted_shareholdings += len(shareholding_entities)

            logger.info(
                f"Completed shareholding processing for {filing.accession_number}: "
                f"{len(shareholding_entities)} entities extracted in {processing_time:.2f}s"
            )

            return True, processing_results

        except Exception as e:
            logger.error(f"Shareholding pipeline failed for {filing.accession_number}: {e}", exc_info=True)
            return False, {"error": str(e)}

    async def _extract_shareholding_relationships(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any]
    ) -> List[ShareholdingEntity]:
        """Extract shareholding relationships from filing data."""
        all_entities = []

        # Extract from main document text
        main_text = extracted_data.get("text", "")
        if main_text:
            text_entities = self.shareholding_classifier.extract_shareholding_relationships(
                main_text,
                {
                    "filing": filing,
                    "cik": filing.cik_number,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date
                }
            )
            all_entities.extend(text_entities)

        # Extract from document sections
        sections = extracted_data.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").lower()
            # Focus on sections likely to contain shareholding information
            if any(keyword in section_title for keyword in [
                "ownership", "shareholder", "beneficial", "insider", "director", "officer",
                "security", "equity", "stock", "share", "principal"
            ]):
                section_text = section.get("text", "")
                if section_text:
                    section_entities = self.shareholding_classifier.extract_shareholding_relationships(
                        section_text,
                        {
                            "filing": filing,
                            "section": section_title,
                            "cik": filing.cik_number
                        }
                    )
                    all_entities.extend(section_entities)

        # Extract from tables with shareholding data
        tables = extracted_data.get("tables", [])
        for table_idx, table in enumerate(tables):
            table_title = table.get("title", "").lower()
            if any(keyword in table_title for keyword in [
                "ownership", "shareholder", "beneficial", "insider", "director"
            ]):
                table_entities = await self._extract_from_shareholding_table(table, filing, table_idx)
                all_entities.extend(table_entities)

        # Process legacy shareholder data if available
        if hasattr(filing, 'shareholders') and filing.shareholders:
            legacy_entities = await self._process_legacy_shareholding_data(filing.shareholders, filing)
            all_entities.extend(legacy_entities)

        # Remove duplicates and low-confidence entities
        filtered_entities = self._filter_and_deduplicate_entities(all_entities)

        return filtered_entities

    async def _extract_from_shareholding_table(
        self,
        table: Dict[str, Any],
        filing: SECFiling,
        table_idx: int
    ) -> List[ShareholdingEntity]:
        """Extract shareholding entities from table data."""
        entities = []
        rows = table.get("rows", [])

        for row_idx, row in enumerate(rows):
            if isinstance(row, dict):
                entity_data = self._parse_shareholding_table_row(row, table, filing)
                if entity_data:
                    # Classify the entity name
                    classification = self.shareholding_classifier.classify_entity(
                        entity_data["name"],
                        {
                            "table_context": table.get("title", ""),
                            "filing": filing,
                            "row_data": row
                        }
                    )

                    if classification.entity_type.value in ["shareholder_person", "shareholder_entity"]:
                        entity = ShareholdingEntity(
                            name=entity_data["name"],
                            entity_type=classification.entity_type,
                            shares=entity_data.get("shares"),
                            percentage=entity_data.get("percentage"),
                            share_class=entity_data.get("share_class"),
                            context=f"Table {table_idx + 1}: {table.get('title', 'Unknown')}",
                            confidence=classification.confidence * 0.85  # Table data is slightly less reliable
                        )
                        entities.append(entity)

        return entities

    def _parse_shareholding_table_row(
        self,
        row: Dict[str, Any],
        table: Dict[str, Any],
        filing: SECFiling
    ) -> Optional[Dict[str, Any]]:
        """Parse a table row for shareholding information."""
        entity_data = {}

        # Common column name patterns for shareholding tables
        name_patterns = ["name", "shareholder", "owner", "holder", "beneficial owner", "person", "entity"]
        share_patterns = ["shares", "owned", "holding", "number", "amount", "quantity"]
        percentage_patterns = ["percentage", "%", "percent", "ownership", "stake"]
        class_patterns = ["class", "type", "series", "common", "preferred"]

        for key, value in row.items():
            key_lower = str(key).lower().strip()
            value_str = str(value).strip()

            # Skip empty values
            if not value_str or value_str.lower() in ["", "n/a", "none", "-"]:
                continue

            # Extract name
            if any(pattern in key_lower for pattern in name_patterns):
                # Validate that this looks like a name
                classification = self.shareholding_classifier.classify_entity(value_str)
                if classification.entity_type.value in ["shareholder_person", "shareholder_entity"]:
                    entity_data["name"] = value_str

            # Extract shares
            elif any(pattern in key_lower for pattern in share_patterns):
                try:
                    # Clean and parse share numbers
                    clean_shares = value_str.replace(",", "").replace("$", "")
                    if clean_shares.replace(".", "").isdigit():
                        entity_data["shares"] = int(float(clean_shares))
                except (ValueError, TypeError):
                    pass

            # Extract percentage
            elif any(pattern in key_lower for pattern in percentage_patterns):
                try:
                    # Clean and parse percentage
                    clean_pct = value_str.replace("%", "").replace(",", "")
                    if clean_pct.replace(".", "").isdigit():
                        entity_data["percentage"] = float(clean_pct)
                except (ValueError, TypeError):
                    pass

            # Extract share class
            elif any(pattern in key_lower for pattern in class_patterns):
                if any(share_type in value_str.lower() for share_type in ["common", "preferred", "class"]):
                    entity_data["share_class"] = value_str

        return entity_data if entity_data.get("name") else None

    async def _process_legacy_shareholding_data(
        self,
        shareholders: List[Dict[str, Any]],
        filing: SECFiling
    ) -> List[ShareholdingEntity]:
        """Process shareholding data from legacy extraction."""
        entities = []

        for shareholder in shareholders:
            name = shareholder.get("name", "").strip()
            if not name:
                continue

            # Classify the shareholder
            classification = self.shareholding_classifier.classify_entity(
                name,
                {"legacy_data": True, "filing": filing}
            )

            if classification.entity_type.value in ["shareholder_person", "shareholder_entity"]:
                # Parse shares and percentage
                shares = None
                percentage = None

                shares_str = shareholder.get("shares", "")
                if shares_str:
                    try:
                        shares = int(str(shares_str).replace(",", ""))
                    except (ValueError, TypeError):
                        pass

                percentage_str = shareholder.get("percentage", "")
                if percentage_str:
                    try:
                        percentage = float(str(percentage_str).replace("%", "").replace(",", ""))
                    except (ValueError, TypeError):
                        pass

                entity = ShareholdingEntity(
                    name=name,
                    entity_type=classification.entity_type,
                    shares=shares,
                    percentage=percentage,
                    context=shareholder.get("context", "Legacy extraction"),
                    confidence=classification.confidence * 0.7  # Legacy data less reliable
                )
                entities.append(entity)

        return entities

    def _filter_and_deduplicate_entities(self, entities: List[ShareholdingEntity]) -> List[ShareholdingEntity]:
        """Filter out low-confidence entities and remove duplicates."""
        # Filter by confidence threshold
        filtered = [entity for entity in entities if entity.confidence >= 0.5]

        # Deduplicate by name (case-insensitive)
        seen_names = {}
        deduplicated = []

        for entity in filtered:
            name_key = entity.name.lower().strip()

            if name_key not in seen_names:
                seen_names[name_key] = entity
                deduplicated.append(entity)
            else:
                # Keep the entity with higher confidence
                existing = seen_names[name_key]
                if entity.confidence > existing.confidence:
                    # Replace in both dict and list
                    seen_names[name_key] = entity
                    for i, existing_entity in enumerate(deduplicated):
                        if existing_entity.name.lower().strip() == name_key:
                            deduplicated[i] = entity
                            break

        return deduplicated

    def _validate_shareholding_entities(self, entities: List[ShareholdingEntity]) -> Dict[str, Any]:
        """Validate extracted shareholding entities."""
        return self.shareholding_classifier.validate_shareholding_data(entities)

    async def _create_shareholding_rag_chunks(
        self,
        filing: SECFiling,
        shareholding_entities: List[ShareholdingEntity],
        extracted_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create RAG-optimized chunks for shareholding data."""
        chunks = []

        if not shareholding_entities:
            return chunks

        # Create summary chunk for all shareholdings in this filing
        summary_chunk = self._create_shareholding_summary_chunk(filing, shareholding_entities)
        if summary_chunk:
            chunks.append(summary_chunk)

        # Create individual chunks for significant shareholdings
        for entity in shareholding_entities:
            if entity.confidence > 0.7 and (entity.percentage and entity.percentage >= 1.0):
                individual_chunk = self._create_individual_shareholding_chunk(filing, entity)
                if individual_chunk:
                    chunks.append(individual_chunk)

        # Create insider-specific chunks for Form 3/4/5
        if getattr(filing, 'is_insider_filing', False):
            insider_chunk = self._create_insider_filing_chunk(filing, shareholding_entities)
            if insider_chunk:
                chunks.append(insider_chunk)

        return chunks

    def _create_shareholding_summary_chunk(
        self,
        filing: SECFiling,
        entities: List[ShareholdingEntity]
    ) -> Optional[Dict[str, Any]]:
        """Create a summary chunk for all shareholdings in the filing."""
        if not entities:
            return None

        # Sort entities by percentage (descending)
        sorted_entities = sorted(
            [e for e in entities if e.percentage],
            key=lambda x: x.percentage or 0,
            reverse=True
        )

        # Create summary text
        company_name = filing.company_name or "Company"
        ticker = getattr(filing, 'ticker_symbol', '') or getattr(filing, 'issuer_ticker', '')
        ticker_text = f" ({ticker})" if ticker else ""

        summary_parts = [
            f"{company_name}{ticker_text} Shareholding Summary as of {filing.filing_date.strftime('%B %d, %Y')}:"
        ]

        # Add top shareholders
        for i, entity in enumerate(sorted_entities[:5]):
            percentage_text = f"{entity.percentage:.1f}%" if entity.percentage else "N/A"
            shares_text = f"{entity.shares:,} shares" if entity.shares else ""

            if shares_text and percentage_text != "N/A":
                summary_parts.append(f"• {entity.name}: {shares_text} ({percentage_text})")
            elif percentage_text != "N/A":
                summary_parts.append(f"• {entity.name}: {percentage_text}")
            else:
                summary_parts.append(f"• {entity.name}: shareholding reported")

        # Add filing context
        summary_parts.append(f"Source: {filing.form_type} filing {filing.accession_number}")

        chunk_text = " ".join(summary_parts)

        return {
            "content": chunk_text,
            "metadata": {
                "source": "shareholding_summary",
                "filing_id": filing.id if hasattr(filing, 'id') else None,
                "accession_number": filing.accession_number,
                "company_name": company_name,
                "ticker": ticker,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                "shareholding_count": len(entities),
                "chunk_type": "shareholding_summary"
            },
            "chunk_id": f"{filing.accession_number}_shareholding_summary",
            "token_count": len(chunk_text.split())
        }

    def _create_individual_shareholding_chunk(
        self,
        filing: SECFiling,
        entity: ShareholdingEntity
    ) -> Optional[Dict[str, Any]]:
        """Create a chunk for an individual significant shareholding."""
        company_name = filing.company_name or "Company"
        ticker = getattr(filing, 'ticker_symbol', '') or getattr(filing, 'issuer_ticker', '')
        ticker_text = f" ({ticker})" if ticker else ""

        # Create detailed text for this shareholding
        chunk_parts = []

        # Basic ownership information
        if entity.percentage and entity.shares:
            chunk_parts.append(
                f"{entity.name} owns {entity.shares:,} shares of {company_name}{ticker_text}, "
                f"representing {entity.percentage:.2f}% ownership"
            )
        elif entity.percentage:
            chunk_parts.append(
                f"{entity.name} owns {entity.percentage:.2f}% of {company_name}{ticker_text}"
            )
        elif entity.shares:
            chunk_parts.append(
                f"{entity.name} owns {entity.shares:,} shares of {company_name}{ticker_text}"
            )
        else:
            chunk_parts.append(
                f"{entity.name} has a reported shareholding in {company_name}{ticker_text}"
            )

        # Add share class if available
        if entity.share_class:
            chunk_parts.append(f"Share class: {entity.share_class}")

        # Add filing context
        chunk_parts.append(
            f"Reported in {filing.form_type} filing on {filing.filing_date.strftime('%B %d, %Y')} "
            f"(Accession: {filing.accession_number})"
        )

        # Add context if available
        if entity.context:
            chunk_parts.append(f"Context: {entity.context[:100]}")

        chunk_text = ". ".join(chunk_parts) + "."

        return {
            "content": chunk_text,
            "metadata": {
                "source": "individual_shareholding",
                "filing_id": filing.id if hasattr(filing, 'id') else None,
                "accession_number": filing.accession_number,
                "company_name": company_name,
                "ticker": ticker,
                "shareholder_name": entity.name,
                "shareholder_type": entity.entity_type.value,
                "percentage": entity.percentage,
                "shares": entity.shares,
                "share_class": entity.share_class,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                "chunk_type": "individual_shareholding",
                "confidence": entity.confidence
            },
            "chunk_id": f"{filing.accession_number}_{entity.name.replace(' ', '_')}_shareholding",
            "token_count": len(chunk_text.split())
        }

    def _create_insider_filing_chunk(
        self,
        filing: SECFiling,
        entities: List[ShareholdingEntity]
    ) -> Optional[Dict[str, Any]]:
        """Create a specialized chunk for insider filings (Form 3/4/5)."""
        if not getattr(filing, 'is_insider_filing', False):
            return None

        issuer_name = getattr(filing, 'issuer_name', filing.company_name)
        ticker = getattr(filing, 'issuer_ticker', '')
        ticker_text = f" ({ticker})" if ticker else ""
        reporting_owner = getattr(filing, 'reporting_owner_name', 'Unknown')

        chunk_parts = [
            f"Insider Filing: {reporting_owner} filed {filing.form_type} for {issuer_name}{ticker_text}"
        ]

        # Add shareholding details if available
        if entities:
            for entity in entities:
                if entity.percentage:
                    chunk_parts.append(f"{entity.name} owns {entity.percentage:.2f}% of {issuer_name}")
                elif entity.shares:
                    chunk_parts.append(f"{entity.name} owns {entity.shares:,} shares of {issuer_name}")

        chunk_parts.append(
            f"Filed on {filing.filing_date.strftime('%B %d, %Y')} "
            f"(Accession: {filing.accession_number})"
        )

        chunk_text = ". ".join(chunk_parts) + "."

        return {
            "content": chunk_text,
            "metadata": {
                "source": "insider_filing",
                "filing_id": filing.id if hasattr(filing, 'id') else None,
                "accession_number": filing.accession_number,
                "issuer_name": issuer_name,
                "issuer_cik": getattr(filing, 'issuer_cik', filing.cik_number),
                "ticker": ticker,
                "reporting_owner": reporting_owner,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                "is_insider_filing": True,
                "chunk_type": "insider_filing"
            },
            "chunk_id": f"{filing.accession_number}_insider_filing",
            "token_count": len(chunk_text.split())
        }

    async def _process_enhanced_rag_chunks(self, filing: SECFiling, chunks: List[Dict[str, Any]]):
        """Process shareholding chunks through the RAG pipeline."""
        if not chunks:
            return

        # Add the chunks to the filing's extracted data for RAG processing
        enhanced_extracted_data = {
            "text": "",  # Standard text processing handled separately
            "tables": [],
            "shareholding_chunks": chunks,
            "metadata": {"shareholding_optimized": True}
        }

        # Process through RAG pipeline
        try:
            success, rag_results = await self.rag_pipeline.process_filing(filing, enhanced_extracted_data)
            if success:
                logger.info(f"Successfully processed {len(chunks)} shareholding chunks through RAG pipeline")
            else:
                logger.warning(f"RAG processing failed for shareholding chunks: {rag_results.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error processing shareholding chunks through RAG: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return {
            "processed_filings": self.processed_filings,
            "extracted_shareholdings": self.extracted_shareholdings,
            "validation_failures": self.validation_failures,
            "average_shareholdings_per_filing": (
                self.extracted_shareholdings / self.processed_filings
                if self.processed_filings > 0 else 0
            ),
            "classifier_stats": self.shareholding_classifier.get_statistics(),
            "neo4j_stats": self.neo4j_store.get_shareholding_statistics()
        }

    def close(self):
        """Close pipeline resources."""
        if self.neo4j_store:
            self.neo4j_store.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()