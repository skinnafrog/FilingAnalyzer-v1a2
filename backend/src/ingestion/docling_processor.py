"""
Document processor using Docling for extracting structured content from SEC filings.
Handles PDFs, HTMLs, and other document formats with table and image extraction.
"""
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.document import ConversionResult

from ..config.settings import Settings, get_settings
from ..models.filing import SECFiling, ProcessingStage
from .issuer_extractor import IssuerExtractor
from ..knowledge.shareholding_pipeline import ShareholdingPipeline

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Process documents using Docling for structured extraction."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Docling processor."""
        self.settings = settings or get_settings()

        # Docling configuration
        self.enable_ocr = self.settings.DOCLING_ENABLE_OCR
        self.extract_tables = self.settings.DOCLING_EXTRACT_TABLES
        self.extract_images = self.settings.DOCLING_EXTRACT_IMAGES
        self.max_workers = self.settings.DOCLING_MAX_WORKERS

        # Processing paths
        self.processed_path = Path(self.settings.PROCESSED_STORAGE_PATH)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Initialize Docling converter
        self._init_converter()

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0

        # Initialize issuer extractor for Form 3/4/5 filings
        self.issuer_extractor = IssuerExtractor()

        # Initialize shareholding pipeline for optimized knowledge graph processing
        self.shareholding_pipeline = ShareholdingPipeline(settings)

    def _init_converter(self):
        """Initialize Docling converter with pipeline options."""
        try:
            # Configure PDF pipeline options for Docling v2
            pdf_pipeline_options = PdfPipelineOptions(
                do_ocr=self.enable_ocr,
                do_table_structure=self.extract_tables,
                table_structure_options={
                    "mode": TableFormerMode.ACCURATE if self.extract_tables else TableFormerMode.FAST,
                    "do_cell_matching": True
                }
            )

            # Create pipeline options dict for Docling v2
            pipeline_options = {
                InputFormat.PDF: pdf_pipeline_options,
            }

            # Initialize converter - checking Docling API version
            try:
                # Try new API first (if supported)
                self.converter = DocumentConverter(
                    allowed_formats=[
                        InputFormat.PDF,
                        InputFormat.HTML,
                        InputFormat.DOCX,
                        InputFormat.XLSX,
                        InputFormat.IMAGE
                    ],
                    pipeline_options=pipeline_options
                )
            except TypeError as e:
                # Fallback to older API if pipeline_options not supported
                logger.warning(f"Docling API does not support pipeline_options, using default: {e}")
                self.converter = DocumentConverter(
                    allowed_formats=[
                        InputFormat.PDF,
                        InputFormat.HTML,
                        InputFormat.DOCX,
                        InputFormat.XLSX,
                        InputFormat.IMAGE
                    ]
                )

            logger.info("Docling converter initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            # Fallback to basic converter
            self.converter = DocumentConverter()

    async def process_filing(
        self,
        filing: SECFiling,
        file_path: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a filing document with Docling.

        Args:
            filing: SECFiling object
            file_path: Path to downloaded document

        Returns:
            Tuple of (success, extracted_data)
        """
        logger.info(f"Processing {filing.accession_number} with Docling")

        # Update filing status
        filing.current_stage = ProcessingStage.DOCLING_EXTRACTION
        filing.processing_attempts += 1

        start_time = datetime.utcnow()

        try:
            # Process document in thread pool (Docling is CPU-bound)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._process_document_sync,
                file_path,
                filing.form_type
            )

            # Parse extraction results
            extracted_data = self._parse_extraction_results(result, filing)

            # Update filing with extracted data
            filing.extracted_text = extracted_data.get("text", "")
            filing.extracted_tables = extracted_data.get("tables", [])
            filing.extracted_images = extracted_data.get("images", [])
            filing.extracted_metadata = extracted_data.get("metadata", {})

            # Extract financial data if applicable
            if filing.form_type in ["10-K", "10-Q", "20-F"]:
                financial_data = await self._extract_financial_data(extracted_data)
                filing.financial_statements = financial_data.get("statements", {})
                filing.key_metrics = financial_data.get("metrics", {})

            # Extract shareholder information
            shareholder_data = await self._extract_shareholder_info(extracted_data)
            filing.shareholders = shareholder_data.get("shareholders", [])
            filing.equity_issuances = shareholder_data.get("equity_issuances", [])

            # Extract issuer information for Form 3/4/5 filings
            issuer_data = await self.issuer_extractor.extract_issuer_info(extracted_data, filing.form_type)
            if issuer_data.get("is_insider_filing"):
                filing.issuer_name = issuer_data.get("issuer_name")
                filing.issuer_cik = issuer_data.get("issuer_cik")
                filing.issuer_ticker = issuer_data.get("issuer_ticker")
                filing.reporting_owner_name = issuer_data.get("reporting_owner_name")
                filing.is_insider_filing = True
                logger.info(
                    f"Extracted issuer info for {filing.accession_number}: "
                    f"Issuer={issuer_data.get('issuer_name')}, "
                    f"Owner={issuer_data.get('reporting_owner_name')}"
                )

            # Process with shareholding pipeline for optimized knowledge graph ingestion
            shareholding_success = await self._process_with_shareholding_pipeline(filing, extracted_data)
            if shareholding_success:
                logger.info(f"Successfully processed {filing.accession_number} with shareholding pipeline")
            else:
                logger.warning(f"Shareholding pipeline processing failed for {filing.accession_number}")

            # Save processed data
            processed_file = await self._save_processed_data(filing, extracted_data)
            filing.processed_file_path = str(processed_file)

            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            filing.processing_time_seconds = processing_time
            filing.current_stage = ProcessingStage.TEXT_CHUNKING

            self.processed_count += 1
            self.total_processing_time += processing_time

            logger.info(
                f"Successfully processed {filing.accession_number} "
                f"({len(filing.extracted_text)} chars, "
                f"{len(filing.extracted_tables)} tables) in {processing_time:.1f}s"
            )

            return True, extracted_data

        except Exception as e:
            self.failed_count += 1
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Failed to process {filing.accession_number}: {error_msg}", exc_info=True)
            filing.mark_failed(error_msg)
            return False, {"error": error_msg}

    def _process_document_sync(self, file_path: str, form_type: str) -> ConversionResult:
        """
        Synchronous document processing with Docling.

        Args:
            file_path: Path to document
            form_type: SEC form type

        Returns:
            Docling conversion result
        """
        try:
            path = Path(file_path)

            # Convert document
            result = self.converter.convert(path)

            logger.debug(f"Docling conversion complete for {path.name}")
            return result

        except Exception as e:
            logger.error(f"Docling conversion failed: {e}", exc_info=True)
            raise

    def _parse_extraction_results(
        self,
        result: ConversionResult,
        filing: SECFiling
    ) -> Dict[str, Any]:
        """
        Parse Docling extraction results into structured data.

        Args:
            result: Docling conversion result
            filing: SECFiling object

        Returns:
            Structured extraction data
        """
        extracted_data = {
            "text": "",
            "tables": [],
            "images": [],
            "metadata": {},
            "sections": []
        }

        try:
            # Extract main text content
            if hasattr(result, 'document'):
                extracted_data["text"] = result.document.export_to_markdown()

                # Extract metadata
                if hasattr(result.document, 'metadata'):
                    extracted_data["metadata"] = {
                        "page_count": getattr(result.document.metadata, 'page_count', 0),
                        "title": getattr(result.document.metadata, 'title', ''),
                        "author": getattr(result.document.metadata, 'author', ''),
                        "creation_date": str(getattr(result.document.metadata, 'creation_date', ''))
                    }

                # Extract sections/headings
                if hasattr(result.document, 'sections'):
                    for section in result.document.sections:
                        extracted_data["sections"].append({
                            "title": section.title,
                            "level": section.level,
                            "text": section.text[:500]  # Preview
                        })

            # Extract tables
            if self.extract_tables and hasattr(result, 'tables'):
                for idx, table in enumerate(result.tables):
                    table_data = {
                        "index": idx,
                        "title": getattr(table, 'title', f"Table {idx + 1}"),
                        "rows": [],
                        "metadata": {}
                    }

                    # Convert table to structured format
                    if hasattr(table, 'to_dataframe'):
                        df = table.to_dataframe()
                        table_data["rows"] = df.to_dict('records')
                        table_data["metadata"] = {
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "columns": df.columns.tolist()
                        }
                    elif hasattr(table, 'data'):
                        table_data["rows"] = table.data

                    extracted_data["tables"].append(table_data)

                    # Look for financial tables
                    if self._is_financial_table(table_data):
                        table_data["is_financial"] = True

            # Extract images if enabled
            if self.extract_images and hasattr(result, 'images'):
                for idx, image in enumerate(result.images):
                    image_data = {
                        "index": idx,
                        "caption": getattr(image, 'caption', f"Image {idx + 1}"),
                        "type": getattr(image, 'type', 'unknown'),
                        "page": getattr(image, 'page', 0)
                    }
                    extracted_data["images"].append(image_data)

        except Exception as e:
            logger.error(f"Error parsing extraction results: {e}", exc_info=True)

        return extracted_data

    def _is_financial_table(self, table_data: Dict[str, Any]) -> bool:
        """
        Check if a table contains financial data.

        Args:
            table_data: Extracted table data

        Returns:
            True if table appears to contain financial data
        """
        financial_keywords = [
            "revenue", "income", "expense", "asset", "liability",
            "equity", "cash", "earnings", "profit", "loss",
            "balance sheet", "income statement", "cash flow",
            "stockholders", "shareholders"
        ]

        # Check table title
        title = table_data.get("title", "").lower()
        if any(keyword in title for keyword in financial_keywords):
            return True

        # Check column names
        columns = table_data.get("metadata", {}).get("columns", [])
        columns_text = " ".join(str(col).lower() for col in columns)
        if any(keyword in columns_text for keyword in financial_keywords):
            return True

        # Check first few rows
        rows = table_data.get("rows", [])[:3]
        for row in rows:
            row_text = " ".join(str(v).lower() for v in row.values())
            if any(keyword in row_text for keyword in financial_keywords):
                return True

        return False

    async def _extract_financial_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract financial statements and metrics from processed data.

        Args:
            extracted_data: Extracted document data

        Returns:
            Financial data dictionary
        """
        financial_data = {
            "statements": {},
            "metrics": {}
        }

        try:
            # Process financial tables
            financial_tables = [
                table for table in extracted_data.get("tables", [])
                if table.get("is_financial", False)
            ]

            for table in financial_tables:
                title = table.get("title", "").lower()

                # Categorize financial statements
                if "balance sheet" in title or "financial position" in title:
                    financial_data["statements"]["balance_sheet"] = table
                elif "income" in title or "operations" in title or "earnings" in title:
                    financial_data["statements"]["income_statement"] = table
                elif "cash flow" in title:
                    financial_data["statements"]["cash_flow"] = table
                elif "equity" in title or "stockholders" in title:
                    financial_data["statements"]["equity_statement"] = table

            # Extract key metrics from text
            text = extracted_data.get("text", "")
            metrics = self._extract_metrics_from_text(text)
            financial_data["metrics"] = metrics

        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")

        return financial_data

    async def _extract_shareholder_info(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract shareholder and equity holder information.

        Args:
            extracted_data: Extracted document data

        Returns:
            Shareholder data dictionary
        """
        shareholder_data = {
            "shareholders": [],
            "equity_issuances": []
        }

        try:
            text = extracted_data.get("text", "")

            # Look for shareholder patterns
            shareholder_patterns = [
                r"([A-Z][A-Za-z\s&,\.]+(?:LLC|LP|Inc|Corp|Company|Fund|Capital|Partners|Management|Group))[^.]*?owns?\s+(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%[^.]*?(?:owned|held)\s+by\s+([A-Z][A-Za-z\s&,\.]+(?:LLC|LP|Inc|Corp|Company|Fund|Capital|Partners|Management|Group))",
                r"beneficial\s+owner[^.]*?([A-Z][A-Za-z\s&,\.]+)\s+[^.]*?(\d+(?:,\d+)*)\s+shares"
            ]

            for pattern in shareholder_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        shareholder = {
                            "name": groups[0] if pattern.index(pattern) != 1 else groups[1],
                            "percentage": groups[1] if pattern.index(pattern) != 1 else groups[0],
                            "context": match.group(0)[:200]
                        }
                        shareholder_data["shareholders"].append(shareholder)

            # Look for equity issuance information
            issuance_patterns = [
                r"(?:issued|granted|awarded)[^.]*?(\d+(?:,\d+)*)\s+shares[^.]*?(?:on|dated|as of)\s+([A-Z][a-z]+\s+\d+,?\s+\d{4})",
                r"([A-Z][a-z]+\s+\d+,?\s+\d{4})[^.]*?(?:issued|granted|awarded)[^.]*?(\d+(?:,\d+)*)\s+shares"
            ]

            for pattern in issuance_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    issuance = {
                        "shares": match.group(1),
                        "date": match.group(2),
                        "context": match.group(0)[:200]
                    }
                    shareholder_data["equity_issuances"].append(issuance)

            # Check tables for ownership data
            for table in extracted_data.get("tables", []):
                title = table.get("title", "").lower()
                if "ownership" in title or "shareholder" in title or "beneficial" in title:
                    rows = table.get("rows", [])
                    for row in rows:
                        # Extract shareholder from table row
                        shareholder = self._extract_shareholder_from_row(row)
                        if shareholder:
                            shareholder_data["shareholders"].append(shareholder)

        except Exception as e:
            logger.error(f"Error extracting shareholder info: {e}")

        return shareholder_data

    def _extract_shareholder_from_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract shareholder information from a table row.

        Args:
            row: Table row data

        Returns:
            Shareholder data or None
        """
        try:
            # Common column name patterns
            name_keys = ["name", "shareholder", "owner", "beneficial owner", "holder"]
            share_keys = ["shares", "shares owned", "number of shares", "common stock"]
            percent_keys = ["percentage", "%", "percent", "ownership %", "% of total"]

            shareholder = {}

            # Find name
            for key in row.keys():
                key_lower = str(key).lower()
                for name_key in name_keys:
                    if name_key in key_lower:
                        shareholder["name"] = str(row[key])
                        break

            # Find shares
            for key in row.keys():
                key_lower = str(key).lower()
                for share_key in share_keys:
                    if share_key in key_lower:
                        shareholder["shares"] = str(row[key])
                        break

            # Find percentage
            for key in row.keys():
                key_lower = str(key).lower()
                for percent_key in percent_keys:
                    if percent_key in key_lower:
                        shareholder["percentage"] = str(row[key])
                        break

            # Return if we found meaningful data
            if shareholder.get("name") and (shareholder.get("shares") or shareholder.get("percentage")):
                return shareholder

        except Exception as e:
            logger.debug(f"Error extracting shareholder from row: {e}")

        return None

    def _extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract key financial metrics from text.

        Args:
            text: Document text

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        try:
            # Revenue patterns
            revenue_patterns = [
                r"(?:total\s+)?revenue[^\$]*?\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)",
                r"(?:net\s+)?sales[^\$]*?\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)"
            ]

            for pattern in revenue_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics["revenue"] = match.group(1)
                    break

            # Earnings patterns
            earnings_patterns = [
                r"net\s+(?:income|earnings)[^\$]*?\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)",
                r"(?:diluted\s+)?earnings\s+per\s+share[^\$]*?\$\s*([\d,]+(?:\.\d+)?)"
            ]

            for pattern in earnings_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    key = "net_income" if "income" in pattern else "eps"
                    metrics[key] = match.group(1)

            # Asset patterns
            asset_pattern = r"total\s+assets[^\$]*?\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)"
            match = re.search(asset_pattern, text, re.IGNORECASE)
            if match:
                metrics["total_assets"] = match.group(1)

        except Exception as e:
            logger.debug(f"Error extracting metrics: {e}")

        return metrics

    async def _process_with_shareholding_pipeline(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any]
    ) -> bool:
        """
        Process filing with shareholding-optimized pipeline for enhanced knowledge graph ingestion.

        Args:
            filing: SECFiling object
            extracted_data: Extracted document data

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            logger.debug(f"Processing {filing.accession_number} with shareholding pipeline")

            # Prepare document for shareholding pipeline
            document_metadata = {
                "accession_number": filing.accession_number,
                "company_name": filing.company_name,
                "company_cik": filing.company_cik,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
                "is_insider_filing": getattr(filing, 'is_insider_filing', False),
                "issuer_name": getattr(filing, 'issuer_name', None),
                "issuer_cik": getattr(filing, 'issuer_cik', None),
                "issuer_ticker": getattr(filing, 'issuer_ticker', None),
                "reporting_owner_name": getattr(filing, 'reporting_owner_name', None)
            }

            # Process through shareholding pipeline
            success = await self.shareholding_pipeline.process_document(
                text_content=extracted_data.get("text", ""),
                tables=extracted_data.get("tables", []),
                metadata=document_metadata
            )

            if success:
                logger.debug(f"Shareholding pipeline processing completed for {filing.accession_number}")

                # Update filing to indicate advanced processing completion
                filing.processed_with_shareholding_pipeline = True

                return True
            else:
                logger.warning(f"Shareholding pipeline processing failed for {filing.accession_number}")
                return False

        except Exception as e:
            logger.error(f"Error in shareholding pipeline processing for {filing.accession_number}: {e}", exc_info=True)
            return False

    async def _save_processed_data(
        self,
        filing: SECFiling,
        extracted_data: Dict[str, Any]
    ) -> Path:
        """
        Save processed data to disk.

        Args:
            filing: SECFiling object
            extracted_data: Extracted data

        Returns:
            Path to saved file
        """
        # Create filing directory
        filing_dir = self.processed_path / filing.accession_number
        filing_dir.mkdir(exist_ok=True)

        # Save as JSON
        processed_file = filing_dir / f"{filing.accession_number}_processed.json"

        # Prepare data for saving
        save_data = {
            "accession_number": filing.accession_number,
            "company_name": filing.company_name,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date.isoformat(),
            "processed_at": datetime.utcnow().isoformat(),
            "extraction": extracted_data
        }

        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # Also save plain text for easy access
        text_file = filing_dir / f"{filing.accession_number}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(extracted_data.get("text", ""))

        logger.debug(f"Saved processed data to {processed_file}")
        return processed_file

    async def process_batch(
        self,
        filings: List[Tuple[SECFiling, str]],
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Process multiple filings concurrently.

        Args:
            filings: List of (filing, file_path) tuples
            max_concurrent: Maximum concurrent processing

        Returns:
            Processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(filing: SECFiling, file_path: str):
            async with semaphore:
                return await self.process_filing(filing, file_path)

        logger.info(f"Starting batch processing of {len(filings)} filings")

        # Create processing tasks
        tasks = [
            process_with_semaphore(filing, file_path)
            for filing, file_path in filings
        ]

        # Execute processing
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = []
        failed = []

        for (filing, _), result in zip(filings, results):
            if isinstance(result, Exception):
                failed.append({
                    "filing": filing.accession_number,
                    "error": str(result)
                })
            else:
                success, data = result
                if success:
                    successful.append({
                        "filing": filing.accession_number,
                        "text_length": len(data.get("text", "")),
                        "table_count": len(data.get("tables", []))
                    })
                else:
                    failed.append({
                        "filing": filing.accession_number,
                        "error": data.get("error", "Unknown error")
                    })

        return {
            "total": len(filings),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(filings) if filings else 0,
            "successful_processing": successful,
            "failed_processing": failed
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0 else 0
            ),
            "average_processing_time": (
                self.total_processing_time / self.processed_count
                if self.processed_count > 0 else 0
            )
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)