"""
Issuer extraction module for Form 3/4/5 SEC filings.
Extracts issuer company information from insider ownership reports,
distinguishing between reporting owners and issuer companies.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class IssuerExtractor:
    """Extract issuer company information from Form 3/4/5 filings."""

    def __init__(self):
        """Initialize issuer extractor."""
        self.form_345_types = {"3", "4", "5", "Form 3", "Form 4", "Form 5"}

        # Issuer identification patterns for Form 3/4/5
        self.issuer_patterns = [
            # Form 3/4/5 specific: "3. Issuer Name and Ticker" section with HTML
            r'3\.\s*Issuer\s+Name.*?<a[^>]*>([A-Z][A-Za-z0-9\s&,\.\-]+(?:INC|CORP|COMPANY|LLC|LP|LTD|CO)\.?)</a>',

            # Alternative HTML pattern for issuer name in anchor tags
            r'<a[^>]*getcompany[^>]*>([A-Z][A-Za-z0-9\s&,\.\-]+(?:INC|CORP|COMPANY|LLC|LP|LTD|CO)\.?)</a>',

            # "Issuer Name: COMPANY NAME INC" format
            r"issuer\s+name:\s*([A-Z][A-Za-z0-9\s&,\.\-]+(?:Inc|Corp|Company|LLC|LP|Ltd|Co)\.?)",

            # "Company: COMPANY NAME" in header sections
            r"company:\s*([A-Z][A-Za-z0-9\s&,\.\-]+(?:Inc|Corp|Company|LLC|LP|Ltd|Co)\.?)",

            # Securities of issuer patterns
            r"securities\s+of\s+([A-Z][A-Za-z0-9\s&,\.\-]+(?:Inc|Corp|Company|LLC|LP|Ltd|Co)\.?)",

            # Issuer information section
            r"issuer\s+information[^:]*:\s*([A-Z][A-Za-z0-9\s&,\.\-]+(?:Inc|Corp|Company|LLC|LP|Ltd|Co)\.?)",

            # Table headers with issuer company
            r"(?:subject\s+)?company[^:]*:\s*([A-Z][A-Za-z0-9\s&,\.\-]+(?:Inc|Corp|Company|LLC|LP|Ltd|Co)\.?)",
        ]

        # CIK patterns
        self.cik_patterns = [
            # Form 3/4/5 specific: CIK in URL within anchor tag
            r'<a[^>]*CIK=(\d{10})[^>]*>',

            # Standard patterns
            r"(?:issuer\s+)?(?:central\s+index\s+key|cik):\s*(\d{10})",
            r"cik\s+no\.?\s*(\d{10})",
            r"central\s+index\s+key\s+number:\s*(\d{10})",
        ]

        # Ticker symbol patterns
        self.ticker_patterns = [
            # Form 3/4/5 specific: ticker in brackets after issuer name
            r'\[\s*<span[^>]*>([A-Z]{1,5})</span>\s*\]',

            # Alternative bracket formats
            r'\[\s*([A-Z]{1,5})\s*\]',

            # Standard patterns
            r"(?:trading\s+symbol|ticker\s+symbol|symbol):\s*([A-Z]{1,5})",
            r"nasdaq\s+symbol:\s*([A-Z]{1,5})",
            r"nyse\s+symbol:\s*([A-Z]{1,5})",
            r"common\s+stock\s+symbol:\s*([A-Z]{1,5})",
        ]

        # Reporting owner patterns (to distinguish from issuer)
        self.reporting_owner_patterns = [
            r"reporting\s+owner\s+name:\s*([A-Za-z\s,\.]+?)(?:\n|$|This)",
            r"owner\s+name:\s*([A-Za-z\s,\.]+?)(?:\n|$|This)",
            r"name\s+of\s+reporting\s+person:\s*([A-Za-z\s,\.]+?)(?:\n|$|This)",
            r"reporting\s+person:\s*([A-Za-z\s,\.]+?)(?:\n|$|This)",
        ]

    def is_form_345(self, form_type: str) -> bool:
        """
        Check if this is a Form 3/4/5 filing.

        Args:
            form_type: SEC form type

        Returns:
            True if Form 3/4/5
        """
        return any(form_type.strip() == ft for ft in self.form_345_types)

    async def extract_issuer_info(
        self,
        extracted_data: Dict[str, Any],
        form_type: str
    ) -> Dict[str, Any]:
        """
        Extract issuer information from Form 3/4/5 filing content.

        Args:
            extracted_data: Document data from Docling processor
            form_type: SEC form type

        Returns:
            Dictionary containing issuer information
        """
        logger.info(f"Extracting issuer information from {form_type}")

        issuer_data = {
            "issuer_name": None,
            "issuer_cik": None,
            "issuer_ticker": None,
            "reporting_owner_name": None,
            "is_insider_filing": False,
            "extraction_confidence": 0.0,
            "extraction_sources": []
        }

        # Only process Form 3/4/5 filings
        if not self.is_form_345(form_type):
            logger.debug(f"Skipping issuer extraction for non-insider filing: {form_type}")
            return issuer_data

        issuer_data["is_insider_filing"] = True

        try:
            # Extract from main text
            main_text = extracted_data.get("text", "")
            if main_text:
                text_results = await self._extract_from_text(main_text)
                self._merge_results(issuer_data, text_results, "main_text")

            # Extract from tables
            tables = extracted_data.get("tables", [])
            if tables:
                table_results = await self._extract_from_tables(tables)
                self._merge_results(issuer_data, table_results, "tables")

            # Extract from sections
            sections = extracted_data.get("sections", [])
            if sections:
                section_results = await self._extract_from_sections(sections)
                self._merge_results(issuer_data, section_results, "sections")

            # Calculate confidence based on completeness
            issuer_data["extraction_confidence"] = self._calculate_confidence(issuer_data)

            logger.info(
                f"Issuer extraction completed - "
                f"Issuer: {issuer_data['issuer_name']}, "
                f"CIK: {issuer_data['issuer_cik']}, "
                f"Ticker: {issuer_data['issuer_ticker']}, "
                f"Confidence: {issuer_data['extraction_confidence']:.2f}"
            )

        except Exception as e:
            logger.error(f"Error during issuer extraction: {e}", exc_info=True)
            issuer_data["extraction_error"] = str(e)

        return issuer_data

    async def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract issuer information from document text.

        Args:
            text: Document text content

        Returns:
            Extracted issuer information
        """
        results = {}

        try:
            # Clean text for better pattern matching
            text_clean = re.sub(r'\s+', ' ', text)

            # Extract issuer company name
            for pattern in self.issuer_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
                if match:
                    issuer_name = match.group(1).strip()
                    # Clean up common artifacts
                    issuer_name = re.sub(r'\s+', ' ', issuer_name)
                    results["issuer_name"] = issuer_name
                    logger.debug(f"Found issuer name via pattern: {issuer_name}")
                    break

            # Extract CIK
            for pattern in self.cik_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    cik = match.group(1).zfill(10)  # Ensure 10 digits
                    results["issuer_cik"] = cik
                    logger.debug(f"Found issuer CIK: {cik}")
                    break

            # Extract ticker symbol
            for pattern in self.ticker_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    ticker = match.group(1).upper()
                    results["issuer_ticker"] = ticker
                    logger.debug(f"Found issuer ticker: {ticker}")
                    break

            # Extract reporting owner name
            for pattern in self.reporting_owner_patterns:
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    owner_name = match.group(1).strip()
                    owner_name = re.sub(r'\s+', ' ', owner_name)
                    results["reporting_owner_name"] = owner_name
                    logger.debug(f"Found reporting owner: {owner_name}")
                    break

        except Exception as e:
            logger.error(f"Error extracting from text: {e}")

        return results

    async def _extract_from_tables(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract issuer information from document tables.

        Args:
            tables: List of table data

        Returns:
            Extracted issuer information
        """
        results = {}

        try:
            for table in tables:
                title = table.get("title", "").lower()

                # Look for issuer information tables
                if any(keyword in title for keyword in ["issuer", "company", "security"]):
                    table_results = self._extract_from_table(table)
                    results.update(table_results)

        except Exception as e:
            logger.error(f"Error extracting from tables: {e}")

        return results

    def _extract_from_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract issuer information from a single table.

        Args:
            table: Table data

        Returns:
            Extracted information
        """
        results = {}

        try:
            rows = table.get("rows", [])

            for row in rows:
                if isinstance(row, dict):
                    # Look for key-value pairs in table rows
                    for key, value in row.items():
                        key_lower = str(key).lower()
                        value_str = str(value).strip()

                        # Company name
                        if "company" in key_lower or "issuer" in key_lower:
                            if self._is_valid_company_name(value_str):
                                results["issuer_name"] = value_str

                        # CIK
                        elif "cik" in key_lower:
                            cik_match = re.search(r'\d{10}', value_str)
                            if cik_match:
                                results["issuer_cik"] = cik_match.group(0)

                        # Ticker
                        elif "symbol" in key_lower or "ticker" in key_lower:
                            ticker_match = re.search(r'[A-Z]{1,5}', value_str)
                            if ticker_match:
                                results["issuer_ticker"] = ticker_match.group(0)

                        # Reporting owner
                        elif "owner" in key_lower or "person" in key_lower:
                            if self._is_valid_person_name(value_str):
                                results["reporting_owner_name"] = value_str

        except Exception as e:
            logger.debug(f"Error extracting from table: {e}")

        return results

    async def _extract_from_sections(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract issuer information from document sections.

        Args:
            sections: List of document sections

        Returns:
            Extracted issuer information
        """
        results = {}

        try:
            for section in sections:
                title = section.get("title", "").lower()
                text = section.get("text", "")

                # Focus on relevant sections
                if any(keyword in title for keyword in ["issuer", "company", "reporting", "cover"]):
                    section_results = await self._extract_from_text(text)
                    results.update(section_results)

        except Exception as e:
            logger.error(f"Error extracting from sections: {e}")

        return results

    def _merge_results(
        self,
        issuer_data: Dict[str, Any],
        new_results: Dict[str, Any],
        source: str
    ) -> None:
        """
        Merge extraction results with preference for completeness.

        Args:
            issuer_data: Main issuer data dictionary
            new_results: New results to merge
            source: Source of the new results
        """
        for key, value in new_results.items():
            if value and (not issuer_data.get(key) or len(str(value)) > len(str(issuer_data.get(key, "")))):
                issuer_data[key] = value
                issuer_data["extraction_sources"].append(f"{key}:{source}")

    def _calculate_confidence(self, issuer_data: Dict[str, Any]) -> float:
        """
        Calculate extraction confidence based on completeness.

        Args:
            issuer_data: Issuer data dictionary

        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0

        # Key information weights
        if issuer_data.get("issuer_name"):
            score += 0.4
        if issuer_data.get("issuer_cik"):
            score += 0.3
        if issuer_data.get("issuer_ticker"):
            score += 0.2
        if issuer_data.get("reporting_owner_name"):
            score += 0.1

        return min(score, 1.0)

    def _is_valid_company_name(self, name: str) -> bool:
        """
        Validate if string looks like a company name.

        Args:
            name: Potential company name

        Returns:
            True if valid company name
        """
        if not name or len(name) < 3:
            return False

        # Must contain company indicators
        company_indicators = ["inc", "corp", "company", "llc", "lp", "ltd", "co"]
        name_lower = name.lower()

        return any(indicator in name_lower for indicator in company_indicators)

    def _is_valid_person_name(self, name: str) -> bool:
        """
        Validate if string looks like a person name.

        Args:
            name: Potential person name

        Returns:
            True if valid person name
        """
        if not name or len(name) < 3:
            return False

        # Should not contain company indicators
        company_indicators = ["inc", "corp", "company", "llc", "lp", "ltd"]
        name_lower = name.lower()

        # Should contain typical name patterns
        name_patterns = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
            r'[A-Z][a-z]+,\s+[A-Z][a-z]+',  # Last, First
        ]

        has_company_indicator = any(indicator in name_lower for indicator in company_indicators)
        has_name_pattern = any(re.search(pattern, name) for pattern in name_patterns)

        return has_name_pattern and not has_company_indicator

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "supported_forms": list(self.form_345_types),
            "issuer_patterns": len(self.issuer_patterns),
            "cik_patterns": len(self.cik_patterns),
            "ticker_patterns": len(self.ticker_patterns),
            "reporting_owner_patterns": len(self.reporting_owner_patterns)
        }