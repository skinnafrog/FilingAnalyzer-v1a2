"""
RSS feed monitoring service for SEC filings.
Polls SEC RSS feeds and discovers new filings for processing.
"""
import asyncio
import feedparser
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
import logging
import re

from ..config.settings import Settings, get_settings
from ..models.filing import SECFiling, RSSFeedEntry, FilingType, FilingStatus, ProcessingStage

logger = logging.getLogger(__name__)


class RSSMonitor:
    """Monitor SEC RSS feeds for new filings."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize RSS monitor with configuration."""
        self.settings = settings or get_settings()
        self.feed_url = self.settings.RSS_FEED_URL
        self.poll_interval = self.settings.RSS_POLL_INTERVAL
        self.user_agent = self.settings.SEC_USER_AGENT
        self.feed_limit = self.settings.RSS_FEED_LIMIT

        # Track processed filings to avoid duplicates
        self.processed_accessions: set = set()
        self.last_poll_time: Optional[datetime] = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # HTTP client configuration
        self.http_timeout = httpx.Timeout(30.0, connect=10.0)
        self.http_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

    async def poll_feed(self, feed_url: Optional[str] = None) -> List[SECFiling]:
        """
        Poll RSS feed for new SEC filings.

        Args:
            feed_url: Optional custom feed URL to poll

        Returns:
            List of discovered SEC filings
        """
        url = feed_url or self.feed_url
        logger.info(f"Polling RSS feed: {url}")

        try:
            # Fetch RSS feed with proper headers
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/atom+xml, application/rss+xml, application/xml, text/xml",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }

            async with httpx.AsyncClient(
                timeout=self.http_timeout,
                limits=self.http_limits,
                follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            # Parse feed
            feed = feedparser.parse(response.text)

            if feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")

            # Extract filings from feed entries
            filings = await self._parse_feed_entries(feed.entries)

            # Update tracking
            self.last_poll_time = datetime.utcnow()
            self.consecutive_errors = 0

            logger.info(f"Discovered {len(filings)} new filings from RSS feed")
            return filings

        except httpx.HTTPStatusError as e:
            self.consecutive_errors += 1
            logger.error(f"HTTP error polling RSS feed: {e.response.status_code} - {e.response.text}")
            raise

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(f"Error polling RSS feed: {str(e)}", exc_info=True)
            raise

    async def _parse_feed_entries(self, entries: List[Dict[str, Any]]) -> List[SECFiling]:
        """
        Parse RSS feed entries into SECFiling objects.

        Args:
            entries: Raw feed entries from feedparser

        Returns:
            List of SECFiling objects
        """
        filings = []
        processed_count = 0

        for entry in entries[:self.feed_limit]:
            try:
                # Extract EDGAR-specific fields from namespaced elements
                rss_entry = self._extract_edgar_fields(entry)

                # Skip if already processed
                if rss_entry.accession_number in self.processed_accessions:
                    continue

                # Convert to SECFiling
                filing = rss_entry.to_sec_filing()

                # Extract additional URLs from entry
                filing = await self._enrich_filing_urls(filing, entry)

                # Mark as discovered
                filing.status = FilingStatus.DISCOVERED
                filing.current_stage = ProcessingStage.RSS_DISCOVERY

                filings.append(filing)
                self.processed_accessions.add(filing.accession_number)
                processed_count += 1

                logger.debug(
                    f"Discovered filing: {filing.company_name} - "
                    f"{filing.form_type} ({filing.accession_number})"
                )

            except Exception as e:
                logger.warning(f"Error parsing feed entry: {str(e)}", exc_info=True)
                continue

        logger.info(f"Parsed {processed_count} new filings from {len(entries)} feed entries")
        return filings

    def _extract_edgar_fields(self, entry: Dict[str, Any]) -> RSSFeedEntry:
        """
        Extract EDGAR-specific fields from RSS entry.

        Args:
            entry: Raw feedparser entry

        Returns:
            RSSFeedEntry with extracted fields
        """
        # Get basic fields
        rss_entry = RSSFeedEntry(
            title=entry.get("title", ""),
            link=entry.get("link", ""),
            published=self._parse_datetime(entry.get("published")),
            updated=self._parse_datetime(entry.get("updated")),
            summary=entry.get("summary")
        )

        # Extract EDGAR namespace fields (edgar:*)
        edgar_fields = {
            "edgar_xbrlfiling": {},
            "edgar_assistantdirector": None,
            "edgar_assignedsic": None,
            "edgar_fiscalyearend": None
        }

        # Different feed formats may use different namespaces
        for key, value in entry.items():
            if key.startswith("edgar_"):
                edgar_fields[key] = value

        # Extract company and filing information
        xbrl_filing = edgar_fields.get("edgar_xbrlfiling", {})

        rss_entry.company_name = xbrl_filing.get("edgar_companyname", entry.get("title", "").split(" - ")[0])
        rss_entry.cik_number = str(xbrl_filing.get("edgar_ciknumber", "")).zfill(10)
        rss_entry.form_type = xbrl_filing.get("edgar_formtype", "")
        rss_entry.filing_date = xbrl_filing.get("edgar_filingdate", "")
        rss_entry.accession_number = xbrl_filing.get("edgar_accessionnumber", "")
        rss_entry.file_number = xbrl_filing.get("edgar_filenumber")
        rss_entry.acceptance_datetime = xbrl_filing.get("edgar_acceptancedatetime")
        rss_entry.period = xbrl_filing.get("edgar_period")

        # Additional metadata
        rss_entry.assistant_director = edgar_fields.get("edgar_assistantdirector")
        rss_entry.assigned_sic = edgar_fields.get("edgar_assignedsic")
        rss_entry.fiscal_year_end = edgar_fields.get("edgar_fiscalyearend")

        # Extract XBRL files if present
        xbrl_files = xbrl_filing.get("edgar_xbrlfiles", {})
        if isinstance(xbrl_files, dict):
            rss_entry.xbrl_files = [
                {"name": k, "url": v} for k, v in xbrl_files.items()
            ]

        return rss_entry

    async def _enrich_filing_urls(self, filing: SECFiling, entry: Dict[str, Any]) -> SECFiling:
        """
        Enrich filing with additional URLs extracted from feed entry.

        Args:
            filing: Base SECFiling object
            entry: Raw feed entry

        Returns:
            Enriched SECFiling with additional URLs
        """
        # Build various SEC URLs from accession number
        if filing.accession_number:
            acc_no_dash = filing.accession_number.replace("-", "")
            cik_padded = filing.cik_number.zfill(10) if filing.cik_number else "0000000000"

            base_url = "https://www.sec.gov/Archives/edgar/data"
            filing_base = f"{base_url}/{cik_padded}/{acc_no_dash}"

            # Main filing index
            filing.filing_html_url = f"{filing_base}/{filing.accession_number}-index.html"

            # Try to determine primary document URL
            if filing.form_type in [FilingType.FORM_10K, FilingType.FORM_10Q, FilingType.FORM_8K]:
                # Common naming patterns for primary documents
                doc_patterns = [
                    f"{filing.accession_number}.htm",
                    f"{filing.accession_number}-99.1.htm",
                    f"{filing.accession_number[0:10]}-{filing.accession_number[10:12]}-{filing.accession_number[12:]}.htm"
                ]

                for pattern in doc_patterns:
                    filing.document_urls.append(f"{filing_base}/{pattern}")

            # XBRL instance document
            filing.filing_xml_url = f"{filing_base}/{filing.accession_number[0:10]}-{filing.accession_number[10:12]}-{filing.accession_number[12:]}_htm.xml"

            # XBRL ZIP file for structured data
            filing.xbrl_zip_url = f"{filing_base}/{filing.accession_number[0:10]}-{filing.accession_number[10:12]}-{filing.accession_number[12:]}-xbrl.zip"

        # Add any XBRL file URLs from the RSS entry
        xbrl_filing = entry.get("edgar_xbrlfiling", {})
        xbrl_files = xbrl_filing.get("edgar_xbrlfiles", {})

        if isinstance(xbrl_files, dict):
            for file_type, file_url in xbrl_files.items():
                if file_url and file_url not in filing.document_urls:
                    filing.document_urls.append(file_url)

        return filing

    def _parse_datetime(self, date_str: Optional[str]) -> datetime:
        """
        Parse datetime from various formats.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime or current time if parsing fails
        """
        if not date_str:
            return datetime.utcnow()

        # Try various date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%m/%d/%Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.replace("Z", "+0000"), fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return datetime.utcnow()

    async def continuous_monitor(
        self,
        callback=None,
        max_iterations: Optional[int] = None
    ) -> None:
        """
        Continuously monitor RSS feed at configured intervals.

        Args:
            callback: Async function to call with discovered filings
            max_iterations: Maximum number of polling iterations (for testing)
        """
        iteration = 0
        logger.info(
            f"Starting continuous RSS monitoring "
            f"(interval: {self.poll_interval}s, limit: {self.feed_limit} entries)"
        )

        while True:
            try:
                # Check if we should stop
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations}), stopping monitor")
                    break

                # Poll feed
                filings = await self.poll_feed()

                # Process discovered filings
                if filings and callback:
                    try:
                        await callback(filings)
                    except Exception as e:
                        logger.error(f"Error in filing callback: {str(e)}", exc_info=True)

                # Check for excessive errors
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive errors ({self.consecutive_errors}), "
                        f"pausing for extended period"
                    )
                    await asyncio.sleep(self.poll_interval * 5)
                    self.consecutive_errors = 0

                iteration += 1

                # Wait for next polling interval
                logger.debug(f"Waiting {self.poll_interval}s until next poll...")
                await asyncio.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("RSS monitor interrupted by user")
                break

            except Exception as e:
                logger.error(f"Unexpected error in continuous monitor: {str(e)}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

        logger.info("RSS monitor stopped")

    async def test_connection(self) -> bool:
        """
        Test RSS feed connection and parsing.

        Returns:
            True if connection successful
        """
        try:
            logger.info("Testing RSS feed connection...")

            headers = {"User-Agent": self.user_agent}

            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.get(self.feed_url, headers=headers)
                response.raise_for_status()

            feed = feedparser.parse(response.text)

            if feed.bozo:
                logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
                return False

            entry_count = len(feed.entries)
            logger.info(f"RSS feed connection successful. Found {entry_count} entries")

            # Try parsing first entry
            if entry_count > 0:
                test_entry = self._extract_edgar_fields(feed.entries[0])
                logger.info(
                    f"Sample entry: {test_entry.company_name} - "
                    f"{test_entry.form_type} ({test_entry.accession_number})"
                )

            return True

        except Exception as e:
            logger.error(f"RSS feed connection test failed: {str(e)}")
            return False

    def reset_processed_cache(self):
        """Reset the cache of processed accession numbers."""
        self.processed_accessions.clear()
        logger.info("Cleared processed accessions cache")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitor status.

        Returns:
            Status dictionary
        """
        return {
            "feed_url": self.feed_url,
            "poll_interval": self.poll_interval,
            "last_poll_time": self.last_poll_time.isoformat() if self.last_poll_time else None,
            "processed_count": len(self.processed_accessions),
            "consecutive_errors": self.consecutive_errors,
            "is_healthy": self.consecutive_errors < self.max_consecutive_errors
        }