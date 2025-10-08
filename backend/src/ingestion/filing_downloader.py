"""
SEC filing downloader with rate limiting and retry logic.
Handles document downloads while respecting SEC rate limits.
"""
import asyncio
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
import hashlib
import mimetypes
from urllib.parse import urlparse, unquote
import re
import json

from ..config.settings import Settings, get_settings
from ..models.filing import SECFiling, FilingStatus, ProcessingStage

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for SEC API compliance."""

    def __init__(self, max_requests_per_second: float = 10.0):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_second: Maximum requests per second
        """
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)

            self.last_request_time = asyncio.get_event_loop().time()


class FilingDownloader:
    """Download SEC filings with rate limiting and error handling."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize filing downloader."""
        self.settings = settings or get_settings()

        # HTTP client configuration
        self.user_agent = self.settings.SEC_USER_AGENT
        self.rate_limiter = RateLimiter(1.0 / self.settings.SEC_RATE_LIMIT_DELAY)
        self.max_retries = self.settings.SEC_MAX_RETRIES
        self.retry_delay = self.settings.SEC_RETRY_DELAY

        # Storage paths
        self.storage_path = Path(self.settings.FILING_STORAGE_PATH)
        self.temp_path = Path(self.settings.TEMP_STORAGE_PATH)
        self.max_file_size = self.settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)

        # HTTP client settings
        self.http_timeout = httpx.Timeout(60.0, connect=10.0)
        self.http_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)

        # Statistics
        self.total_downloads = 0
        self.failed_downloads = 0
        self.total_bytes_downloaded = 0

    async def download_filing(
        self,
        filing: SECFiling,
        prefer_html: bool = False
    ) -> Tuple[bool, str]:
        """
        Download filing documents and save to disk.

        Args:
            filing: SECFiling object with URLs
            prefer_html: Whether to prefer HTML over text documents

        Returns:
            Tuple of (success, file_path or error_message)
        """
        logger.info(
            f"Starting download for {filing.company_name} "
            f"{filing.form_type} ({filing.accession_number})"
        )

        # Update filing status
        filing.status = FilingStatus.DOWNLOADING
        filing.current_stage = ProcessingStage.DOCUMENT_DOWNLOAD
        filing.download_attempts += 1

        start_time = datetime.utcnow()

        try:
            # Determine which URL to download
            download_url = await self._select_download_url(filing, prefer_html)

            if not download_url:
                error_msg = "No valid download URL found"
                filing.mark_failed(error_msg)
                return False, error_msg

            # Download the document
            file_path = await self._download_document(
                download_url,
                filing.accession_number,
                filing.form_type
            )

            # Update filing with download info
            filing.raw_file_path = str(file_path)
            filing.download_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            filing.file_size_bytes = file_path.stat().st_size

            # Mark stage complete
            filing.current_stage = ProcessingStage.DOCLING_EXTRACTION

            self.total_downloads += 1
            logger.info(
                f"Successfully downloaded {filing.accession_number} "
                f"({filing.file_size_bytes / 1024:.1f} KB in {filing.download_time_seconds:.1f}s)"
            )

            return True, str(file_path)

        except Exception as e:
            self.failed_downloads += 1
            error_msg = f"Download failed: {str(e)}"
            logger.error(f"Failed to download {filing.accession_number}: {error_msg}", exc_info=True)
            filing.mark_failed(error_msg)
            return False, error_msg

    async def _select_download_url(
        self,
        filing: SECFiling,
        prefer_html: bool
    ) -> Optional[str]:
        """
        Select the best URL to download for a filing.

        Args:
            filing: SECFiling with potential URLs
            prefer_html: Whether to prefer HTML format

        Returns:
            Selected download URL or None
        """
        # Priority order for downloads
        url_candidates = []

        # ALWAYS try to resolve the actual document from the index first
        # This applies to ALL form types, not just 10-K, 10-Q, 8-K
        index_url = await self._resolve_index_document_url(filing)
        if index_url:
            logger.info(f"Resolved primary document URL from index: {index_url}")
            url_candidates.append(index_url)

        # Only use filing URL if it's not an index page
        if filing.filing_url and 'index' not in str(filing.filing_url).lower():
            url_candidates.append(str(filing.filing_url))

        # Then try HTML version if preferred and not an index
        if prefer_html and filing.filing_html_url and 'index' not in str(filing.filing_html_url).lower():
            url_candidates.append(str(filing.filing_html_url))

        # Add document URLs that are not index pages
        url_candidates.extend([
            str(url) for url in filing.document_urls
            if 'index' not in str(url).lower()
        ])

        # Test URLs and return first valid one
        for url in url_candidates:
            if await self._test_url(url):
                logger.debug(f"Selected download URL: {url}")
                return url

        return None

    async def _resolve_index_document_url(self, filing: SECFiling) -> Optional[str]:
        """
        Resolve the primary document URL from SEC filing index.

        Args:
            filing: SECFiling object

        Returns:
            Primary document URL or None
        """
        try:
            # Build index URL - try HTML first for parsing
            acc_no_dash = filing.accession_number.replace("-", "")
            cik_padded = filing.cik_number.zfill(10)

            # First try JSON API
            index_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_padded}/{acc_no_dash}/{filing.accession_number}-index.json"
            )

            await self.rate_limiter.acquire()
            headers = {"User-Agent": self.user_agent}

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True
            ) as client:
                # Try JSON endpoint first
                try:
                    response = await client.get(index_url, headers=headers)
                    if response.status_code == 200:
                        index_data = response.json()
                        # Find primary document
                        for item in index_data.get("directory", {}).get("item", []):
                            if item.get("type") in ["10-K", "10-Q", "8-K", filing.form_type]:
                                doc_name = item.get("name")
                                if doc_name:
                                    return (
                                        f"https://www.sec.gov/Archives/edgar/data/"
                                        f"{cik_padded}/{acc_no_dash}/{doc_name}"
                                    )
                except:
                    pass  # Fall through to HTML parsing

                # Fall back to HTML parsing
                html_index_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_padded}/{acc_no_dash}/{filing.accession_number}-index.html"
                )

                response = await client.get(html_index_url, headers=headers)
                if response.status_code == 200:
                    # Parse the HTML to find actual document links
                    html_content = response.text
                    document_urls = await self._parse_edgar_index_html(html_content, filing)
                    if document_urls:
                        return document_urls[0]  # Return the primary document

        except Exception as e:
            logger.debug(f"Could not resolve index document: {e}")

        return None

    async def _parse_edgar_index_html(self, html_content: str, filing: SECFiling) -> List[str]:
        """
        Parse EDGAR index HTML to extract actual document URLs.

        Args:
            html_content: HTML content of the index page
            filing: SECFiling object

        Returns:
            List of document URLs
        """
        document_urls = []

        try:
            # Import BeautifulSoup for HTML parsing
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the document table
            tables = soup.find_all('table', {'class': 'tableFile'})

            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Check if this is a document row
                        type_cell = cells[2] if len(cells) > 2 else None
                        doc_link = row.find('a', href=True)

                        if doc_link and type_cell:
                            doc_type = type_cell.get_text(strip=True)
                            # Look for the primary document (not exhibits or graphics)
                            if doc_type == filing.form_type or doc_type in ['4', '10-K', '10-Q', '8-K', 'DEF 14A', 'DEFA14A']:
                                href = doc_link['href']
                                # Convert relative URL to absolute
                                if href.startswith('/'):
                                    doc_url = f"https://www.sec.gov{href}"
                                else:
                                    # Build absolute URL
                                    acc_no_dash = filing.accession_number.replace("-", "")
                                    cik_padded = filing.cik_number.zfill(10)
                                    doc_url = (
                                        f"https://www.sec.gov/Archives/edgar/data/"
                                        f"{cik_padded}/{acc_no_dash}/{href}"
                                    )
                                document_urls.append(doc_url)
                                logger.debug(f"Found document URL: {doc_url}")

            # If no specific document found, look for any .htm or .html files
            if not document_urls:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if ('.htm' in href or '.xml' in href) and 'index' not in href.lower():
                        if href.startswith('/Archives/'):
                            doc_url = f"https://www.sec.gov{href}"
                            document_urls.append(doc_url)
                            logger.debug(f"Found fallback document URL: {doc_url}")

        except Exception as e:
            logger.error(f"Error parsing EDGAR index HTML: {e}")

        return document_urls

    async def _test_url(self, url: str) -> bool:
        """
        Test if a URL is accessible.

        Args:
            url: URL to test

        Returns:
            True if URL is accessible
        """
        try:
            await self.rate_limiter.acquire()

            headers = {"User-Agent": self.user_agent}

            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.head(url, headers=headers, follow_redirects=True)
                return response.status_code == 200

        except Exception:
            return False

    async def _download_document(
        self,
        url: str,
        accession_number: str,
        form_type: str
    ) -> Path:
        """
        Download document from URL with retry logic.

        Args:
            url: Document URL
            accession_number: Filing accession number
            form_type: Filing form type

        Returns:
            Path to downloaded file
        """
        # Determine file extension from URL or content type
        parsed_url = urlparse(url)
        url_path = unquote(parsed_url.path)
        extension = Path(url_path).suffix or ".html"

        # Create subdirectory for filing
        filing_dir = self.storage_path / accession_number
        filing_dir.mkdir(exist_ok=True)

        # Generate filename
        safe_form_type = re.sub(r'[^\w\-_]', '', form_type)
        filename = f"{accession_number}_{safe_form_type}{extension}"
        file_path = filing_dir / filename

        # Check if already downloaded
        if file_path.exists():
            logger.debug(f"File already exists: {file_path}")
            return file_path

        # Download with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()

                headers = {
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate",
                }

                async with httpx.AsyncClient(
                    timeout=self.http_timeout,
                    limits=self.http_limits,
                    follow_redirects=True
                ) as client:
                    # Stream download for large files
                    async with client.stream("GET", url, headers=headers) as response:
                        response.raise_for_status()

                        # Check content type
                        content_type = response.headers.get("content-type", "")
                        if not extension or extension == ".html":
                            # Determine extension from content type
                            ext = mimetypes.guess_extension(content_type.split(";")[0])
                            if ext:
                                filename = f"{accession_number}_{safe_form_type}{ext}"
                                file_path = filing_dir / filename

                        # Check file size
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > self.max_file_size:
                            raise ValueError(
                                f"File too large: {int(content_length) / 1024 / 1024:.1f} MB "
                                f"(max: {self.settings.MAX_FILE_SIZE_MB} MB)"
                            )

                        # Download to temp file first
                        temp_file = self.temp_path / f"{accession_number}_{attempt}.tmp"
                        total_bytes = 0

                        with open(temp_file, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                total_bytes += len(chunk)

                                # Check size during download
                                if total_bytes > self.max_file_size:
                                    temp_file.unlink(missing_ok=True)
                                    raise ValueError(
                                        f"File too large during download: "
                                        f"{total_bytes / 1024 / 1024:.1f} MB"
                                    )

                        # Move to final location
                        temp_file.rename(file_path)
                        self.total_bytes_downloaded += total_bytes

                        logger.debug(
                            f"Downloaded {total_bytes / 1024:.1f} KB "
                            f"from {url} to {file_path}"
                        )

                        return file_path

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning(f"Download attempt {attempt + 1} failed: {last_error}")

                if e.response.status_code == 404:
                    break  # Don't retry on 404

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Download attempt {attempt + 1} failed: {last_error}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.debug(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        raise Exception(f"Failed to download after {self.max_retries} attempts: {last_error}")

    async def download_batch(
        self,
        filings: List[SECFiling],
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download multiple filings concurrently.

        Args:
            filings: List of filings to download
            max_concurrent: Maximum concurrent downloads

        Returns:
            Dictionary with download results
        """
        max_concurrent = max_concurrent or self.settings.MAX_CONCURRENT_DOWNLOADS
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(filing: SECFiling):
            async with semaphore:
                return await self.download_filing(filing)

        logger.info(f"Starting batch download of {len(filings)} filings")

        # Create download tasks
        tasks = [download_with_semaphore(filing) for filing in filings]

        # Execute downloads
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = []
        failed = []

        for filing, result in zip(filings, results):
            if isinstance(result, Exception):
                failed.append({
                    "filing": filing.accession_number,
                    "error": str(result)
                })
            else:
                success, path_or_error = result
                if success:
                    successful.append({
                        "filing": filing.accession_number,
                        "path": path_or_error
                    })
                else:
                    failed.append({
                        "filing": filing.accession_number,
                        "error": path_or_error
                    })

        return {
            "total": len(filings),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(filings) if filings else 0,
            "successful_downloads": successful,
            "failed_downloads": failed
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get download statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_downloads": self.total_downloads,
            "failed_downloads": self.failed_downloads,
            "success_rate": (
                (self.total_downloads - self.failed_downloads) / self.total_downloads
                if self.total_downloads > 0 else 0
            ),
            "total_bytes_downloaded": self.total_bytes_downloaded,
            "total_mb_downloaded": self.total_bytes_downloaded / 1024 / 1024
        }

    async def cleanup_old_files(self, days: int = 7):
        """
        Clean up old downloaded files.

        Args:
            days: Files older than this many days will be deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cleaned_count = 0
        cleaned_bytes = 0

        for filing_dir in self.storage_path.iterdir():
            if filing_dir.is_dir():
                for file_path in filing_dir.iterdir():
                    if file_path.is_file():
                        file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_age < cutoff_date:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleaned_count += 1
                            cleaned_bytes += file_size

                # Remove empty directories
                if not any(filing_dir.iterdir()):
                    filing_dir.rmdir()

        logger.info(
            f"Cleaned up {cleaned_count} files "
            f"({cleaned_bytes / 1024 / 1024:.1f} MB) older than {days} days"
        )

        return {
            "files_removed": cleaned_count,
            "bytes_removed": cleaned_bytes
        }