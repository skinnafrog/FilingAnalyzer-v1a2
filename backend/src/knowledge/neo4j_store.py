"""
Neo4j knowledge graph storage for SEC filings.
Stores companies, filings, and relationships in a graph database.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
import asyncio

from ..config.settings import Settings, get_settings
from ..models.filing import SECFiling

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Manage Neo4j knowledge graph storage."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Neo4j connection."""
        self.settings = settings or get_settings()
        self.uri = self.settings.NEO4J_URI
        self.user = self.settings.NEO4J_USER
        self.password = self.settings.NEO4J_PASSWORD
        self.database = self.settings.NEO4J_DATABASE

        # Initialize driver
        self.driver = None
        self._connect()

        # Create constraints and indexes
        self._initialize_schema()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            raise

    def _initialize_schema(self):
        """Create constraints and indexes for optimal performance."""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints for unique identifiers
                constraints = [
                    "CREATE CONSTRAINT company_cik IF NOT EXISTS FOR (c:Company) REQUIRE c.cik IS UNIQUE",
                    "CREATE CONSTRAINT filing_accession IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession_number IS UNIQUE",
                    "CREATE CONSTRAINT person_name_company IF NOT EXISTS FOR (p:Person) REQUIRE (p.name, p.company_cik) IS UNIQUE"
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.info(f"Created constraint: {constraint.split('CONSTRAINT')[1].split('IF')[0].strip()}")
                    except ClientError as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Constraint creation warning: {e}")

                # Create indexes for search performance
                indexes = [
                    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
                    "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
                    "CREATE INDEX company_exchange IF NOT EXISTS FOR (c:Company) ON (c.exchange)",
                    "CREATE INDEX company_industry IF NOT EXISTS FOR (c:Company) ON (c.industry)",
                    "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
                    "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON (f.filing_date)",
                    "CREATE INDEX filing_type IF NOT EXISTS FOR (f:Filing) ON (f.form_type)",
                    "CREATE INDEX filing_period IF NOT EXISTS FOR (f:Filing) ON (f.period_date)",
                    "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)"
                ]

                for index in indexes:
                    try:
                        session.run(index)
                        logger.info(f"Created index: {index.split('INDEX')[1].split('IF')[0].strip()}")
                    except ClientError as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Index creation warning: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")

    def store_filing(self, filing: SECFiling, company_data: Dict[str, Any]) -> bool:
        """
        Store filing and company data in Neo4j.

        Args:
            filing: SECFiling object
            company_data: Company information dict

        Returns:
            Success boolean
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Create or update Company node with comprehensive data
                company_query = """
                MERGE (c:Company {cik: $cik})
                SET c.name = $name,
                    c.ticker = $ticker,
                    c.exchange = $exchange,
                    c.sic_code = $sic_code,
                    c.industry = $industry,
                    c.sector = $sector,
                    c.state = $state,
                    c.state_of_incorporation = $state_of_incorporation,
                    c.business_address = $business_address,
                    c.mailing_address = $mailing_address,
                    c.phone = $phone,
                    c.website = $website,
                    c.fiscal_year_end = $fiscal_year_end,
                    c.irs_number = $irs_number,
                    c.updated_at = datetime()
                RETURN c
                """

                company_result = session.run(
                    company_query,
                    cik=filing.cik_number,
                    name=filing.company_name,
                    ticker=getattr(filing, 'ticker_symbol', None) or company_data.get("ticker"),
                    exchange=company_data.get("exchange"),
                    sic_code=company_data.get("sic_code"),
                    industry=company_data.get("industry"),
                    sector=company_data.get("sector"),
                    state=company_data.get("state"),
                    state_of_incorporation=company_data.get("state_of_incorporation"),
                    business_address=company_data.get("business_address"),
                    mailing_address=company_data.get("mailing_address"),
                    phone=company_data.get("phone"),
                    website=company_data.get("website"),
                    fiscal_year_end=company_data.get("fiscal_year_end"),
                    irs_number=company_data.get("irs_number")
                )

                # Create Filing node with comprehensive metadata
                filing_query = """
                MERGE (f:Filing {accession_number: $accession_number})
                SET f.form_type = $form_type,
                    f.filing_date = $filing_date,
                    f.period_of_report = $period,
                    f.period_date = $period_date,
                    f.document_count = $doc_count,
                    f.file_url = $file_url,
                    f.filing_html_url = $filing_html_url,
                    f.filing_xml_url = $filing_xml_url,
                    f.xbrl_zip_url = $xbrl_zip_url,
                    f.file_number = $file_number,
                    f.film_number = $film_number,
                    f.acceptance_datetime = $acceptance_datetime,
                    f.items = $items,
                    f.size_bytes = $size_bytes,
                    f.is_xbrl = $is_xbrl,
                    f.is_inline_xbrl = $is_inline_xbrl,
                    f.primary_document = $primary_document,
                    f.primary_doc_description = $primary_doc_description,
                    f.processed_at = datetime()
                RETURN f
                """

                filing_result = session.run(
                    filing_query,
                    accession_number=filing.accession_number,
                    form_type=filing.form_type,
                    filing_date=filing.filing_date.isoformat() if filing.filing_date else None,
                    period=getattr(filing, 'period', None),
                    period_date=getattr(filing, 'period_date', None).isoformat() if getattr(filing, 'period_date', None) else None,
                    doc_count=len(getattr(filing, 'document_urls', [])),
                    file_url=str(getattr(filing, 'filing_url', None)) if getattr(filing, 'filing_url', None) else None,
                    filing_html_url=str(getattr(filing, 'filing_html_url', None)) if getattr(filing, 'filing_html_url', None) else None,
                    filing_xml_url=str(getattr(filing, 'filing_xml_url', None)) if getattr(filing, 'filing_xml_url', None) else None,
                    xbrl_zip_url=str(getattr(filing, 'xbrl_zip_url', None)) if getattr(filing, 'xbrl_zip_url', None) else None,
                    file_number=getattr(filing, 'file_number', None),
                    film_number=getattr(filing, 'film_number', None),
                    acceptance_datetime=getattr(filing, 'acceptance_datetime', None),
                    items=getattr(filing, 'items', None),
                    size_bytes=getattr(filing, 'size_bytes', None),
                    is_xbrl=getattr(filing, 'is_xbrl', False),
                    is_inline_xbrl=getattr(filing, 'is_inline_xbrl', False),
                    primary_document=getattr(filing, 'primary_document', None),
                    primary_doc_description=getattr(filing, 'primary_doc_description', None)
                )

                # Create relationship between Company and Filing
                relationship_query = """
                MATCH (c:Company {cik: $cik})
                MATCH (f:Filing {accession_number: $accession_number})
                MERGE (c)-[r:FILED]->(f)
                SET r.filed_at = datetime()
                RETURN r
                """

                session.run(
                    relationship_query,
                    cik=filing.cik_number,
                    accession_number=filing.accession_number
                )

                # Store financial metrics if available
                if getattr(filing, 'key_metrics', None):
                    self._store_financial_metrics(session, filing)

                # Store shareholders if available
                if getattr(filing, 'shareholders', None):
                    self._store_shareholders(session, filing)

                # Store executive officers if available
                if getattr(filing, 'officers', None):
                    self._store_officers(session, filing, company_data)

                # Store insider transactions if available
                if getattr(filing, 'insider_transactions', None):
                    self._store_insider_transactions(session, filing)

                logger.info(f"Stored filing {filing.accession_number} in Neo4j")
                return True

        except Exception as e:
            logger.error(f"Failed to store filing in Neo4j: {e}")
            return False

    def _store_financial_metrics(self, session, filing: SECFiling):
        """Store financial metrics as separate nodes."""
        try:
            metrics_query = """
            MATCH (f:Filing {accession_number: $accession_number})
            MERGE (m:FinancialMetrics {filing_accession: $accession_number})
            SET m.revenue = $revenue,
                m.net_income = $net_income,
                m.total_assets = $total_assets,
                m.total_liabilities = $total_liabilities,
                m.cash = $cash,
                m.period = $period
            MERGE (f)-[:HAS_METRICS]->(m)
            RETURN m
            """

            metrics = getattr(filing, 'key_metrics', {}) or {}
            session.run(
                metrics_query,
                accession_number=filing.accession_number,
                revenue=metrics.get("revenue"),
                net_income=metrics.get("net_income"),
                total_assets=metrics.get("total_assets"),
                total_liabilities=metrics.get("total_liabilities"),
                cash=metrics.get("cash"),
                period=getattr(filing, 'period', None)
            )

        except Exception as e:
            logger.warning(f"Failed to store financial metrics: {e}")

    def _store_shareholders(self, session, filing: SECFiling):
        """Store shareholder information."""
        try:
            shareholders = getattr(filing, 'shareholders', [])
            for shareholder in shareholders[:50]:  # Limit to top 50
                shareholder_query = """
                MERGE (p:Person {name: $name, company_cik: $cik})
                SET p.shares = $shares,
                    p.percentage = $percentage,
                    p.updated_at = datetime()
                WITH p
                MATCH (c:Company {cik: $cik})
                MERGE (p)-[:OWNS_SHARES_IN]->(c)
                WITH p
                MATCH (f:Filing {accession_number: $accession_number})
                MERGE (f)-[:REPORTS_SHAREHOLDER]->(p)
                RETURN p
                """

                session.run(
                    shareholder_query,
                    name=shareholder.get("name"),
                    cik=filing.cik_number,
                    shares=shareholder.get("shares"),
                    percentage=shareholder.get("percentage"),
                    accession_number=filing.accession_number
                )

        except Exception as e:
            logger.warning(f"Failed to store shareholders: {e}")

    def _store_officers(self, session, filing: SECFiling, company_data: Dict[str, Any]):
        """Store executive officers and directors."""
        try:
            officers = getattr(filing, 'officers', [])
            for officer in officers:
                officer_query = """
                MERGE (p:Person {name: $name, company_cik: $cik})
                SET p.title = $title,
                    p.is_director = $is_director,
                    p.is_officer = $is_officer,
                    p.is_ten_percent_owner = $is_ten_percent_owner,
                    p.officer_since = $officer_since,
                    p.age = $age,
                    p.updated_at = datetime()
                WITH p
                MATCH (c:Company {cik: $cik})
                MERGE (p)-[r:WORKS_FOR]->(c)
                SET r.role = $title,
                    r.is_current = true
                WITH p
                MATCH (f:Filing {accession_number: $accession_number})
                MERGE (f)-[:REPORTS_OFFICER]->(p)
                RETURN p
                """

                session.run(
                    officer_query,
                    name=officer.get("name"),
                    cik=filing.cik_number,
                    title=officer.get("title"),
                    is_director=officer.get("is_director", False),
                    is_officer=officer.get("is_officer", True),
                    is_ten_percent_owner=officer.get("is_ten_percent_owner", False),
                    officer_since=officer.get("officer_since"),
                    age=officer.get("age"),
                    accession_number=filing.accession_number
                )

        except Exception as e:
            logger.warning(f"Failed to store officers: {e}")

    def _store_insider_transactions(self, session, filing: SECFiling):
        """Store insider trading transactions."""
        try:
            transactions = getattr(filing, 'insider_transactions', [])
            for transaction in transactions:
                transaction_query = """
                MERGE (t:Transaction {transaction_id: $transaction_id})
                SET t.transaction_date = $transaction_date,
                    t.transaction_type = $transaction_type,
                    t.shares = $shares,
                    t.price_per_share = $price_per_share,
                    t.total_value = $total_value,
                    t.shares_owned_after = $shares_owned_after,
                    t.is_direct = $is_direct
                WITH t
                MATCH (f:Filing {accession_number: $accession_number})
                MERGE (f)-[:CONTAINS_TRANSACTION]->(t)
                WITH t
                MATCH (p:Person {name: $insider_name, company_cik: $cik})
                MERGE (p)-[:EXECUTED_TRANSACTION]->(t)
                WITH t
                MATCH (c:Company {cik: $cik})
                MERGE (t)-[:TRANSACTION_IN]->(c)
                RETURN t
                """

                session.run(
                    transaction_query,
                    transaction_id=transaction.get("transaction_id", f"{filing.accession_number}_{transaction.get('transaction_date')}_{transaction.get('insider_name')}"),
                    transaction_date=transaction.get("transaction_date"),
                    transaction_type=transaction.get("transaction_type"),
                    shares=transaction.get("shares"),
                    price_per_share=transaction.get("price_per_share"),
                    total_value=transaction.get("total_value"),
                    shares_owned_after=transaction.get("shares_owned_after"),
                    is_direct=transaction.get("is_direct", True),
                    insider_name=transaction.get("insider_name"),
                    cik=filing.cik_number,
                    accession_number=filing.accession_number
                )

        except Exception as e:
            logger.warning(f"Failed to store insider transactions: {e}")

    def create_filing_relationships(self, filing: SECFiling, related_companies: List[str]):
        """
        Create relationships between filings and mentioned companies.

        Args:
            filing: Current filing
            related_companies: List of company names or CIKs mentioned
        """
        try:
            with self.driver.session(database=self.database) as session:
                for company_ref in related_companies:
                    query = """
                    MATCH (f:Filing {accession_number: $accession_number})
                    MATCH (c:Company)
                    WHERE c.name CONTAINS $company_ref OR c.cik = $company_ref
                    MERGE (f)-[:MENTIONS]->(c)
                    RETURN c.name
                    """

                    result = session.run(
                        query,
                        accession_number=filing.accession_number,
                        company_ref=company_ref
                    )

                    for record in result:
                        logger.debug(f"Created MENTIONS relationship with {record['c.name']}")

        except Exception as e:
            logger.warning(f"Failed to create filing relationships: {e}")

    def get_company_filings(self, cik: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent filings for a company.

        Args:
            cik: Company CIK number
            limit: Maximum number of filings to return

        Returns:
            List of filing dictionaries
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (c:Company {cik: $cik})-[:FILED]->(f:Filing)
                RETURN f.accession_number as accession_number,
                       f.form_type as form_type,
                       f.filing_date as filing_date,
                       f.period_of_report as period,
                       f.file_url as url,
                       f.filing_html_url as html_url,
                       f.size_bytes as size,
                       f.is_xbrl as is_xbrl
                ORDER BY f.filing_date DESC
                LIMIT $limit
                """

                result = session.run(query, cik=cik, limit=limit)
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to get company filings: {e}")
            return []

    def get_related_companies(self, cik: str, relationship_type: str = "MENTIONS") -> List[Dict[str, Any]]:
        """
        Get companies related through filings.

        Args:
            cik: Company CIK number
            relationship_type: Type of relationship to query

        Returns:
            List of related company dictionaries
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (c1:Company {cik: $cik})-[:FILED]->(f:Filing)-[:%s]->(c2:Company)
                RETURN DISTINCT c2.name as name, c2.cik as cik, COUNT(f) as mention_count
                ORDER BY mention_count DESC
                LIMIT 20
                """ % relationship_type

                result = session.run(query, cik=cik)
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to get related companies: {e}")
            return []

    def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for companies by name.

        Args:
            query: Search query string

        Returns:
            List of matching companies
        """
        try:
            with self.driver.session(database=self.database) as session:
                cypher_query = """
                MATCH (c:Company)
                WHERE toLower(c.name) CONTAINS toLower($query)
                   OR toLower(c.ticker) CONTAINS toLower($query)
                   OR c.cik = $query
                RETURN c.name as name,
                       c.cik as cik,
                       c.ticker as ticker,
                       c.exchange as exchange,
                       c.industry as industry,
                       c.sector as sector
                LIMIT 10
                """

                result = session.run(cypher_query, query=query)
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to search companies: {e}")
            return []

    def get_company_details(self, cik: str) -> Dict[str, Any]:
        """
        Get comprehensive company information.

        Args:
            cik: Company CIK number

        Returns:
            Dictionary with full company details
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (c:Company {cik: $cik})
                OPTIONAL MATCH (c)<-[:WORKS_FOR]-(officer:Person)
                WHERE officer.is_officer = true OR officer.is_director = true
                OPTIONAL MATCH (c)<-[:OWNS_SHARES_IN]-(shareholder:Person)
                OPTIONAL MATCH (c)-[:FILED]->(f:Filing)
                WITH c,
                     COLLECT(DISTINCT {
                         name: officer.name,
                         title: officer.title,
                         is_director: officer.is_director,
                         is_officer: officer.is_officer
                     }) as officers,
                     COLLECT(DISTINCT {
                         name: shareholder.name,
                         shares: shareholder.shares,
                         percentage: shareholder.percentage
                     }) as shareholders,
                     COUNT(DISTINCT f) as filing_count,
                     MAX(f.filing_date) as last_filing_date
                RETURN c.name as name,
                       c.cik as cik,
                       c.ticker as ticker,
                       c.exchange as exchange,
                       c.sic_code as sic_code,
                       c.industry as industry,
                       c.sector as sector,
                       c.state as state,
                       c.state_of_incorporation as state_of_incorporation,
                       c.business_address as business_address,
                       c.mailing_address as mailing_address,
                       c.phone as phone,
                       c.website as website,
                       c.fiscal_year_end as fiscal_year_end,
                       c.irs_number as irs_number,
                       officers,
                       shareholders,
                       filing_count,
                       last_filing_date
                """

                result = session.run(query, cik=cik)
                record = result.single()

                if record:
                    return dict(record)
                else:
                    return {}

        except Exception as e:
            logger.error(f"Failed to get company details: {e}")
            return {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            with self.driver.session(database=self.database) as session:
                stats_query = """
                MATCH (c:Company)
                WITH COUNT(c) as company_count
                MATCH (f:Filing)
                WITH company_count, COUNT(f) as filing_count
                MATCH (p:Person)
                WITH company_count, filing_count, COUNT(p) as person_count
                MATCH ()-[r:FILED]->()
                WITH company_count, filing_count, person_count, COUNT(r) as filed_relationships
                MATCH ()-[m:MENTIONS]->()
                RETURN company_count, filing_count, person_count, filed_relationships, COUNT(m) as mention_relationships
                """

                result = session.run(stats_query)
                record = result.single()

                if record:
                    return {
                        "companies": record["company_count"],
                        "filings": record["filing_count"],
                        "persons": record["person_count"],
                        "filed_relationships": record["filed_relationships"],
                        "mention_relationships": record["mention_relationships"]
                    }
                else:
                    return {
                        "companies": 0,
                        "filings": 0,
                        "persons": 0,
                        "filed_relationships": 0,
                        "mention_relationships": 0
                    }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()