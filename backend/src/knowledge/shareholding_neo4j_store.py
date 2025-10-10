"""
Enhanced Neo4j knowledge graph storage optimized for shareholding data.
Provides specialized schema and operations for complex shareholding relationships,
ownership tracking, and temporal analysis.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError

from ..config.settings import Settings, get_settings
from ..models.filing import SECFiling
from .shareholding_classifier import ShareholdingClassifier, ShareholdingEntity, EntityType

logger = logging.getLogger(__name__)


class ShareholdingNeo4jStore:
    """Enhanced Neo4j store optimized for shareholding data and relationships."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize shareholding-optimized Neo4j store."""
        self.settings = settings or get_settings()
        self.uri = self.settings.NEO4J_URI
        self.user = self.settings.NEO4J_USER
        self.password = self.settings.NEO4J_PASSWORD
        self.database = self.settings.NEO4J_DATABASE

        # Initialize driver
        self.driver = None
        self._connect()

        # Initialize shareholding classifier
        self.classifier = ShareholdingClassifier()

        # Create enhanced schema
        self._initialize_shareholding_schema()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j for shareholding operations")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Neo4j connection error: {e}")
            raise

    def _initialize_shareholding_schema(self):
        """Create enhanced schema optimized for shareholding relationships."""
        try:
            with self.driver.session(database=self.database) as session:
                # Enhanced constraints for shareholding entities
                constraints = [
                    # Core entity constraints
                    "CREATE CONSTRAINT company_cik IF NOT EXISTS FOR (c:Company) REQUIRE c.cik IS UNIQUE",
                    "CREATE CONSTRAINT filing_accession IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession_number IS UNIQUE",

                    # Enhanced person constraints
                    "CREATE CONSTRAINT shareholder_person_unique IF NOT EXISTS FOR (p:ShareholderPerson) REQUIRE (p.name, p.context_company_cik) IS UNIQUE",
                    "CREATE CONSTRAINT shareholder_entity_unique IF NOT EXISTS FOR (e:ShareholderEntity) REQUIRE (e.name, e.context_company_cik) IS UNIQUE",

                    # Shareholding-specific constraints
                    "CREATE CONSTRAINT share_position_unique IF NOT EXISTS FOR (sp:SharePosition) REQUIRE (sp.holder_name, sp.company_cik, sp.as_of_date, sp.filing_accession) IS UNIQUE",
                    "CREATE CONSTRAINT issuance_event_unique IF NOT EXISTS FOR (ie:IssuanceEvent) REQUIRE (ie.event_id) IS UNIQUE",
                    "CREATE CONSTRAINT share_transaction_unique IF NOT EXISTS FOR (st:ShareTransaction) REQUIRE (st.transaction_id) IS UNIQUE",
                ]

                for constraint in constraints:
                    try:
                        session.run(constraint)
                        constraint_name = constraint.split('CONSTRAINT')[1].split('IF')[0].strip()
                        logger.info(f"Created constraint: {constraint_name}")
                    except ClientError as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Constraint creation warning: {e}")

                # Enhanced indexes for shareholding queries
                indexes = [
                    # Company indexes
                    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
                    "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
                    "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",

                    # Filing indexes
                    "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON (f.filing_date)",
                    "CREATE INDEX filing_type IF NOT EXISTS FOR (f:Filing) ON (f.form_type)",
                    "CREATE INDEX filing_insider IF NOT EXISTS FOR (f:Filing) ON (f.is_insider_filing)",

                    # Shareholder indexes
                    "CREATE INDEX shareholder_person_name IF NOT EXISTS FOR (p:ShareholderPerson) ON (p.name)",
                    "CREATE INDEX shareholder_entity_name IF NOT EXISTS FOR (e:ShareholderEntity) ON (e.name)",
                    "CREATE INDEX shareholder_entity_type IF NOT EXISTS FOR (e:ShareholderEntity) ON (e.entity_type)",

                    # Share position indexes (critical for shareholding queries)
                    "CREATE INDEX share_position_percentage IF NOT EXISTS FOR (sp:SharePosition) ON (sp.percentage)",
                    "CREATE INDEX share_position_shares IF NOT EXISTS FOR (sp:SharePosition) ON (sp.shares)",
                    "CREATE INDEX share_position_date IF NOT EXISTS FOR (sp:SharePosition) ON (sp.as_of_date)",
                    "CREATE INDEX share_position_class IF NOT EXISTS FOR (sp:SharePosition) ON (sp.share_class)",

                    # Issuance and transaction indexes
                    "CREATE INDEX issuance_date IF NOT EXISTS FOR (ie:IssuanceEvent) ON (ie.event_date)",
                    "CREATE INDEX issuance_type IF NOT EXISTS FOR (ie:IssuanceEvent) ON (ie.event_type)",
                    "CREATE INDEX transaction_date IF NOT EXISTS FOR (st:ShareTransaction) ON (st.transaction_date)",
                    "CREATE INDEX transaction_type IF NOT EXISTS FOR (st:ShareTransaction) ON (st.transaction_type)",
                ]

                for index in indexes:
                    try:
                        session.run(index)
                        index_name = index.split('INDEX')[1].split('IF')[0].strip()
                        logger.info(f"Created index: {index_name}")
                    except ClientError as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Index creation warning: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize shareholding schema: {e}")

    def store_shareholding_filing(self, filing: SECFiling, company_data: Dict[str, Any], extracted_data: Dict[str, Any]) -> bool:
        """
        Store filing with enhanced shareholding relationship extraction.

        Args:
            filing: SECFiling object
            company_data: Company information
            extracted_data: Extracted document data from Docling

        Returns:
            Success boolean
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Store basic company and filing (similar to original)
                self._store_company_and_filing(session, filing, company_data)

                # Extract and store shareholding relationships
                shareholding_entities = self._extract_shareholding_from_filing(filing, extracted_data)

                if shareholding_entities:
                    self._store_shareholding_entities(session, filing, shareholding_entities)
                    logger.info(f"Stored {len(shareholding_entities)} shareholding entities for {filing.accession_number}")

                # Handle Form 3/4/5 specific data
                if getattr(filing, 'is_insider_filing', False):
                    self._store_insider_filing_data(session, filing)

                return True

        except Exception as e:
            logger.error(f"Failed to store shareholding filing: {e}")
            return False

    def _store_company_and_filing(self, session, filing: SECFiling, company_data: Dict[str, Any]):
        """Store basic company and filing nodes."""
        # Enhanced company storage with shareholding context
        company_query = """
        MERGE (c:Company {cik: $cik})
        SET c.name = $name,
            c.ticker = $ticker,
            c.exchange = $exchange,
            c.sector = $sector,
            c.industry = $industry,
            c.market_cap = $market_cap,
            c.total_shares_outstanding = $total_shares,
            c.updated_at = datetime()
        RETURN c
        """

        session.run(
            company_query,
            cik=filing.cik_number,
            name=filing.company_name,
            ticker=getattr(filing, 'ticker_symbol', None) or company_data.get("ticker"),
            exchange=company_data.get("exchange"),
            sector=company_data.get("sector"),
            industry=company_data.get("industry"),
            market_cap=company_data.get("market_cap"),
            total_shares=company_data.get("total_shares_outstanding")
        )

        # Enhanced filing storage with issuer information
        filing_query = """
        MERGE (f:Filing {accession_number: $accession_number})
        SET f.form_type = $form_type,
            f.filing_date = $filing_date,
            f.period_date = $period_date,
            f.is_insider_filing = $is_insider_filing,
            f.issuer_cik = $issuer_cik,
            f.issuer_name = $issuer_name,
            f.reporting_owner_name = $reporting_owner_name,
            f.processed_at = datetime()
        RETURN f
        """

        session.run(
            filing_query,
            accession_number=filing.accession_number,
            form_type=filing.form_type,
            filing_date=filing.filing_date.isoformat() if filing.filing_date else None,
            period_date=getattr(filing, 'period_date', None).isoformat() if getattr(filing, 'period_date', None) else None,
            is_insider_filing=getattr(filing, 'is_insider_filing', False),
            issuer_cik=getattr(filing, 'issuer_cik', None),
            issuer_name=getattr(filing, 'issuer_name', None),
            reporting_owner_name=getattr(filing, 'reporting_owner_name', None)
        )

        # Create company-filing relationship
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

    def _extract_shareholding_from_filing(self, filing: SECFiling, extracted_data: Dict[str, Any]) -> List[ShareholdingEntity]:
        """Extract shareholding relationships from filing content."""
        shareholding_entities = []

        # Extract from main text
        main_text = extracted_data.get("text", "")
        if main_text:
            text_entities = self.classifier.extract_shareholding_relationships(
                main_text,
                {"filing": filing, "cik": filing.cik_number}
            )
            shareholding_entities.extend(text_entities)

        # Extract from processed shareholders (if available)
        if hasattr(filing, 'shareholders') and filing.shareholders:
            for shareholder in filing.shareholders:
                entity = self._process_legacy_shareholder(shareholder, filing)
                if entity:
                    shareholding_entities.append(entity)

        # Extract from tables
        tables = extracted_data.get("tables", [])
        for table in tables:
            title = table.get("title", "").lower()
            if any(keyword in title for keyword in ["ownership", "shareholder", "beneficial", "insider"]):
                table_entities = self._extract_shareholding_from_table(table, filing)
                shareholding_entities.extend(table_entities)

        return shareholding_entities

    def _process_legacy_shareholder(self, shareholder: Dict[str, Any], filing: SECFiling) -> Optional[ShareholdingEntity]:
        """Process shareholder data from legacy extraction."""
        name = shareholder.get("name", "").strip()
        if not name:
            return None

        # Classify the shareholder name
        classification = self.classifier.classify_entity(
            name,
            {"filing_context": filing.form_type, "cik": filing.cik_number}
        )

        if classification.entity_type == EntityType.INVALID:
            return None

        # Convert percentage string to float
        percentage_str = shareholder.get("percentage", "")
        percentage = None
        if percentage_str:
            try:
                percentage = float(str(percentage_str).replace("%", "").replace(",", ""))
            except (ValueError, TypeError):
                pass

        # Convert shares string to int
        shares_str = shareholder.get("shares", "")
        shares = None
        if shares_str:
            try:
                shares = int(str(shares_str).replace(",", ""))
            except (ValueError, TypeError):
                pass

        return ShareholdingEntity(
            name=name,
            entity_type=classification.entity_type,
            shares=shares,
            percentage=percentage,
            context=shareholder.get("context", ""),
            confidence=classification.confidence
        )

    def _extract_shareholding_from_table(self, table: Dict[str, Any], filing: SECFiling) -> List[ShareholdingEntity]:
        """Extract shareholding entities from table data."""
        entities = []
        rows = table.get("rows", [])

        for row in rows:
            if isinstance(row, dict):
                # Look for shareholding patterns in table rows
                name = None
                shares = None
                percentage = None

                # Common column name patterns
                for key, value in row.items():
                    key_lower = str(key).lower()

                    if any(name_key in key_lower for name_key in ["name", "shareholder", "owner", "holder"]):
                        name = str(value).strip()
                    elif any(share_key in key_lower for share_key in ["shares", "owned", "holding"]):
                        try:
                            shares = int(str(value).replace(",", ""))
                        except (ValueError, TypeError):
                            pass
                    elif any(pct_key in key_lower for pct_key in ["percentage", "%", "percent"]):
                        try:
                            percentage = float(str(value).replace("%", "").replace(",", ""))
                        except (ValueError, TypeError):
                            pass

                if name:
                    classification = self.classifier.classify_entity(name)
                    if classification.entity_type != EntityType.INVALID:
                        entities.append(ShareholdingEntity(
                            name=name,
                            entity_type=classification.entity_type,
                            shares=shares,
                            percentage=percentage,
                            context=f"Table: {table.get('title', 'Unknown')}",
                            confidence=classification.confidence * 0.8  # Table data slightly less reliable
                        ))

        return entities

    def _store_shareholding_entities(self, session, filing: SECFiling, entities: List[ShareholdingEntity]):
        """Store shareholding entities and their relationships."""
        for entity in entities:
            if entity.confidence < 0.5:  # Skip low-confidence entities
                continue

            # Determine node label based on entity type
            if entity.entity_type == EntityType.SHAREHOLDER_PERSON:
                self._store_shareholder_person(session, entity, filing)
            elif entity.entity_type == EntityType.SHAREHOLDER_ENTITY:
                self._store_shareholder_entity(session, entity, filing)

    def _store_shareholder_person(self, session, entity: ShareholdingEntity, filing: SECFiling):
        """Store shareholder person with position."""
        # Store person node
        person_query = """
        MERGE (p:ShareholderPerson {name: $name, context_company_cik: $context_cik})
        SET p.updated_at = datetime(),
            p.classification_confidence = $confidence
        RETURN p
        """

        session.run(
            person_query,
            name=entity.name,
            context_cik=filing.cik_number,
            confidence=entity.confidence
        )

        # Store share position if we have shareholding data
        if entity.shares is not None or entity.percentage is not None:
            self._store_share_position(session, entity, filing)

    def _store_shareholder_entity(self, session, entity: ShareholdingEntity, filing: SECFiling):
        """Store shareholder entity with position."""
        # Determine entity type (institutional, corporate, etc.)
        entity_type = "institutional"  # Default
        if any(corp_indicator in entity.name.lower() for corp_indicator in ["corp", "inc", "company", "llc"]):
            entity_type = "corporate"
        elif any(fund_indicator in entity.name.lower() for fund_indicator in ["fund", "trust", "pension", "endowment"]):
            entity_type = "fund"

        entity_query = """
        MERGE (e:ShareholderEntity {name: $name, context_company_cik: $context_cik})
        SET e.entity_type = $entity_type,
            e.updated_at = datetime(),
            e.classification_confidence = $confidence
        RETURN e
        """

        session.run(
            entity_query,
            name=entity.name,
            context_cik=filing.cik_number,
            entity_type=entity_type,
            confidence=entity.confidence
        )

        # Store share position
        if entity.shares is not None or entity.percentage is not None:
            self._store_share_position(session, entity, filing)

    def _store_share_position(self, session, entity: ShareholdingEntity, filing: SECFiling):
        """Store detailed share position information."""
        position_id = f"{entity.name}_{filing.cik_number}_{filing.filing_date.strftime('%Y%m%d')}_{filing.accession_number}"

        position_query = """
        MERGE (sp:SharePosition {
            holder_name: $holder_name,
            company_cik: $company_cik,
            as_of_date: $as_of_date,
            filing_accession: $filing_accession
        })
        SET sp.shares = $shares,
            sp.percentage = $percentage,
            sp.share_class = $share_class,
            sp.position_value = $position_value,
            sp.context = $context,
            sp.confidence = $confidence,
            sp.created_at = datetime()
        RETURN sp
        """

        session.run(
            position_query,
            holder_name=entity.name,
            company_cik=filing.cik_number,
            as_of_date=filing.filing_date.isoformat(),
            filing_accession=filing.accession_number,
            shares=entity.shares,
            percentage=entity.percentage,
            share_class=getattr(entity, 'share_class', 'common'),
            position_value=None,  # Calculate if we have share price
            context=entity.context,
            confidence=entity.confidence
        )

        # Create relationships
        self._create_shareholding_relationships(session, entity, filing)

    def _create_shareholding_relationships(self, session, entity: ShareholdingEntity, filing: SECFiling):
        """Create relationships between shareholders, positions, companies, and filings."""
        # Connect shareholder to position
        if entity.entity_type == EntityType.SHAREHOLDER_PERSON:
            shareholder_position_query = """
            MATCH (p:ShareholderPerson {name: $holder_name, context_company_cik: $company_cik})
            MATCH (sp:SharePosition {holder_name: $holder_name, company_cik: $company_cik, filing_accession: $filing_accession})
            MERGE (p)-[r:HOLDS_POSITION]->(sp)
            SET r.as_of_date = $as_of_date
            """
        else:
            shareholder_position_query = """
            MATCH (e:ShareholderEntity {name: $holder_name, context_company_cik: $company_cik})
            MATCH (sp:SharePosition {holder_name: $holder_name, company_cik: $company_cik, filing_accession: $filing_accession})
            MERGE (e)-[r:HOLDS_POSITION]->(sp)
            SET r.as_of_date = $as_of_date
            """

        session.run(
            shareholder_position_query,
            holder_name=entity.name,
            company_cik=filing.cik_number,
            filing_accession=filing.accession_number,
            as_of_date=filing.filing_date.isoformat()
        )

        # Connect position to company
        position_company_query = """
        MATCH (sp:SharePosition {holder_name: $holder_name, company_cik: $company_cik, filing_accession: $filing_accession})
        MATCH (c:Company {cik: $company_cik})
        MERGE (sp)-[r:POSITION_IN]->(c)
        """

        session.run(
            position_company_query,
            holder_name=entity.name,
            company_cik=filing.cik_number,
            filing_accession=filing.accession_number
        )

        # Connect filing to position
        filing_position_query = """
        MATCH (f:Filing {accession_number: $filing_accession})
        MATCH (sp:SharePosition {holder_name: $holder_name, company_cik: $company_cik, filing_accession: $filing_accession})
        MERGE (f)-[r:REPORTS_POSITION]->(sp)
        SET r.reported_at = datetime()
        """

        session.run(
            filing_position_query,
            filing_accession=filing.accession_number,
            holder_name=entity.name,
            company_cik=filing.cik_number
        )

    def _store_insider_filing_data(self, session, filing: SECFiling):
        """Store additional data for Form 3/4/5 insider filings."""
        if not getattr(filing, 'is_insider_filing', False):
            return

        # Create relationship between reporting owner and issuer company
        if getattr(filing, 'reporting_owner_name', None) and getattr(filing, 'issuer_cik', None):
            insider_query = """
            MERGE (p:ShareholderPerson {name: $owner_name, context_company_cik: $issuer_cik})
            SET p.is_insider = true,
                p.updated_at = datetime()
            WITH p
            MATCH (c:Company {cik: $issuer_cik})
            MERGE (p)-[r:INSIDER_OF]->(c)
            SET r.first_reported = coalesce(r.first_reported, $filing_date),
                r.last_reported = $filing_date
            WITH p
            MATCH (f:Filing {accession_number: $accession_number})
            MERGE (f)-[fr:FILED_BY_INSIDER]->(p)
            """

            session.run(
                insider_query,
                owner_name=filing.reporting_owner_name,
                issuer_cik=filing.issuer_cik,
                filing_date=filing.filing_date.isoformat(),
                accession_number=filing.accession_number
            )

    def get_company_shareholding_summary(self, cik: str, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive shareholding summary for a company.

        Args:
            cik: Company CIK
            as_of_date: Optional date filter

        Returns:
            Shareholding summary with positions, changes, and statistics
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Get current major shareholders
                shareholders_query = """
                MATCH (c:Company {cik: $cik})<-[:POSITION_IN]-(sp:SharePosition)<-[:HOLDS_POSITION]-(holder)
                WHERE ($as_of_date IS NULL OR sp.as_of_date <= $as_of_date)
                WITH holder, sp, c
                ORDER BY sp.as_of_date DESC
                WITH holder, COLLECT(sp)[0] as latest_position, c
                WHERE latest_position.percentage >= 1.0
                RETURN
                    labels(holder)[0] as holder_type,
                    holder.name as holder_name,
                    latest_position.shares as shares,
                    latest_position.percentage as percentage,
                    latest_position.as_of_date as as_of_date,
                    latest_position.filing_accession as filing_accession
                ORDER BY latest_position.percentage DESC
                LIMIT 20
                """

                result = session.run(shareholders_query, cik=cik, as_of_date=as_of_date)
                shareholders = [dict(record) for record in result]

                # Get insider information
                insiders_query = """
                MATCH (c:Company {cik: $cik})<-[:INSIDER_OF]-(insider:ShareholderPerson)
                OPTIONAL MATCH (insider)-[:HOLDS_POSITION]->(sp:SharePosition)-[:POSITION_IN]->(c)
                WHERE ($as_of_date IS NULL OR sp.as_of_date <= $as_of_date)
                WITH insider, sp
                ORDER BY sp.as_of_date DESC
                WITH insider, COLLECT(sp)[0] as latest_position
                RETURN
                    insider.name as insider_name,
                    COALESCE(latest_position.shares, 0) as shares,
                    COALESCE(latest_position.percentage, 0.0) as percentage,
                    latest_position.as_of_date as as_of_date
                ORDER BY latest_position.percentage DESC
                """

                result = session.run(insiders_query, cik=cik, as_of_date=as_of_date)
                insiders = [dict(record) for record in result]

                # Get summary statistics
                stats_query = """
                MATCH (c:Company {cik: $cik})<-[:POSITION_IN]-(sp:SharePosition)
                WHERE ($as_of_date IS NULL OR sp.as_of_date <= $as_of_date)
                WITH sp
                ORDER BY sp.as_of_date DESC
                WITH COLLECT(DISTINCT sp.holder_name) as unique_holders,
                     SUM(sp.percentage) as total_reported_percentage,
                     COUNT(sp) as total_positions,
                     MAX(sp.as_of_date) as latest_report_date
                RETURN
                    SIZE(unique_holders) as unique_shareholders,
                    total_reported_percentage,
                    total_positions,
                    latest_report_date
                """

                result = session.run(stats_query, cik=cik, as_of_date=as_of_date)
                stats = dict(result.single()) if result.peek() else {}

                return {
                    "company_cik": cik,
                    "as_of_date": as_of_date,
                    "major_shareholders": shareholders,
                    "insiders": insiders,
                    "statistics": stats,
                    "query_timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get shareholding summary: {e}")
            return {}

    def search_shareholders(self, query: str, min_percentage: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for shareholders across all companies.

        Args:
            query: Shareholder name query
            min_percentage: Minimum ownership percentage

        Returns:
            List of shareholding positions
        """
        try:
            with self.driver.session(database=self.database) as session:
                search_query = """
                MATCH (holder)-[:HOLDS_POSITION]->(sp:SharePosition)-[:POSITION_IN]->(c:Company)
                WHERE toLower(holder.name) CONTAINS toLower($query)
                  AND sp.percentage >= $min_percentage
                WITH holder, sp, c
                ORDER BY sp.as_of_date DESC
                WITH holder, c, COLLECT(sp)[0] as latest_position
                RETURN
                    labels(holder)[0] as holder_type,
                    holder.name as holder_name,
                    c.name as company_name,
                    c.cik as company_cik,
                    c.ticker as ticker,
                    latest_position.shares as shares,
                    latest_position.percentage as percentage,
                    latest_position.as_of_date as as_of_date,
                    latest_position.filing_accession as filing_accession
                ORDER BY latest_position.percentage DESC
                LIMIT 50
                """

                result = session.run(search_query, query=query, min_percentage=min_percentage)
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to search shareholders: {e}")
            return []

    def get_shareholding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive shareholding statistics."""
        try:
            with self.driver.session(database=self.database) as session:
                stats_query = """
                MATCH (c:Company)
                WITH COUNT(c) as company_count
                MATCH (sp:ShareholderPerson)
                WITH company_count, COUNT(sp) as person_count
                MATCH (se:ShareholderEntity)
                WITH company_count, person_count, COUNT(se) as entity_count
                MATCH (pos:SharePosition)
                WITH company_count, person_count, entity_count, COUNT(pos) as position_count
                MATCH (f:Filing {is_insider_filing: true})
                RETURN
                    company_count,
                    person_count,
                    entity_count,
                    position_count,
                    COUNT(f) as insider_filing_count
                """

                result = session.run(stats_query)
                record = result.single()

                if record:
                    return dict(record)
                else:
                    return {
                        "company_count": 0,
                        "person_count": 0,
                        "entity_count": 0,
                        "position_count": 0,
                        "insider_filing_count": 0
                    }

        except Exception as e:
            logger.error(f"Failed to get shareholding statistics: {e}")
            return {}

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("ShareholdingNeo4jStore connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()