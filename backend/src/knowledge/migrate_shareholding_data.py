"""
Data migration script for cleaning up incorrect Person node classifications
and migrating to the new shareholding-optimized schema.
"""
import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .shareholding_classifier import ShareholdingClassifier, EntityType
from .shareholding_neo4j_store import ShareholdingNeo4jStore
from .neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)


class ShareholdingDataMigrator:
    """Migrate existing knowledge graph data to shareholding-optimized schema."""

    def __init__(self, settings=None):
        """Initialize data migrator."""
        self.classifier = ShareholdingClassifier()
        self.old_store = Neo4jStore(settings)
        self.new_store = ShareholdingNeo4jStore(settings)

        self.migration_stats = {
            "total_persons_analyzed": 0,
            "invalid_persons_removed": 0,
            "valid_persons_migrated": 0,
            "entities_reclassified": 0,
            "shareholding_positions_created": 0,
            "errors": 0
        }

    async def run_migration(self, dry_run: bool = True, batch_size: int = 100) -> Dict[str, Any]:
        """
        Run the complete data migration.

        Args:
            dry_run: If True, analyze but don't make changes
            batch_size: Number of nodes to process in each batch

        Returns:
            Migration results and statistics
        """
        logger.info(f"Starting shareholding data migration (dry_run={dry_run})")

        try:
            # Step 1: Analyze existing Person nodes
            person_analysis = await self._analyze_existing_persons()
            logger.info(f"Found {person_analysis['total_persons']} Person nodes")

            # Step 2: Clean up invalid Person nodes
            if not dry_run:
                cleanup_results = await self._cleanup_invalid_persons(person_analysis['invalid_persons'])
                self.migration_stats.update(cleanup_results)

            # Step 3: Migrate valid Person nodes to new schema
            if not dry_run:
                migration_results = await self._migrate_valid_persons(person_analysis['valid_persons'], batch_size)
                self.migration_stats.update(migration_results)

            # Step 4: Create shareholding position nodes
            if not dry_run:
                position_results = await self._create_shareholding_positions()
                self.migration_stats.update(position_results)

            # Step 5: Validation
            validation_results = await self._validate_migration()

            migration_report = {
                "migration_completed": not dry_run,
                "migration_stats": self.migration_stats,
                "person_analysis": person_analysis,
                "validation_results": validation_results,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info("Migration completed successfully")
            return migration_report

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            self.migration_stats["errors"] += 1
            return {"error": str(e), "migration_stats": self.migration_stats}

    async def _analyze_existing_persons(self) -> Dict[str, Any]:
        """Analyze existing Person nodes to classify them properly."""
        logger.info("Analyzing existing Person nodes...")

        try:
            with self.old_store.driver.session(database=self.old_store.database) as session:
                # Get all Person nodes with their data
                query = """
                MATCH (p:Person)
                RETURN p.name as name,
                       p.company_cik as company_cik,
                       p.shares as shares,
                       p.percentage as percentage,
                       elementId(p) as node_id,
                       p.title as title,
                       p.is_director as is_director,
                       p.is_officer as is_officer
                LIMIT 1000
                """

                result = session.run(query)
                persons = [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Error querying existing persons: {e}")
            return {"total_persons": 0, "valid_persons": [], "invalid_persons": [], "error": str(e)}

        # Classify each person
        valid_persons = []
        invalid_persons = []

        for person in persons:
            name = person.get("name", "").strip()
            if not name:
                invalid_persons.append({**person, "reason": "empty_name"})
                continue

            # Classify using our new classifier
            classification = self.classifier.classify_entity(
                name,
                {
                    "company_cik": person.get("company_cik"),
                    "title": person.get("title"),
                    "is_director": person.get("is_director"),
                    "is_officer": person.get("is_officer")
                }
            )

            if classification.entity_type == EntityType.INVALID:
                invalid_persons.append({
                    **person,
                    "reason": "invalid_classification",
                    "confidence": classification.confidence,
                    "validation_flags": classification.validation_flags
                })
            elif classification.entity_type in [EntityType.SHAREHOLDER_PERSON, EntityType.SHAREHOLDER_ENTITY]:
                valid_persons.append({
                    **person,
                    "new_entity_type": classification.entity_type.value,
                    "confidence": classification.confidence,
                    "validation_flags": classification.validation_flags
                })
            else:
                # Other entity types we're not handling yet
                invalid_persons.append({
                    **person,
                    "reason": "unsupported_entity_type",
                    "entity_type": classification.entity_type.value
                })

        self.migration_stats["total_persons_analyzed"] = len(persons)

        logger.info(f"Analysis complete: {len(valid_persons)} valid, {len(invalid_persons)} invalid")

        return {
            "total_persons": len(persons),
            "valid_persons": valid_persons,
            "invalid_persons": invalid_persons,
            "validation_summary": {
                "valid_count": len(valid_persons),
                "invalid_count": len(invalid_persons),
                "invalid_reasons": self._summarize_invalid_reasons(invalid_persons)
            }
        }

    def _summarize_invalid_reasons(self, invalid_persons: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize reasons for invalid person classifications."""
        reasons = {}
        for person in invalid_persons:
            reason = person.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    async def _cleanup_invalid_persons(self, invalid_persons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Remove invalid Person nodes from the knowledge graph."""
        logger.info(f"Cleaning up {len(invalid_persons)} invalid Person nodes...")

        removed_count = 0
        error_count = 0

        try:
            with self.old_store.driver.session(database=self.old_store.database) as session:
                for person in invalid_persons:
                    try:
                        # Delete the invalid person node and its relationships
                        delete_query = """
                        MATCH (p:Person {name: $name, company_cik: $company_cik})
                        DETACH DELETE p
                        """

                        session.run(
                            delete_query,
                            name=person["name"],
                            company_cik=person["company_cik"]
                        )

                        removed_count += 1
                        logger.debug(f"Removed invalid person: {person['name']} (reason: {person.get('reason', 'unknown')})")

                    except Exception as e:
                        logger.error(f"Error removing person {person['name']}: {e}")
                        error_count += 1

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            error_count += len(invalid_persons)

        logger.info(f"Cleanup complete: {removed_count} removed, {error_count} errors")

        return {
            "invalid_persons_removed": removed_count,
            "cleanup_errors": error_count
        }

    async def _migrate_valid_persons(self, valid_persons: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """Migrate valid Person nodes to the new shareholding schema."""
        logger.info(f"Migrating {len(valid_persons)} valid Person nodes...")

        migrated_count = 0
        reclassified_count = 0
        error_count = 0

        try:
            # Process in batches
            for i in range(0, len(valid_persons), batch_size):
                batch = valid_persons[i:i + batch_size]

                with self.new_store.driver.session(database=self.new_store.database) as session:
                    for person in batch:
                        try:
                            new_entity_type = person["new_entity_type"]

                            if new_entity_type == "shareholder_person":
                                # Create ShareholderPerson node
                                create_query = """
                                MERGE (sp:ShareholderPerson {name: $name, context_company_cik: $company_cik})
                                SET sp.title = $title,
                                    sp.is_director = $is_director,
                                    sp.is_officer = $is_officer,
                                    sp.classification_confidence = $confidence,
                                    sp.migrated_at = datetime(),
                                    sp.updated_at = datetime()
                                RETURN sp
                                """

                                session.run(
                                    create_query,
                                    name=person["name"],
                                    company_cik=person["company_cik"],
                                    title=person.get("title"),
                                    is_director=person.get("is_director", False),
                                    is_officer=person.get("is_officer", False),
                                    confidence=person["confidence"]
                                )

                            elif new_entity_type == "shareholder_entity":
                                # Create ShareholderEntity node
                                create_query = """
                                MERGE (se:ShareholderEntity {name: $name, context_company_cik: $company_cik})
                                SET se.entity_type = "institutional",
                                    se.classification_confidence = $confidence,
                                    se.migrated_at = datetime(),
                                    se.updated_at = datetime()
                                RETURN se
                                """

                                session.run(
                                    create_query,
                                    name=person["name"],
                                    company_cik=person["company_cik"],
                                    confidence=person["confidence"]
                                )

                                reclassified_count += 1

                            migrated_count += 1

                        except Exception as e:
                            logger.error(f"Error migrating person {person['name']}: {e}")
                            error_count += 1

                logger.info(f"Migrated batch {i//batch_size + 1}: {len(batch)} persons")

        except Exception as e:
            logger.error(f"Error during migration: {e}")
            error_count += len(valid_persons) - migrated_count

        logger.info(f"Migration complete: {migrated_count} migrated, {reclassified_count} reclassified, {error_count} errors")

        return {
            "valid_persons_migrated": migrated_count,
            "entities_reclassified": reclassified_count,
            "migration_errors": error_count
        }

    async def _create_shareholding_positions(self) -> Dict[str, Any]:
        """Create SharePosition nodes for migrated shareholders with ownership data."""
        logger.info("Creating shareholding position nodes...")

        positions_created = 0
        error_count = 0

        try:
            with self.new_store.driver.session(database=self.new_store.database) as session:
                # Find migrated shareholders with shareholding data
                query = """
                MATCH (holder)
                WHERE (holder:ShareholderPerson OR holder:ShareholderEntity)
                  AND holder.migrated_at IS NOT NULL
                WITH holder
                MATCH (f:Filing)-[:REPORTS_SHAREHOLDER]->(oldPerson:Person {name: holder.name})
                WHERE oldPerson.shares IS NOT NULL OR oldPerson.percentage IS NOT NULL
                RETURN DISTINCT
                    holder.name as holder_name,
                    holder.context_company_cik as company_cik,
                    oldPerson.shares as shares,
                    oldPerson.percentage as percentage,
                    f.accession_number as filing_accession,
                    f.filing_date as filing_date,
                    labels(holder)[0] as holder_type
                """

                result = session.run(query)
                shareholding_data = [dict(record) for record in result]

                # Create SharePosition nodes
                for data in shareholding_data:
                    try:
                        position_query = """
                        MERGE (sp:SharePosition {
                            holder_name: $holder_name,
                            company_cik: $company_cik,
                            filing_accession: $filing_accession
                        })
                        SET sp.shares = $shares,
                            sp.percentage = $percentage,
                            sp.as_of_date = $filing_date,
                            sp.created_at = datetime(),
                            sp.migrated_from_legacy = true
                        WITH sp
                        MATCH (holder {name: $holder_name, context_company_cik: $company_cik})
                        WHERE (holder:ShareholderPerson OR holder:ShareholderEntity)
                        MERGE (holder)-[r:HOLDS_POSITION]->(sp)
                        WITH sp
                        MATCH (c:Company {cik: $company_cik})
                        MERGE (sp)-[r2:POSITION_IN]->(c)
                        WITH sp
                        MATCH (f:Filing {accession_number: $filing_accession})
                        MERGE (f)-[r3:REPORTS_POSITION]->(sp)
                        RETURN sp
                        """

                        session.run(
                            position_query,
                            holder_name=data["holder_name"],
                            company_cik=data["company_cik"],
                            filing_accession=data["filing_accession"],
                            shares=data["shares"],
                            percentage=data["percentage"],
                            filing_date=data["filing_date"]
                        )

                        positions_created += 1

                    except Exception as e:
                        logger.error(f"Error creating position for {data['holder_name']}: {e}")
                        error_count += 1

        except Exception as e:
            logger.error(f"Error creating shareholding positions: {e}")
            error_count += 1

        logger.info(f"Position creation complete: {positions_created} created, {error_count} errors")

        return {
            "shareholding_positions_created": positions_created,
            "position_creation_errors": error_count
        }

    async def _validate_migration(self) -> Dict[str, Any]:
        """Validate the migration results."""
        logger.info("Validating migration results...")

        validation_results = {}

        try:
            with self.new_store.driver.session(database=self.new_store.database) as session:
                # Count new node types
                count_query = """
                MATCH (sp:ShareholderPerson) WITH COUNT(sp) as person_count
                MATCH (se:ShareholderEntity) WITH person_count, COUNT(se) as entity_count
                MATCH (pos:SharePosition) WITH person_count, entity_count, COUNT(pos) as position_count
                RETURN person_count, entity_count, position_count
                """

                result = session.run(count_query)
                counts = dict(result.single()) if result.peek() else {}

                # Check for remaining invalid Person nodes
                invalid_check_query = """
                MATCH (p:Person)
                WHERE p.name =~ '.*[0-9]+.*' OR p.name =~ '^[0-9\\.]+$' OR LENGTH(p.name) > 100
                RETURN COUNT(p) as remaining_invalid_persons
                """

                result = session.run(invalid_check_query)
                invalid_count = result.single()["remaining_invalid_persons"]

                validation_results = {
                    "new_shareholder_persons": counts.get("person_count", 0),
                    "new_shareholder_entities": counts.get("entity_count", 0),
                    "new_share_positions": counts.get("position_count", 0),
                    "remaining_invalid_persons": invalid_count,
                    "migration_successful": invalid_count == 0
                }

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results = {"error": str(e)}

        return validation_results

    def get_migration_report(self) -> Dict[str, Any]:
        """Get detailed migration report."""
        return {
            "migration_statistics": self.migration_stats,
            "classifier_info": self.classifier.get_statistics(),
            "timestamp": datetime.utcnow().isoformat()
        }

    def close(self):
        """Close connections."""
        if self.old_store:
            self.old_store.close()
        if self.new_store:
            self.new_store.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


async def main():
    """Main migration script."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate shareholding data to new schema")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't make changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run migration
    migrator = ShareholdingDataMigrator()

    try:
        results = await migrator.run_migration(
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )

        print("\n" + "="*80)
        print("SHAREHOLDING DATA MIGRATION RESULTS")
        print("="*80)

        if args.dry_run:
            print("🔍 DRY RUN - No changes made")
        else:
            print("✅ MIGRATION COMPLETED")

        print(f"\nStatistics:")
        for key, value in results.get("migration_stats", {}).items():
            print(f"  {key}: {value}")

        if "person_analysis" in results:
            analysis = results["person_analysis"]
            print(f"\nPerson Analysis:")
            print(f"  Total persons analyzed: {analysis['total_persons']}")
            print(f"  Valid persons: {len(analysis['valid_persons'])}")
            print(f"  Invalid persons: {len(analysis['invalid_persons'])}")

            if analysis["invalid_persons"]:
                print(f"\nInvalid person reasons:")
                for reason, count in analysis["validation_summary"]["invalid_reasons"].items():
                    print(f"    {reason}: {count}")

        if "validation_results" in results:
            validation = results["validation_results"]
            print(f"\nValidation Results:")
            for key, value in validation.items():
                print(f"  {key}: {value}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        migrator.close()


if __name__ == "__main__":
    asyncio.run(main())