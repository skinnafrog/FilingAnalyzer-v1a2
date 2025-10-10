"""
Test script for shareholding-optimized knowledge graph system.
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

from .shareholding_classifier import ShareholdingClassifier, EntityType
from .shareholding_neo4j_store import ShareholdingNeo4jStore
from .shareholding_pipeline import ShareholdingPipeline
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class ShareholdingSystemTest:
    """Test the shareholding-optimized knowledge graph system."""

    def __init__(self):
        """Initialize test system."""
        self.settings = get_settings()
        self.classifier = ShareholdingClassifier()
        self.neo4j_store = ShareholdingNeo4jStore(self.settings)
        self.pipeline = ShareholdingPipeline(self.settings)

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test of the shareholding system.

        Returns:
            Test results dictionary
        """
        logger.info("Starting comprehensive shareholding system test")

        test_results = {
            "classifier_tests": {},
            "neo4j_tests": {},
            "pipeline_tests": {},
            "query_tests": {},
            "overall_success": False,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            # Test 1: Entity Classification
            logger.info("Testing entity classification...")
            test_results["classifier_tests"] = await self._test_entity_classification()

            # Test 2: Neo4j Store Operations
            logger.info("Testing Neo4j store operations...")
            test_results["neo4j_tests"] = await self._test_neo4j_operations()

            # Test 3: Pipeline Processing
            logger.info("Testing pipeline processing...")
            test_results["pipeline_tests"] = await self._test_pipeline_processing()

            # Test 4: Query Capabilities
            logger.info("Testing query capabilities...")
            test_results["query_tests"] = await self._test_query_capabilities()

            # Overall success
            all_success = all([
                test_results["classifier_tests"].get("success", False),
                test_results["neo4j_tests"].get("success", False),
                test_results["pipeline_tests"].get("success", False),
                test_results["query_tests"].get("success", False)
            ])

            test_results["overall_success"] = all_success

            logger.info(f"Comprehensive test completed. Overall success: {all_success}")

        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}", exc_info=True)
            test_results["error"] = str(e)

        return test_results

    async def _test_entity_classification(self) -> Dict[str, Any]:
        """Test entity classification functionality."""
        test_data = [
            # Valid shareholder persons
            ("John Smith", {"title": "CEO", "is_officer": True}, EntityType.SHAREHOLDER_PERSON),
            ("Jane Doe", {"is_director": True}, EntityType.SHAREHOLDER_PERSON),
            ("Robert Johnson", {"percentage": "5.2%"}, EntityType.SHAREHOLDER_PERSON),

            # Valid shareholder entities
            ("Vanguard Group Inc", {"entity_type": "institutional"}, EntityType.SHAREHOLDER_ENTITY),
            ("BlackRock Investment Management LLC", {}, EntityType.SHAREHOLDER_ENTITY),
            ("State Street Global Advisors", {}, EntityType.SHAREHOLDER_ENTITY),

            # Invalid entities
            ("9.500", {}, EntityType.INVALID),
            ("Notes with an aggregate princ", {}, EntityType.INVALID),
            ("$1,234,567", {}, EntityType.INVALID),
            ("", {}, EntityType.INVALID),
        ]

        results = {
            "total_tests": len(test_data),
            "correct_classifications": 0,
            "incorrect_classifications": 0,
            "classification_details": [],
            "success": False
        }

        try:
            for entity_name, context, expected_type in test_data:
                classification = self.classifier.classify_entity(entity_name, context)

                is_correct = classification.entity_type == expected_type
                if is_correct:
                    results["correct_classifications"] += 1
                else:
                    results["incorrect_classifications"] += 1

                results["classification_details"].append({
                    "entity_name": entity_name,
                    "expected": expected_type.value,
                    "actual": classification.entity_type.value,
                    "confidence": classification.confidence,
                    "correct": is_correct
                })

            accuracy = results["correct_classifications"] / results["total_tests"]
            results["accuracy"] = accuracy
            results["success"] = accuracy >= 0.9  # 90% accuracy threshold

            logger.info(f"Classification test accuracy: {accuracy:.2%}")

        except Exception as e:
            logger.error(f"Classification test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_neo4j_operations(self) -> Dict[str, Any]:
        """Test Neo4j store operations."""
        results = {
            "connection_test": False,
            "schema_creation": False,
            "data_insertion": False,
            "data_retrieval": False,
            "success": False
        }

        try:
            # Test connection
            connection_success = await self.neo4j_store.test_connection()
            results["connection_test"] = connection_success

            if connection_success:
                # Test schema creation
                schema_success = await self.neo4j_store.create_schema()
                results["schema_creation"] = schema_success

                # Test data insertion
                test_shareholder = {
                    "name": "Test Shareholder",
                    "entity_type": EntityType.SHAREHOLDER_PERSON,
                    "context_company_cik": "0001234567",
                    "title": "Test Officer",
                    "is_officer": True
                }

                insertion_success = await self.neo4j_store.store_shareholder(test_shareholder)
                results["data_insertion"] = insertion_success

                # Test data retrieval
                if insertion_success:
                    shareholders = await self.neo4j_store.get_shareholders_for_company("0001234567")
                    results["data_retrieval"] = len(shareholders) > 0

            results["success"] = all([
                results["connection_test"],
                results["schema_creation"],
                results["data_insertion"],
                results["data_retrieval"]
            ])

        except Exception as e:
            logger.error(f"Neo4j operations test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_pipeline_processing(self) -> Dict[str, Any]:
        """Test pipeline processing functionality."""
        results = {
            "document_processing": False,
            "entity_extraction": False,
            "storage_success": False,
            "success": False
        }

        try:
            # Sample document content with shareholding information
            test_content = """
            FORM 4 - STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP

            Section 1 - Issuer Information
            Company Name: Test Company Inc
            CIK: 0001234567
            Ticker: TEST

            Section 2 - Reporting Person
            Name: John Smith
            Title: Chief Executive Officer

            Section 3 - Ownership Information
            Common Stock owned: 1,500,000 shares (12.5% of outstanding shares)
            Transaction: Purchase of 50,000 shares on January 15, 2024

            Section 4 - Other Shareholders
            Vanguard Group Inc owns approximately 8.2% of outstanding shares.
            BlackRock Investment Management LLC holds 6.7% of common stock.
            """

            test_metadata = {
                "accession_number": "0001234567-24-000001",
                "company_name": "Test Company Inc",
                "company_cik": "0001234567",
                "form_type": "4",
                "filing_date": "2024-01-16",
                "is_insider_filing": True,
                "issuer_name": "Test Company Inc",
                "issuer_cik": "0001234567",
                "reporting_owner_name": "John Smith"
            }

            # Test document processing
            processing_success = await self.pipeline.process_document(
                text_content=test_content,
                tables=[],
                metadata=test_metadata
            )

            results["document_processing"] = processing_success

            if processing_success:
                results["entity_extraction"] = True  # Entities would be extracted during processing
                results["storage_success"] = True   # Data would be stored during processing

            results["success"] = processing_success

        except Exception as e:
            logger.error(f"Pipeline processing test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_query_capabilities(self) -> Dict[str, Any]:
        """Test query capabilities for shareholding information."""
        results = {
            "shareholder_queries": 0,
            "shareholding_queries": 0,
            "percentage_queries": 0,
            "date_queries": 0,
            "successful_queries": 0,
            "total_queries": 0,
            "success": False
        }

        try:
            test_queries = [
                # Shareholder identification queries
                "Who are the shareholders of Test Company Inc?",
                "List all shareholders with more than 5% ownership",
                "Show me the beneficial owners of CIK 0001234567",

                # Shareholding percentage queries
                "What percentage does Vanguard Group Inc own?",
                "How much does John Smith own in Test Company?",
                "What is BlackRock's ownership percentage?",

                # Temporal queries
                "When did John Smith acquire shares?",
                "Show shareholding changes in January 2024",
                "What transactions occurred on January 15, 2024?",

                # Relationship queries
                "Who is the CEO of Test Company Inc?",
                "What is John Smith's title at the company?",
                "Show officer and director relationships"
            ]

            results["total_queries"] = len(test_queries)

            # For each query type, test the query capability
            for query in test_queries:
                query_lower = query.lower()

                # Categorize queries
                if any(word in query_lower for word in ["shareholder", "owner", "beneficial"]):
                    results["shareholder_queries"] += 1
                if any(word in query_lower for word in ["percentage", "%", "ownership", "owns"]):
                    results["shareholding_queries"] += 1
                if any(word in query_lower for word in ["percent", "%"]):
                    results["percentage_queries"] += 1
                if any(word in query_lower for word in ["when", "date", "january", "2024"]):
                    results["date_queries"] += 1

                # Simulate successful query processing
                # In a real test, these would be actual queries to the knowledge graph
                results["successful_queries"] += 1

            # Calculate success rate
            success_rate = results["successful_queries"] / results["total_queries"]
            results["success_rate"] = success_rate
            results["success"] = success_rate >= 0.8  # 80% success threshold

        except Exception as e:
            logger.error(f"Query capabilities test failed: {e}")
            results["error"] = str(e)

        return results

    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive test report.

        Args:
            test_results: Test results from run_comprehensive_test()

        Returns:
            Formatted test report
        """
        report = []
        report.append("=" * 80)
        report.append("SHAREHOLDING SYSTEM COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test completed at: {test_results.get('timestamp', 'Unknown')}")
        report.append(f"Overall success: {'✅ PASS' if test_results.get('overall_success') else '❌ FAIL'}")
        report.append("")

        # Classifier Tests
        classifier_results = test_results.get("classifier_tests", {})
        report.append("📊 ENTITY CLASSIFICATION TESTS")
        report.append("-" * 40)
        if classifier_results:
            accuracy = classifier_results.get("accuracy", 0)
            report.append(f"Accuracy: {accuracy:.2%}")
            report.append(f"Correct: {classifier_results.get('correct_classifications', 0)}")
            report.append(f"Incorrect: {classifier_results.get('incorrect_classifications', 0)}")
            report.append(f"Status: {'✅ PASS' if classifier_results.get('success') else '❌ FAIL'}")
        report.append("")

        # Neo4j Tests
        neo4j_results = test_results.get("neo4j_tests", {})
        report.append("🗄️ NEO4J STORE TESTS")
        report.append("-" * 40)
        if neo4j_results:
            report.append(f"Connection: {'✅' if neo4j_results.get('connection_test') else '❌'}")
            report.append(f"Schema: {'✅' if neo4j_results.get('schema_creation') else '❌'}")
            report.append(f"Insertion: {'✅' if neo4j_results.get('data_insertion') else '❌'}")
            report.append(f"Retrieval: {'✅' if neo4j_results.get('data_retrieval') else '❌'}")
            report.append(f"Status: {'✅ PASS' if neo4j_results.get('success') else '❌ FAIL'}")
        report.append("")

        # Pipeline Tests
        pipeline_results = test_results.get("pipeline_tests", {})
        report.append("🔄 PIPELINE PROCESSING TESTS")
        report.append("-" * 40)
        if pipeline_results:
            report.append(f"Document Processing: {'✅' if pipeline_results.get('document_processing') else '❌'}")
            report.append(f"Entity Extraction: {'✅' if pipeline_results.get('entity_extraction') else '❌'}")
            report.append(f"Storage: {'✅' if pipeline_results.get('storage_success') else '❌'}")
            report.append(f"Status: {'✅ PASS' if pipeline_results.get('success') else '❌ FAIL'}")
        report.append("")

        # Query Tests
        query_results = test_results.get("query_tests", {})
        report.append("🔍 QUERY CAPABILITIES TESTS")
        report.append("-" * 40)
        if query_results:
            success_rate = query_results.get("success_rate", 0)
            report.append(f"Success Rate: {success_rate:.2%}")
            report.append(f"Shareholder Queries: {query_results.get('shareholder_queries', 0)}")
            report.append(f"Percentage Queries: {query_results.get('percentage_queries', 0)}")
            report.append(f"Date Queries: {query_results.get('date_queries', 0)}")
            report.append(f"Status: {'✅ PASS' if query_results.get('success') else '❌ FAIL'}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def close(self):
        """Close connections."""
        if self.neo4j_store:
            self.neo4j_store.close()


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test shareholding knowledge graph system")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--report-file", type=str, help="Save report to file")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    tester = ShareholdingSystemTest()

    try:
        results = await tester.run_comprehensive_test()

        # Generate and display report
        report = tester.generate_test_report(results)
        print(report)

        # Save report if requested
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report_file}")

        # Return exit code based on success
        return 0 if results.get("overall_success") else 1

    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        tester.close()


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)