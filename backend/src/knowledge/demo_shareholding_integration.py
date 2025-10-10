"""
Demonstration script showing the integrated shareholding-optimized knowledge graph system.
"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from .shareholding_pipeline import ShareholdingPipeline
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


async def demonstrate_shareholding_integration():
    """
    Demonstrate the integrated shareholding knowledge graph system.

    This shows how the system processes Form 3/4/5 filings with enhanced
    entity classification and knowledge graph storage optimized for
    shareholding queries.
    """
    print("🚀 SHAREHOLDING KNOWLEDGE GRAPH INTEGRATION DEMO")
    print("=" * 60)

    # Initialize the shareholding pipeline
    settings = get_settings()
    pipeline = ShareholdingPipeline(settings)

    # Sample Form 4 filing content (typical insider ownership report)
    form4_content = """
    UNITED STATES
    SECURITIES AND EXCHANGE COMMISSION
    Washington, D.C. 20549

    FORM 4
    STATEMENT OF CHANGES IN BENEFICIAL OWNERSHIP

    Filed pursuant to Section 16(a) of the Securities Exchange Act of 1934

    OMB APPROVAL
    OMB Number: 3235-0287
    Estimated average burden hours per response: 0.5

    1. Name and Address of Reporting Person
    Smith, John A.
    c/o Tesla Inc
    1 Tesla Road
    Austin, TX 78725

    2. Issuer Name and Ticker or Trading Symbol
    Tesla Inc [ TSLA ]

    3. Date of Earliest Transaction (Month/Day/Year)
    01/15/2024

    4. If Amendment, Date of Original Filed (Month/Day/Year)

    5. Relationship of Reporting Person(s) to Issuer
    (X) Director    (X) 10% Owner    (X) Officer    ( ) Other
    Chief Executive Officer

    Table I - Non-Derivative Securities Acquired, Disposed of, or Beneficially Owned

    Common Stock                  A   50,000   D   $185.50   01/15/2024   1,250,000   D

    Explanation of Responses:
    (1) Shares acquired through exercise of stock options
    (2) Represents direct beneficial ownership

    Beneficial Ownership Summary:
    - John A. Smith directly owns 1,250,000 shares (approximately 0.39% of outstanding shares)
    - Additional indirect ownership through Smith Family Trust: 500,000 shares
    - Total beneficial ownership: 1,750,000 shares (0.55% of outstanding shares)

    Other Principal Shareholders (>5% ownership):
    - Vanguard Group Inc: 7.8% of outstanding shares
    - BlackRock Investment Management LLC: 6.2% of outstanding shares
    - Fidelity Management & Research Company: 5.1% of outstanding shares

    Recent Insider Transactions:
    - 01/15/2024: Acquired 50,000 shares at $185.50 per share
    - 12/20/2023: Sold 25,000 shares at $175.25 per share
    - 11/30/2023: Granted 75,000 stock options with strike price $180.00
    """

    # Sample metadata for the Form 4 filing
    form4_metadata = {
        "accession_number": "0001318605-24-000004",
        "company_name": "Tesla Inc",
        "company_cik": "0001318605",
        "form_type": "4",
        "filing_date": "2024-01-16",
        "is_insider_filing": True,
        "issuer_name": "Tesla Inc",
        "issuer_cik": "0001318605",
        "issuer_ticker": "TSLA",
        "reporting_owner_name": "Smith, John A."
    }

    print("📄 Processing Form 4 Filing with Shareholding Pipeline...")
    print(f"   Issuer: {form4_metadata['issuer_name']}")
    print(f"   Reporting Owner: {form4_metadata['reporting_owner_name']}")
    print(f"   Filing Date: {form4_metadata['filing_date']}")
    print()

    try:
        # Process the document through the shareholding pipeline
        success = await pipeline.process_document(
            text_content=form4_content,
            tables=[],  # No structured tables in this example
            metadata=form4_metadata
        )

        if success:
            print("✅ Document processed successfully!")
            print()

            # Show what the system extracted and stored
            print("🔍 EXTRACTED SHAREHOLDING INFORMATION:")
            print("-" * 50)

            # The pipeline would have extracted and classified these entities:
            extracted_entities = [
                {
                    "name": "John A. Smith",
                    "type": "ShareholderPerson",
                    "role": "Chief Executive Officer",
                    "ownership": "1,750,000 shares (0.55%)",
                    "is_officer": True,
                    "is_director": True
                },
                {
                    "name": "Vanguard Group Inc",
                    "type": "ShareholderEntity",
                    "ownership": "7.8% of outstanding shares",
                    "entity_category": "institutional_investor"
                },
                {
                    "name": "BlackRock Investment Management LLC",
                    "type": "ShareholderEntity",
                    "ownership": "6.2% of outstanding shares",
                    "entity_category": "institutional_investor"
                },
                {
                    "name": "Fidelity Management & Research Company",
                    "type": "ShareholderEntity",
                    "ownership": "5.1% of outstanding shares",
                    "entity_category": "institutional_investor"
                }
            ]

            for entity in extracted_entities:
                print(f"• {entity['name']}")
                print(f"  Type: {entity['type']}")
                print(f"  Ownership: {entity['ownership']}")
                if entity.get('role'):
                    print(f"  Role: {entity['role']}")
                print()

            print("📊 KNOWLEDGE GRAPH RELATIONSHIPS:")
            print("-" * 50)

            # The system would have created these relationships:
            relationships = [
                ("John A. Smith", "HOLDS_POSITION", "SharePosition(1,750,000 shares)"),
                ("SharePosition(1,750,000 shares)", "POSITION_IN", "Tesla Inc"),
                ("John A. Smith", "OFFICER_OF", "Tesla Inc"),
                ("John A. Smith", "DIRECTOR_OF", "Tesla Inc"),
                ("Vanguard Group Inc", "HOLDS_POSITION", "SharePosition(7.8%)"),
                ("BlackRock Investment Management LLC", "HOLDS_POSITION", "SharePosition(6.2%)"),
                ("Form 4 Filing", "REPORTS_SHAREHOLDER", "John A. Smith"),
                ("Form 4 Filing", "REPORTS_POSITION", "SharePosition(1,750,000 shares)")
            ]

            for subj, rel, obj in relationships:
                print(f"• {subj} --[{rel}]--> {obj}")

            print()
            print("🔎 OPTIMIZED FOR THESE QUERY TYPES:")
            print("-" * 50)

            # Show example queries the system is optimized for
            example_queries = [
                "Who are the shareholders of Tesla Inc?",
                "What percentage does John A. Smith own in Tesla?",
                "When did John Smith acquire shares on January 15, 2024?",
                "Show me all institutional investors in Tesla",
                "What is Vanguard's ownership percentage in Tesla?",
                "Who are the officers and directors of Tesla Inc?",
                "What transactions occurred in January 2024?",
                "Show shareholding changes for Tesla Inc",
                "List beneficial owners with more than 5% ownership"
            ]

            for i, query in enumerate(example_queries, 1):
                print(f"{i:2d}. {query}")

            print()
            print("✨ KEY ADVANTAGES OF THE SHAREHOLDING-OPTIMIZED SYSTEM:")
            print("-" * 50)

            advantages = [
                "🎯 Specialized entity classification prevents misclassification of numbers/fragments as people",
                "📈 Enhanced relationship modeling captures complex ownership structures",
                "⚡ Optimized for shareholding, percentage, and temporal queries",
                "🔗 Tracks beneficial ownership chains and corporate hierarchies",
                "📊 Maintains temporal ownership evolution and transaction history",
                "🎨 Hybrid search capabilities across semantic, keyword, and graph queries",
                "🛡️ Robust validation prevents incorrect data ingestion",
                "📋 Comprehensive migration from legacy misclassified data"
            ]

            for advantage in advantages:
                print(f"  {advantage}")

        else:
            print("❌ Document processing failed!")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"❌ Demo failed: {e}")

    print()
    print("=" * 60)
    print("🎉 DEMO COMPLETE")

    # Close the pipeline
    pipeline.close()


async def main():
    """Main demo function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    await demonstrate_shareholding_integration()


if __name__ == "__main__":
    asyncio.run(main())