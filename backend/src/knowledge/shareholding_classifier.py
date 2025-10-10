"""
Shareholding-optimized entity classification system for SEC filings.
Specialized for accurate identification and categorization of shareholding relationships,
ownership data, and related financial entities.
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity types for shareholding classification."""
    SHAREHOLDER_PERSON = "shareholder_person"
    SHAREHOLDER_ENTITY = "shareholder_entity"
    COMPANY = "company"
    SHARE_CLASS = "share_class"
    SHARE_POSITION = "share_position"
    ISSUANCE_EVENT = "issuance_event"
    SHARE_TRANSACTION = "share_transaction"
    BENEFICIAL_OWNERSHIP = "beneficial_ownership"
    INVALID = "invalid"


class ShareClassType(str, Enum):
    """Share class types."""
    COMMON = "common"
    PREFERRED = "preferred"
    CLASS_A = "class_a"
    CLASS_B = "class_b"
    CLASS_C = "class_c"
    OPTIONS = "options"
    WARRANTS = "warrants"
    CONVERTIBLE = "convertible"


class TransactionType(str, Enum):
    """Share transaction types."""
    PURCHASE = "purchase"
    SALE = "sale"
    GRANT = "grant"
    EXERCISE = "exercise"
    CONVERSION = "conversion"
    DISTRIBUTION = "distribution"
    GIFT = "gift"
    INHERITANCE = "inheritance"


@dataclass
class ClassificationResult:
    """Result of entity classification."""
    entity_type: EntityType
    confidence: float
    extracted_data: Dict[str, Any]
    validation_flags: List[str]
    source_pattern: Optional[str] = None


@dataclass
class ShareholdingEntity:
    """Structured shareholding entity."""
    name: str
    entity_type: EntityType
    shares: Optional[int] = None
    percentage: Optional[float] = None
    share_class: Optional[ShareClassType] = None
    as_of_date: Optional[datetime] = None
    context: Optional[str] = None
    confidence: float = 0.0


class ShareholdingClassifier:
    """Classify and validate shareholding-related entities."""

    def __init__(self):
        """Initialize shareholding classifier."""
        self.person_name_patterns = [
            # Standard person name patterns
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)*\s+[A-Z][a-z]+$',  # John A. Smith
            r'^[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+(?:\s+(?:Jr|Sr|III?|IV)\.?)?$',  # John A. Smith Jr.
            r'^[A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)*$',  # Smith, John A.
        ]

        self.entity_indicators = [
            # Corporate entity indicators
            r'\b(?:Inc|Corp|Corporation|Company|LLC|LP|Ltd|Co|Trust|Fund|Management|Group|Holdings|Partners|Advisors|Capital|Investments)\b\.?',
            r'\b(?:Foundation|Endowment|Pension|Retirement|401k|IRA|Estate|Family|Charitable)\b',
            r'\b(?:Investment|Asset|Wealth|Portfolio|Advisory|Securities|Financial)\b',
        ]

        self.shareholding_extraction_patterns = {
            'ownership_with_percentage': [
                r'(?P<name>[\w\s,\.&]+?)\s+(?:owns|holds|beneficially owns)\s+(?:approximately\s+)?(?P<shares>[\d,]+)\s+shares?.*?(?P<percentage>\d+(?:\.\d+)?)\s*%',
                r'(?P<name>[\w\s,\.&]+?):\s*(?P<shares>[\d,]+)\s+shares?\s*\((?P<percentage>\d+(?:\.\d+)?)\s*%\)',
                r'(?P<percentage>\d+(?:\.\d+)?)\s*%.*?owned\s+by\s+(?P<name>[\w\s,\.&]+?)(?:\s+holding\s+(?P<shares>[\d,]+)\s+shares?)?',
            ],

            'issuance_events': [
                r'(?:issued|granted|awarded)\s+(?P<shares>[\d,]+)\s+shares?\s+(?:of\s+)?(?P<share_class>[\w\s]+?)?\s+(?:to\s+)?(?P<recipient>[\w\s,\.&]+?)\s+(?:on|dated)\s+(?P<date>[A-Z][a-z]+\s+\d+,?\s+\d{4})(?:\s+(?:for|as|in)\s+(?P<reason>[^.]+?))?',
                r'(?P<recipient>[\w\s,\.&]+?)\s+(?:received|was granted)\s+(?P<shares>[\d,]+)\s+(?P<share_class>[\w\s]+?)?\s*shares?\s+(?:on\s+)?(?P<date>[A-Z][a-z]+\s+\d+,?\s+\d{4})',
            ],

            'share_transactions': [
                r'(?P<person>[\w\s,\.&]+?)\s+(?P<transaction_type>purchased|sold|acquired|disposed of)\s+(?P<shares>[\d,]+)\s+shares?\s+(?:at\s+\$?(?P<price>[\d,\.]+))?.*?(?:on\s+)?(?P<date>[A-Z][a-z]+\s+\d+,?\s+\d{4})',
                r'(?P<transaction_type>Purchase|Sale|Acquisition|Disposition)\s+of\s+(?P<shares>[\d,]+)\s+shares?\s+by\s+(?P<person>[\w\s,\.&]+?)(?:\s+at\s+\$?(?P<price>[\d,\.]+))?',
            ],

            'beneficial_ownership': [
                r'(?P<owner>[\w\s,\.&]+?)\s+beneficially\s+owns.*?through\s+(?:its\s+)?(?:ownership\s+of\s+)?(?P<through_entity>[\w\s,\.&]+?)(?:\s+which\s+owns\s+(?P<shares>[\d,]+)\s+shares?)?',
                r'(?P<ultimate_owner>[\w\s,\.&]+?)\s+controls\s+(?P<shares>[\d,]+)\s+shares?\s+through\s+(?P<controlled_entity>[\w\s,\.&]+?)',
            ],

            'share_classes': [
                r'(?P<shares>[\d,]+)\s+shares?\s+of\s+(?P<share_class>Class\s+[A-Z]|[Cc]ommon|[Pp]referred|[Oo]ptions|[Ww]arrants)\s+[Ss]tock',
                r'(?P<share_class>Class\s+[A-Z]|Common|Preferred)\s+shares?:\s*(?P<shares>[\d,]+)',
            ]
        }

        # Invalid entity patterns (things that shouldn't be classified as persons/entities)
        self.invalid_patterns = [
            r'^\d+(?:\.\d+)?$',  # Pure numbers
            r'^[\d,]+$',  # Numbers with commas
            r'^[^a-zA-Z]*$',  # No letters
            r'^\$[\d,\.]+$',  # Dollar amounts
            r'^(?:and|or|the|of|in|at|on|for|with|by|to|from)$',  # Common words
            r'^(?:shares?|stock|securities|percentage|percent|ownership)$',  # Financial terms
            r'.{100,}',  # Very long strings (likely document fragments)
            r'^\s*$',  # Whitespace only
        ]

    def classify_entity(self, text: str, context: Dict[str, Any] = None) -> ClassificationResult:
        """
        Classify an entity for shareholding purposes.

        Args:
            text: Text to classify
            context: Additional context for classification

        Returns:
            ClassificationResult with type and confidence
        """
        text = str(text).strip()
        context = context or {}

        # Check for invalid patterns first
        if self._is_invalid_entity(text):
            return ClassificationResult(
                entity_type=EntityType.INVALID,
                confidence=1.0,
                extracted_data={"text": text, "reason": "invalid_pattern"},
                validation_flags=["invalid_pattern"]
            )

        # Check if it's a person name
        person_result = self._classify_person(text, context)
        if person_result.confidence > 0.7:
            return person_result

        # Check if it's a corporate entity
        entity_result = self._classify_corporate_entity(text, context)
        if entity_result.confidence > 0.7:
            return entity_result

        # Check if it's a share class
        share_class_result = self._classify_share_class(text, context)
        if share_class_result.confidence > 0.8:
            return share_class_result

        # Default to invalid with low confidence
        return ClassificationResult(
            entity_type=EntityType.INVALID,
            confidence=0.9,
            extracted_data={"text": text, "reason": "no_clear_classification"},
            validation_flags=["unclassified"]
        )

    def extract_shareholding_relationships(self, text: str, filing_context: Dict[str, Any] = None) -> List[ShareholdingEntity]:
        """
        Extract structured shareholding relationships from text.

        Args:
            text: Text to extract from
            filing_context: Filing metadata for context

        Returns:
            List of ShareholdingEntity objects
        """
        relationships = []
        filing_context = filing_context or {}

        for pattern_type, patterns in self.shareholding_extraction_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

                for match in matches:
                    try:
                        entity = self._process_shareholding_match(match, pattern_type, text, filing_context)
                        if entity and entity.confidence > 0.5:
                            relationships.append(entity)
                    except Exception as e:
                        logger.debug(f"Error processing shareholding match: {e}")

        return self._deduplicate_relationships(relationships)

    def _is_invalid_entity(self, text: str) -> bool:
        """Check if text matches invalid entity patterns."""
        for pattern in self.invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _classify_person(self, text: str, context: Dict[str, Any]) -> ClassificationResult:
        """Classify if text represents a person name."""
        confidence = 0.0
        validation_flags = []
        extracted_data = {"name": text}

        # Check person name patterns
        for pattern in self.person_name_patterns:
            if re.match(pattern, text):
                confidence += 0.4
                validation_flags.append("name_pattern_match")
                break

        # Additional person indicators
        if len(text.split()) in [2, 3, 4]:  # Reasonable name length
            confidence += 0.2

        if text[0].isupper() and not text.isupper():  # Proper capitalization
            confidence += 0.1

        # Check for titles that indicate person
        person_titles = r'\b(?:Mr|Mrs|Ms|Dr|Prof|CEO|CFO|CTO|President|Director|Chairman|Executive|Officer)\b\.?'
        if re.search(person_titles, context.get('surrounding_text', ''), re.IGNORECASE):
            confidence += 0.3
            validation_flags.append("person_title_context")

        # Negative indicators
        entity_pattern = '|'.join(self.entity_indicators)
        if re.search(entity_pattern, text, re.IGNORECASE):
            confidence -= 0.4
            validation_flags.append("entity_indicator_found")

        return ClassificationResult(
            entity_type=EntityType.SHAREHOLDER_PERSON,
            confidence=min(confidence, 1.0),
            extracted_data=extracted_data,
            validation_flags=validation_flags
        )

    def _classify_corporate_entity(self, text: str, context: Dict[str, Any]) -> ClassificationResult:
        """Classify if text represents a corporate entity."""
        confidence = 0.0
        validation_flags = []
        extracted_data = {"name": text}

        # Check for entity indicators
        entity_pattern = '|'.join(self.entity_indicators)
        if re.search(entity_pattern, text, re.IGNORECASE):
            confidence += 0.6
            validation_flags.append("entity_indicator_match")

        # Check for institutional investor patterns
        institutional_patterns = [
            r'\b(?:Vanguard|BlackRock|Fidelity|State Street|Capital|Advisors|Management|Partners)\b',
            r'\b(?:Investment|Asset|Wealth|Portfolio|Advisory|Securities|Financial)\b.*\b(?:Group|Company|Corporation|LLC|LP)\b',
            r'\b(?:Pension|Retirement|Endowment|Foundation|Trust|Fund)\b',
        ]

        for pattern in institutional_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence += 0.3
                validation_flags.append("institutional_pattern")
                break

        # Length and structure checks
        if 10 <= len(text) <= 100:  # Reasonable entity name length
            confidence += 0.1

        return ClassificationResult(
            entity_type=EntityType.SHAREHOLDER_ENTITY,
            confidence=min(confidence, 1.0),
            extracted_data=extracted_data,
            validation_flags=validation_flags
        )

    def _classify_share_class(self, text: str, context: Dict[str, Any]) -> ClassificationResult:
        """Classify if text represents a share class."""
        confidence = 0.0
        validation_flags = []
        extracted_data = {"share_class": text}

        share_class_patterns = [
            (r'\bClass\s+[A-Z]\b', ShareClassType.CLASS_A, 0.9),
            (r'\b[Cc]ommon\s+[Ss]tock\b', ShareClassType.COMMON, 0.8),
            (r'\b[Pp]referred\s+[Ss]tock\b', ShareClassType.PREFERRED, 0.8),
            (r'\b[Oo]ptions?\b', ShareClassType.OPTIONS, 0.7),
            (r'\b[Ww]arrants?\b', ShareClassType.WARRANTS, 0.7),
        ]

        for pattern, share_type, pattern_confidence in share_class_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                confidence = pattern_confidence
                extracted_data["share_type"] = share_type
                validation_flags.append(f"share_class_{share_type}")
                break

        return ClassificationResult(
            entity_type=EntityType.SHARE_CLASS,
            confidence=confidence,
            extracted_data=extracted_data,
            validation_flags=validation_flags
        )

    def _process_shareholding_match(self, match: re.Match, pattern_type: str, source_text: str, context: Dict[str, Any]) -> Optional[ShareholdingEntity]:
        """Process a regex match into a ShareholdingEntity."""
        try:
            groups = match.groupdict()

            if pattern_type == 'ownership_with_percentage':
                name = groups.get('name', '').strip()
                shares_str = groups.get('shares', '').replace(',', '')
                percentage_str = groups.get('percentage', '')

                # Validate the name
                name_classification = self.classify_entity(name, {'surrounding_text': source_text})
                if name_classification.entity_type == EntityType.INVALID:
                    return None

                return ShareholdingEntity(
                    name=name,
                    entity_type=name_classification.entity_type,
                    shares=int(shares_str) if shares_str.isdigit() else None,
                    percentage=float(percentage_str) if percentage_str else None,
                    context=match.group(0)[:200],
                    confidence=name_classification.confidence * 0.9
                )

            elif pattern_type == 'issuance_events':
                recipient = groups.get('recipient', '').strip()
                shares_str = groups.get('shares', '').replace(',', '')

                recipient_classification = self.classify_entity(recipient, {'surrounding_text': source_text})
                if recipient_classification.entity_type == EntityType.INVALID:
                    return None

                return ShareholdingEntity(
                    name=recipient,
                    entity_type=recipient_classification.entity_type,
                    shares=int(shares_str) if shares_str.isdigit() else None,
                    context=f"Issuance: {match.group(0)[:200]}",
                    confidence=recipient_classification.confidence * 0.8
                )

            # Handle other pattern types...
            return None

        except Exception as e:
            logger.debug(f"Error processing shareholding match: {e}")
            return None

    def _deduplicate_relationships(self, relationships: List[ShareholdingEntity]) -> List[ShareholdingEntity]:
        """Remove duplicate shareholding relationships."""
        seen = set()
        deduplicated = []

        for rel in relationships:
            # Create a key based on name and entity type
            key = (rel.name.lower().strip(), rel.entity_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
            else:
                # If we've seen this entity, keep the one with higher confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.name.lower().strip(), existing.entity_type) == key:
                        if rel.confidence > existing.confidence:
                            deduplicated[i] = rel
                        break

        return deduplicated

    def validate_shareholding_data(self, shareholding_entities: List[ShareholdingEntity]) -> Dict[str, Any]:
        """
        Validate a collection of shareholding entities.

        Args:
            shareholding_entities: List of entities to validate

        Returns:
            Validation report with statistics and issues
        """
        report = {
            "total_entities": len(shareholding_entities),
            "valid_entities": 0,
            "invalid_entities": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "entity_type_distribution": {},
            "validation_issues": []
        }

        for entity in shareholding_entities:
            # Count valid/invalid
            if entity.confidence > 0.5:
                report["valid_entities"] += 1
            else:
                report["invalid_entities"] += 1

            # Confidence distribution
            if entity.confidence > 0.8:
                report["confidence_distribution"]["high"] += 1
            elif entity.confidence > 0.5:
                report["confidence_distribution"]["medium"] += 1
            else:
                report["confidence_distribution"]["low"] += 1

            # Entity type distribution
            entity_type = entity.entity_type.value
            report["entity_type_distribution"][entity_type] = report["entity_type_distribution"].get(entity_type, 0) + 1

            # Specific validation issues
            if entity.confidence < 0.3:
                report["validation_issues"].append(f"Low confidence entity: {entity.name} ({entity.confidence:.2f})")

        return report

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "supported_entity_types": [e.value for e in EntityType],
            "shareholding_pattern_types": list(self.shareholding_extraction_patterns.keys()),
            "total_extraction_patterns": sum(len(patterns) for patterns in self.shareholding_extraction_patterns.values()),
            "person_name_patterns": len(self.person_name_patterns),
            "entity_indicators": len(self.entity_indicators),
            "invalid_patterns": len(self.invalid_patterns)
        }