# Form 3/4/5 Insider Filing Issue

## Problem Description

When querying about SEC Form 3, 4, or 5 filings (insider ownership reports), the AI chat cannot correctly identify the company because:

1. **RSS Feed Structure**: The SEC RSS feed lists the REPORTING PERSON's name (e.g., "Sutherland Gregory David") as the "company name" for these forms
2. **Missing Issuer Data**: The actual company whose securities are being reported is not captured during ingestion
3. **Incomplete Extraction**: Our current pipeline doesn't extract the issuer company information from the filing content

## Example

Filing: `0002091230-25-000002`
- What's stored as "company": Sutherland Gregory David (the person filing)
- What's missing: The actual company whose stock they own

## Technical Details

### Current Behavior

```python
# In RSS feed for Form 3/4/5:
<company-name>Sutherland Gregory David</company-name>  # This is the filer, not the company
```

### What We Need

The actual issuer information is in the filing's XML/HTML but requires special parsing:
- Issuer CIK
- Issuer Name
- Issuer Trading Symbol

## Temporary Workaround

For now, when analyzing Form 3/4/5 filings:

1. The system will show the reporting person's name
2. To find the actual company, users need to:
   - Click "View on SEC.gov" to see the full filing
   - Look for the "Issuer" section in the document

## Permanent Solution (TODO)

### Option 1: Enhanced Parsing

Update `docling_processor.py` to:
1. Detect Form 3/4/5 filings
2. Extract issuer information from the XML structure
3. Store both reporter and issuer information separately

### Option 2: Dual Company Model

Modify database schema to support:
```python
class Filing:
    reporting_owner_id = Column(Integer, ForeignKey("persons.id"))  # For Form 3/4/5
    issuer_company_id = Column(Integer, ForeignKey("companies.id"))  # The actual company
```

### Option 3: Post-Processing Enhancement

Add a background task that:
1. Identifies Form 3/4/5 filings
2. Fetches the XML version
3. Extracts issuer details
4. Updates the filing record

## Implementation Priority

This is a known limitation that affects:
- Form 3 (Initial Statement of Beneficial Ownership)
- Form 4 (Statement of Changes in Beneficial Ownership)
- Form 5 (Annual Statement of Beneficial Ownership)

These forms represent a significant portion of SEC filings but are less critical for financial analysis than 10-K, 10-Q, and 8-K forms.

## Current Impact

- Users asking about Form 3/4/5 filings will get the filer's name, not the company
- The AI can still analyze the ownership data if it's in the chunks
- Full company context requires viewing the original filing on SEC.gov

## Recommended Action

1. **Short term**: Document this limitation in user guide
2. **Medium term**: Implement issuer extraction for Form 3/4/5
3. **Long term**: Consider separate handling for all insider trading forms

---

**Note**: This issue does NOT affect Form 10-K, 10-Q, 8-K, or other standard company filings where the company name is correctly captured from the RSS feed.