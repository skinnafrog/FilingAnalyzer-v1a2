-- Migration to add issuer company support for Form 3/4/5 filings
-- This allows us to distinguish between the reporting owner (person filing)
-- and the issuer company (whose securities are being reported)

-- Add issuer fields to filings table
ALTER TABLE filings ADD COLUMN IF NOT EXISTS issuer_company_id INTEGER REFERENCES companies(id);
ALTER TABLE filings ADD COLUMN IF NOT EXISTS issuer_cik VARCHAR(10);
ALTER TABLE filings ADD COLUMN IF NOT EXISTS issuer_name VARCHAR(255);
ALTER TABLE filings ADD COLUMN IF NOT EXISTS issuer_ticker VARCHAR(10);
ALTER TABLE filings ADD COLUMN IF NOT EXISTS reporting_owner_name VARCHAR(255);
ALTER TABLE filings ADD COLUMN IF NOT EXISTS is_insider_filing BOOLEAN DEFAULT FALSE;

-- Create index for issuer lookups
CREATE INDEX IF NOT EXISTS idx_filings_issuer_company ON filings(issuer_company_id);
CREATE INDEX IF NOT EXISTS idx_filings_issuer_cik ON filings(issuer_cik);
CREATE INDEX IF NOT EXISTS idx_filings_insider ON filings(is_insider_filing);

-- Update existing Form 3/4/5 filings to mark them as insider filings
UPDATE filings
SET is_insider_filing = TRUE,
    reporting_owner_name = (SELECT name FROM companies WHERE id = filings.company_id)
WHERE form_type IN ('3', '4', '5', '3/A', '4/A', '5/A');

COMMENT ON COLUMN filings.issuer_company_id IS 'The actual company whose securities are being reported (for Form 3/4/5)';
COMMENT ON COLUMN filings.reporting_owner_name IS 'The person/entity filing the report (for Form 3/4/5)';
COMMENT ON COLUMN filings.is_insider_filing IS 'True for Form 3/4/5 insider ownership reports';