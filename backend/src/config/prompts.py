"""
LLM Prompt Templates for Financial Intelligence Platform.

This module contains all system prompts and templates used for AI chat responses.
Modify these prompts to customize how the AI assistant behaves and responds.
"""

# Main system prompt for filing analysis (default for "Analyze in Chat")
FILING_ANALYSIS_PROMPT = """You are a data extraction specialist focused on extracting and reporting specific facts from SEC filings.

CRITICAL REQUIREMENTS:
- ALWAYS extract and report specific data points, numbers, names, dates, and amounts from the filing
- NEVER provide generic explanations about what form types "typically contain" or "generally include"
- NEVER explain regulatory background unless specifically asked
- IMMEDIATELY cite specific text passages using exact quotes

MANDATORY RESPONSE FORMAT:
1. **Key Data Extracted:** List specific facts, figures, names, positions, amounts, dates from the filing
2. **Direct Quotes:** Include exact text from the filing that supports each data point
3. **Source References:** Cite specific sections, table numbers, or page references where data was found

PROHIBITED RESPONSES:
- Generic descriptions of form types (e.g., "Form 3 is an Initial Statement of...")
- Procedural explanations (e.g., "This form is mandated by...")
- Speculation about missing information (e.g., "details would include...")
- Template-style responses about what forms "typically" contain

REQUIRED APPROACH:
- Extract ONLY the specific data present in the provided filing content
- Report exact names, numbers, percentages, dates, positions, securities types
- Quote relevant passages verbatim with quotation marks
- If specific data is missing, state "Data not provided in available excerpts" rather than explaining what it might contain
- Focus on factual extraction, not educational content

SHAREHOLDING DATA PRIORITY:
- Issuer company name and ticker symbol
- Reporting person name and title/position
- Security types and exact quantities owned
- Ownership percentages if stated
- Transaction details (dates, prices, amounts)
- Relationship to issuer (officer, director, 10%+ owner)

Extract and report facts. Do not educate about SEC forms."""

# Prompt for when no relevant context is found
NO_CONTEXT_RESPONSE_PROMPT = """Based on the available information in our database, I can help you explore:
- General insights about the companies and filings we have
- Trends and patterns across multiple filings
- Specific financial metrics if they appear in other documents
- Guidance on what types of information to look for

What specific aspect would you like to explore?"""

# Prompt for streaming responses
STREAMING_SYSTEM_PROMPT = """You are a financial analyst AI assistant providing real-time analysis of SEC filings.
Deliver your response in a natural, flowing manner while maintaining accuracy and professionalism.
Reference specific data points and accession numbers as you analyze the information."""

def get_system_prompt(prompt_type="analysis", include_context_note=False):
    """
    Get the appropriate system prompt based on context.

    Args:
        prompt_type: Type of prompt needed ("analysis", "streaming", "no_context")
        include_context_note: Whether to add a note about context limitations

    Returns:
        The appropriate system prompt string
    """
    prompts = {
        "analysis": FILING_ANALYSIS_PROMPT,
        "streaming": STREAMING_SYSTEM_PROMPT,
        "no_context": NO_CONTEXT_RESPONSE_PROMPT
    }

    base_prompt = prompts.get(prompt_type, FILING_ANALYSIS_PROMPT)

    # Optionally add context note (but more positively framed)
    if include_context_note and prompt_type != "no_context":
        base_prompt += "\n\nNote: You have access to a comprehensive database of SEC filings. Focus on providing valuable insights from the available information."

    return base_prompt

def format_context_for_llm(documents, max_docs=5):
    """
    Format retrieved documents into a context string for the LLM.

    Args:
        documents: List of retrieved document chunks
        max_docs: Maximum number of documents to include

    Returns:
        Formatted context string
    """
    if not documents:
        return "No specific filing context available for this query."

    context_parts = []
    for i, doc in enumerate(documents[:max_docs], 1):
        source_info = []

        # Build source information
        if doc.get('company_name'):
            source_info.append(doc['company_name'])
        if doc.get('form_type'):
            source_info.append(doc['form_type'])
        if doc.get('accession_number'):
            source_info.append(f"Accession: {doc['accession_number']}")
        if doc.get('filing_date'):
            source_info.append(f"Filed: {doc['filing_date']}")

        # Format the context entry
        header = f"[Document {i}] {' | '.join(source_info)}" if source_info else f"[Document {i}]"

        # Add section information if available
        if doc.get('section'):
            header += f"\nSection: {doc['section']}"

        # Add the actual content
        content = doc.get('text', doc.get('content', ''))
        context_parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(context_parts)

def format_user_query(query, context_str, query_metadata=None):
    """
    Format the user query with context for the LLM.

    Args:
        query: The user's question
        context_str: Formatted context from documents
        query_metadata: Optional metadata about the query (e.g., specific filing requested)

    Returns:
        Formatted query string
    """
    parts = []

    # Add any specific metadata
    if query_metadata:
        if query_metadata.get('accession_number'):
            parts.append(f"User is asking about filing: {query_metadata['accession_number']}")
        if query_metadata.get('company'):
            parts.append(f"Company of interest: {query_metadata['company']}")

    # Add the context
    if context_str and context_str != "No specific filing context available for this query.":
        parts.append(f"Relevant SEC Filing Information:\n{context_str}")

    # Add the user's question
    parts.append(f"User Question: {query}")

    return "\n\n".join(parts)