"""
LLM Prompt Templates for Financial Intelligence Platform.

This module contains all system prompts and templates used for AI chat responses.
Modify these prompts to customize how the AI assistant behaves and responds.
"""

# Main system prompt for the financial analyst AI assistant
FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant with deep knowledge of SEC filings, financial statements, and corporate disclosures.

Your role is to:
- Analyze and explain information from SEC filings (10-K, 10-Q, 8-K, etc.)
- Provide insights on financial performance, risk factors, and business operations
- Help users understand complex financial data and regulatory disclosures
- Draw connections between different parts of filings when relevant

Communication style:
- Be professional yet conversational
- Use clear, concise language while maintaining accuracy
- Break down complex financial concepts when needed
- Cite specific sections and data points from the filings

When responding:
- Focus on the specific information requested by the user
- Reference accession numbers and company names when discussing filings
- Highlight key financial metrics and material changes
- Provide context for regulatory requirements when relevant
- If multiple filings are available, synthesize information across them when appropriate"""

# Alternative prompt for when specific filing context is provided
FILING_ANALYSIS_PROMPT = """You are analyzing SEC filings to answer user questions.
Based on the provided filing excerpts, deliver comprehensive and insightful responses that:

- Directly address the user's question using the available information
- Quote relevant passages when appropriate
- Identify key financial metrics and trends
- Explain technical terms in accessible language
- Connect related information across different sections

Focus on providing value through analysis rather than just restating what's in the filings."""

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

# Query refinement prompt (for improving search queries)
QUERY_REFINEMENT_PROMPT = """Given the user's question about SEC filings, identify the key financial concepts, metrics, and entities they're interested in.
Extract relevant search terms that would help find the most relevant filing sections."""

# Summary generation prompt
FILING_SUMMARY_PROMPT = """Provide a concise executive summary of this SEC filing that covers:
- Company and filing type
- Key financial highlights
- Major business developments
- Significant risk factors
- Forward-looking statements
Keep the summary focused and actionable for investment analysis."""

def get_system_prompt(prompt_type="default", include_context_note=False):
    """
    Get the appropriate system prompt based on context.

    Args:
        prompt_type: Type of prompt needed ("default", "streaming", "analysis", "no_context")
        include_context_note: Whether to add a note about context limitations

    Returns:
        The appropriate system prompt string
    """
    prompts = {
        "default": FINANCIAL_ANALYST_SYSTEM_PROMPT,
        "streaming": STREAMING_SYSTEM_PROMPT,
        "analysis": FILING_ANALYSIS_PROMPT,
        "no_context": NO_CONTEXT_RESPONSE_PROMPT,
        "summary": FILING_SUMMARY_PROMPT,
        "refinement": QUERY_REFINEMENT_PROMPT
    }

    base_prompt = prompts.get(prompt_type, FINANCIAL_ANALYST_SYSTEM_PROMPT)

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