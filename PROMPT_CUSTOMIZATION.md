# AI Chat Prompt Customization Guide

This guide explains how to customize the AI assistant's behavior and responses in the Financial Intelligence Platform.

## Quick Start

The AI chat prompts are centrally managed in `/backend/src/config/prompts.py`. You can modify these prompts to change how the AI responds to queries about SEC filings.

## Available Prompts

### 1. Main Financial Analyst Prompt (`FINANCIAL_ANALYST_SYSTEM_PROMPT`)

This is the primary system prompt used for standard chat interactions. It defines the AI's role as a financial analyst and sets the tone for responses.

**Current behavior:**
- Acts as an expert financial analyst
- Focuses on SEC filings and financial data
- Professional yet conversational tone
- Cites specific data points

**To customize:** Modify the `FINANCIAL_ANALYST_SYSTEM_PROMPT` variable to change the AI's personality, expertise level, or communication style.

### 2. Filing Analysis Prompt (`FILING_ANALYSIS_PROMPT`)

Used when specific filing context is provided with the query. This prompt emphasizes analytical insights over simple information retrieval.

**Current behavior:**
- Provides comprehensive analysis
- Quotes relevant passages
- Identifies trends and metrics
- Explains technical terms

**To customize:** Adjust this prompt to change the depth of analysis or focus areas.

### 3. No Context Response (`NO_CONTEXT_RESPONSE_PROMPT`)

Used when no relevant filings are found for a query. Instead of saying "no information available," it offers helpful alternatives.

**Current behavior:**
- Suggests exploring available data
- Offers to look at related information
- Remains helpful and engaged

**To customize:** Change this to provide different fallback behaviors.

### 4. Streaming Response Prompt (`STREAMING_SYSTEM_PROMPT`)

Used for real-time streaming responses, optimized for natural flow.

## How to Customize Prompts

### Step 1: Locate the Prompts File
```bash
cd backend/src/config
vi prompts.py  # or use your preferred editor
```

### Step 2: Modify the Desired Prompt

Example - Making responses more concise:
```python
FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a concise financial analyst AI.
Provide brief, bullet-point responses focusing on key metrics and insights.
Limit responses to essential information only."""
```

Example - Adding industry-specific knowledge:
```python
FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a specialized financial analyst
focusing on [YOUR INDUSTRY]. You understand industry-specific metrics like
[METRIC 1], [METRIC 2], and regulatory requirements specific to [INDUSTRY]."""
```

### Step 3: Restart the Backend Service
```bash
docker-compose restart backend
```

## Common Customizations

### Remove "No Information Available" Messages

The original prompt contained:
```python
"If the context doesn't contain information about the requested filing, clearly state that."
```

This has been removed and replaced with a more helpful approach that explores available data.

### Change Response Length

Add length instructions to any prompt:
```python
"Keep responses under 200 words unless specifically asked for more detail."
```

### Add Custom Expertise

Include domain-specific knowledge:
```python
"You specialize in analyzing tech company financials, with deep understanding of
SaaS metrics, R&D expenditures, and software revenue recognition."
```

### Adjust Formality Level

For more casual responses:
```python
"Communicate in a friendly, approachable manner while maintaining accuracy.
Use simple language and avoid jargon unless necessary."
```

For more formal responses:
```python
"Maintain a formal, professional tone suitable for institutional investors.
Use precise financial terminology and cite specific regulatory requirements."
```

## Testing Your Changes

After modifying prompts, test them through the chat interface:

1. Open the Filings Explorer
2. Click "Analyze in Chat" on any filing
3. Ask various questions to see how the AI responds
4. Fine-tune prompts based on the responses

## Advanced Customization

### Context Formatting

The `format_context_for_llm()` function in `prompts.py` controls how filing data is presented to the AI. Modify this to:
- Change how sources are cited
- Adjust the amount of context provided
- Add or remove metadata fields

### Query Processing

The `format_user_query()` function preprocesses user questions. Customize this to:
- Extract specific entities from queries
- Add automatic context enrichment
- Implement query rewriting

## Best Practices

1. **Test Incrementally**: Make small changes and test thoroughly
2. **Preserve Core Functionality**: Keep financial analysis capabilities intact
3. **Document Changes**: Comment your modifications for future reference
4. **Monitor Performance**: Check response quality after changes
5. **Backup Original**: Keep a copy of the original prompts.py file

## Troubleshooting

### AI Still Mentions "No Information Available"

1. Check that you've restarted the backend service
2. Verify the prompts.py file was saved correctly
3. Clear any cached responses (if applicable)

### Responses Are Too Long/Short

Adjust the `max_tokens` parameter in Settings or add explicit length instructions in prompts.

### AI Doesn't Follow Instructions

Ensure your prompt changes are clear and unambiguous. The AI follows the system prompt strictly, so be specific about desired behaviors.

## Example: Custom Prompt for Your Use Case

Here's a complete example of customizing for a more natural chat experience:

```python
FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are an intelligent assistant helping users
understand SEC filings and financial data.

Your approach:
- Start responses directly with valuable insights
- Focus on what the data reveals rather than what it doesn't
- Draw connections between different pieces of information
- Highlight important trends and anomalies
- Use natural, conversational language

When analyzing filings:
- Reference specific sections and page numbers when available
- Quote key passages that support your analysis
- Explain financial metrics in context
- Identify potential risks and opportunities

Never start responses with phrases like "The provided context" or "Based on the
available information." Instead, dive directly into the analysis."""
```

## Support

For questions or issues with prompt customization:
1. Check the logs: `docker-compose logs backend`
2. Verify syntax in prompts.py
3. Ensure all services are running: `docker-compose ps`