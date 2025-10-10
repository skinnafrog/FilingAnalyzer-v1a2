# Session Management and Context Preservation

This document describes the session management system implemented for the AI chat interface to maintain conversation context and filing references across multiple queries.

## Overview

The session management system addresses the issue where the AI chat would "forget" which filing was being discussed between queries. It maintains:
- Conversation history across multiple interactions
- Current filing context (when analyzing a specific filing)
- Session persistence in browser localStorage
- Automatic context inclusion in RAG queries

## Architecture

### Backend Components

#### 1. Session Manager (`backend/src/api/session_manager.py`)

The core session management module that provides:

- **ChatSession**: Data class representing a complete chat session
  - Stores session ID, timestamps, current filing context, and conversation history
  - Automatically limits history to last 20 messages for memory efficiency
  - Provides methods to format context for LLM consumption

- **SessionManager**: Singleton service managing all active sessions
  - Maintains sessions in memory with OrderedDict (LRU-like behavior)
  - Configurable TTL (default 24 hours) for session expiration
  - Automatic cleanup of expired sessions
  - Maximum concurrent sessions limit (default 1000)

Key features:
```python
# Create or retrieve session
session = session_manager.get_or_create_session(session_id)

# Update filing context when analyzing a specific filing
session_manager.update_filing_context(session_id, filing_data)

# Add conversation turn
session_manager.add_interaction(
    session_id=session_id,
    user_message=query,
    assistant_response=response,
    sources=retrieved_docs
)

# Get formatted context for LLM
context = session_manager.get_conversation_context(session_id)
```

#### 2. API Integration (`backend/src/api/main.py`)

Both `/api/chat` and `/api/chat/stream` endpoints now:
1. Get or create a session for each request
2. Extract filing context from queries mentioning specific accession numbers
3. Include full conversation history in LLM prompts
4. Preserve filing context for subsequent queries
5. Automatically filter RAG retrieval based on current filing context

Example flow:
```python
# User asks: "Analyze filing 0001326801-24-000090"
# System:
# 1. Creates session
# 2. Extracts accession number
# 3. Loads filing details from database
# 4. Stores filing context in session
# 5. Filters RAG retrieval to this specific filing

# User then asks: "What was the revenue?"
# System:
# 1. Retrieves existing session
# 2. Includes filing context in prompt
# 3. Filters RAG retrieval to same filing
# 4. Provides contextual response
```

#### 3. RAG Pipeline Updates (`backend/src/knowledge/rag_pipeline.py`)

Updated to handle both dictionary and object formats for chat history, ensuring compatibility with session manager's context format.

### Frontend Components

#### Session Persistence (`frontend/index.html`)

The frontend now:
1. Stores session ID in localStorage for persistence across page reloads
2. Displays active session indicator with current filing context
3. Sends session ID with all chat requests
4. Clears session when user clicks "Clear Chat"

Key features:
```javascript
// Session stored in localStorage
let sessionId = localStorage.getItem('chatSessionId') || null;

// Session indicator shows active context
<span class="session-indicator">Active session - Filing: 0001326801-24-000090</span>

// Session cleared with chat
function clearChat() {
    sessionId = null;
    localStorage.removeItem('chatSessionId');
}
```

## Usage Examples

### Example 1: Analyzing a Specific Filing

1. User clicks "Analyze in Chat" from Filing Details
2. System pre-fills query: "Analyze the SEC filing with accession number 0001326801-24-000090"
3. Session created with filing context
4. Subsequent questions automatically filtered to this filing

### Example 2: Multi-Turn Conversation

```
User: "Tell me about Meta's latest 10-K"
AI: [Provides overview of Meta's filing]

User: "What were the main risk factors?"
AI: [Remembers Meta context, provides risk factors from same filing]

User: "Compare revenue to previous year"
AI: [Still maintains Meta context, provides comparison]
```

### Example 3: Session Persistence

1. User analyzes a filing
2. Browser is refreshed or tab closed
3. User returns to chat
4. Session automatically restored with filing context intact

## Configuration

### Session Manager Settings

In `session_manager.py`:
```python
SessionManager(
    max_sessions=1000,        # Maximum concurrent sessions
    session_ttl_hours=24      # Hours before session expires
)
```

### Conversation History Limit

In `ChatSession.add_message()`:
```python
# Keep only last 20 messages for memory efficiency
if len(self.conversation_history) > 20:
    self.conversation_history = self.conversation_history[-20:]
```

## API Endpoints

### POST /api/chat

Request with session:
```json
{
    "query": "What was the revenue?",
    "session_id": "uuid-here",
    "filters": {}
}
```

Response includes session:
```json
{
    "response": "The revenue was...",
    "session_id": "uuid-here",
    "sources": [...],
    "timestamp": "2024-01-01T00:00:00Z",
    "metadata": {...}
}
```

### Health Check

GET `/health` now includes session manager stats:
```json
{
    "status": "healthy",
    "services": {
        "session_manager": {
            "total_sessions": 5,
            "max_sessions": 1000,
            "session_ttl_hours": 24
        }
    }
}
```

## Testing

A test script is provided at `test_session_memory.py` to verify:
1. Session creation and maintenance
2. Context preservation across queries
3. Session isolation (new sessions don't share context)
4. Filing context retention

Run tests:
```bash
docker-compose exec backend python test_session_memory.py
```

## Benefits

1. **Improved User Experience**: Natural multi-turn conversations without repeating context
2. **Efficient Retrieval**: Automatic filtering to relevant filing when context is established
3. **Persistence**: Sessions survive page refreshes
4. **Resource Management**: Automatic cleanup of old sessions
5. **Scalability**: Configurable limits and TTL settings

## Future Enhancements

1. **Redis Backend**: Move session storage to Redis for horizontal scaling
2. **Session Export**: Allow users to save/export conversation history
3. **Multi-Filing Context**: Support comparing multiple filings in same session
4. **User Authentication**: Tie sessions to authenticated users
5. **Analytics**: Track session metrics and usage patterns

## Troubleshooting

### Session Not Persisting

1. Check browser localStorage is enabled
2. Verify backend session manager is running (check `/health` endpoint)
3. Check session TTL hasn't expired

### Context Not Maintained

1. Ensure session_id is being sent with requests
2. Check filing accession number format matches pattern: `\d{10}-\d{2}-\d{6}`
3. Verify filing exists in database

### Memory Issues

1. Reduce max_sessions if server has limited memory
2. Decrease conversation history limit
3. Reduce session TTL for faster cleanup

---

Implementation completed as requested to solve the issue: "upon manually submitting a query following its initial analysis of a given filing, it unexpectedly 'forgets' which filing we were discussing already."