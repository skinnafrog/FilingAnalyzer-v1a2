"""
Session management for maintaining chat context across interactions.

This module handles:
- Session creation and management
- Conversation history storage
- Filing context preservation
- Memory cleanup
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import asyncio
from collections import OrderedDict
import json


@dataclass
class FilingContext:
    """Context about the filing being discussed"""
    accession_number: str
    company_name: str
    form_type: str
    filing_date: str
    ticker_symbol: Optional[str] = None
    cik_number: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    filing_context: Optional[FilingContext] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatSession:
    """Represents a complete chat session"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    current_filing: Optional[FilingContext] = None
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, sources: Optional[List] = None):
        """Add a message to conversation history"""
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            filing_context=self.current_filing,
            sources=sources or []
        )
        self.conversation_history.append(turn)
        self.last_activity = datetime.utcnow()

        # Keep only last 20 messages for memory efficiency
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def set_filing_context(self, filing_data: Dict[str, Any]):
        """Set the current filing being discussed"""
        self.current_filing = FilingContext(
            accession_number=filing_data.get("accession_number", ""),
            company_name=filing_data.get("company_name", ""),
            form_type=filing_data.get("form_type", ""),
            filing_date=filing_data.get("filing_date", ""),
            ticker_symbol=filing_data.get("ticker_symbol"),
            cik_number=filing_data.get("cik_number"),
            metadata=filing_data.get("metadata", {})
        )
        self.last_activity = datetime.utcnow()

    def get_context_for_llm(self, include_system_context: bool = True) -> List[Dict[str, str]]:
        """Get formatted conversation history for LLM"""
        messages = []

        # Add system context about current filing if available
        if include_system_context and self.current_filing:
            system_context = (
                f"Current filing context: {self.current_filing.company_name} "
                f"{self.current_filing.form_type} (Accession: {self.current_filing.accession_number}, "
                f"Filed: {self.current_filing.filing_date})"
            )
            if self.current_filing.ticker_symbol:
                system_context += f", Ticker: {self.current_filing.ticker_symbol}"

            messages.append({
                "role": "system",
                "content": system_context
            })

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })

        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_filing": {
                "accession_number": self.current_filing.accession_number,
                "company_name": self.current_filing.company_name,
                "form_type": self.current_filing.form_type,
                "filing_date": self.current_filing.filing_date,
                "ticker_symbol": self.current_filing.ticker_symbol,
                "cik_number": self.current_filing.cik_number,
                "metadata": self.current_filing.metadata
            } if self.current_filing else None,
            "conversation_history": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp.isoformat(),
                    "sources": turn.sources
                }
                for turn in self.conversation_history
            ],
            "metadata": self.metadata
        }


class SessionManager:
    """Manages chat sessions with memory and context preservation"""

    def __init__(self, max_sessions: int = 1000, session_ttl_hours: int = 24):
        """
        Initialize session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions to maintain
            session_ttl_hours: Hours before a session expires due to inactivity
        """
        self.sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._cleanup_task = None

    async def start(self):
        """Start the session manager with periodic cleanup"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop(self):
        """Stop the session manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break

    def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Create a new chat session"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Clean up if we're at capacity
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            self.sessions.popitem(last=False)

        session = ChatSession(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing session or None if not found/expired"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check if session has expired
        if datetime.utcnow() - session.last_activity > self.session_ttl:
            del self.sessions[session_id]
            return None

        # Move to end (most recently used)
        self.sessions.move_to_end(session_id)

        return session

    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session

        return self.create_session(session_id)

    def update_filing_context(self, session_id: str, filing_data: Dict[str, Any]):
        """Update the filing context for a session"""
        session = self.get_session(session_id)
        if session:
            session.set_filing_context(filing_data)

    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Add a complete interaction (user message + assistant response) to session"""
        session = self.get_session(session_id)
        if session:
            session.add_message("user", user_message)
            session.add_message("assistant", assistant_response, sources)

    def get_conversation_context(
        self,
        session_id: str,
        include_filing_context: bool = True
    ) -> List[Dict[str, str]]:
        """Get formatted conversation context for LLM"""
        session = self.get_session(session_id)
        if not session:
            return []

        return session.get_context_for_llm(include_filing_context)

    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.utcnow()
        expired = []

        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_ttl:
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]

        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        return {
            "total_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "session_ttl_hours": self.session_ttl.total_seconds() / 3600,
            "oldest_session": min(
                (s.created_at for s in self.sessions.values()),
                default=None
            ),
            "newest_session": max(
                (s.created_at for s in self.sessions.values()),
                default=None
            )
        }


# Global session manager instance
session_manager = SessionManager()