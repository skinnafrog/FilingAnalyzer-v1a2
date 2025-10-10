"""
FastAPI server for Financial Intelligence Platform.
Provides REST API and WebSocket endpoints for chat interface.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import uuid

from ..config.settings import get_settings
from ..knowledge.rag_pipeline import RAGPipeline
from ..models.filing import SECFiling
from ..database import init_db, get_db
from ..database.models import Filing, Company, FilingDocument, FilingChunk
from .session_manager import session_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Celery tasks for ingestion control
try:
    from celery import Celery

    # Create a Celery client configured for the API
    celery_client = Celery('financial_intel')
    settings = get_settings()
    celery_client.conf.update(
        broker_url=settings.CELERY_BROKER_URL,
        result_backend=settings.CELERY_RESULT_BACKEND,
    )

    # Import task functions
    from ..tasks import is_market_hours, get_ingestion_status
    celery_available = True

    # Create a reference to the manual_ingest task
    manual_ingest = celery_client.signature('src.tasks.manual_ingest', queue='ingestion')

except ImportError:
    celery_available = False
    logger.warning("Celery tasks not available")

# Create FastAPI app
app = FastAPI(
    title="Financial Intelligence Platform API",
    description="AI-powered SEC filing analysis with chat interface",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
settings = get_settings()
rag_pipeline = RAGPipeline(settings)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and create tables."""
    try:
        init_db()
        logger.info("Database initialized successfully")

        # Start session manager
        await session_manager.start()
        logger.info("Session manager started")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")


# Request/Response models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[List[ChatMessage]] = []
    filters: Optional[Dict[str, Any]] = {}  # e.g., {"company": "AAPL", "form_type": "10-K"}
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]  # Retrieved document chunks
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any]


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = {}


class FilingInfo(BaseModel):
    accession_number: str
    company_name: str
    cik_number: Optional[str] = None
    ticker_symbol: Optional[str] = None
    form_type: str
    filing_date: str
    period_date: Optional[str] = None
    document_count: int
    status: str
    file_size_mb: Optional[float] = None
    processing_stage: Optional[str] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None


# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)


manager = ConnectionManager()


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Financial Intelligence Platform API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/chat",
            "search": "/api/search",
            "filings": "/api/filings",
            "websocket": "/ws/{client_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "api": "running",
            "rag_pipeline": rag_pipeline.get_statistics() if rag_pipeline else None,
            "session_manager": session_manager.get_stats()
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for querying SEC filings.
    Supports both standard and streaming responses.
    """
    try:
        # Get or create session
        session = session_manager.get_or_create_session(request.session_id)
        session_id = session.session_id

        # Get conversation context from session
        conversation_context = session_manager.get_conversation_context(session_id)

        # Process query through RAG pipeline
        logger.info(f"Processing chat query: {request.query[:100]}... (session: {session_id})")

        # Check if query mentions analyzing a specific filing
        import re
        if "analyze" in request.query.lower() or request.filters.get("accession_number"):
            acc_number = request.filters.get("accession_number")
            if not acc_number:
                # Try to extract from query
                acc_pattern = r'\d{10}-\d{2}-\d{6}'
                match = re.search(acc_pattern, request.query)
                if match:
                    acc_number = match.group()

            if acc_number:
                # Retrieve filing details for context
                db = next(get_db())
                filing = db.query(Filing).filter(Filing.accession_number == acc_number).first()
                if filing:
                    # Set filing context in session
                    filing_context = {
                        "accession_number": filing.accession_number,
                        "company_name": filing.company.name if filing.company else "Unknown",
                        "form_type": filing.form_type,
                        "filing_date": filing.filing_date.isoformat() if filing.filing_date else "",
                        "ticker_symbol": filing.company.ticker if filing.company else None,
                        "cik_number": filing.company.cik if filing.company else None
                    }
                    session_manager.update_filing_context(session_id, filing_context)
                    # Add this as a filter for retrieval
                    request.filters["accession_number"] = acc_number
                db.close()

        # Retrieve relevant documents - prioritize current filing context if available
        filters = request.filters.copy()
        if session.current_filing and not filters.get("accession_number"):
            # If we have a filing context but no explicit filter, add it
            filters["accession_number"] = session.current_filing.accession_number

        retrieved_docs = await rag_pipeline.retrieve(
            query=request.query,
            filters=filters,
            limit=5
        )

        # Generate response with full conversation context
        response = await rag_pipeline.generate_response(
            query=request.query,
            context=retrieved_docs,
            chat_history=conversation_context  # Use session conversation context
        )

        # Add interaction to session
        session_manager.add_interaction(
            session_id=session_id,
            user_message=request.query,
            assistant_response=response,
            sources=retrieved_docs
        )

        # Format sources
        sources = [
            {
                "accession_number": doc.get("accession_number"),
                "company_name": doc.get("company_name"),
                "form_type": doc.get("form_type"),
                "excerpt": doc.get("text", "")[:200],
                "relevance_score": doc.get("score", 0)
            }
            for doc in retrieved_docs
        ]

        return ChatResponse(
            response=response,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            metadata={
                "model": settings.LLM_MODEL,
                "tokens_used": len(response.split()),  # Rough estimate
                "retrieval_count": len(retrieved_docs)
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    """
    async def generate():
        try:
            # Get or create session
            session = session_manager.get_or_create_session(request.session_id)
            session_id = session.session_id

            # Get conversation context from session
            conversation_context = session_manager.get_conversation_context(session_id)

            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            # Check for filing context
            import re
            if "analyze" in request.query.lower() or request.filters.get("accession_number"):
                acc_number = request.filters.get("accession_number")
                if not acc_number:
                    acc_pattern = r'\d{10}-\d{2}-\d{6}'
                    match = re.search(acc_pattern, request.query)
                    if match:
                        acc_number = match.group()

                if acc_number:
                    db = next(get_db())
                    filing = db.query(Filing).filter(Filing.accession_number == acc_number).first()
                    if filing:
                        filing_context = {
                            "accession_number": filing.accession_number,
                            "company_name": filing.company.name if filing.company else "Unknown",
                            "form_type": filing.form_type,
                            "filing_date": filing.filing_date.isoformat() if filing.filing_date else "",
                            "ticker_symbol": filing.company.ticker if filing.company else None,
                            "cik_number": filing.company.cik if filing.company else None
                        }
                        session_manager.update_filing_context(session_id, filing_context)
                        request.filters["accession_number"] = acc_number
                    db.close()

            # Retrieve relevant documents with session context
            filters = request.filters.copy()
            if session.current_filing and not filters.get("accession_number"):
                filters["accession_number"] = session.current_filing.accession_number

            retrieved_docs = await rag_pipeline.retrieve(
                query=request.query,
                filters=filters,
                limit=5
            )

            # Send sources
            sources = [
                {
                    "accession_number": doc.get("accession_number"),
                    "company_name": doc.get("company_name"),
                    "form_type": doc.get("form_type"),
                    "excerpt": doc.get("text", "")[:200]
                }
                for doc in retrieved_docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Stream response generation with session context
            full_response = ""
            async for chunk in rag_pipeline.generate_response_stream(
                query=request.query,
                context=retrieved_docs,
                chat_history=conversation_context  # Use session conversation context
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming

            # Add interaction to session after streaming completes
            session_manager.add_interaction(
                session_id=session_id,
                user_message=request.query,
                assistant_response=full_response,
                sources=retrieved_docs
            )

            # Send completion message
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search endpoint for finding relevant documents.
    """
    try:
        results = await rag_pipeline.retrieve(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "accession_number": doc.get("accession_number"),
                    "company_name": doc.get("company_name"),
                    "form_type": doc.get("form_type"),
                    "filing_date": doc.get("filing_date"),
                    "excerpt": doc.get("text", "")[:300],
                    "score": doc.get("score", 0)
                }
                for doc in results
            ]
        }

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filings", response_model=List[FilingInfo])
async def get_filings(
    limit: int = 20,
    offset: int = 0,
    company: Optional[str] = None,
    form_type: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get list of available filings from database with pagination and filtering.
    """
    try:
        # Build query
        query = db.query(Filing).join(Company)

        # Apply filters
        if company:
            query = query.filter(Company.name.ilike(f"%{company}%"))
        if form_type:
            query = query.filter(Filing.form_type == form_type)
        if status:
            query = query.filter(Filing.status == status)
        if start_date:
            query = query.filter(Filing.filing_date >= datetime.fromisoformat(start_date))
        if end_date:
            query = query.filter(Filing.filing_date <= datetime.fromisoformat(end_date))

        # Order by filing date and apply pagination
        filings = query.order_by(Filing.filing_date.desc()).offset(offset).limit(limit).all()

        # Convert to response model with enhanced information
        result = []
        for filing in filings:
            file_size_mb = None
            if hasattr(filing, 'file_size_bytes') and filing.file_size_bytes:
                file_size_mb = filing.file_size_bytes / (1024 * 1024)

            result.append(FilingInfo(
                accession_number=filing.accession_number,
                company_name=filing.company.name if filing.company else "Unknown",
                cik_number=filing.company.cik if filing.company else None,
                ticker_symbol=filing.company.ticker if filing.company and hasattr(filing.company, 'ticker') else None,
                form_type=filing.form_type,
                filing_date=filing.filing_date.isoformat() if filing.filing_date else "",
                period_date=filing.period_date.isoformat() if hasattr(filing, 'period_date') and filing.period_date else None,
                document_count=filing.document_count or 1,
                status=filing.status or "completed",
                file_size_mb=file_size_mb,
                processing_stage=filing.current_stage if hasattr(filing, 'current_stage') else None,
                chunk_count=filing.chunk_count if hasattr(filing, 'chunk_count') else None,
                error_message=filing.error_message if hasattr(filing, 'error_message') else None
            ))

        return result

    except Exception as e:
        logger.error(f"Filings error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filings/stats")
async def get_filing_stats(db: Session = Depends(get_db)):
    """
    Get filing statistics from database.
    """
    try:
        from sqlalchemy import func
        import os
        import shutil

        total_filings = db.query(func.count(Filing.id)).scalar() or 0
        total_companies = db.query(func.count(Company.id)).scalar() or 0

        # Get total chunks for RAG
        total_chunks = db.query(func.count(FilingChunk.id)).scalar() or 0

        # Get filing type distribution
        filing_types = db.query(
            Filing.form_type,
            func.count(Filing.form_type)
        ).group_by(Filing.form_type).all()

        # Get recent activity
        recent_filings = db.query(Filing).order_by(Filing.created_at.desc()).limit(1).first()
        last_updated = recent_filings.created_at.isoformat() if recent_filings and hasattr(recent_filings, 'created_at') else None

        # Get processing status distribution
        status_distribution = db.query(
            Filing.status,
            func.count(Filing.status)
        ).group_by(Filing.status).all()

        # Calculate storage usage
        storage_info = {}

        # Get filing storage size
        filings_dir = os.path.join("data", "filings")
        if os.path.exists(filings_dir):
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(filings_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
                        file_count += 1

            storage_info["filings_storage_mb"] = round(total_size / (1024 * 1024), 2)
            storage_info["total_files"] = file_count
        else:
            storage_info["filings_storage_mb"] = 0
            storage_info["total_files"] = 0

        # Get disk usage for data directory
        data_dir = "data"
        if os.path.exists(data_dir):
            disk_usage = shutil.disk_usage(data_dir)
            storage_info["disk_total_gb"] = round(disk_usage.total / (1024**3), 2)
            storage_info["disk_used_gb"] = round(disk_usage.used / (1024**3), 2)
            storage_info["disk_free_gb"] = round(disk_usage.free / (1024**3), 2)
            storage_info["disk_usage_percent"] = round((disk_usage.used / disk_usage.total) * 100, 1)

        # Get filings processed today
        from datetime import datetime, timedelta
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        filings_today = db.query(func.count(Filing.id)).filter(
            Filing.created_at >= today_start
        ).scalar() or 0

        return {
            "total_filings": total_filings,
            "total_companies": total_companies,
            "total_chunks": total_chunks,
            "filings_today": filings_today,
            "filing_types": {ft[0]: ft[1] for ft in filing_types if ft[0]},
            "status_distribution": {st[0]: st[1] for st in status_distribution if st[0]},
            "storage": storage_info,
            "last_updated": last_updated
        }

    except Exception as e:
        logger.error(f"Stats error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filings/{accession_number}")
async def get_filing_detail(accession_number: str, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific filing.
    """
    try:
        filing = db.query(Filing).filter(Filing.accession_number == accession_number).first()

        if not filing:
            raise HTTPException(status_code=404, detail="Filing not found")

        # Get associated documents if available
        documents = []
        if hasattr(filing, 'documents'):
            documents = [
                {
                    "id": doc.id,
                    "document_type": doc.document_type if hasattr(doc, 'document_type') else None,
                    "url": doc.url if hasattr(doc, 'url') else None,
                    "processed": doc.processed if hasattr(doc, 'processed') else False
                }
                for doc in filing.documents
            ]

        return {
            "accession_number": filing.accession_number,
            "company": {
                "name": filing.company.name if filing.company else "Unknown",
                "cik": filing.company.cik if filing.company else None,
                "ticker": filing.company.ticker if filing.company and hasattr(filing.company, 'ticker') else None
            },
            "form_type": filing.form_type,
            "filing_date": filing.filing_date.isoformat() if filing.filing_date else None,
            "period_date": filing.period_date.isoformat() if hasattr(filing, 'period_date') and filing.period_date else None,
            "acceptance_datetime": filing.acceptance_datetime.isoformat() if hasattr(filing, 'acceptance_datetime') and filing.acceptance_datetime else None,
            "filing_url": filing.filing_url if hasattr(filing, 'filing_url') else None,
            "status": filing.status,
            "processing_stage": filing.current_stage if hasattr(filing, 'current_stage') else None,
            "chunk_count": filing.chunk_count if hasattr(filing, 'chunk_count') else 0,
            "file_size_bytes": filing.file_size_bytes if hasattr(filing, 'file_size_bytes') else None,
            "documents": documents,
            "error_message": filing.error_message if hasattr(filing, 'error_message') else None,
            "created_at": filing.created_at.isoformat() if hasattr(filing, 'created_at') and filing.created_at else None,
            "updated_at": filing.updated_at.isoformat() if hasattr(filing, 'updated_at') and filing.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Filing detail error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingestion/trigger")
async def trigger_ingestion(limit: int = 5):
    """
    Manually trigger SEC filing ingestion.
    """
    if not celery_available:
        raise HTTPException(
            status_code=503,
            detail="Celery service is not available"
        )

    try:
        result = manual_ingest.apply_async(kwargs={'limit': limit})
        return {
            "status": "triggered",
            "task_id": result.id if result else None,
            "limit": limit,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Failed to trigger ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingestion/status")
async def get_ingestion_pipeline_status():
    """
    Get current ingestion pipeline status.
    """
    try:
        if celery_available:
            status = get_ingestion_status()
            market_open = is_market_hours()
        else:
            status = {"status": "unavailable", "celery": False}
            market_open = False

        return {
            "pipeline_status": status,
            "market_hours": market_open,
            "automatic_ingestion": celery_available,
            "poll_interval": settings.RSS_POLL_INTERVAL,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}", exc_info=True)
        return {
            "pipeline_status": {"status": "error", "error": str(e)},
            "market_hours": False,
            "automatic_ingestion": False,
            "timestamp": datetime.utcnow()
        }


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time chat.
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "chat":
                # Process chat message
                query = message.get("query", "")

                # Send acknowledgment
                await manager.send_message(
                    json.dumps({"type": "processing", "status": "retrieving"}),
                    client_id
                )

                # Retrieve documents
                retrieved_docs = await rag_pipeline.retrieve(query=query)

                # Send sources
                await manager.send_message(
                    json.dumps({
                        "type": "sources",
                        "sources": [
                            {
                                "company": doc.get("company_name"),
                                "form_type": doc.get("form_type")
                            }
                            for doc in retrieved_docs[:3]
                        ]
                    }),
                    client_id
                )

                # Generate and stream response
                async for chunk in rag_pipeline.generate_response_stream(
                    query=query,
                    context=retrieved_docs
                ):
                    await manager.send_message(
                        json.dumps({"type": "content", "content": chunk}),
                        client_id
                    )
                    await asyncio.sleep(0.01)

                # Send completion
                await manager.send_message(
                    json.dumps({"type": "complete"}),
                    client_id
                )

            elif message.get("type") == "ping":
                # Respond to ping
                await manager.send_message(
                    json.dumps({"type": "pong"}),
                    client_id
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )