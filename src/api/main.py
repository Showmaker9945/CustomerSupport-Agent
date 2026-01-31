"""
FastAPI application for CustomerSupport-Agent.

Provides REST API and WebSocket endpoints for real-time customer support.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    status,
    Depends,
    BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from ..config import settings
from ..conversation.support_agent import get_support_agent
from ..tools.support_tools import get_ticket_store, TicketStore

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatMessage(BaseModel):
    """Incoming chat message."""
    user_id: str = Field(..., min_length=1, description="User identifier")
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")
    session_id: Optional[str] = Field(None, description="Session identifier for continuity")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    message: str
    intent: str
    sentiment: str
    sentiment_polarity: float
    frustration_score: float
    sources: List[str]
    escalated: bool
    ticket_created: Optional[str]
    timestamp: str


class FeedbackRequest(BaseModel):
    """Feedback on AI response."""
    user_id: str
    session_id: str
    message_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, max_length=1000)


class TicketListResponse(BaseModel):
    """Response for user tickets."""
    user_id: str
    tickets: List[Dict]
    count: int


class ConversationHistoryResponse(BaseModel):
    """Response for conversation history."""
    user_id: str
    messages: List[Dict]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time chat.

    Handles connection lifecycle, message broadcasting, and session tracking.
    """

    def __init__(self):
        """Initialize connection manager."""
        # Active WebSocket connections per user
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # Session tracking
        self.sessions: Dict[str, Dict] = {}

        # User sessions mapping
        self.user_sessions: Dict[str, Set[str]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str, session_id: str) -> bool:
        """
        Connect a WebSocket for a user.

        Args:
            websocket: WebSocket connection
            user_id: User identifier
            session_id: Session identifier

        Returns:
            True if connected successfully
        """
        await websocket.accept()

        async with self._lock:
            # Initialize user connections set if needed
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()

            # Check connection limit per user
            if len(self.active_connections[user_id]) >= settings.max_ws_connections_per_user:
                await websocket.close(code=1008, reason="Too many connections")
                return False

            # Add connection
            self.active_connections[user_id].add(websocket)

            # Track session
            self.sessions[session_id] = {
                "user_id": user_id,
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "message_count": 0
            }

            # Track user's sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)

        logger.info(f"WebSocket connected: user={user_id}, session={session_id}")
        return True

    async def disconnect(self, websocket: WebSocket, user_id: str, session_id: str):
        """
        Disconnect a WebSocket.

        Args:
            websocket: WebSocket connection
            user_id: User identifier
            session_id: Session identifier
        """
        async with self._lock:
            # Remove connection
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]

            # Remove session tracking
            if session_id in self.sessions:
                # Update final stats
                self.sessions[session_id]["disconnected_at"] = datetime.now(timezone.utc).isoformat()
                del self.sessions[session_id]

            # Remove from user sessions
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

        logger.info(f"WebSocket disconnected: user={user_id}, session={session_id}")

    async def send_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message dictionary
            websocket: Target WebSocket
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

    async def broadcast_to_user(self, user_id: str, message: dict):
        """
        Broadcast a message to all connections for a user.

        Args:
            user_id: User identifier
            message: Message dictionary
        """
        if user_id not in self.active_connections:
            return

        # Send to all connections for this user
        disconnected = []
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
                disconnected.append(websocket)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.active_connections[user_id].discard(ws)

    async def send_typing_indicator(self, user_id: str, typing: bool):
        """
        Send typing indicator to user's connections.

        Args:
            user_id: User identifier
            typing: Whether agent is typing
        """
        await self.broadcast_to_user(user_id, {
            "type": "typing",
            "typing": typing,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def update_session_activity(self, session_id: str):
        """
        Update session activity timestamp.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
            self.sessions[session_id]["message_count"] += 1

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        return self.sessions.get(session_id)

    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get active session IDs for a user."""
        return list(self.user_sessions.get(user_id, set()))

    async def cleanup_stale_sessions(self, timeout_hours: int = 24):
        """
        Clean up stale sessions older than timeout.

        Args:
            timeout_hours: Hours before session is considered stale
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
        stale_sessions = []

        async with self._lock:
            for session_id, session_data in list(self.sessions.items()):
                try:
                    last_activity = datetime.fromisoformat(session_data["last_activity"])
                    if last_activity < cutoff:
                        stale_sessions.append(session_id)
                except (KeyError, ValueError):
                    continue

        # Log stale sessions (actual cleanup happens on disconnect)
        if stale_sessions:
            logger.info(f"Found {len(stale_sessions)} stale sessions")


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Customer Support AI Agent with real-time chat capabilities"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Rate Limiting
# ============================================================================

try:
    from shared.rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    logger.info("Rate limiting enabled")
except ImportError:
    logger.warning("Shared rate limiting module not available - rate limiting disabled")


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def cleanup_task():
    """Background task to clean up stale sessions."""
    while True:
        try:
            await manager.cleanup_stale_sessions(timeout_hours=settings.session_timeout_hours)
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(3600)


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Install security filter to prevent API key exposure in logs
    try:
        from shared.security import SensitiveDataFilter
        root_logger = logging.getLogger()
        root_logger.addFilter(SensitiveDataFilter())
        logger.info("Security filter installed - API keys will be redacted from logs")
    except ImportError:
        logger.warning("Shared security module not available - API keys may be exposed in logs")

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Start background cleanup task
    asyncio.create_task(cleanup_task())

    # Initialize support agent
    get_support_agent()

    logger.info("Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")
    logger.info("Application shutdown complete")


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    Real-time chat endpoint via WebSocket.

    Connection URL: ws://localhost:8000/ws/chat/{user_id}

    Message format (client → server):
    {
        "type": "message",
        "content": "Your message here",
        "session_id": "optional-session-id"
    }

    Response format (server → client):
    {
        "type": "response",
        "content": "Agent response",
        "metadata": {...}
    }
    """
    # Generate session ID
    session_id = str(uuid.uuid4())

    # Connect WebSocket
    connected = await manager.connect(websocket, user_id, session_id)
    if not connected:
        return

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Get support agent
        agent = get_support_agent()

        # Message loop
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type", "message")
            content = data.get("content", "")

            if message_type == "ping":
                # Respond to ping
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                continue

            if message_type != "message" or not content:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                continue

            # Update session activity
            manager.update_session_activity(session_id)

            # Send typing indicator
            await manager.send_typing_indicator(user_id, True)

            try:
                # Get response from agent
                response = agent.chat(user_id=user_id, message=content)

                # Send typing indicator (done)
                await manager.send_typing_indicator(user_id, False)

                # Send response
                await websocket.send_json({
                    "type": "response",
                    "content": response.message,
                    "metadata": {
                        "intent": response.intent,
                        "sentiment": {
                            "label": response.sentiment.label,
                            "polarity": response.sentiment.polarity,
                            "frustration_score": response.sentiment.frustration_score
                        },
                        "sources": response.sources,
                        "escalated": response.escalated,
                        "ticket_created": response.ticket_created
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await manager.send_typing_indicator(user_id, False)
                await websocket.send_json({
                    "type": "error",
                    "error": "Failed to process message",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket, user_id, session_id)


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Send a message to the support agent",
    description="Alternative to WebSocket for simple request-response interactions"
)
async def chat(message: ChatMessage) -> ChatResponse:
    """
    Process a chat message and return agent response.

    This is a REST alternative to the WebSocket endpoint for simple
    request-response patterns.
    """
    try:
        # Get agent
        agent = get_support_agent()

        # Use user_id from request
        user_id = message.user_id

        # Get response
        response = agent.chat(user_id=user_id, message=message.content)

        return ChatResponse(
            message=response.message,
            intent=response.intent,
            sentiment=response.sentiment.label,
            sentiment_polarity=response.sentiment.polarity,
            frustration_score=response.sentiment.frustration_score,
            sources=response.sources,
            escalated=response.escalated,
            ticket_created=response.ticket_created,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@app.get(
    "/users/{user_id}/tickets",
    response_model=TicketListResponse,
    responses={
        404: {"model": ErrorResponse}
    },
    summary="Get user's support tickets",
    description="Retrieve all support tickets for a specific user"
)
async def get_user_tickets(
    user_id: str,
    status: Optional[str] = None
) -> TicketListResponse:
    """
    Get all tickets for a user.

    Query parameters:
    - status: Optional filter by ticket status (open, resolved, etc.)
    """
    try:
        ticket_store: TicketStore = get_ticket_store()
        tickets = ticket_store.get_user_tickets(user_id, status=status)

        return TicketListResponse(
            user_id=user_id,
            tickets=[t.to_dict() for t in tickets],
            count=len(tickets)
        )

    except Exception as e:
        logger.error(f"Get tickets error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve tickets: {str(e)}"
        )


@app.get(
    "/users/{user_id}/history",
    response_model=ConversationHistoryResponse,
    responses={
        404: {"model": ErrorResponse}
    },
    summary="Get user's conversation history",
    description="Retrieve recent conversation history for a user"
)
async def get_conversation_history(
    user_id: str,
    limit: int = 20
) -> ConversationHistoryResponse:
    """
    Get conversation history for a user.

    Query parameters:
    - limit: Maximum number of messages to return (default: 20)
    """
    try:
        agent = get_support_agent()
        messages = agent.get_conversation_history(user_id, limit=limit)

        return ConversationHistoryResponse(
            user_id=user_id,
            messages=messages,
            count=len(messages)
        )

    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.post(
    "/feedback",
    summary="Submit feedback on AI response",
    description="Rate the quality of an AI response for improvement"
)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on an AI response.

    Used for improving the system over time.
    """
    try:
        # In production, store in database
        logger.info(
            f"Feedback received: user={feedback.user_id}, "
            f"rating={feedback.rating}, session={feedback.session_id}"
        )

        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check system health and component status"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.
    """
    components = {
        "api": "healthy",
        "agent": "healthy",
        "knowledge_base": "healthy",
        "sentiment_analyzer": "healthy"
    }

    try:
        # Check agent
        agent = get_support_agent()
        if agent is None:
            components["agent"] = "unhealthy"

        # Check FAQ store
        if agent.faq_store is None:
            components["knowledge_base"] = "unhealthy"

        # Check sentiment analyzer
        if agent.sentiment_analyzer is None:
            components["sentiment_analyzer"] = "disabled"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        components["api"] = "unhealthy"

    all_healthy = all(status == "healthy" for status in components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.app_version,
        components=components
    )


@app.get(
    "/sessions/{session_id}",
    summary="Get session information",
    description="Retrieve information about a WebSocket session"
)
async def get_session_info(session_id: str):
    """
    Get session information.
    """
    session = manager.get_session_info(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return session


@app.get(
    "/users/{user_id}/sessions",
    summary="Get user's active sessions",
    description="Retrieve all active session IDs for a user"
)
async def get_user_sessions(user_id: str):
    """
    Get active sessions for a user.
    """
    sessions = manager.get_user_sessions(user_id)

    return {
        "user_id": user_id,
        "active_sessions": sessions,
        "count": len(sessions)
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", summary="API information", description="Get API information and endpoints")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "websocket": "/ws/chat/{user_id}",
            "chat": "/chat (POST)",
            "tickets": "/users/{user_id}/tickets (GET)",
            "history": "/users/{user_id}/history (GET)",
            "feedback": "/feedback (POST)",
            "health": "/health (GET)",
            "docs": "/docs (Swagger UI)"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
