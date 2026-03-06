"""
CustomerSupport-Agent FastAPI 入口。

新特性：
- 多 Agent 对话入口（/chat）
- HITL 恢复入口（/runs/{thread_id}/resume）
- SSE 流式事件（/chat/stream）
- WebSocket 事件流（/ws/chat/{user_id}）
- 知识库重建（/knowledge/reindex）
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..conversation.support_agent import get_support_agent, peek_support_agent
from ..tools.support_tools import TicketStore, get_ticket_store

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """REST 聊天请求。"""

    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=6000)
    thread_id: Optional[str] = Field(None, description="LangGraph 线程 ID")
    session_id: Optional[str] = Field(None, description="向后兼容字段，将映射到 thread_id")


class ChatResponse(BaseModel):
    """聊天响应（兼容旧字段 + 新字段）。"""

    message: str
    intent: str
    sentiment: str
    sentiment_polarity: float
    frustration_score: float
    sources: List[str]
    escalated: bool
    ticket_created: Optional[str]
    timestamp: str
    thread_id: str
    run_status: str
    interrupts: List[Dict[str, Any]]
    citations: List[str]
    active_agent: str
    trace_id: str


class ResumeDecision(BaseModel):
    """HITL 决策。"""

    type: str = Field(..., description="approve | edit | reject")
    edited_action: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ResumeRequest(BaseModel):
    """恢复请求体。"""

    decisions: List[ResumeDecision] = Field(default_factory=list)


class ReindexRequest(BaseModel):
    """知识库重建请求。"""

    clear_existing: bool = False


class FeedbackRequest(BaseModel):
    """反馈请求。"""

    user_id: str
    session_id: str
    message_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=1000)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]


class ConnectionManager:
    """管理 WebSocket 会话与连接。"""

    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str, session_id: str) -> bool:
        await websocket.accept()
        async with self._lock:
            self.active_connections.setdefault(user_id, set())
            if len(self.active_connections[user_id]) >= settings.max_ws_connections_per_user:
                await websocket.close(code=1008, reason="Too many connections")
                return False

            self.active_connections[user_id].add(websocket)
            self.sessions[session_id] = {
                "user_id": user_id,
                "connected_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "message_count": 0,
            }
            self.user_sessions.setdefault(user_id, set()).add(session_id)
        return True

    async def disconnect(self, websocket: WebSocket, user_id: str, session_id: str) -> None:
        async with self._lock:
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]

            if session_id in self.sessions:
                self.sessions[session_id]["disconnected_at"] = datetime.now(timezone.utc).isoformat()
                del self.sessions[session_id]

            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

    async def send(self, websocket: WebSocket, payload: Dict[str, Any]) -> None:
        await websocket.send_json(payload)

    async def send_typing_indicator(self, user_id: str, typing: bool) -> None:
        payload = {
            "type": "typing",
            "typing": typing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for websocket in list(self.active_connections.get(user_id, set())):
            with suppress(Exception):
                await websocket.send_json(payload)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def get_user_sessions(self, user_id: str) -> List[str]:
        return list(self.user_sessions.get(user_id, set()))

    async def cleanup_stale_sessions(self, timeout_hours: int = 24) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
        stale = []
        async with self._lock:
            for session_id, session in list(self.sessions.items()):
                try:
                    last_activity = datetime.fromisoformat(session["last_activity"])
                except Exception:
                    continue
                if last_activity < cutoff:
                    stale.append(session_id)
        if stale:
            logger.info(f"Stale sessions detected: {len(stale)}")


manager = ConnectionManager()


async def _cleanup_task() -> None:
    while True:
        try:
            await manager.cleanup_stale_sessions(timeout_hours=settings.session_timeout_hours)
        except Exception as error:
            logger.error(f"Cleanup task error: {error}")
        await asyncio.sleep(3600)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="LangGraph 多 Agent 客服服务",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from shared.rate_limit import RateLimitExceeded, limiter, rate_limit_exception_handler

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
except ImportError:
    logger.warning("Rate limiting module not available.")


@app.on_event("startup")
async def startup_event() -> None:
    with suppress(Exception):
        from shared.security import SensitiveDataFilter

        logging.getLogger().addFilter(SensitiveDataFilter())
    asyncio.create_task(_cleanup_task())
    asyncio.create_task(asyncio.to_thread(get_support_agent))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    agent = peek_support_agent()
    if agent:
        with suppress(Exception):
            agent.close()


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage) -> ChatResponse:
    """核心聊天接口。"""
    agent = get_support_agent()
    thread_id = message.thread_id or message.session_id
    response = agent.chat(user_id=message.user_id, message=message.content, thread_id=thread_id)
    return ChatResponse(
        message=response.message,
        intent=response.intent,
        sentiment=response.sentiment.label,
        sentiment_polarity=response.sentiment.polarity,
        frustration_score=response.sentiment.frustration_score,
        sources=response.sources,
        escalated=response.escalated,
        ticket_created=response.ticket_created,
        timestamp=datetime.now(timezone.utc).isoformat(),
        thread_id=response.thread_id or "",
        run_status=response.run_status,
        interrupts=response.interrupts,
        citations=response.citations,
        active_agent=response.active_agent,
        trace_id=response.trace_id or "",
    )


@app.post("/runs/{thread_id}/resume", response_model=ChatResponse)
async def resume_run(thread_id: str, request: ResumeRequest) -> ChatResponse:
    """恢复 HITL 中断。"""
    agent = get_support_agent()
    decisions = [decision.model_dump(exclude_none=True) for decision in request.decisions]
    response = agent.resume(thread_id=thread_id, decisions=decisions)
    return ChatResponse(
        message=response.message,
        intent=response.intent,
        sentiment=response.sentiment.label,
        sentiment_polarity=response.sentiment.polarity,
        frustration_score=response.sentiment.frustration_score,
        sources=response.sources,
        escalated=response.escalated,
        ticket_created=response.ticket_created,
        timestamp=datetime.now(timezone.utc).isoformat(),
        thread_id=response.thread_id or thread_id,
        run_status=response.run_status,
        interrupts=response.interrupts,
        citations=response.citations,
        active_agent=response.active_agent,
        trace_id=response.trace_id or "",
    )


@app.get("/chat/stream")
async def chat_stream(user_id: str, content: str, thread_id: Optional[str] = None) -> EventSourceResponse:
    """SSE 事件流（token/node/interrupt）。"""
    agent = get_support_agent()

    async def event_generator():
        for event in agent.stream_chat(user_id=user_id, message=content, thread_id=thread_id):
            yield {"event": event.get("type", "message"), "data": json.dumps(event, ensure_ascii=False)}
            await asyncio.sleep(0)

    return EventSourceResponse(event_generator())


@app.post("/knowledge/reindex")
async def knowledge_reindex(request: ReindexRequest):
    """触发知识库重建。"""
    agent = get_support_agent()
    result = agent.reindex_knowledge(clear_existing=request.clear_existing)
    return {
        "status": "success",
        "message": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str) -> None:
    session_id = str(uuid.uuid4())
    if not await manager.connect(websocket, user_id, session_id):
        return

    await manager.send(
        websocket,
        {
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    agent = get_support_agent()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")
            thread_id = data.get("thread_id")

            if msg_type == "ping":
                await manager.send(websocket, {"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
                continue

            if msg_type == "resume":
                decisions = data.get("decisions", [])
                response = agent.resume(thread_id=thread_id, decisions=decisions)
                await manager.send(websocket, {"type": "resume_result", "payload": response.to_dict()})
                continue

            content = data.get("content", "")
            if not content:
                await manager.send(websocket, {"type": "error", "error": "content is required"})
                continue

            await manager.send_typing_indicator(user_id, True)
            for event in agent.stream_chat(user_id=user_id, message=content, thread_id=thread_id):
                await manager.send(websocket, event)
            await manager.send_typing_indicator(user_id, False)

            if session_id in manager.sessions:
                manager.sessions[session_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
                manager.sessions[session_id]["message_count"] += 1
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, session={session_id}")
    finally:
        await manager.disconnect(websocket, user_id, session_id)


@app.get("/users/{user_id}/tickets")
async def get_user_tickets(user_id: str, status: Optional[str] = None):
    ticket_store: TicketStore = get_ticket_store()
    tickets = ticket_store.get_user_tickets(user_id, status=status)
    return {"user_id": user_id, "tickets": [ticket.to_dict() for ticket in tickets], "count": len(tickets)}


@app.get("/users/{user_id}/history")
async def get_conversation_history(user_id: str, limit: int = 20):
    agent = get_support_agent()
    messages = agent.get_conversation_history(user_id=user_id, limit=limit)
    return {"user_id": user_id, "messages": messages, "count": len(messages)}


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    logger.info(
        f"Feedback received: user={feedback.user_id}, rating={feedback.rating}, session={feedback.session_id}"
    )
    return {
        "status": "success",
        "message": "感谢你的反馈，我们会持续优化。",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    components = {"api": "healthy", "agent": "healthy", "knowledge_base": "healthy", "memory": "healthy"}
    agent = peek_support_agent()
    if agent is None:
        components["agent"] = "initializing"
    return HealthResponse(
        status="healthy" if all(value == "healthy" for value in components.values()) else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.app_version,
        components=components,
    )


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    session = manager.get_session_info(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.get("/users/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    sessions = manager.get_user_sessions(user_id)
    return {"user_id": user_id, "active_sessions": sessions, "count": len(sessions)}


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now(timezone.utc).isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务错误",
            "detail": str(exc) if settings.debug else "请求处理失败",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "chat": "/chat (POST)",
            "resume": "/runs/{thread_id}/resume (POST)",
            "sse": "/chat/stream (GET)",
            "websocket": "/ws/chat/{user_id}",
            "knowledge_reindex": "/knowledge/reindex (POST)",
            "tickets": "/users/{user_id}/tickets (GET)",
            "history": "/users/{user_id}/history (GET)",
            "health": "/health (GET)",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
