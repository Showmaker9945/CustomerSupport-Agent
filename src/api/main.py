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
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import settings
from ..conversation.support_agent import get_support_agent, peek_support_agent
from ..db.repositories import (
    get_conversation_thread,
    list_thread_messages,
    list_user_conversation_messages,
)
from ..db.demo_seed import load_seed_tickets
from ..tools.support_tools import (
    get_latest_invoice_record,
    get_subscription_record,
    list_ticket_records,
)

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """REST 聊天请求。"""

    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, max_length=6000)
    thread_id: Optional[str] = Field(None, description="LangGraph 线程 ID")
    session_id: Optional[str] = Field(None, description="向后兼容字段，将映射到 thread_id")
    debug: bool = Field(False, description="是否返回轻量调试信息（路径、耗时、LangSmith 链接）。")


class SentimentPayload(BaseModel):
    label: str
    polarity: float
    frustration_score: float


class ResultPayload(BaseModel):
    escalated: bool
    ticket_created: Optional[str]
    sources: List[str]
    citations: List[str]


class ApprovalToolPayload(BaseModel):
    id: Optional[str]
    tool: str
    tool_label: str
    reason: str
    args_preview: Dict[str, Any] = Field(default_factory=dict)
    allowed_decisions: List[str] = Field(default_factory=list)


class ApprovalPayload(BaseModel):
    count: int
    required_decisions: int
    tools: List[ApprovalToolPayload]
    message: str


class DebugPayload(BaseModel):
    trace_id: Optional[str]
    route_path: List[str]
    node_timings: List[Dict[str, Any]] = Field(default_factory=list)
    total_duration_ms: Optional[float] = None
    langsmith: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """更适合演示与调试的聊天响应。"""

    message: str
    thread_id: str
    run_status: str
    active_agent: str
    intent: str
    sentiment: SentimentPayload
    result: ResultPayload
    next_action: str
    approval: Optional[ApprovalPayload] = None
    debug: Optional[DebugPayload] = None
    timestamp: str


class ResumeDecision(BaseModel):
    """HITL 决策。"""

    type: str = Field(..., description="approve | edit | reject")
    edited_action: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ResumeRequest(BaseModel):
    """恢复请求体。"""

    decisions: List[ResumeDecision] = Field(default_factory=list)
    debug: bool = Field(False, description="是否返回轻量调试信息（路径、耗时、LangSmith 链接）。")


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


class ThreadPayload(BaseModel):
    thread_id: str
    user_id: str
    status: str
    title: str
    message_count: int
    rolling_summary: str
    last_active_agent: Optional[str]
    pending_role: Optional[str]
    pending_approval: bool
    graph_thread_id: Optional[str]
    last_graph_node: Optional[str]
    last_checkpoint_at: Optional[str]
    trace_id: Optional[str]
    created_at: str
    updated_at: str
    last_message_at: Optional[str]


class ThreadDetailResponse(BaseModel):
    thread: ThreadPayload


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    title: str
    status: str
    visible_only: bool
    limit: Optional[int]
    count: int
    messages: List[Dict[str, Any]]


def _load_demo_tickets(user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """只读接口的兜底数据源，避免本地数据库异常时接口长时间卡住。"""
    tickets = [ticket for ticket in load_seed_tickets() if ticket.get("user_id") == user_id]
    if status:
        tickets = [ticket for ticket in tickets if ticket.get("status") == status]
    tickets.sort(
        key=lambda ticket: (ticket.get("created_at", ""), ticket.get("ticket_id", "")),
        reverse=True,
    )
    return tickets


def _build_chat_response(payload: Dict[str, Any]) -> ChatResponse:
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return ChatResponse(**payload)


def _build_thread_payload(thread: Dict[str, Any]) -> ThreadPayload:
    return ThreadPayload(
        thread_id=thread["thread_id"],
        user_id=thread["user_id"],
        status=thread.get("status", "active"),
        title=thread.get("title") or "",
        message_count=int(thread.get("message_count") or 0),
        rolling_summary=thread.get("rolling_summary") or "",
        last_active_agent=thread.get("last_active_agent"),
        pending_role=thread.get("pending_role") or None,
        pending_approval=bool(thread.get("pending_role") and thread.get("pending_state")),
        graph_thread_id=thread.get("graph_thread_id"),
        last_graph_node=thread.get("last_graph_node"),
        last_checkpoint_at=thread.get("last_checkpoint_at"),
        trace_id=thread.get("trace_id"),
        created_at=thread.get("created_at") or "",
        updated_at=thread.get("updated_at") or "",
        last_message_at=thread.get("last_message_at"),
    )


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    with suppress(Exception):
        from shared.security import SensitiveDataFilter

        logging.getLogger().addFilter(SensitiveDataFilter())

    cleanup_task = asyncio.create_task(_cleanup_task())
    warmup_task = asyncio.create_task(asyncio.to_thread(get_support_agent))
    app.state.cleanup_task = cleanup_task
    app.state.warmup_task = warmup_task

    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task

        if not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

        agent = peek_support_agent()
        if agent:
            with suppress(Exception):
                agent.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="LangGraph 多 Agent 客服服务",
    lifespan=lifespan,
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


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(message: ChatMessage) -> ChatResponse:
    """核心聊天接口。"""
    agent = get_support_agent()
    thread_id = message.thread_id or message.session_id
    response = agent.chat(user_id=message.user_id, message=message.content, thread_id=thread_id)
    payload = response.to_dict(include_debug=message.debug)
    payload["thread_id"] = response.thread_id or ""
    return _build_chat_response(payload)


@app.post("/runs/{thread_id}/resume", response_model=ChatResponse, response_model_exclude_none=True)
async def resume_run(thread_id: str, request: ResumeRequest) -> ChatResponse:
    """恢复 HITL 中断。"""
    agent = get_support_agent()
    decisions = [decision.model_dump(exclude_none=True) for decision in request.decisions]
    response = agent.resume(thread_id=thread_id, decisions=decisions)
    payload = response.to_dict(include_debug=request.debug)
    payload["thread_id"] = response.thread_id or thread_id
    return _build_chat_response(payload)


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
    try:
        tickets = list_ticket_records(user_id, status=status)
    except Exception as error:
        logger.warning(f"Ticket query fallback engaged for user={user_id}: {error}")
        tickets = _load_demo_tickets(user_id, status=status)
    return {"user_id": user_id, "tickets": tickets, "count": len(tickets)}


@app.get("/users/{user_id}/subscription")
async def get_user_subscription(user_id: str):
    subscription = get_subscription_record(user_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="未找到该用户的订阅信息")
    return {"user_id": user_id, "subscription": subscription}


@app.get("/users/{user_id}/invoices/latest")
async def get_user_latest_invoice(user_id: str):
    invoice = get_latest_invoice_record(user_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="未找到该用户的最新账单")
    return {"user_id": user_id, "invoice": invoice}


@app.get("/users/{user_id}/history")
async def get_conversation_history(user_id: str, limit: int = 20, thread_id: Optional[str] = None):
    messages = list_user_conversation_messages(
        user_id=user_id,
        limit=max(1, limit),
        thread_id=thread_id,
    )
    return {"user_id": user_id, "messages": messages, "count": len(messages)}


@app.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread_detail(thread_id: str) -> ThreadDetailResponse:
    """查询单个线程的摘要元数据，不触发 Agent 初始化。"""
    thread = get_conversation_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="未找到对应线程")
    return ThreadDetailResponse(thread=_build_thread_payload(thread))


@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_transcript(
    thread_id: str,
    limit: Optional[int] = None,
    visible_only: bool = True,
) -> ThreadMessagesResponse:
    """按线程查询 transcript，默认只返回用户可见消息。"""
    thread = get_conversation_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="未找到对应线程")

    normalized_limit = max(1, limit) if limit is not None else None
    messages = list_thread_messages(
        thread_id,
        limit=normalized_limit,
        visible_only=visible_only,
    )
    return ThreadMessagesResponse(
        thread_id=thread_id,
        title=thread.get("title") or "",
        status=thread.get("status", "active"),
        visible_only=visible_only,
        limit=normalized_limit,
        count=len(messages),
        messages=messages,
    )


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
            "subscription": "/users/{user_id}/subscription (GET)",
            "latest_invoice": "/users/{user_id}/invoices/latest (GET)",
            "history": "/users/{user_id}/history (GET)",
            "thread_detail": "/threads/{thread_id} (GET)",
            "thread_messages": "/threads/{thread_id}/messages (GET)",
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
