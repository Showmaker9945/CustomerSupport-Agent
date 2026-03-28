"""结构化业务数据的建库、种子数据与查询封装。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, inspect, select, text

from .demo_seed import load_seed_bundle
from .models import (
    Base,
    ConversationMessage,
    ConversationThread,
    Invoice,
    InvoiceItem,
    KnowledgeChunk,
    KnowledgeDocument,
    Subscription,
    TicketRecord,
    User,
    UserMemoryRecord,
)
from .session import get_engine, session_scope

_bootstrap_lock = Lock()
_ticket_lock = Lock()
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_SPACE_PATTERN = re.compile(r"\s+")


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _normalize_ws(text: str) -> str:
    """压缩文本中的多余空白，便于标题和 token 估算。"""
    return _SPACE_PATTERN.sub(" ", (text or "").strip())


def estimate_text_tokens(text: str) -> int:
    """轻量估算 transcript token 数，避免引入额外 tokenizer 依赖。"""
    normalized = _normalize_ws(text)
    if not normalized:
        return 0

    chinese_tokens = len(_CJK_PATTERN.findall(normalized))
    english_tokens = sum(max(1, (len(word) + 3) // 4) for word in _WORD_PATTERN.findall(normalized))

    residual = _WORD_PATTERN.sub("", normalized)
    residual = _CJK_PATTERN.sub("", residual)
    residual = _SPACE_PATTERN.sub("", residual)
    residual_tokens = (len(residual) + 3) // 4 if residual else 0

    return chinese_tokens + english_tokens + residual_tokens


def _conversation_title(content: str, limit: int = 48) -> str:
    """从第一条用户消息生成简短线程标题。"""
    normalized = _normalize_ws(content)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _ensure_business_schema() -> None:
    """为已有数据库补齐本轮新增的短期记忆列。"""
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "conversation_threads" not in tables:
        return

    columns = {column["name"] for column in inspector.get_columns("conversation_threads")}
    ddl_by_column = {
        "pending_role": "ALTER TABLE conversation_threads ADD COLUMN pending_role VARCHAR(64)",
        "pending_state_json": "ALTER TABLE conversation_threads ADD COLUMN pending_state_json JSON",
        "trace_id": "ALTER TABLE conversation_threads ADD COLUMN trace_id VARCHAR(64)",
    }
    with engine.begin() as connection:
        for column, ddl in ddl_by_column.items():
            if column not in columns:
                connection.execute(text(ddl))


def ensure_business_database(seed_demo: bool = True) -> None:
    """确保业务表存在，并在需要时自动注入演示数据。"""
    Base.metadata.create_all(get_engine())
    _ensure_business_schema()
    if not seed_demo:
        return

    with _bootstrap_lock:
        with session_scope() as session:
            existing_users = session.scalar(select(func.count()).select_from(User)) or 0
            if existing_users > 0:
                return
        seed_demo_data(clear_existing=False)


def reset_business_database(seed_demo: bool = True) -> None:
    """删除并重建全部业务表。"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    if seed_demo:
        seed_demo_data(clear_existing=False)


def seed_demo_data(clear_existing: bool = False) -> Dict[str, int]:
    """向业务库写入演示数据，并返回导入统计。"""
    Base.metadata.create_all(get_engine())
    bundle = load_seed_bundle()
    stats = {"users": 0, "subscriptions": 0, "invoices": 0, "invoice_items": 0, "tickets": 0}

    with session_scope() as session:
        if clear_existing:
            session.execute(delete(InvoiceItem))
            session.execute(delete(Invoice))
            session.execute(delete(Subscription))
            session.execute(delete(TicketRecord))
            session.execute(delete(User))

        for user_id, payload in bundle["accounts"].items():
            user = session.get(User, user_id) or User(user_id=user_id)
            user.name = payload["name"]
            user.email = payload["email"]
            user.plan = payload.get("plan", "Free")
            user.status = payload.get("status", "active")
            user.member_since = payload.get("member_since", "")
            user.company = payload.get("company")
            user.usage_json = payload.get("usage", {})
            user.billing_json = payload.get("billing", {})
            user.support_history_json = payload.get("support_history", {})
            session.add(user)
            stats["users"] += 1

        for user_id, payload in bundle["subscriptions"].items():
            subscription = session.scalar(
                select(Subscription).where(Subscription.user_id == user_id)
            ) or Subscription(user_id=user_id)
            subscription.plan_name = payload["plan_name"]
            subscription.plan_code = payload["plan_code"]
            subscription.status = payload.get("status", "active")
            subscription.billing_cycle = payload.get("billing_cycle", "monthly")
            subscription.renewal_date = payload.get("renewal_date")
            subscription.currency = payload.get("currency", "CNY")
            subscription.amount = float(payload.get("amount", 0.0))
            subscription.auto_renew = bool(payload.get("auto_renew", False))
            subscription.seat_quota = int(payload.get("seat_quota", 0))
            subscription.seats_used = int(payload.get("seats_used", 0))
            subscription.benefits_json = payload.get("benefits", [])
            session.add(subscription)
            stats["subscriptions"] += 1

        for invoice_group in bundle["invoices"].values():
            for payload in invoice_group:
                invoice_id = payload["invoice_id"]
                invoice = session.get(Invoice, invoice_id) or Invoice(invoice_id=invoice_id)
                invoice.user_id = payload["user_id"]
                invoice.issued_at = payload["issued_at"]
                invoice.period_start = payload["period_start"]
                invoice.period_end = payload["period_end"]
                invoice.status = payload.get("status", "paid")
                invoice.currency = payload.get("currency", "CNY")
                invoice.total_amount = float(payload.get("total_amount", 0.0))
                invoice.payment_method = payload.get("payment_method")
                invoice.summary = payload.get("summary", "")
                invoice.explanation_json = payload.get("explanation", [])
                session.add(invoice)
                session.flush()
                session.execute(delete(InvoiceItem).where(InvoiceItem.invoice_id == invoice_id))
                for item in payload.get("line_items", []):
                    session.add(
                        InvoiceItem(
                            invoice_id=invoice_id,
                            name=item["name"],
                            amount=float(item.get("amount", 0.0)),
                            type=item.get("type", "subscription"),
                            reason=item.get("reason", ""),
                        )
                    )
                    stats["invoice_items"] += 1
                stats["invoices"] += 1

        for payload in bundle["tickets"]:
            ticket = session.get(TicketRecord, payload["ticket_id"]) or TicketRecord(ticket_id=payload["ticket_id"])
            ticket.user_id = payload["user_id"]
            ticket.subject = payload["subject"]
            ticket.description = payload["description"]
            ticket.category = payload.get("category", "general")
            ticket.status = payload.get("status", "open")
            ticket.priority = payload.get("priority", "medium")
            ticket.assigned_to = payload.get("assigned_to")
            ticket.created_at = payload.get("created_at", _now_iso())
            ticket.updated_at = payload.get("updated_at", ticket.created_at)
            ticket.resolved_at = payload.get("resolved_at")
            ticket.tags_json = payload.get("tags", [])
            ticket.notes_json = payload.get("notes", [])
            ticket.metadata_json = payload.get("metadata", {})
            session.add(ticket)
            stats["tickets"] += 1

    return stats


def _user_to_record(user: User) -> Dict[str, Any]:
    """将 User ORM 对象转换为接口层可直接使用的字典。"""
    return {
        "user_id": user.user_id,
        "name": user.name,
        "email": user.email,
        "plan": user.plan,
        "status": user.status,
        "member_since": user.member_since,
        "company": user.company,
        "usage": user.usage_json or {},
        "billing": user.billing_json or {},
        "support_history": user.support_history_json or {},
    }


def _subscription_to_record(subscription: Subscription) -> Dict[str, Any]:
    """将订阅 ORM 对象转换为字典。"""
    return {
        "user_id": subscription.user_id,
        "plan_name": subscription.plan_name,
        "plan_code": subscription.plan_code,
        "status": subscription.status,
        "billing_cycle": subscription.billing_cycle,
        "renewal_date": subscription.renewal_date,
        "currency": subscription.currency,
        "amount": subscription.amount,
        "auto_renew": subscription.auto_renew,
        "seat_quota": subscription.seat_quota,
        "seats_used": subscription.seats_used,
        "benefits": subscription.benefits_json or [],
    }


def _invoice_to_record(invoice: Invoice) -> Dict[str, Any]:
    """将账单 ORM 对象转换为字典。"""
    return {
        "invoice_id": invoice.invoice_id,
        "user_id": invoice.user_id,
        "issued_at": invoice.issued_at,
        "period_start": invoice.period_start,
        "period_end": invoice.period_end,
        "status": invoice.status,
        "currency": invoice.currency,
        "total_amount": invoice.total_amount,
        "payment_method": invoice.payment_method,
        "summary": invoice.summary,
        "line_items": [
            {
                "name": item.name,
                "amount": item.amount,
                "type": item.type,
                "reason": item.reason,
            }
            for item in invoice.items
        ],
        "explanation": invoice.explanation_json or [],
    }


def _ticket_to_record(ticket: TicketRecord) -> Dict[str, Any]:
    """将工单 ORM 对象转换为字典。"""
    return {
        "ticket_id": ticket.ticket_id,
        "user_id": ticket.user_id,
        "subject": ticket.subject,
        "description": ticket.description,
        "category": ticket.category,
        "status": ticket.status,
        "priority": ticket.priority,
        "assigned_to": ticket.assigned_to,
        "created_at": ticket.created_at,
        "updated_at": ticket.updated_at,
        "resolved_at": ticket.resolved_at,
        "tags": ticket.tags_json or [],
        "notes": ticket.notes_json or [],
        "metadata": ticket.metadata_json or {},
    }


def _thread_to_record(thread: ConversationThread) -> Dict[str, Any]:
    """将对话线程 ORM 对象转换为字典。"""
    return {
        "thread_id": thread.thread_id,
        "user_id": thread.user_id,
        "status": thread.status,
        "title": thread.title,
        "rolling_summary": thread.rolling_summary or "",
        "message_count": thread.message_count,
        "last_active_agent": thread.last_active_agent,
        "pending_role": thread.pending_role,
        "pending_state": thread.pending_state_json or {},
        "trace_id": thread.trace_id,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "last_message_at": thread.last_message_at,
    }


def _message_to_record(message: ConversationMessage) -> Dict[str, Any]:
    """将对话消息 ORM 对象转换为字典。"""
    return {
        "id": message.id,
        "thread_id": message.thread_id,
        "user_id": message.user_id,
        "role": message.role,
        "content": message.content,
        "timestamp": message.created_at,
        "intent": message.intent,
        "active_agent": message.active_agent,
        "run_status": message.run_status,
        "visible": message.visible,
        "estimated_tokens": message.estimated_tokens,
        "metadata": message.metadata_json or {},
    }


def _knowledge_document_to_record(document: KnowledgeDocument) -> Dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "source_path": document.source_path,
        "title": document.title,
        "doc_type": document.doc_type,
        "checksum": document.checksum,
        "status": document.status,
        "version": document.version,
        "metadata": document.metadata_json or {},
        "ingested_at": document.ingested_at,
        "updated_at": document.updated_at,
    }


def _knowledge_chunk_to_record(chunk: KnowledgeChunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "chunk_level": chunk.chunk_level,
        "chunk_index": chunk.chunk_index,
        "child_index": chunk.child_index,
        "section_path": chunk.section_path,
        "title": chunk.title,
        "category": chunk.category,
        "content": chunk.content,
        "char_count": chunk.char_count,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata_json or {},
    }


def _memory_record_to_payload(memory: UserMemoryRecord) -> Dict[str, Any]:
    payload = dict(memory.payload_json or {})
    payload.setdefault("memory_id", memory.memory_id)
    payload.setdefault("user_id", memory.user_id)
    payload.setdefault("memory_type", memory.memory_type)
    payload.setdefault("status", memory.status)
    payload.setdefault("category", memory.category)
    payload.setdefault("field", memory.field)
    payload.setdefault("value", memory.value_text)
    payload.setdefault("summary", memory.summary)
    payload.setdefault("content", memory.content)
    payload.setdefault("issue_code", memory.issue_code)
    payload.setdefault("importance", memory.importance)
    payload.setdefault("created_at", memory.created_at)
    payload.setdefault("updated_at", memory.updated_at)
    return payload


def get_user_record(user_id: str) -> Optional[Dict[str, Any]]:
    """按用户 ID 查询用户主档。"""
    ensure_business_database()
    with session_scope() as session:
        user = session.get(User, user_id)
        return _user_to_record(user) if user else None


def get_subscription_record(user_id: str) -> Optional[Dict[str, Any]]:
    """按用户 ID 查询订阅记录。"""
    ensure_business_database()
    with session_scope() as session:
        subscription = session.scalar(select(Subscription).where(Subscription.user_id == user_id))
        return _subscription_to_record(subscription) if subscription else None


def get_latest_invoice_record(user_id: str) -> Optional[Dict[str, Any]]:
    """查询用户最近一张账单。"""
    ensure_business_database()
    with session_scope() as session:
        invoice = session.scalar(
            select(Invoice)
            .where(Invoice.user_id == user_id)
            .order_by(Invoice.issued_at.desc(), Invoice.invoice_id.desc())
        )
        return _invoice_to_record(invoice) if invoice else None


def get_invoice_record(invoice_id: str) -> Optional[Dict[str, Any]]:
    """按账单号查询账单。"""
    ensure_business_database()
    with session_scope() as session:
        invoice = session.get(Invoice, str(invoice_id).strip().upper())
        return _invoice_to_record(invoice) if invoice else None


def _generate_ticket_id(session) -> str:
    """生成线程安全的工单号。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    prefix = f"TKT-{timestamp}-"
    existing_count = session.scalar(
        select(func.count()).select_from(TicketRecord).where(TicketRecord.ticket_id.like(f"{prefix}%"))
    ) or 0
    return f"TKT-{timestamp}-{existing_count + 1:04d}"


def create_ticket_record(
    *,
    user_id: str,
    subject: str,
    description: str,
    priority: str = "medium",
    category: str = "general",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建新的工单记录。"""
    ensure_business_database()
    with _ticket_lock:
        with session_scope() as session:
            now = _now_iso()
            ticket = TicketRecord(
                ticket_id=_generate_ticket_id(session),
                user_id=user_id,
                subject=subject,
                description=description,
                category=category,
                status="open",
                priority=str(priority or "medium").lower(),
                created_at=now,
                updated_at=now,
                metadata_json=metadata or {},
                tags_json=[],
                notes_json=[],
            )
            session.add(ticket)
            session.flush()
            return _ticket_to_record(ticket)


def get_ticket_record(ticket_id: str) -> Optional[Dict[str, Any]]:
    """按工单号查询工单。"""
    ensure_business_database()
    with session_scope() as session:
        ticket = session.get(TicketRecord, ticket_id)
        return _ticket_to_record(ticket) if ticket else None


def update_ticket_record(
    *,
    ticket_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    assigned_to: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """更新工单状态、备注、处理人与标签。"""
    ensure_business_database()
    with session_scope() as session:
        ticket = session.get(TicketRecord, ticket_id)
        if ticket is None:
            return None

        if status:
            normalized = str(status).lower()
            ticket.status = normalized
            if normalized == "resolved":
                ticket.resolved_at = _now_iso()
        if assigned_to:
            ticket.assigned_to = assigned_to
        if notes:
            ticket.notes_json = [*(ticket.notes_json or []), notes]
        if tags:
            ticket.tags_json = sorted(set([*(ticket.tags_json or []), *tags]))
        ticket.updated_at = _now_iso()
        session.add(ticket)
        session.flush()
        return _ticket_to_record(ticket)


def list_ticket_records(user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出指定用户的工单，支持按状态过滤。"""
    ensure_business_database()
    with session_scope() as session:
        query = select(TicketRecord).where(TicketRecord.user_id == user_id)
        if status:
            query = query.where(TicketRecord.status == status)
        query = query.order_by(TicketRecord.created_at.desc(), TicketRecord.ticket_id.desc())
        return [_ticket_to_record(ticket) for ticket in session.scalars(query).all()]


def create_or_touch_conversation_thread(
    *,
    thread_id: str,
    user_id: str,
    status: str = "active",
    title: Optional[str] = None,
    last_active_agent: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    pending_role: Optional[str] = None,
    pending_state: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """创建或更新一条对话线程元数据。"""
    ensure_business_database()
    with session_scope() as session:
        now = _now_iso()
        thread = session.get(ConversationThread, thread_id)
        if thread is None:
            thread = ConversationThread(
                thread_id=thread_id,
                user_id=user_id,
                status=status or "active",
                title=_conversation_title(title or "") if title else "",
                rolling_summary=(rolling_summary or "").strip(),
                message_count=0,
                last_active_agent=last_active_agent,
                pending_role=pending_role,
                pending_state_json=pending_state or {},
                trace_id=trace_id,
                created_at=now,
                updated_at=now,
                last_message_at=None,
            )
        else:
            thread.user_id = user_id
            if title and not thread.title:
                thread.title = _conversation_title(title)
            if rolling_summary is not None:
                thread.rolling_summary = rolling_summary.strip()
            if last_active_agent is not None:
                thread.last_active_agent = last_active_agent
            if pending_role is not None:
                thread.pending_role = pending_role
            if pending_state is not None:
                thread.pending_state_json = pending_state
            if trace_id is not None:
                thread.trace_id = trace_id
            if status:
                thread.status = status
            thread.updated_at = now
        session.add(thread)
        session.flush()
        return _thread_to_record(thread)


def append_conversation_message(
    *,
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    visible: bool = True,
    intent: Optional[str] = None,
    active_agent: Optional[str] = None,
    run_status: Optional[str] = None,
    estimated_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    thread_status: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """向短期 transcript 写入一条消息，并同步更新线程元数据。"""
    ensure_business_database()
    with session_scope() as session:
        now = _now_iso()
        thread = session.get(ConversationThread, thread_id)
        if thread is None:
            thread = ConversationThread(
                thread_id=thread_id,
                user_id=user_id,
                status=thread_status or run_status or "active",
                title=_conversation_title(content) if role == "user" else "",
                rolling_summary="",
                message_count=0,
                last_active_agent=active_agent,
                pending_role=None,
                pending_state_json={},
                trace_id=trace_id,
                created_at=now,
                updated_at=now,
                last_message_at=now,
            )
        else:
            thread.user_id = user_id
            if role == "user" and not thread.title:
                thread.title = _conversation_title(content)
            if active_agent is not None:
                thread.last_active_agent = active_agent
            if trace_id is not None:
                thread.trace_id = trace_id
            if thread_status:
                thread.status = thread_status
            elif run_status in {"completed", "interrupted", "error"}:
                thread.status = run_status
            elif role == "user":
                thread.status = "active"
            thread.updated_at = now
            thread.last_message_at = now

        message = ConversationMessage(
            thread_id=thread_id,
            user_id=user_id,
            role=role,
            content=content,
            visible=visible,
            intent=intent,
            active_agent=active_agent,
            run_status=run_status,
            estimated_tokens=estimated_tokens if estimated_tokens is not None else estimate_text_tokens(content),
            metadata_json=metadata or {},
            created_at=now,
        )
        thread.message_count = int(thread.message_count or 0) + 1
        session.add(thread)
        session.add(message)
        session.flush()
        return _message_to_record(message)


def mark_conversation_thread_status(
    thread_id: str,
    *,
    status: str,
    last_active_agent: Optional[str] = None,
    rolling_summary: Optional[str] = None,
    pending_role: Optional[str] = None,
    pending_state: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """更新线程状态，不额外写入消息。"""
    ensure_business_database()
    with session_scope() as session:
        thread = session.get(ConversationThread, thread_id)
        if thread is None:
            return None
        thread.status = status
        thread.updated_at = _now_iso()
        if last_active_agent is not None:
            thread.last_active_agent = last_active_agent
        if rolling_summary is not None:
            thread.rolling_summary = rolling_summary.strip()
        if pending_role is not None:
            thread.pending_role = pending_role
        if pending_state is not None:
            thread.pending_state_json = pending_state
        if trace_id is not None:
            thread.trace_id = trace_id
        session.add(thread)
        session.flush()
        return _thread_to_record(thread)


def list_thread_messages(
    thread_id: str,
    *,
    limit: Optional[int] = None,
    visible_only: bool = True,
) -> List[Dict[str, Any]]:
    """按线程读取消息，默认仅返回用户可见 transcript。"""
    ensure_business_database()
    with session_scope() as session:
        query = select(ConversationMessage).where(ConversationMessage.thread_id == thread_id)
        if visible_only:
            query = query.where(ConversationMessage.visible.is_(True))
        query = query.order_by(ConversationMessage.id.desc())
        if limit is not None:
            query = query.limit(max(1, int(limit)))
        rows = list(session.scalars(query).all())
        rows.reverse()
        return [_message_to_record(message) for message in rows]


def get_conversation_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """读取单个对话线程元数据。"""
    ensure_business_database()
    with session_scope() as session:
        thread = session.get(ConversationThread, thread_id)
        return _thread_to_record(thread) if thread else None


def list_user_conversation_messages(
    user_id: str,
    *,
    limit: int = 20,
    thread_id: Optional[str] = None,
    visible_only: bool = True,
) -> List[Dict[str, Any]]:
    """按用户读取最近消息，可选限制在线程内。"""
    ensure_business_database()
    with session_scope() as session:
        query = select(ConversationMessage).where(ConversationMessage.user_id == user_id)
        if thread_id:
            query = query.where(ConversationMessage.thread_id == thread_id)
        if visible_only:
            query = query.where(ConversationMessage.visible.is_(True))
        query = query.order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc()).limit(
            max(1, int(limit))
        )
        rows = list(session.scalars(query).all())
        rows.reverse()
        return [_message_to_record(message) for message in rows]


def build_recent_context_window(
    *,
    thread_id: str,
    recent_turns: int,
    max_messages: int,
    max_tokens: int,
) -> Dict[str, Any]:
    """构建线程级短期上下文窗口，供主图提示词注入使用。"""
    ensure_business_database()
    with session_scope() as session:
        thread = session.get(ConversationThread, thread_id)
        if thread is None:
            return {
                "thread_id": thread_id,
                "rolling_summary": "",
                "messages": [],
                "text": "",
                "estimated_tokens": 0,
            }

        fetch_limit = max(24, max_messages * 2, recent_turns * 4)
        query = (
            select(ConversationMessage)
            .where(
                ConversationMessage.thread_id == thread_id,
                ConversationMessage.visible.is_(True),
            )
            .order_by(ConversationMessage.id.desc())
            .limit(fetch_limit)
        )
        rows = list(session.scalars(query).all())

        effective_cap = max(2, min(max_messages, max(2, recent_turns * 2)))
        kept: List[ConversationMessage] = []
        token_total = 0
        for message in rows:
            message_tokens = max(1, int(message.estimated_tokens or estimate_text_tokens(message.content)))
            if kept and len(kept) >= effective_cap:
                break
            if kept and token_total + message_tokens > max_tokens:
                break
            kept.append(message)
            token_total += message_tokens

        kept.reverse()
        history_lines = []
        role_map = {
            "user": "用户",
            "assistant": "客服",
            "system": "系统",
            "interrupt": "审批",
        }
        for message in kept:
            label = role_map.get(message.role, message.role)
            history_lines.append(f"{label}：{_normalize_ws(message.content)}")

        rolling_summary = (thread.rolling_summary or "").strip()
        parts: List[str] = []
        if rolling_summary:
            parts.append(f"更早对话摘要：{rolling_summary}")
        if history_lines:
            parts.append("最近对话：\n" + "\n".join(history_lines))

        return {
            "thread_id": thread_id,
            "rolling_summary": rolling_summary,
            "messages": [_message_to_record(message) for message in kept],
            "text": "\n".join(parts).strip(),
            "estimated_tokens": token_total,
        }


def save_pending_conversation_state(
    *,
    thread_id: str,
    user_id: str,
    pending_role: str,
    pending_state: Dict[str, Any],
    trace_id: Optional[str],
    status: str = "interrupted",
    last_active_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """持久化待审批线程状态，便于跨进程 resume。"""
    return create_or_touch_conversation_thread(
        thread_id=thread_id,
        user_id=user_id,
        status=status,
        last_active_agent=last_active_agent or pending_role,
        pending_role=pending_role,
        pending_state=pending_state,
        trace_id=trace_id,
    )


def clear_pending_conversation_state(
    thread_id: str,
    *,
    status: str,
    last_active_agent: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """清空线程中的待审批状态。"""
    return mark_conversation_thread_status(
        thread_id,
        status=status,
        last_active_agent=last_active_agent,
        pending_role="",
        pending_state={},
        trace_id=trace_id,
    )


def delete_user_conversations(user_id: str) -> int:
    """删除指定用户的全部短期 transcript，用于重置会话。"""
    ensure_business_database()
    with session_scope() as session:
        thread_ids = list(
            session.scalars(
                select(ConversationThread.thread_id).where(ConversationThread.user_id == user_id)
            ).all()
        )
        if thread_ids:
            session.execute(delete(ConversationMessage).where(ConversationMessage.thread_id.in_(thread_ids)))
        session.execute(delete(ConversationThread).where(ConversationThread.user_id == user_id))
        return len(thread_ids)


def replace_knowledge_corpus(
    *,
    documents: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    clear_existing: bool = True,
) -> Dict[str, int]:
    """Replace the persisted document knowledge metadata with the latest corpus snapshot."""
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        if clear_existing:
            session.execute(delete(KnowledgeChunk))
            session.execute(delete(KnowledgeDocument))

        for payload in documents:
            document = session.get(KnowledgeDocument, payload["doc_id"]) or KnowledgeDocument(doc_id=payload["doc_id"])
            document.source_path = payload["source_path"]
            document.title = payload["title"]
            document.doc_type = payload.get("doc_type", "markdown")
            document.checksum = payload["checksum"]
            document.status = payload.get("status", "active")
            document.version = int(payload.get("version", 1))
            document.metadata_json = payload.get("metadata", {})
            document.ingested_at = payload["ingested_at"]
            document.updated_at = payload.get("updated_at", payload["ingested_at"])
            session.add(document)

        if documents and not clear_existing:
            doc_ids = [payload["doc_id"] for payload in documents]
            session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.doc_id.in_(doc_ids)))

        for payload in chunks:
            session.add(
                KnowledgeChunk(
                    chunk_id=payload["chunk_id"],
                    doc_id=payload["doc_id"],
                    parent_chunk_id=payload.get("parent_chunk_id"),
                    chunk_level=payload.get("chunk_level", "child"),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    child_index=payload.get("child_index"),
                    section_path=payload.get("section_path", ""),
                    title=payload.get("title", ""),
                    category=payload.get("category", "general"),
                    content=payload["content"],
                    char_count=int(payload.get("char_count", len(payload["content"]))),
                    token_count=int(payload.get("token_count", estimate_text_tokens(payload["content"]))),
                    metadata_json=payload.get("metadata", {}),
                )
            )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "parent_chunks": sum(1 for chunk in chunks if chunk.get("chunk_level") == "parent"),
        "child_chunks": sum(1 for chunk in chunks if chunk.get("chunk_level") == "child"),
    }


def list_knowledge_documents() -> List[Dict[str, Any]]:
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        rows = list(session.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.title.asc())).all())
        return [_knowledge_document_to_record(row) for row in rows]


def list_knowledge_chunks(
    *,
    doc_id: Optional[str] = None,
    chunk_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        query = select(KnowledgeChunk)
        if doc_id:
            query = query.where(KnowledgeChunk.doc_id == doc_id)
        if chunk_level:
            query = query.where(KnowledgeChunk.chunk_level == chunk_level)
        query = query.order_by(KnowledgeChunk.doc_id.asc(), KnowledgeChunk.chunk_index.asc(), KnowledgeChunk.child_index.asc())
        rows = list(session.scalars(query).all())
        return [_knowledge_chunk_to_record(row) for row in rows]


def upsert_user_memory_record(
    *,
    user_id: str,
    memory_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    ensure_business_database(seed_demo=False)
    now = _now_iso()
    with session_scope() as session:
        memory = session.get(UserMemoryRecord, memory_id)
        created_at = now
        if memory is None:
            memory = UserMemoryRecord(memory_id=memory_id, created_at=now, updated_at=now)
        else:
            created_at = memory.created_at

        memory.user_id = user_id
        memory.memory_type = payload.get("memory_type", "memory")
        memory.status = payload.get("status", "active")
        memory.category = payload.get("category")
        memory.field = payload.get("field")
        value = payload.get("value")
        memory.value_text = "" if value is None else str(value)
        memory.summary = payload.get("summary", "")
        memory.content = payload.get("content", "")
        memory.issue_code = payload.get("issue_code")
        memory.importance = float(payload.get("importance", 0.0) or 0.0)
        stored_payload = dict(payload)
        stored_payload.setdefault("memory_id", memory_id)
        stored_payload.setdefault("user_id", user_id)
        stored_payload.setdefault("created_at", created_at)
        stored_payload.setdefault("updated_at", now)
        memory.payload_json = stored_payload
        memory.created_at = stored_payload["created_at"]
        memory.updated_at = stored_payload["updated_at"]
        session.add(memory)
        session.flush()
        return _memory_record_to_payload(memory)


def get_user_memory_record(user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        memory = session.get(UserMemoryRecord, memory_id)
        if memory is None or memory.user_id != user_id:
            return None
        return _memory_record_to_payload(memory)


def list_user_memory_records(
    user_id: str,
    *,
    limit: Optional[int] = None,
    memory_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        query = select(UserMemoryRecord).where(UserMemoryRecord.user_id == user_id)
        if memory_type:
            query = query.where(UserMemoryRecord.memory_type == memory_type)
        query = query.order_by(UserMemoryRecord.updated_at.desc(), UserMemoryRecord.memory_id.asc())
        if limit is not None:
            query = query.limit(max(1, int(limit)))
        rows = list(session.scalars(query).all())
        return [_memory_record_to_payload(row) for row in rows]


def delete_user_memory_record(user_id: str, memory_id: str) -> bool:
    ensure_business_database(seed_demo=False)
    with session_scope() as session:
        memory = session.get(UserMemoryRecord, memory_id)
        if memory is None or memory.user_id != user_id:
            return False
        session.delete(memory)
        return True
