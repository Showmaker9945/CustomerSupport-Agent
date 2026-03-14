"""结构化业务数据的建库、种子数据与查询封装。"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, select

from .demo_seed import load_seed_bundle
from .models import Base, Invoice, InvoiceItem, Subscription, TicketRecord, User
from .session import get_engine, session_scope

_bootstrap_lock = Lock()
_ticket_lock = Lock()


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def ensure_business_database(seed_demo: bool = True) -> None:
    """确保业务表存在，并在需要时自动注入演示数据。"""
    Base.metadata.create_all(get_engine())
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
