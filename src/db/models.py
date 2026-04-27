"""业务实体对应的 SQLAlchemy ORM 模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class User(Base):
    """用户主档，汇总基础资料、使用情况与客服历史。"""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(256), nullable=False)
    plan: Mapped[str] = mapped_column(String(64), nullable=False, default="Free")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    member_since: Mapped[str] = mapped_column(String(32), nullable=False)
    company: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    usage_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    billing_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    support_history_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    subscription: Mapped[Optional["Subscription"]] = relationship(back_populates="user", uselist=False)
    invoices: Mapped[List["Invoice"]] = relationship(back_populates="user")
    tickets: Mapped[List["TicketRecord"]] = relationship(back_populates="user")


class Subscription(Base):
    """订阅信息，描述套餐、计费周期与席位使用情况。"""

    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), unique=True, index=True)
    plan_name: Mapped[str] = mapped_column(String(128), nullable=False)
    plan_code: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    billing_cycle: Mapped[str] = mapped_column(String(32), nullable=False)
    renewal_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    currency: Mapped[str] = mapped_column(String(16), nullable=False, default="CNY")
    amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    seat_quota: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    seats_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    benefits_json: Mapped[List[str]] = mapped_column(JSON, default=list)

    user: Mapped[User] = relationship(back_populates="subscription")


class Invoice(Base):
    """账单主表，存储账期、总金额与摘要。"""

    __tablename__ = "invoices"

    invoice_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    issued_at: Mapped[str] = mapped_column(String(32), nullable=False)
    period_start: Mapped[str] = mapped_column(String(32), nullable=False)
    period_end: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    currency: Mapped[str] = mapped_column(String(16), nullable=False, default="CNY")
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payment_method: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    summary: Mapped[str] = mapped_column(String(256), nullable=False)
    explanation_json: Mapped[List[str]] = mapped_column(JSON, default=list)

    user: Mapped[User] = relationship(back_populates="invoices")
    items: Mapped[List["InvoiceItem"]] = relationship(
        back_populates="invoice",
        cascade="all, delete-orphan",
        order_by="InvoiceItem.id",
    )


class InvoiceItem(Base):
    """账单明细项。"""

    __tablename__ = "invoice_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(ForeignKey("invoices.invoice_id"), index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    invoice: Mapped[Invoice] = relationship(back_populates="items")


class TicketRecord(Base):
    """客服工单记录。"""

    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    assigned_to: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    resolved_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tags_json: Mapped[List[str]] = mapped_column(JSON, default=list)
    notes_json: Mapped[List[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="tickets")


class KnowledgeDocument(Base):
    """Knowledge document metadata persisted for the document-backed RAG index."""

    __tablename__ = "knowledge_documents"

    doc_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False, default="markdown")
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)

    chunks: Mapped[List["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunk.chunk_index",
    )


class KnowledgeChunk(Base):
    """Persisted parent/child chunk metadata for the document knowledge base."""

    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("knowledge_documents.doc_id"), index=True)
    parent_chunk_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    chunk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="child")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    child_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")


class UserMemoryRecord(Base):
    """Persistent structured long-term memory for each user."""

    __tablename__ = "user_memories"

    memory_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    field: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    value_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    issue_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ConversationThread(Base):
    """短期对话线程，保存 thread 级元信息与滚动上下文摘要。"""

    __tablename__ = "conversation_threads"

    thread_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    rolling_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_active_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    pending_role: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    pending_state_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    graph_thread_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_graph_node: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_checkpoint_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    last_message_at: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    messages: Mapped[List["ConversationMessage"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.id",
    )


class ConversationMessage(Base):
    """短期对话消息明细，按线程保存用户与助手的原始 transcript。"""

    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("conversation_threads.thread_id"), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    visible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    intent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    active_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    run_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    estimated_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)

    thread: Mapped[ConversationThread] = relationship(back_populates="messages")
