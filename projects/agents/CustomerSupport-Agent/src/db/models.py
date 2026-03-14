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
