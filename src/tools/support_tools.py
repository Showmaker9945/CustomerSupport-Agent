"""
Support tools for customer service agent.

Provides ticket management, account lookup, and FAQ search tools
that can be used by LangChain agents.
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from ..config import settings
from ..db.demo_seed import load_seed_accounts, load_seed_invoices, load_seed_subscriptions
from ..db.repositories import (
    create_ticket_record,
    ensure_business_database,
    get_invoice_record as repo_get_invoice_record,
    get_latest_invoice_record as repo_get_latest_invoice_record,
    get_subscription_record as repo_get_subscription_record,
    get_ticket_record,
    get_user_record as repo_get_user_record,
    list_ticket_records,
    update_ticket_record,
)
from ..knowledge.faq_store import FAQStore, create_faq_store

logger = logging.getLogger(__name__)


def _load_mock_accounts() -> Dict[str, Any]:
    """Load seed accounts from external JSON file."""
    try:
        return load_seed_accounts()
    except Exception as e:
        logger.warning(f"Failed to load demo seed accounts: {e}")
        return {}


class TicketStatus(str, Enum):
    """Ticket status values."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(str, Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Ticket:
    """Support ticket data model."""
    ticket_id: str
    user_id: str
    subject: str
    description: str
    category: str = "general"
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.MEDIUM
    assigned_to: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ticket to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ticket":
        """Create ticket from dictionary."""
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = TicketStatus(data["status"])
        if isinstance(data.get("priority"), str):
            data["priority"] = TicketPriority(data["priority"])
        return cls(**data)


class TicketStore:
    """
    In-memory ticket store with optional file persistence.

    For production, replace with database implementation.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        """
        Initialize ticket store.

        Args:
            persist_path: Optional path to persist tickets
        """
        self.persist_path = persist_path or Path("./data/tickets.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        self._tickets: Dict[str, Ticket] = {}
        self._user_tickets: Dict[str, List[str]] = {}  # user_id -> [ticket_ids]
        self._ticket_counter = 0  # Counter for unique IDs
        # Re-entrant lock prevents deadlock when nested helper methods also lock.
        self._lock = threading.RLock()

        # Load existing tickets if file exists
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load tickets from disk."""
        if not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ticket_data in data.get("tickets", []):
                ticket = Ticket.from_dict(ticket_data)
                self._tickets[ticket.ticket_id] = ticket

                # Update user index
                if ticket.user_id not in self._user_tickets:
                    self._user_tickets[ticket.user_id] = []
                self._user_tickets[ticket.user_id].append(ticket.ticket_id)

            logger.info(f"Loaded {len(self._tickets)} tickets from disk")

        except Exception as e:
            logger.error(f"Failed to load tickets: {e}")

    def _save_to_disk(self) -> None:
        """Persist tickets to disk."""
        try:
            data = {
                "tickets": [ticket.to_dict() for ticket in self._tickets.values()],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save tickets: {e}")

    def _generate_ticket_id(self) -> str:
        """
        Generate unique ticket ID.

        Note: The counter is safely incremented within a lock to ensure
        uniqueness across concurrent requests.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        # Use atomic counter for uniqueness (thread-safe with lock)
        with self._lock:
            self._ticket_counter += 1
            return f"TKT-{timestamp}-{self._ticket_counter:04d}"

    def create_ticket(
        self,
        user_id: str,
        subject: str,
        description: str,
        priority: str = "medium",
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Ticket:
        """
        Create a new support ticket.

        Args:
            user_id: User ID
            subject: Ticket subject
            description: Ticket description
            priority: Priority level (low, medium, high, urgent)
            metadata: Optional metadata

        Returns:
            Created ticket
        """
        with self._lock:
            ticket = Ticket(
                ticket_id=self._generate_ticket_id(),
                user_id=user_id,
                subject=subject,
                description=description,
                category=category,
                priority=TicketPriority(priority.lower()),
                metadata=metadata or {}
            )

            self._tickets[ticket.ticket_id] = ticket

            # Update user index
            if user_id not in self._user_tickets:
                self._user_tickets[user_id] = []
            self._user_tickets[user_id].append(ticket.ticket_id)

            self._save_to_disk()
            logger.info(f"Created ticket {ticket.ticket_id} for user {user_id}")

            return ticket

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """
        Get ticket by ID.

        Args:
            ticket_id: Ticket ID

        Returns:
            Ticket or None if not found
        """
        return self._tickets.get(ticket_id)

    def update_ticket(
        self,
        ticket_id: str,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        assigned_to: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Ticket]:
        """
        Update ticket.

        Args:
            ticket_id: Ticket ID
            status: New status
            notes: Notes to add
            assigned_to: Assign to agent
            tags: Tags to add

        Returns:
            Updated ticket or None if not found
        """
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return None

            if status:
                ticket.status = TicketStatus(status)
                if status == "resolved":
                    ticket.resolved_at = datetime.now(timezone.utc).isoformat()

            if assigned_to:
                ticket.assigned_to = assigned_to

            if notes:
                ticket.notes.append(notes)

            if tags:
                ticket.tags.extend(tags)
                ticket.tags = list(set(ticket.tags))  # Remove duplicates

            ticket.updated_at = datetime.now(timezone.utc).isoformat()

            self._save_to_disk()
            logger.info(f"Updated ticket {ticket_id}")

            return ticket

    def get_user_tickets(
        self,
        user_id: str,
        status: Optional[str] = None
    ) -> List[Ticket]:
        """
        Get all tickets for a user.

        Args:
            user_id: User ID
            status: Optional status filter

        Returns:
            List of tickets
        """
        ticket_ids = self._user_tickets.get(user_id, [])
        tickets = [self._tickets[tid] for tid in ticket_ids if tid in self._tickets]

        if status:
            tickets = [t for t in tickets if t.status.value == status]

        # Sort by created_at descending
        tickets.sort(key=lambda t: t.created_at, reverse=True)
        return tickets

    def search_tickets(
        self,
        query: str,
        limit: int = 10
    ) -> List[Ticket]:
        """
        Search tickets by subject/description.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching tickets
        """
        query_lower = query.lower()
        results = []

        for ticket in self._tickets.values():
            if (query_lower in ticket.subject.lower() or
                query_lower in ticket.description.lower()):
                results.append(ticket)
                if len(results) >= limit:
                    break

        return results


MOCK_ACCOUNTS: Dict[str, Any] = _load_mock_accounts()
try:
    MOCK_SUBSCRIPTIONS = load_seed_subscriptions()
except Exception as error:
    logger.warning(f"Failed to load demo seed subscriptions: {error}")
    MOCK_SUBSCRIPTIONS = {}

try:
    MOCK_INVOICES = load_seed_invoices()
except Exception as error:
    logger.warning(f"Failed to load demo seed invoices: {error}")
    MOCK_INVOICES = {}


def get_account_record(user_id: str) -> Optional[Dict[str, Any]]:
    """Get account data from the business database."""
    try:
        ensure_business_database()
        account = repo_get_user_record(user_id)
        if account:
            return account
    except Exception as error:
        logger.error(f"Business account lookup failed for {user_id}: {error}")

    account = MOCK_ACCOUNTS.get(user_id)
    if not account:
        return None
    return json.loads(json.dumps(account, ensure_ascii=False))


def get_subscription_record(user_id: str) -> Optional[Dict[str, Any]]:
    """Get subscription data from the business database."""
    try:
        ensure_business_database()
        record = repo_get_subscription_record(user_id)
        if record:
            return record
    except Exception as error:
        logger.error(f"Business subscription lookup failed for {user_id}: {error}")

    record = MOCK_SUBSCRIPTIONS.get(user_id)
    if record:
        return json.loads(json.dumps(record, ensure_ascii=False))

    account = get_account_record(user_id)
    if not account:
        return None

    status = str(account.get("status", "active"))
    return {
        "user_id": user_id,
        "plan_name": f"{account.get('plan', 'Unknown')} 套餐",
        "plan_code": str(account.get("plan", "unknown")).lower(),
        "status": status,
        "billing_cycle": "monthly",
        "renewal_date": None if status != "active" else (
            datetime.now(timezone.utc) + timedelta(days=30)
        ).date().isoformat(),
        "currency": "USD",
        "amount": 0.0,
        "auto_renew": status == "active",
        "seat_quota": account.get("usage", {}).get("team_limit", 1),
        "seats_used": account.get("usage", {}).get("team_members", 1),
        "benefits": ["基础支持"],
    }


def get_latest_invoice_record(user_id: str) -> Optional[Dict[str, Any]]:
    """Get latest invoice for a user."""
    try:
        ensure_business_database()
        record = repo_get_latest_invoice_record(user_id)
        if record:
            return record
    except Exception as error:
        logger.error(f"Business latest invoice lookup failed for {user_id}: {error}")

    invoices = MOCK_INVOICES.get(user_id, [])
    if not invoices:
        return None
    sorted_invoices = sorted(invoices, key=lambda item: item.get("issued_at", ""), reverse=True)
    return json.loads(json.dumps(sorted_invoices[0], ensure_ascii=False))


def get_invoice_record(invoice_id: str) -> Optional[Dict[str, Any]]:
    """Get invoice by invoice ID."""
    normalized = str(invoice_id).strip().upper()
    try:
        ensure_business_database()
        record = repo_get_invoice_record(normalized)
        if record:
            return record
    except Exception as error:
        logger.error(f"Business invoice lookup failed for {invoice_id}: {error}")

    for invoices in MOCK_INVOICES.values():
        for invoice in invoices:
            if str(invoice.get("invoice_id", "")).upper() == normalized:
                return json.loads(json.dumps(invoice, ensure_ascii=False))
    return None


def _format_money(amount: float, currency: str) -> str:
    if str(currency).upper() == "CNY":
        return f"¥{amount:.2f}"
    return f"{str(currency).upper()} {amount:.2f}"


def _record_to_ticket(record: Dict[str, Any]) -> Ticket:
    return Ticket.from_dict(record)


def _ticket_next_step(ticket: Ticket) -> str:
    if ticket.status == TicketStatus.OPEN:
        return "下一步：客服会先完成分诊，通常会在 24 小时内给出首轮回复。"
    if ticket.status == TicketStatus.IN_PROGRESS:
        return "下一步：工单正在处理中，请留意邮件或站内通知。"
    if ticket.status == TicketStatus.WAITING_CUSTOMER:
        return "下一步：当前等待你补充信息，补充后会继续处理。"
    if ticket.status == TicketStatus.RESOLVED:
        return "下一步：如果问题仍存在，可以直接回复当前工单继续跟进。"
    return "下一步：如需继续处理，可补充说明后重新联系支持团队。"


# Global FAQ store instance
_faq_store: Optional[FAQStore] = None


def get_faq_store() -> FAQStore:
    """Get or create global FAQ store instance."""
    global _faq_store
    if _faq_store is None:
        _faq_store = create_faq_store()
    return _faq_store


def reset_faq_store() -> None:
    """Reset the global FAQ store instance (useful for testing)."""
    global _faq_store
    _faq_store = None


# Global ticket store instance
_ticket_store: Optional[TicketStore] = None


def get_ticket_store() -> TicketStore:
    """Get or create global ticket store instance."""
    global _ticket_store
    if _ticket_store is None:
        _ticket_store = TicketStore()
    return _ticket_store


def reset_ticket_store() -> None:
    """Reset the global ticket store instance (useful for testing)."""
    global _ticket_store
    _ticket_store = None


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

@tool
def search_faq(query: str, category: Optional[str] = None) -> str:
    """
    查询/搜索知识库中的常见问题答案。

    Args:
        query: 用户问题或关键词
        category: 可选分类过滤（billing/account/technical 等）

    Returns:
        带相关度的格式化检索结果
    """
    try:
        store = get_faq_store()
        results = store.search_hybrid(query, category=category, top_k=settings.rag_top_k)
        trace = store.get_last_query_trace()

        if not results:
            return f"未找到与该问题相关的知识：{query}"

        output_parts = [f"找到 {len(results)} 条相关知识（混合检索+重排）：\n"]

        if settings.debug:
            strategy_parts = []
            if trace.get("effective_category"):
                strategy_parts.append(f"分类推断：{trace['effective_category']}")
            if trace.get("sub_queries"):
                strategy_parts.append(f"子问题：{len(trace['sub_queries'])} 个")
            if trace.get("rewrite_used"):
                strategy_parts.append("启用改写兜底")
            if strategy_parts:
                output_parts.append("检索策略：" + " | ".join(strategy_parts))

        for i, result in enumerate(results, 1):
            output_parts.append(
                f"\n{i}. {result.question}\n"
                f"   答案：{result.answer}\n"
                f"   分类：{result.category} | 相关度：{result.confidence:.0%}\n"
                f"   来源：FAQ::{result.category}"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"FAQ search error: {e}")
        return f"知识检索失败：{str(e)}"


@tool
def reindex_knowledge_base(clear_existing: bool = False) -> str:
    """
    重建并更新 FAQ 知识库索引。

    Args:
        clear_existing: 是否先清空现有集合再重建

    Returns:
        重建结果摘要
    """
    try:
        store = get_faq_store()
        stats = store.reindex(clear_existing=clear_existing)
        return (
            f"知识库索引已刷新。\n"
            f"FAQ 总量：{stats.get('total_faqs', 0)}\n"
            f"分类数：{stats.get('category_count', 0)}\n"
            f"集合：{stats.get('collection_name', settings.collection_name)}"
        )
    except Exception as e:
        logger.error(f"Reindex knowledge base error: {e}")
        return f"知识库重建失败：{str(e)}"
@tool
def create_ticket(
    user_id: str,
    subject: str,
    description: str,
    priority: str = "medium",
    category: str = "general",
) -> str:
    """
    为用户创建新的客服工单。

    Args:
        user_id: 用户唯一标识
        subject: 问题主题
        description: 问题详情
        priority: 优先级（low/medium/high/urgent）
        category: 工单分类（general/billing/account/technical/escalation）

    Returns:
        创建结果和工单号
    """
    try:
        ticket = _record_to_ticket(
            create_ticket_record(
                user_id=user_id,
                subject=subject,
                description=description,
                priority=priority,
                category=category,
            )
        )

        return (
            f"工单创建成功。\n"
            f"工单号：{ticket.ticket_id}\n"
            f"分类：{ticket.category}\n"
            f"状态：{ticket.status.value}\n"
            f"优先级：{ticket.priority.value}\n"
            f"创建时间：{ticket.created_at}\n"
            f"{_ticket_next_step(ticket)}\n"
            f"来源：Support Ticket Store"
        )

    except Exception as e:
        logger.error(f"Create ticket error: {e}")
        return f"创建工单失败：{str(e)}"
@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    查询工单当前状态。

    Args:
        ticket_id: 工单号（如 TKT-20250131-0001）

    Returns:
        工单详情文本
    """
    try:
        record = get_ticket_record(ticket_id)
        ticket = _record_to_ticket(record) if record else None

        if not ticket:
            return f"未找到工单 {ticket_id}，请检查工单号是否正确。"

        output = [
            f"工单号：{ticket.ticket_id}",
            f"主题：{ticket.subject}",
            f"分类：{ticket.category}",
            f"状态：{ticket.status.value}",
            f"优先级：{ticket.priority.value}",
            f"创建时间：{ticket.created_at}",
            f"更新时间：{ticket.updated_at}",
        ]

        if ticket.assigned_to:
            output.append(f"处理人：{ticket.assigned_to}")

        if ticket.resolved_at:
            output.append(f"解决时间：{ticket.resolved_at}")

        if ticket.tags:
            output.append(f"标签：{', '.join(ticket.tags)}")

        if ticket.notes:
            output.append("\n备注：")
            for note in ticket.notes:
                output.append(f"- {note}")

        output.append(_ticket_next_step(ticket))
        output.append("来源：Support Ticket Store")
        return "\n".join(output)

    except Exception as e:
        logger.error(f"Get ticket status error: {e}")
        return f"查询工单失败：{str(e)}"
@tool
def update_ticket(
    ticket_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None
) -> str:
    """
    更新工单状态或补充备注。

    Args:
        ticket_id: 工单号
        status: 新状态（open/in_progress/waiting_customer/resolved/closed）
        notes: 备注内容

    Returns:
        更新结果
    """
    try:
        record = update_ticket_record(
            ticket_id=ticket_id,
            status=status,
            notes=notes,
        )
        ticket = _record_to_ticket(record) if record else None

        if not ticket:
            return f"未找到工单 {ticket_id}。"

        result = f"工单 {ticket_id} 更新成功。\n"
        result += f"分类：{ticket.category}\n"
        result += f"新状态：{ticket.status.value}\n"
        result += f"更新时间：{ticket.updated_at}"

        if notes:
            result += f"\n新增备注：{notes}"

        result += f"\n{_ticket_next_step(ticket)}"
        result += "\n来源：Support Ticket Store"

        return result

    except Exception as e:
        logger.error(f"Update ticket error: {e}")
        return f"更新工单失败：{str(e)}"
@tool
def get_user_tickets(user_id: str, status: Optional[str] = None) -> str:
    """
    查询用户的工单列表。

    Args:
        user_id: 用户唯一标识
        status: 可选状态过滤（open/resolved 等）

    Returns:
        工单列表摘要
    """
    try:
        tickets = [_record_to_ticket(record) for record in list_ticket_records(user_id, status=status)]

        if not tickets:
            return f"用户 {user_id} 暂无工单记录。"

        output = [f"用户 {user_id} 共找到 {len(tickets)} 条工单：\n"]

        for ticket in tickets:
            output.append(
                f"{ticket.ticket_id} - {ticket.subject}\n"
                f"分类：{ticket.category} | "
                f"状态：{ticket.status.value} | "
                f"优先级：{ticket.priority.value} | "
                f"创建日期：{ticket.created_at[:10]}"
            )

        output.append("\n来源：Support Ticket Store")
        return "\n".join(output)

    except Exception as e:
        logger.error(f"Get user tickets error: {e}")
        return f"查询用户工单失败：{str(e)}"
@tool
def lookup_account(user_id: str) -> str:
    """
    查询用户账户信息（套餐、使用量、账单等）。

    Args:
        user_id: 用户唯一标识

    Returns:
        账户详情文本
    """
    try:
        account = get_account_record(user_id)

        if not account:
            return f"未找到用户 {user_id} 的账户信息。"

        subscription = get_subscription_record(user_id)

        output = [
            f"{account['name']} 的账户信息",
            "",
            f"用户ID：{account['user_id']}",
            f"邮箱：{account['email']}",
            f"套餐：{account['plan']}",
            f"状态：{account['status'].replace('_', ' ').title()}",
            f"注册时间：{account['member_since']}"
        ]

        if account.get('company'):
            output.append(f"公司：{account['company']}")

        output.extend([
            "",
            "使用情况：",
            f"  存储：{account['usage']['storage_used']} / {account['usage']['storage_limit']}",
            f"  API 调用：{account['usage']['api_calls_this_month']} / {account['usage']['api_limit']}",
            f"  团队成员：{account['usage']['team_members']} / {account['usage']['team_limit']}"
        ])

        if account.get('billing'):
            output.append("")
            output.append("账单信息：")
            billing = account['billing']
            output.append(f"  本期账单：{billing['invoice_amount']}")
            if billing.get('last_payment'):
                output.append(f"  最近支付：{billing['last_payment']}")
            if billing.get('payment_method'):
                output.append(f"  支付方式：{billing['payment_method']}")

        if subscription:
            output.append("")
            output.append("订阅摘要：")
            output.append(f"  当前订阅：{subscription['plan_name']}")
            output.append(f"  计费周期：{subscription['billing_cycle']}")
            if subscription.get("renewal_date"):
                output.append(f"  下次续费：{subscription['renewal_date']}")
            output.append(f"  自动续费：{'开启' if subscription.get('auto_renew') else '关闭'}")
            output.append(
                f"  席位使用：{subscription.get('seats_used', 0)} / {subscription.get('seat_quota', 0)}"
            )

        output.append("")
        output.append("客服历史：")
        support = account['support_history']
        output.append(f"  总工单数：{support['total_tickets']}")
        output.append(f"  已解决：{support['resolved_tickets']}")
        output.append(f"  平均解决时长：{support['avg_resolution_time']}")
        output.append("")
        output.append("来源：Account Profile")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Account lookup error: {e}")
        return f"查询账户失败：{str(e)}"
@tool
def get_subscription_status(user_id: str) -> str:
    """
    查询用户当前订阅状态、套餐与续费时间。

    Args:
        user_id: 用户唯一标识

    Returns:
        订阅摘要文本
    """
    try:
        subscription = get_subscription_record(user_id)
        if not subscription:
            return f"未找到用户 {user_id} 的订阅信息。"

        renewal = subscription.get("renewal_date") or "当前无需续费"
        amount = _format_money(subscription.get("amount", 0.0), subscription.get("currency", "CNY"))
        benefits = "、".join(subscription.get("benefits", [])) or "基础支持"

        return (
            f"订阅状态查询成功。\n"
            f"用户：{user_id}\n"
            f"当前套餐：{subscription['plan_name']}\n"
            f"状态：{subscription['status']}\n"
            f"计费周期：{subscription['billing_cycle']}\n"
            f"下次续费：{renewal}\n"
            f"当前费用：{amount}\n"
            f"自动续费：{'开启' if subscription.get('auto_renew') else '关闭'}\n"
            f"席位使用：{subscription.get('seats_used', 0)} / {subscription.get('seat_quota', 0)}\n"
            f"权益摘要：{benefits}\n"
            f"下一步：如需变更套餐，可继续说明是升级、降级还是关闭自动续费。\n"
            f"来源：Subscription Ledger"
        )
    except Exception as e:
        logger.error(f"Get subscription status error: {e}")
        return f"查询订阅失败：{str(e)}"


@tool
def get_latest_invoice(user_id: str) -> str:
    """
    查询用户最近一笔账单。

    Args:
        user_id: 用户唯一标识

    Returns:
        最新账单摘要
    """
    try:
        invoice = get_latest_invoice_record(user_id)
        if not invoice:
            return f"未找到用户 {user_id} 的账单记录。"

        line_summaries = []
        for item in invoice.get("line_items", []):
            line_summaries.append(
                f"- {item['name']}：{_format_money(item['amount'], invoice.get('currency', 'CNY'))}（{item['reason']}）"
            )
        lines = "\n".join(line_summaries) if line_summaries else "- 无明细"

        return (
            f"最近账单查询成功。\n"
            f"账单号：{invoice['invoice_id']}\n"
            f"账单摘要：{invoice['summary']}\n"
            f"出账时间：{invoice['issued_at']}\n"
            f"账单周期：{invoice['period_start']} 至 {invoice['period_end']}\n"
            f"账单状态：{invoice['status']}\n"
            f"总金额：{_format_money(invoice['total_amount'], invoice.get('currency', 'CNY'))}\n"
            f"支付方式：{invoice.get('payment_method', '未记录')}\n"
            f"费用明细：\n{lines}\n"
            f"下一步：如果你怀疑扣费异常，我可以继续解释金额构成，或为你创建账单异常工单。\n"
            f"来源：Billing Ledger"
        )
    except Exception as e:
        logger.error(f"Get latest invoice error: {e}")
        return f"查询最新账单失败：{str(e)}"


@tool
def explain_invoice_charge(invoice_id: str) -> str:
    """
    查询并解释指定账单的扣费原因与金额构成。

    Args:
        invoice_id: 账单号（如 INV-202603-0001）

    Returns:
        账单解释文本
    """
    try:
        invoice = get_invoice_record(invoice_id)
        if not invoice:
            return f"未找到账单 {invoice_id}。"

        explanations = "\n".join(f"- {line}" for line in invoice.get("explanation", [])) or "- 暂无解释信息"
        line_items = "\n".join(
            f"- {item['name']}：{_format_money(item['amount'], invoice.get('currency', 'CNY'))}"
            for item in invoice.get("line_items", [])
        ) or "- 无明细"

        return (
            f"账单扣费说明。\n"
            f"账单号：{invoice['invoice_id']}\n"
            f"总金额：{_format_money(invoice['total_amount'], invoice.get('currency', 'CNY'))}\n"
            f"费用构成：\n{line_items}\n"
            f"扣费解释：\n{explanations}\n"
            f"下一步：如果你认为其中某项费用不合理，我可以继续为你创建账单异常工单并转人工复核。\n"
            f"来源：Billing Explanation Engine"
        )
    except Exception as e:
        logger.error(f"Explain invoice charge error: {e}")
        return f"解释账单失败：{str(e)}"


@tool
def escalate_to_human(user_id: str, reason: str, conversation_summary: str) -> str:
    """
    将对话升级给人工客服。

    当问题复杂、高风险，或用户明确要求人工处理时使用。

    Args:
        user_id: 用户唯一标识
        reason: 升级原因
        conversation_summary: 对话摘要

    Returns:
        升级结果和工单信息
    """
    try:
        ticket = _record_to_ticket(
            create_ticket_record(
                user_id=user_id,
                subject=f"人工升级：{reason}",
                description=conversation_summary,
                priority="high",
                category="escalation",
                metadata={"escalated": True, "reason": reason},
            )
        )

        updated = update_ticket_record(
            ticket_id=ticket.ticket_id,
            status="in_progress",
            notes=f"由AI升级，原因：{reason}",
        )
        if updated:
            ticket = _record_to_ticket(updated)

        account = get_account_record(user_id)
        user_name = account['name'] if account else "用户"

        return (
            f"已升级到人工客服。\n\n"
            f"已创建工单：{ticket.ticket_id}\n"
            f"工单分类：{ticket.category}\n"
            f"优先级：high\n"
            f"升级原因：{reason}\n\n"
            f"交接摘要：{conversation_summary[:120]}\n"
            f"{user_name}，人工客服会在 2 小时内接手，并优先通过邮件与你联系。\n"
            f"下一步：如果你有新的补充信息，可以继续回复本线程，人工客服会一并查看。\n"
            f"来源：Human Escalation Queue"
        )

    except Exception as e:
        logger.error(f"Escalation error: {e}")
        return f"升级人工客服失败：{str(e)}"
# List of all tools for LangChain agent
ALL_TOOLS = [
    search_faq,
    reindex_knowledge_base,
    create_ticket,
    get_ticket_status,
    update_ticket,
    get_user_tickets,
    lookup_account,
    get_subscription_status,
    get_latest_invoice,
    explain_invoice_charge,
    escalate_to_human
]


def get_tool_by_name(name: str) -> Optional[Any]:
    """Get a tool by name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    """Demonstrate support tools usage."""
    print("=" * 60)
    print("Support Tools Demo")
    print("=" * 60)

    # Test FAQ search
    print("\n1. FAQ Search:")
    result = search_faq.invoke({"query": "password reset", "category": None})
    print(result[:200] + "...")

    # Test ticket creation
    print("\n2. Create Ticket:")
    result = create_ticket.invoke({
        "user_id": "demo_user",
        "subject": "Demo ticket",
        "description": "This is a demonstration ticket.",
        "priority": "medium"
    })
    print(result)

    # Test account lookup
    print("\n3. Account Lookup:")
    result = lookup_account.invoke({"user_id": "user_001"})
    print(result[:300] + "...")

    print("\n" + "=" * 60)
