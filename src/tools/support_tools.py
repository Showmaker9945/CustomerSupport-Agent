"""
Support tools for customer service agent.

Provides ticket management, account lookup, and FAQ search tools
that can be used by LangChain agents.
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

from ..config import settings
from ..knowledge.faq_store import FAQStore, create_faq_store

logger = logging.getLogger(__name__)


# Load mock accounts from external file
def _load_mock_accounts() -> Dict[str, Any]:
    """Load mock account data from JSON file."""
    mock_path = Path(__file__).parent.parent.parent / "data" / "mock_accounts.json"
    try:
        if mock_path.exists():
            with open(mock_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load mock accounts from {mock_path}: {e}")
    # Built-in fallback so tools and tests still work without external file.
    return {
        "user_001": {
            "user_id": "user_001",
            "name": "Alice Johnson",
            "email": "alice.johnson@example.com",
            "plan": "Pro",
            "status": "active",
            "member_since": "2023-01-15",
            "company": "Acme Labs",
            "usage": {
                "storage_used": "42GB",
                "storage_limit": "100GB",
                "api_calls_this_month": 12840,
                "api_limit": 50000,
                "team_members": 8,
                "team_limit": 20,
            },
            "billing": {
                "invoice_amount": "$99.00",
                "last_payment": "2026-02-01",
                "payment_method": "Visa **** 4242",
            },
            "support_history": {
                "total_tickets": 12,
                "resolved_tickets": 11,
                "avg_resolution_time": "6h",
            },
        }
    }


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


# Mock account database for demo - loaded from external file
MOCK_ACCOUNTS: Dict[str, Any] = _load_mock_accounts()


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

        if not results:
            return f"未找到与该问题相关的知识：{query}"

        output_parts = [f"找到 {len(results)} 条相关知识（混合检索+重排）：\n"]

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
    priority: str = "medium"
) -> str:
    """
    为用户创建新的客服工单。

    Args:
        user_id: 用户唯一标识
        subject: 问题主题
        description: 问题详情
        priority: 优先级（low/medium/high/urgent）

    Returns:
        创建结果和工单号
    """
    try:
        store = get_ticket_store()
        ticket = store.create_ticket(
            user_id=user_id,
            subject=subject,
            description=description,
            priority=priority
        )

        return (
            f"工单创建成功。\n"
            f"工单号：{ticket.ticket_id}\n"
            f"状态：{ticket.status.value}\n"
            f"优先级：{ticket.priority.value}\n"
            f"创建时间：{ticket.created_at}\n"
            f"后续进展会通过邮件通知你。"
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
        store = get_ticket_store()
        ticket = store.get_ticket(ticket_id)

        if not ticket:
            return f"未找到工单 {ticket_id}，请检查工单号是否正确。"

        output = [
            f"工单号：{ticket.ticket_id}",
            f"主题：{ticket.subject}",
            f"状态：{ticket.status.value}",
            f"优先级：{ticket.priority.value}",
            f"创建时间：{ticket.created_at}",
            f"更新时间：{ticket.updated_at}"
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
        store = get_ticket_store()
        ticket = store.update_ticket(
            ticket_id=ticket_id,
            status=status,
            notes=notes
        )

        if not ticket:
            return f"未找到工单 {ticket_id}。"

        result = f"工单 {ticket_id} 更新成功。\n"
        result += f"新状态：{ticket.status.value}\n"
        result += f"更新时间：{ticket.updated_at}"

        if notes:
            result += f"\n新增备注：{notes}"

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
        store = get_ticket_store()
        tickets = store.get_user_tickets(user_id, status=status)

        if not tickets:
            return f"用户 {user_id} 暂无工单记录。"

        output = [f"用户 {user_id} 共找到 {len(tickets)} 条工单：\n"]

        for ticket in tickets:
            output.append(
                f"{ticket.ticket_id} - {ticket.subject}\n"
                f"状态：{ticket.status.value} | "
                f"优先级：{ticket.priority.value} | "
                f"创建日期：{ticket.created_at[:10]}"
            )

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
        account = MOCK_ACCOUNTS.get(user_id)

        if not account:
            return f"未找到用户 {user_id} 的账户信息。"

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

        output.append("")
        output.append("客服历史：")
        support = account['support_history']
        output.append(f"  总工单数：{support['total_tickets']}")
        output.append(f"  已解决：{support['resolved_tickets']}")
        output.append(f"  平均解决时长：{support['avg_resolution_time']}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Account lookup error: {e}")
        return f"查询账户失败：{str(e)}"
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
        store = get_ticket_store()

        ticket = store.create_ticket(
            user_id=user_id,
            subject=f"人工升级：{reason}",
            description=conversation_summary,
            priority="high",
            metadata={"escalated": True, "reason": reason}
        )

        store.update_ticket(
            ticket.ticket_id,
            status="in_progress",
            notes=f"由AI升级，原因：{reason}"
        )

        account = MOCK_ACCOUNTS.get(user_id)
        user_name = account['name'] if account else "用户"

        return (
            f"已升级到人工客服。\n\n"
            f"已创建工单：{ticket.ticket_id}\n"
            f"优先级：high\n"
            f"升级原因：{reason}\n\n"
            f"{user_name}，我们会尽快通过邮件与你联系。"
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

