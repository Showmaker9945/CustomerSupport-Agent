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
    # Fallback to empty dict if file doesn't exist or fails to load
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
        self._lock = threading.Lock()

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
    Search the FAQ knowledge base for relevant answers.

    Args:
        query: The question or topic to search for
        category: Optional category to filter by (billing, account, technical, etc.)

    Returns:
        Formatted FAQ search results with confidence scores
    """
    try:
        store = get_faq_store()
        results = store.search(query, category=category, top_k=3)

        if not results:
            return f"No FAQs found for: {query}"

        output_parts = [f"Found {len(results)} relevant FAQ(s):\n"]

        for i, result in enumerate(results, 1):
            output_parts.append(
                f"\n{i}. **{result.question}**\n"
                f"   {result.answer}\n"
                f"   *Category: {result.category} | Relevance: {result.confidence:.0%}*"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"FAQ search error: {e}")
        return f"Error searching FAQ: {str(e)}"


@tool
def create_ticket(
    user_id: str,
    subject: str,
    description: str,
    priority: str = "medium"
) -> str:
    """
    Create a new support ticket for the user.

    Args:
        user_id: The user's unique identifier
        subject: Brief summary of the issue
        description: Detailed description of the issue
        priority: Priority level (low, medium, high, urgent)

    Returns:
        Confirmation message with ticket ID
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
            f"✓ Support ticket created successfully!\n"
            f"  Ticket ID: {ticket.ticket_id}\n"
            f"  Status: {ticket.status.value}\n"
            f"  Priority: {ticket.priority.value}\n"
            f"  Created: {ticket.created_at}\n"
            f"\n  You'll receive updates at your email address."
        )

    except Exception as e:
        logger.error(f"Create ticket error: {e}")
        return f"Error creating ticket: {str(e)}"


@tool
def get_ticket_status(ticket_id: str) -> str:
    """
    Get the current status of a support ticket.

    Args:
        ticket_id: The ticket ID (e.g., TKT-20250131-001)

    Returns:
        Ticket status and details
    """
    try:
        store = get_ticket_store()
        ticket = store.get_ticket(ticket_id)

        if not ticket:
            return f"Ticket {ticket_id} not found. Please check the ticket ID."

        output = [
            f"Ticket: {ticket.ticket_id}",
            f"Subject: {ticket.subject}",
            f"Status: {ticket.status.value}",
            f"Priority: {ticket.priority.value}",
            f"Created: {ticket.created_at}",
            f"Updated: {ticket.updated_at}"
        ]

        if ticket.assigned_to:
            output.append(f"Assigned to: {ticket.assigned_to}")

        if ticket.resolved_at:
            output.append(f"Resolved: {ticket.resolved_at}")

        if ticket.tags:
            output.append(f"Tags: {', '.join(ticket.tags)}")

        if ticket.notes:
            output.append("\nNotes:")
            for note in ticket.notes:
                output.append(f"  - {note}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Get ticket status error: {e}")
        return f"Error retrieving ticket: {str(e)}"


@tool
def update_ticket(
    ticket_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None
) -> str:
    """
    Update a support ticket's status or add notes.

    Args:
        ticket_id: The ticket ID
        status: New status (open, in_progress, waiting_customer, resolved, closed)
        notes: Additional notes to add to the ticket

    Returns:
        Confirmation of the update
    """
    try:
        store = get_ticket_store()
        ticket = store.update_ticket(
            ticket_id=ticket_id,
            status=status,
            notes=notes
        )

        if not ticket:
            return f"Ticket {ticket_id} not found."

        result = f"✓ Ticket {ticket_id} updated successfully.\n"
        result += f"  New status: {ticket.status.value}\n"
        result += f"  Updated: {ticket.updated_at}"

        if notes:
            result += f"\n  Note added: {notes}"

        return result

    except Exception as e:
        logger.error(f"Update ticket error: {e}")
        return f"Error updating ticket: {str(e)}"


@tool
def get_user_tickets(user_id: str, status: Optional[str] = None) -> str:
    """
    Get all support tickets for a user.

    Args:
        user_id: The user's unique identifier
        status: Optional filter by status (open, resolved, etc.)

    Returns:
        List of user's tickets with summaries
    """
    try:
        store = get_ticket_store()
        tickets = store.get_user_tickets(user_id, status=status)

        if not tickets:
            return f"No tickets found for user {user_id}."

        output = [f"Found {len(tickets)} ticket(s) for user {user_id}:\n"]

        for ticket in tickets:
            status_emoji = {
                "open": "🔵",
                "in_progress": "🟡",
                "waiting_customer": "⏸️",
                "resolved": "✅",
                "closed": "📁"
            }.get(ticket.status.value, "⚪")

            output.append(
                f"{status_emoji} **{ticket.ticket_id}** - {ticket.subject}\n"
                f"   Status: {ticket.status.value} | "
                f"Priority: {ticket.priority.value} | "
                f"Created: {ticket.created_at[:10]}"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Get user tickets error: {e}")
        return f"Error retrieving tickets: {str(e)}"


@tool
def lookup_account(user_id: str) -> str:
    """
    Look up user account information including plan, usage, and billing.

    Args:
        user_id: The user's unique identifier

    Returns:
        User account details
    """
    try:
        # In production, this would query a real database
        account = MOCK_ACCOUNTS.get(user_id)

        if not account:
            return f"Account not found for user ID: {user_id}"

        output = [
            f"📋 Account Information for {account['name']}",
            f"",
            f"**User ID:** {account['user_id']}",
            f"**Email:** {account['email']}",
            f"**Plan:** {account['plan']}",
            f"**Status:** {account['status'].replace('_', ' ').title()}",
            f"**Member Since:** {account['member_since']}"
        ]

        if account.get('company'):
            output.append(f"**Company:** {account['company']}")

        output.extend([
            f"",
            f"**Usage:**",
            f"  Storage: {account['usage']['storage_used']} / {account['usage']['storage_limit']}",
            f"  API Calls: {account['usage']['api_calls_this_month']} / {account['usage']['api_limit']}",
            f"  Team Members: {account['usage']['team_members']} / {account['usage']['team_limit']}"
        ])

        if account.get('billing'):
            output.append(f"")
            output.append(f"**Billing:**")
            billing = account['billing']
            output.append(f"  Invoice Amount: {billing['invoice_amount']}")
            if billing.get('last_payment'):
                output.append(f"  Last Payment: {billing['last_payment']}")
            if billing.get('payment_method'):
                output.append(f"  Payment Method: {billing['payment_method']}")

        output.append(f"")
        output.append(f"**Support History:**")
        support = account['support_history']
        output.append(f"  Total Tickets: {support['total_tickets']}")
        output.append(f"  Resolved: {support['resolved_tickets']}")
        output.append(f"  Avg Resolution Time: {support['avg_resolution_time']}")

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Account lookup error: {e}")
        return f"Error looking up account: {str(e)}"


@tool
def escalate_to_human(user_id: str, reason: str, conversation_summary: str) -> str:
    """
    Escalate the conversation to a human support agent.

    Use this when the issue is too complex, technical, or sensitive for AI,
    or when the user explicitly requests human assistance.

    Args:
        user_id: The user's unique identifier
        reason: Why escalation is needed
        conversation_summary: Summary of the conversation so far

    Returns:
        Escalation confirmation and ticket information
    """
    try:
        store = get_ticket_store()

        # Create escalation ticket
        ticket = store.create_ticket(
            user_id=user_id,
            subject=f"HANDOFF: {reason}",
            description=conversation_summary,
            priority="high",
            metadata={"escalated": True, "reason": reason}
        )

        # Update ticket status
        store.update_ticket(
            ticket.ticket_id,
            status="in_progress",
            notes=f"Escalated from AI assistant. Reason: {reason}"
        )

        account = MOCK_ACCOUNTS.get(user_id)
        user_name = account['name'] if account else "User"

        return (
            f"⚠️ **Escalating to Human Support**\n\n"
            f"I've connected you with our human support team for personalized assistance.\n\n"
            f"**Ticket Created:** {ticket.ticket_id}\n"
            f"**Priority:** High\n"
            f"**Reason:** {reason}\n\n"
            f"📧 {user_name}, you'll receive an email confirmation shortly. "
            f"Our team typically responds within:\n"
            f"  • Pro plans: 2-4 hours\n"
            f"  • Enterprise plans: 1 hour or less\n\n"
            f"Thank you for your patience. Is there anything else I can help with in the meantime?"
        )

    except Exception as e:
        logger.error(f"Escalation error: {e}")
        return f"Error escalating to human: {str(e)}"


# List of all tools for LangChain agent
ALL_TOOLS = [
    search_faq,
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
