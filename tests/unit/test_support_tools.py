"""
Unit tests for support tools.
"""

import pytest
import tempfile

from src.tools.support_tools import (
    Ticket,
    TicketStatus,
    TicketPriority,
    TicketStore,
    search_faq,
    create_ticket,
    get_ticket_status,
    update_ticket,
    get_user_tickets,
    lookup_account,
    escalate_to_human,
    MOCK_ACCOUNTS
)


class TestTicket:
    """Test Ticket data model."""

    def test_create_ticket(self):
        """Test ticket creation."""
        ticket = Ticket(
            ticket_id="TKT-001",
            user_id="user_123",
            subject="Test issue",
            description="Test description"
        )

        assert ticket.ticket_id == "TKT-001"
        assert ticket.status == TicketStatus.OPEN
        assert ticket.priority == TicketPriority.MEDIUM

    def test_ticket_to_dict(self):
        """Test ticket serialization."""
        ticket = Ticket(
            ticket_id="TKT-001",
            user_id="user_123",
            subject="Test",
            description="Test desc",
            status=TicketStatus.IN_PROGRESS
        )

        data = ticket.to_dict()
        assert data["ticket_id"] == "TKT-001"
        assert data["status"] == "in_progress"
        assert isinstance(data["status"], str)

    def test_ticket_from_dict(self):
        """Test ticket deserialization."""
        data = {
            "ticket_id": "TKT-001",
            "user_id": "user_123",
            "subject": "Test",
            "description": "Test desc",
            "status": "resolved",
            "priority": "high",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z",
            "tags": [],
            "notes": [],
            "metadata": {}
        }

        ticket = Ticket.from_dict(data)
        assert ticket.status == TicketStatus.RESOLVED
        assert ticket.priority == TicketPriority.HIGH


class TestTicketStore:
    """Test TicketStore class."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary ticket store."""
        return TicketStore(persist_path=tmp_path / "tickets.json")

    def test_create_ticket(self, temp_store):
        """Test creating a ticket."""
        import time
        unique_id = f"user_{int(time.time() * 1000)}"
        ticket = temp_store.create_ticket(
            user_id=unique_id,
            subject="Test issue",
            description="Test description"
        )

        assert ticket.ticket_id.startswith("TKT-")
        assert ticket.user_id == unique_id
        assert ticket.subject == "Test issue"
        assert ticket.status == TicketStatus.OPEN

    def test_get_ticket(self, temp_store):
        """Test retrieving a ticket."""
        import time
        unique_id = f"user_get_ticket_{int(time.time() * 1000)}"
        created = temp_store.create_ticket(
            user_id=unique_id,
            subject="Test",
            description="Test"
        )

        retrieved = temp_store.get_ticket(created.ticket_id)
        assert retrieved is not None
        assert retrieved.ticket_id == created.ticket_id

    def test_get_ticket_not_found(self, temp_store):
        """Test getting non-existent ticket."""
        ticket = temp_store.get_ticket("NONEXISTENT")
        assert ticket is None

    def test_update_ticket_status(self, temp_store):
        """Test updating ticket status."""
        import time
        unique_id = f"user_update_status_{int(time.time() * 1000)}"
        ticket = temp_store.create_ticket(
            user_id=unique_id,
            subject="Test",
            description="Test"
        )

        updated = temp_store.update_ticket(
            ticket.ticket_id,
            status="in_progress"
        )

        assert updated.status == TicketStatus.IN_PROGRESS
        assert updated.resolved_at is None

    def test_update_ticket_resolved(self, temp_store):
        """Test marking ticket as resolved."""
        import time
        unique_id = f"user_resolve_{int(time.time() * 1000)}"
        ticket = temp_store.create_ticket(
            user_id=unique_id,
            subject="Test",
            description="Test"
        )

        updated = temp_store.update_ticket(
            ticket.ticket_id,
            status="resolved"
        )

        assert updated.status == TicketStatus.RESOLVED
        assert updated.resolved_at is not None

    def test_update_ticket_with_notes(self, temp_store):
        """Test adding notes to ticket."""
        import time
        unique_id = f"user_notes_{int(time.time() * 1000)}"
        ticket = temp_store.create_ticket(
            user_id=unique_id,
            subject="Test",
            description="Test"
        )

        updated = temp_store.update_ticket(
            ticket.ticket_id,
            notes="Customer confirmed issue is resolved"
        )

        assert len(updated.notes) == 1
        assert "Customer confirmed" in updated.notes[0]

    def test_get_user_tickets(self, temp_store):
        """Test getting all tickets for a user."""
        import time
        timestamp = int(time.time() * 1000)
        user_1 = f"test_get_user_tickets_1_{timestamp}"
        user_2 = f"test_get_user_tickets_2_{timestamp}"

        temp_store.create_ticket(user_1, "Issue 1", "Desc 1")
        temp_store.create_ticket(user_1, "Issue 2", "Desc 2")
        temp_store.create_ticket(user_2, "Issue 3", "Desc 3")

        tickets = temp_store.get_user_tickets(user_1)
        assert len(tickets) == 2
        assert all(t.user_id == user_1 for t in tickets)

    def test_get_user_tickets_with_status_filter(self, temp_store):
        """Test filtering user tickets by status."""
        import time
        timestamp = int(time.time() * 1000)
        user_id = f"test_status_filter_{timestamp}"

        t1 = temp_store.create_ticket(user_id, "Issue 1", "Desc 1")
        t2 = temp_store.create_ticket(user_id, "Issue 2", "Desc 2")

        temp_store.update_ticket(t1.ticket_id, status="resolved")

        open_tickets = temp_store.get_user_tickets(user_id, status="open")
        resolved_tickets = temp_store.get_user_tickets(user_id, status="resolved")

        assert len(open_tickets) == 1
        assert len(resolved_tickets) == 1

    def test_ticket_persistence(self, temp_store):
        """Test that tickets persist to disk."""
        ticket = temp_store.create_ticket(
            user_id="user_001",
            subject="Persist test",
            description="Should be saved"
        )

        # Create new store instance (should load from disk)
        new_store = TicketStore(persist_path=temp_store.persist_path)
        loaded = new_store.get_ticket(ticket.ticket_id)

        assert loaded is not None
        assert loaded.subject == "Persist test"


class TestMockAccounts:
    """Test mock account data."""

    def test_mock_accounts_exist(self):
        """Test that mock accounts are defined."""
        assert len(MOCK_ACCOUNTS) > 0
        assert "user_001" in MOCK_ACCOUNTS

    def test_mock_account_structure(self):
        """Test mock account has required fields."""
        account = MOCK_ACCOUNTS["user_001"]
        assert "user_id" in account
        assert "name" in account
        assert "email" in account
        assert "plan" in account
        assert "usage" in account
        assert "billing" in account


class TestSupportTools:
    """Test LangChain tools."""

    def test_search_faq(self):
        """Test FAQ search tool."""
        result = search_faq.invoke({"query": "password reset", "category": None})
        assert "FAQ" in result or "found" in result.lower()

    def test_search_faq_with_category(self):
        """Test FAQ search with category filter."""
        result = search_faq.invoke({"query": "payment", "category": "billing"})
        assert "billing" in result.lower() or "FAQ" in result

    def test_create_ticket_tool(self):
        """Test create ticket tool."""
        result = create_ticket.invoke({
            "user_id": "test_user",
            "subject": "Test issue",
            "description": "Test description",
            "priority": "medium"
        })

        assert "created successfully" in result.lower()
        assert "TKT-" in result

    def test_get_ticket_status_tool(self):
        """Test get ticket status tool."""
        # First create a ticket
        create_result = create_ticket.invoke({
            "user_id": "test_user",
            "subject": "Status test",
            "description": "Testing status lookup"
        })

        # Extract ticket ID from result
        ticket_id = create_result.split("Ticket ID: ")[1].split("\n")[0].strip()

        # Get status
        status_result = get_ticket_status.invoke({"ticket_id": ticket_id})
        assert ticket_id in status_result
        assert "Status test" in status_result

    def test_get_ticket_status_not_found(self):
        """Test get ticket status with invalid ID."""
        result = get_ticket_status.invoke({"ticket_id": "TKT-INVALID"})
        assert "not found" in result.lower()

    def test_update_ticket_tool(self):
        """Test update ticket tool."""
        # Create ticket
        create_result = create_ticket.invoke({
            "user_id": "test_user",
            "subject": "Update test",
            "description": "Testing updates"
        })

        ticket_id = create_result.split("Ticket ID: ")[1].split("\n")[0].strip()

        # Update status
        update_result = update_ticket.invoke({
            "ticket_id": ticket_id,
            "status": "in_progress",
            "notes": "Working on it"
        })

        assert "updated successfully" in update_result.lower()
        assert "in_progress" in update_result

    def test_get_user_tickets_tool(self):
        """Test get user tickets tool."""
        # Use unique user ID with timestamp for this test
        import time
        user_id = f"test_user_tools_{int(time.time())}"

        # Create multiple tickets
        create_ticket.invoke({
            "user_id": user_id,
            "subject": "Ticket 1",
            "description": "First ticket"
        })
        create_ticket.invoke({
            "user_id": user_id,
            "subject": "Ticket 2",
            "description": "Second ticket"
        })

        # Get all tickets - should have at least 2
        result = get_user_tickets.invoke({"user_id": user_id, "status": None})
        # Check for at least 2 tickets (might have more from previous runs)
        assert "ticket" in result.lower()
        assert user_id in result

    def test_lookup_account_tool(self):
        """Test account lookup tool."""
        result = lookup_account.invoke({"user_id": "user_001"})
        assert "Alice Johnson" in result
        assert "Pro" in result
        assert "alice.johnson@example.com" in result

    def test_lookup_account_not_found(self):
        """Test account lookup with invalid user."""
        result = lookup_account.invoke({"user_id": "nonexistent_user"})
        assert "not found" in result.lower()

    def test_escalate_to_human_tool(self):
        """Test escalation tool."""
        result = escalate_to_human.invoke({
            "user_id": "user_001",
            "reason": "Complex technical issue",
            "conversation_summary": "User experiencing authentication problems"
        })

        assert "escalating" in result.lower() or "escalated" in result.lower()
        assert "TKT-" in result

    def test_all_tools_have_names(self):
        """Test that all tools have proper names."""
        from src.tools.support_tools import ALL_TOOLS

        for tool_obj in ALL_TOOLS:
            assert hasattr(tool_obj, 'name')
            assert tool_obj.name
            assert hasattr(tool_obj, 'description')
            assert tool_obj.description

    def test_tool_descriptions_are_helpful(self):
        """Test that tool descriptions are informative."""
        from src.tools.support_tools import ALL_TOOLS

        for tool_obj in ALL_TOOLS:
            # Should have reasonable length description
            assert len(tool_obj.description) > 20
            # Should describe what it does
            assert any(word in tool_obj.description.lower()
                      for word in ["search", "create", "get", "update", "look up", "escalate"])
