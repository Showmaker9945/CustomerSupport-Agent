"""
Unit tests for conversation memory.
"""

import pytest
import tempfile
from pathlib import Path

from src.memory.conversation_memory import ConversationMemory, UserMemoryStore


class TestConversationMemory:
    """Test ConversationMemory class."""

    def test_init(self):
        """Test initialization."""
        memory = ConversationMemory(user_id="test_user")
        assert memory.user_id == "test_user"
        assert memory.messages == []
        assert memory.summary is None

    def test_add_message(self):
        """Test adding messages."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Hello, I need help")
        memory.add_message("assistant", "Hi! How can I help?")

        assert len(memory.messages) == 2
        assert memory.messages[0]["role"] == "user"
        assert memory.messages[0]["content"] == "Hello, I need help"
        assert memory.metadata["turn_count"] == 2

    def test_get_context(self):
        """Test getting conversation context."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")

        context = memory.get_context()
        assert "User: Hello" in context
        assert "Assistant: Hi there!" in context

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        memory = ConversationMemory(user_id="test_user")
        for i in range(10):
            memory.add_message("user", f"Message {i}")

        recent = memory.get_recent_messages(count=3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 7"

    def test_clear(self):
        """Test clearing memory."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Test")
        memory.add_message("assistant", "Response")
        memory.summary = "Test summary"

        memory.clear()
        assert len(memory.messages) == 0
        assert memory.summary is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Test message")

        data = memory.to_dict()
        assert data["user_id"] == "test_user"
        assert len(data["messages"]) == 1
        assert "metadata" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "user_id": "test_user",
            "messages": [
                {"role": "user", "content": "Test", "timestamp": "2024-01-01", "metadata": {}}
            ],
            "summary": "Test summary",
            "metadata": {"message_count": 1}
        }

        memory = ConversationMemory.from_dict(data)
        assert memory.user_id == "test_user"
        assert len(memory.messages) == 1
        assert memory.summary == "Test summary"


class TestUserMemoryStore:
    """Test UserMemoryStore class."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary store."""
        return UserMemoryStore(storage_path=tmp_path)

    def test_get_user_profile_new_user(self, temp_store):
        """Test getting profile for new user."""
        profile = temp_store.get_user_profile("new_user")
        assert profile["name"] is None
        assert profile["sentiments"]["positive"] == 0
        assert "preferences" in profile

    def test_update_user_profile(self, temp_store):
        """Test updating user profile."""
        temp_store.update_user_profile(
            "test_user",
            {"name": "John Doe", "email": "john@example.com"}
        )

        profile = temp_store.get_user_profile("test_user")
        assert profile["name"] == "John Doe"
        assert profile["email"] == "john@example.com"

    def test_save_conversation(self, temp_store):
        """Test saving conversation."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Help needed")
        memory.summary = "Customer had a question"

        conv_id = temp_store.save_conversation(
            "test_user",
            memory,
            title="Support Chat 1"
        )

        assert conv_id is not None
        history = temp_store.get_conversation_history("test_user")
        assert len(history) == 1
        assert history[0]["title"] == "Support Chat 1"

    def test_get_conversation_history(self, temp_store):
        """Test getting conversation history."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Test")

        # Save multiple conversations
        for i in range(5):
            temp_store.save_conversation("test_user", memory)

        history = temp_store.get_conversation_history("test_user", limit=3)
        assert len(history) == 3

    def test_search_history(self, temp_store):
        """Test searching conversation history."""
        memory = ConversationMemory(user_id="test_user")
        memory.add_message("user", "Payment issue")
        memory.summary = "Customer couldn't process payment"

        temp_store.save_conversation(
            "test_user",
            memory,
            title="Payment Problem"
        )

        results = temp_store.search_history("test_user", "payment")
        assert len(results) == 1
        assert "Payment" in results[0]["title"]

    def test_update_sentiment_tracking(self, temp_store):
        """Test sentiment tracking."""
        temp_store.update_sentiment_tracking("test_user", "positive")
        temp_store.update_sentiment_tracking("test_user", "negative")

        profile = temp_store.get_user_profile("test_user")
        assert profile["sentiments"]["positive"] == 1
        assert profile["sentiments"]["negative"] == 1

    def test_add_issue_to_history(self, temp_store):
        """Test adding issues to history."""
        temp_store.add_issue_to_history("test_user", "Login not working")
        temp_store.add_issue_to_history("test_user", "Payment failed", resolved=True)

        profile = temp_store.get_user_profile("test_user")
        assert len(profile["issues"]) == 2
        # First issue should be "Payment failed" (most recent)
        assert profile["issues"][0]["resolved"] is True
        assert profile["issues"][1]["resolved"] is False

    def test_get_user_context(self, temp_store):
        """Test getting formatted user context."""
        temp_store.update_user_profile(
            "test_user",
            {"name": "Jane", "email": "jane@test.com"}
        )
        temp_store.add_issue_to_history("test_user", "Previous issue")

        context = temp_store.get_user_context("test_user")
        assert "Jane" in context
        assert "jane@test.com" in context
        assert "Previous issue" in context
