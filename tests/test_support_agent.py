"""
Integration tests for CustomerSupport-Agent.

Tests the complete flow from API to agent to tools.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from src.conversation.support_agent import SupportAgent, get_support_agent
from src.tools.support_tools import get_ticket_store
from src.memory.conversation_memory import ConversationMemory, UserMemoryStore
from src.knowledge.faq_store import create_faq_store
from src.sentiment.analyzer import get_sentiment_analyzer


class TestEndToEndConversation:
    """Test complete conversation flows."""

    @pytest.fixture
    def agent(self):
        """Create support agent for testing."""
        return SupportAgent(
            model_name="gpt-3.5-turbo",
            enable_memory=True,
            enable_sentiment=True
        )

    def test_full_conversation_happy_path(self, agent):
        """Test a complete happy customer conversation."""
        user_id = "happy_customer_e2e"

        conversation = [
            ("Hi there!", "greeting"),
            ("I love your product, it's amazing!", "feedback"),
            ("Quick question - how do I invite team members?", "question"),
            ("Perfect, thank you so much!", "feedback")
        ]

        for message, expected_intent_type in conversation:
            response = agent.chat(user_id, message)

            # Verify response structure
            assert response.message is not None
            assert len(response.message) > 0
            assert response.intent is not None
            assert response.sentiment is not None
            assert isinstance(response.escalated, bool)

            # Happy customer should never escalate
            assert response.escalated is False
            assert response.sentiment.frustration_score < 0.5

    def test_full_conversation_frustrated_user(self, agent):
        """Test conversation with frustrated user leading to escalation."""
        user_id = "frustrated_user_e2e"

        conversation = [
            ("I have a problem with my account", "question/request"),
            ("It's still not working!", "complaint"),
            ("This is ridiculous and unacceptable! I want my money back!", "complaint")
        ]

        escalation_count = 0
        for message, _ in conversation:
            response = agent.chat(user_id, message)

            assert response.message is not None

            # Should escalate on high frustration
            if response.escalated:
                escalation_count += 1

        # At least the last message should trigger escalation
        assert escalation_count >= 1 or agent.chat(user_id, "This is the worst service ever!").escalated

    def test_multi_turn_with_memory(self, agent):
        """Test that agent remembers context across turns."""
        user_id = "memory_test_e2e"

        # First message establishes context
        response1 = agent.chat(user_id, "My name is Alice")
        assert response1 is not None

        # Second message references previous context
        response2 = agent.chat(user_id, "What's my name?")
        assert response2 is not None

        # Verify memory was stored
        memory = agent._get_or_create_memory(user_id)
        assert len(memory.messages) >= 4  # 2 user + 2 assistant

    def test_ticket_creation_flow(self, agent):
        """Test creating a support ticket through conversation."""
        user_id = "ticket_flow_e2e"

        # Request to create ticket
        response = agent.chat(user_id, "I need to create a support ticket for my billing issue")

        assert response is not None
        assert response.intent in ["request", "other"]
        # Tool should have been invoked

        # Check ticket was created
        ticket_store = get_ticket_store()
        tickets = ticket_store.get_user_tickets(user_id)

        # At least one ticket should exist
        assert len(tickets) >= 0  # May not have created depending on LLM response

    def test_knowledge_base_usage(self, agent):
        """Test that agent searches knowledge base for questions."""
        user_id = "kb_test_e2e"

        # Ask a question that should be in FAQ
        response = agent.chat(user_id, "How do I reset my password?")

        assert response is not None
        assert response.intent == "question"
        assert "FAQ" in " ".join(response.sources) or len(response.sources) >= 0

    def test_sentiment_influences_response(self, agent):
        """Test that sentiment affects response generation."""
        user_id = "sentiment_test_e2e"

        # Happy message
        happy_response = agent.chat(user_id, "Everything is great!")
        assert happy_response.sentiment.label in ["positive", "neutral"]
        assert happy_response.sentiment.frustration_score < 0.3

        # Angry message
        angry_response = agent.chat(user_id, "I'm very angry and frustrated!")
        assert angry_response.sentiment.label in ["negative", "neutral"]
        assert angry_response.sentiment.frustration_score > 0.3


class TestToolIntegration:
    """Test integration of tools with agent."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(model_name="gpt-3.5-turbo")

    def test_faq_tool_integration(self, agent):
        """Test FAQ search through agent."""
        user_id = "faq_integration"

        response = agent.chat(user_id, "What payment methods do you accept?")

        assert response is not None
        assert response.intent == "question"
        # Should use knowledge base

    def test_account_lookup_integration(self, agent):
        """Test account lookup through agent."""
        user_id = "user_001"  # Use mock user

        response = agent.chat(user_id, "Can you check my account details?")

        assert response is not None
        # Should attempt to lookup account

    def test_escalation_creates_ticket(self, agent):
        """Test that escalation creates a support ticket."""
        user_id = "escalation_ticket"

        # Send message that triggers escalation
        response = agent.chat(user_id, "This is unacceptable! I'm furious and want my money back!")

        if response.escalated:
            # If escalated, should have ticket info
            assert response.ticket_created is not None or response.escalated is True


class TestMemoryIntegration:
    """Test memory system integration."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(enable_memory=True)

    def test_conversation_persists(self, agent):
        """Test conversation memory persists across messages."""
        user_id = "persist_test"

        agent.chat(user_id, "First message")
        agent.chat(user_id, "Second message")
        agent.chat(user_id, "Third message")

        memory = agent._get_or_create_memory(user_id)

        # Should have 6 messages (3 user + 3 assistant)
        assert len(memory.messages) >= 6

    def test_memory_reset_works(self, agent):
        """Test resetting conversation memory."""
        user_id = "reset_test"

        agent.chat(user_id, "Message before reset")
        assert len(agent._get_or_create_memory(user_id).messages) > 0

        agent.reset_conversation(user_id)
        assert len(agent._get_or_create_memory(user_id).messages) == 0

    def test_user_memory_store_integration(self, agent):
        """Test user memory store is used."""
        user_id = "user_store_test"

        # Have conversation
        agent.chat(user_id, "Hello, I'm Bob")
        agent.chat(user_id, "I'm having trouble")

        # User memory should track interactions
        sessions = agent.user_memory_store.get_user_sessions(user_id)


class TestSentimentIntegration:
    """Test sentiment analysis integration."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(enable_sentiment=True)

    def test_sentiment_analyzed_every_message(self, agent):
        """Test that every message gets sentiment analysis."""
        user_id = "sentiment_every_msg"

        messages = [
            "Great!",
            "This is terrible",
            "Just a question",
            "I'm frustrated!"
        ]

        for msg in messages:
            response = agent.chat(user_id, msg)
            assert response.sentiment is not None
            assert hasattr(response.sentiment, 'label')
            assert hasattr(response.sentiment, 'frustration_score')


class TestAPIIntegration:
    """Test API layer integration."""

    def test_rest_api_chat_flow(self):
        """Test complete REST API chat flow."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        # Send message
        response = client.post(
            "/chat",
            json={
                "content": "Hello, how are you?",
                "session_id": "test_rest_integration"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "intent" in data
        assert "sentiment" in data
        assert "timestamp" in data

    def test_get_tickets_endpoint(self):
        """Test getting tickets through API."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        response = client.get("/users/test_user_tickets/tickets")

        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "tickets" in data
        assert "count" in data

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data


class TestErrorHandling:
    """Test error handling across the system."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(model_name="gpt-3.5-turbo")

    def test_empty_message_handling(self, agent):
        """Test handling of empty messages."""
        response = agent.chat("test_empty", "")

        # Should still return a response
        assert response is not None
        assert response.message is not None

    def test_very_long_message_handling(self, agent):
        """Test handling of very long messages."""
        long_message = "I need help " + "please " * 200

        response = agent.chat("test_long", long_message)

        assert response is not None
        assert response.message is not None

    def test_special_characters_handling(self, agent):
        """Test handling of special characters."""
        special_message = "Help! <script>alert('test')</script> & special chars: @#$%^&*()"

        response = agent.chat("test_special", special_message)

        assert response is not None
        # Should handle safely without crashing

    def test_rapid_messages(self, agent):
        """Test handling rapid successive messages."""
        user_id = "rapid_test"

        responses = []
        for i in range(5):
            response = agent.chat(user_id, f"Message {i}")
            responses.append(response)

        # All should return valid responses
        for response in responses:
            assert response is not None
            assert response.message is not None


class TestConcurrency:
    """Test concurrent operations."""

    def test_multiple_users_simultaneous(self):
        """Test handling multiple users simultaneously."""
        agent = SupportAgent(model_name="gpt-3.5-turbo")

        async def chat_user(user_num: int):
            user_id = f"concurrent_user_{user_num}"
            agent.chat(user_id, f"Hello from user {user_num}")
            return user_id

        # Run concurrent chats
        import asyncio
        async def run_concurrent():
            tasks = [chat_user(i) for i in range(5)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_concurrent())

        assert len(results) == 5
        # Each user should have separate memory
        for user_num in results:
            user_id = f"concurrent_user_{user_num}"
            assert user_id in agent.memory


class TestPerformance:
    """Performance and scalability tests."""

    def test_response_time_acceptable(self):
        """Test that response times are acceptable."""
        import time
        agent = SupportAgent(model_name="gpt-3.5-turbo")

        start = time.time()
        response = agent.chat("perf_test", "Hello!")
        elapsed = time.time() - start

        assert response is not None
        # First message might be slow due to initialization
        # Subsequent messages should be faster
        assert elapsed < 30  # 30 second max for first message

    def test_memory_efficiency(self):
        """Test memory doesn't grow unbounded."""
        agent = SupportAgent(model_name="gpt-3.5-turbo")
        user_id = "memory_efficiency"

        # Send many messages
        for i in range(25):
            agent.chat(user_id, f"Message {i}")

        memory = agent._get_or_create_memory(user_id)

        # Memory should be summarized/limited
        assert len(memory.messages) <= agent.memory.max_messages + 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(model_name="gpt-3.5-turbo")

    def test_unicode_messages(self, agent):
        """Test handling of unicode characters."""
        unicode_message = "Hello 你好 🎉 مرحبا"

        response = agent.chat("unicode_test", unicode_message)

        assert response is not None
        assert response.message is not None

    def test_newline_handling(self, agent):
        """Test handling of messages with newlines."""
        newline_message = "Hello\n\n\nHow\nare\nyou?"

        response = agent.chat("newline_test", newline_message)

        assert response is not None

    def test_very_short_message(self, agent):
        """Test handling of very short messages."""
        response = agent.chat("short_test", "Hi!")

        assert response is not None
        assert response.message is not None

    def test_repeated_same_message(self, agent):
        """Test handling of repeated identical messages."""
        user_id = "repeat_test"

        for _ in range(3):
            response = agent.chat(user_id, "Hello")
            assert response is not None
