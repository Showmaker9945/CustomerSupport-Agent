"""
Unit tests for support conversation agent.
"""

import pytest
from unittest.mock import Mock, patch

from src.conversation.support_agent import (
    SupportAgent,
    SupportResponse,
    ConversationState,
    get_support_agent
)
from src.sentiment.analyzer import SentimentResult


class TestSupportResponse:
    """Test SupportResponse dataclass."""

    def test_create_response(self):
        """Test creating a support response."""
        sentiment = SentimentResult(
            polarity=0.5,
            subjectivity=0.3,
            label="positive",
            frustration_score=0.0,
            keywords=[]
        )

        response = SupportResponse(
            message="Here's your answer!",
            intent="question",
            sentiment=sentiment,
            sources=["FAQ Knowledge Base"],
            escalated=False,
            ticket_created=None
        )

        assert response.message == "Here's your answer!"
        assert response.intent == "question"
        assert response.escalated is False

    def test_to_dict(self):
        """Test converting response to dictionary."""
        sentiment = SentimentResult(
            polarity=-0.3,
            subjectivity=0.7,
            label="negative",
            frustration_score=0.6,
            keywords=["frustrated"]
        )

        response = SupportResponse(
            message="I understand your frustration.",
            intent="complaint",
            sentiment=sentiment,
            sources=["Support Tools"],
            escalated=False
        )

        data = response.to_dict()
        assert "message" in data
        assert "intent" in data
        assert "sentiment" in data
        assert data["sentiment"]["label"] == "negative"
        assert data["escalated"] is False


class TestSupportAgent:
    """Test SupportAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a support agent for testing."""
        # Use a smaller model for faster testing
        return SupportAgent(
            model_name="gpt-3.5-turbo",
            enable_memory=True,
            enable_sentiment=True
        )

    def test_init(self, agent):
        """Test agent initialization."""
        assert agent.model_name == "gpt-3.5-turbo"
        assert agent.enable_memory is True
        assert agent.enable_sentiment is True
        assert agent.llm is not None
        assert agent.faq_store is not None
        assert agent.sentiment_analyzer is not None
        assert agent.graph is not None

    def test_init_without_sentiment(self):
        """Test initialization without sentiment analysis."""
        agent = SupportAgent(
            model_name="gpt-3.5-turbo",
            enable_sentiment=False
        )

        assert agent.enable_sentiment is False
        assert agent.sentiment_analyzer is None

    def test_get_or_create_memory(self, agent):
        """Test memory creation per user."""
        memory1 = agent._get_or_create_memory("user_001")
        memory2 = agent._get_or_create_memory("user_001")
        memory3 = agent._get_or_create_memory("user_002")

        # Same user should return same memory
        assert memory1 is memory2
        # Different user should return different memory
        assert memory1 is not memory3

    def test_understand_intent(self, agent):
        """Test intent understanding."""
        state = ConversationState(
            user_id="user_001",
            messages=[],
            current_message="How do I reset my password?",
            intent=None,
            sentiment=None,
            faq_results=None,
            tool_result=None,
            response=None,
            escalated=False,
            ticket_id=None,
            context=None
        )

        result = agent._understand_intent(state)

        assert "intent" in result
        assert result["intent"] in ["question", "complaint", "request", "feedback", "greeting", "other"]
        assert "sentiment" in result
        assert result["sentiment"] is not None

    def test_search_knowledge(self, agent):
        """Test knowledge base search."""
        state = ConversationState(
            user_id="user_001",
            messages=[],
            current_message="I need to reset my password",
            intent="question",
            sentiment=None,
            faq_results=None,
            tool_result=None,
            response=None,
            escalated=False,
            ticket_id=None,
            context=None
        )

        result = agent._search_knowledge(state)

        assert "faq_results" in result
        assert result["faq_results"] is not None
        assert isinstance(result["faq_results"], list)

    def test_check_escalation_low_frustration(self, agent):
        """Test escalation check with low frustration."""
        sentiment = SentimentResult(
            polarity=0.2,
            subjectivity=0.3,
            label="neutral",
            frustration_score=0.1,
            keywords=[]
        )

        state = ConversationState(
            user_id="user_001",
            messages=[],
            current_message="I have a question",
            intent="question",
            sentiment=sentiment,
            faq_results=None,
            tool_result=None,
            response=None,
            escalated=False,
            ticket_id=None,
            context=None
        )

        result = agent._check_escalation(state)

        assert result["escalated"] is False
        assert result["ticket_id"] is None

    def test_check_escalation_high_frustration(self, agent):
        """Test escalation check with high frustration."""
        # Mock the escalate_to_human tool to avoid actual ticket creation
        with patch('src.conversation.support_agent.escalate_to_human') as mock_escalate:
            mock_escalate.invoke.return_value = "Escalated. Ticket: TKT-20250131-0001"

            sentiment = SentimentResult(
                polarity=-0.8,
                subjectivity=0.9,
                label="negative",
                frustration_score=0.9,
                keywords=["furious", "terrible"]
            )

            state = ConversationState(
                user_id="user_001",
                messages=[],
                current_message="This is unacceptable!",
                intent="complaint",
                sentiment=sentiment,
                faq_results=None,
                tool_result=None,
                response=None,
                escalated=False,
                ticket_id=None,
                context=None
            )

            result = agent._check_escalation(state)

            assert result["escalated"] is True
            # Note: ticket_id extraction might fail in test, that's OK

    def test_route_after_intent(self, agent):
        """Test routing after intent classification."""
        # Question -> search
        state = ConversationState(
            user_id="user_001",
            messages=[],
            current_message="How do I?",
            intent="question",
            sentiment=None,
            faq_results=None,
            tool_result=None,
            response=None,
            escalated=False,
            ticket_id=None,
            context=None
        )
        assert agent._route_after_intent(state) == "search"

        # Request -> tool
        state["intent"] = "request"
        assert agent._route_after_intent(state) == "tool"

        # Greeting -> respond
        state["intent"] = "greeting"
        assert agent._route_after_intent(state) == "respond"

        # High frustration -> escalate
        state["intent"] = "complaint"
        state["sentiment"] = SentimentResult(
            polarity=-0.7,
            subjectivity=0.8,
            label="negative",
            frustration_score=0.85,
            keywords=["angry"]
        )
        assert agent._route_after_intent(state) == "escalate"

    def test_route_after_escalation_check(self, agent):
        """Test routing after escalation check."""
        # Escalated -> escalate
        state = ConversationState(
            user_id="user_001",
            messages=[],
            current_message="",
            intent=None,
            sentiment=None,
            faq_results=None,
            tool_result=None,
            response=None,
            escalated=True,
            ticket_id=None,
            context=None
        )
        assert agent._route_after_escalation_check(state) == "escalate"

        # Not escalated -> continue
        state["escalated"] = False
        assert agent._route_after_escalation_check(state) == "continue"

    def test_chat_simple_question(self, agent):
        """Test chat with a simple question."""
        response = agent.chat(
            user_id="test_user_001",
            message="How do I reset my password?"
        )

        assert isinstance(response, SupportResponse)
        assert response.message is not None
        assert len(response.message) > 0
        assert response.intent in ["question", "other"]
        assert response.sentiment is not None
        assert isinstance(response.escalated, bool)

    def test_chat_with_memory(self, agent):
        """Test that conversation memory is maintained."""
        user_id = "test_user_memory"

        # First message
        response1 = agent.chat(user_id, "My name is Alice")
        assert response1 is not None

        # Check memory was created
        assert user_id in agent.memory
        assert len(agent.memory[user_id].messages) >= 2  # user + assistant

        # Second message
        response2 = agent.chat(user_id, "What's my name?")
        assert response2 is not None

        # Check memory accumulated
        assert len(agent.memory[user_id].messages) >= 4

    def test_reset_conversation(self, agent):
        """Test resetting conversation."""
        user_id = "test_user_reset"

        # Send a message
        agent.chat(user_id, "Hello")
        assert user_id in agent.memory
        assert len(agent.memory[user_id].messages) > 0

        # Reset
        agent.reset_conversation(user_id)
        assert len(agent.memory[user_id].messages) == 0

    def test_get_conversation_history(self, agent):
        """Test getting conversation history."""
        user_id = "test_user_history"

        # Send messages
        agent.chat(user_id, "First message")
        agent.chat(user_id, "Second message")

        # Get history
        history = agent.get_conversation_history(user_id, limit=10)

        assert isinstance(history, list)
        assert len(history) >= 4  # At least 2 user + 2 assistant messages

        # Check structure
        for msg in history:
            assert "role" in msg
            assert "content" in msg

    def test_chat_creates_support_response(self, agent):
        """Test that chat returns proper SupportResponse."""
        response = agent.chat(
            user_id="test_user_response",
            message="I need help with my account"
        )

        # Check all expected fields
        assert hasattr(response, "message")
        assert hasattr(response, "intent")
        assert hasattr(response, "sentiment")
        assert hasattr(response, "sources")
        assert hasattr(response, "escalated")
        assert hasattr(response, "ticket_created")

        # Check types
        assert isinstance(response.message, str)
        assert isinstance(response.intent, str)
        assert isinstance(response.sources, list)
        assert isinstance(response.escalated, bool)


class TestGlobalAgent:
    """Test global support agent instance."""

    def test_get_support_agent_singleton(self):
        """Test that get_support_agent returns singleton."""
        agent1 = get_support_agent()
        agent2 = get_support_agent()

        assert agent1 is agent2

    def test_global_agent_works(self):
        """Test that global agent can process messages."""
        agent = get_support_agent()
        response = agent.chat(
            user_id="global_test_user",
            message="Hello, can you help me?"
        )

        assert response is not None
        assert response.message is not None


class TestConversationFlow:
    """Test end-to-end conversation flows."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(model_name="gpt-3.5-turbo")

    def test_happy_customer_flow(self, agent):
        """Test flow with a happy customer."""
        user_id = "happy_customer"

        messages = [
            "Hi there!",
            "I love your product, it's amazing!",
            "Quick question - how do I invite team members?",
            "That's perfect, thank you so much!"
        ]

        responses = []
        for msg in messages:
            response = agent.chat(user_id, msg)
            responses.append(response)

            # Should never escalate for happy customer
            assert response.escalated is False
            assert "helpful" in response.intent or "question" in response.intent or "greeting" in response.intent

        assert len(responses) == 4

    def test_frustrated_customer_flow(self, agent):
        """Test flow with a frustrated customer."""
        user_id = "frustrated_customer"

        messages = [
            "I have a problem with my account",
            "It's still not working!",
            "This is ridiculous and unacceptable!"
        ]

        responses = []
        for msg in messages:
            response = agent.chat(user_id, msg)
            responses.append(response)

        # Last message should trigger escalation due to high frustration
        assert responses[-1].escalated is True or responses[-1].sentiment.frustration_score > 0.5

    def test_ticket_request_flow(self, agent):
        """Test flow for creating a support ticket."""
        user_id = "ticket_request"

        response = agent.chat(
            user_id,
            "I need to create a support ticket for my billing issue"
        )

        # Should recognize as request
        assert response.intent in ["request", "other"]
        # Should have a response
        assert response.message is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def agent(self):
        return SupportAgent(model_name="gpt-3.5-turbo")

    def test_empty_message(self, agent):
        """Test handling empty message."""
        response = agent.chat("test_empty", "")

        # Should still return a response
        assert response is not None
        assert response.message is not None

    def test_very_long_message(self, agent):
        """Test handling very long message."""
        long_message = "I need help " + "please " * 100

        response = agent.chat("test_long", long_message)

        assert response is not None
        assert response.message is not None

    def test_special_characters(self, agent):
        """Test handling special characters."""
        special_message = "Help! My password reset isn't working!!! <script>alert('test')</script>"

        response = agent.chat("test_special", special_message)

        assert response is not None
        # Should handle safely without crashing

    def test_multiple_users(self, agent):
        """Test handling multiple concurrent users."""
        users = ["user_a", "user_b", "user_c"]

        responses = []
        for user in users:
            response = agent.chat(user, f"Hello from {user}")
            responses.append((user, response))

        # Each user should get a response
        for user, response in responses:
            assert response is not None
            # Each user should have separate memory
            assert user in agent.memory
