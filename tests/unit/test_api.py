"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, manager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns system status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data


class TestChatEndpoint:
    """Test REST chat endpoint."""

    def test_chat_message(self, client):
        """Test sending a chat message via REST."""
        response = client.post(
            "/chat",
            json={
                "content": "Hello, how are you?",
                "session_id": "test_session_123"
            }
        )

        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "intent" in data
        assert "sentiment" in data
        assert "timestamp" in data

    def test_chat_empty_message(self, client):
        """Test chat with empty message fails validation."""
        response = client.post(
            "/chat",
            json={
                "content": "",
                "session_id": "test"
            }
        )

        assert response.status_code == 422  # Validation error

    def test_chat_long_message(self, client):
        """Test chat with very long message."""
        long_message = "Test " * 2000  # 5000 chars

        response = client.post(
            "/chat",
            json={
                "content": long_message,
                "session_id": "test"
            }
        )

        # Should either succeed or fail with validation error
        assert response.status_code in [200, 422]


class TestTicketsEndpoint:
    """Test tickets endpoint."""

    def test_get_user_tickets(self, client):
        """Test getting user tickets."""
        response = client.get("/users/test_user_001/tickets")

        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "tickets" in data
        assert "count" in data

    def test_get_user_tickets_with_status(self, client):
        """Test getting user tickets filtered by status."""
        response = client.get("/users/test_user_001/tickets?status=open")

        assert response.status_code == 200


class TestHistoryEndpoint:
    """Test conversation history endpoint."""

    def test_get_conversation_history(self, client):
        """Test getting conversation history."""
        response = client.get("/users/test_user_001/history?limit=10")

        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "messages" in data
        assert "count" in data


class TestFeedbackEndpoint:
    """Test feedback endpoint."""

    def test_submit_feedback(self, client):
        """Test submitting feedback."""
        response = client.post(
            "/feedback",
            json={
                "user_id": "test_user",
                "session_id": "test_session",
                "message_id": "msg_123",
                "rating": 5,
                "comment": "Great response!"
            }
        )

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "success"

    def test_feedback_invalid_rating(self, client):
        """Test feedback with invalid rating."""
        response = client.post(
            "/feedback",
            json={
                "user_id": "test_user",
                "session_id": "test_session",
                "message_id": "msg_123",
                "rating": 6  # Invalid (must be 1-5)
            }
        )

        assert response.status_code == 422  # Validation error


class TestSessionEndpoints:
    """Test session information endpoints."""

    def test_get_session_info_not_found(self, client):
        """Test getting info for non-existent session."""
        response = client.get("/sessions/nonexistent_session")

        assert response.status_code == 404

    def test_get_user_sessions(self, client):
        """Test getting user's active sessions."""
        response = client.get("/users/test_user_sessions/sessions")

        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "active_sessions" in data
        assert "count" in data


class TestConnectionManager:
    """Test ConnectionManager class."""

    def test_session_info(self):
        """Test session info tracking."""
        # Get non-existent session
        info = manager.get_session_info("nonexistent")
        assert info is None

    def test_user_sessions(self):
        """Test user session tracking."""
        sessions = manager.get_user_sessions("nonexistent_user")
        assert isinstance(sessions, list)
        assert len(sessions) == 0


class TestErrorHandling:
    """Test error handling."""

    def test_404_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method(self, client):
        """Test using invalid HTTP method."""
        response = client.patch("/chat")
        assert response.status_code == 405  # Method not allowed
