"""
Example REST client for testing the support agent API.

Run this to test REST endpoints.
"""

import httpx
import asyncio
from typing import Optional


class SupportClient:
    """Simple REST client for support agent API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def chat(self, message: str, session_id: Optional[str] = None) -> dict:
        """
        Send a chat message.

        Args:
            message: Message content
            session_id: Optional session identifier

        Returns:
            Response dictionary
        """
        payload = {"content": message}
        if session_id:
            payload["session_id"] = session_id

        response = await self.client.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_tickets(self, user_id: str, status: Optional[str] = None) -> dict:
        """
        Get user's tickets.

        Args:
            user_id: User identifier
            status: Optional status filter

        Returns:
            Tickets dictionary
        """
        params = {"status": status} if status else {}
        response = await self.client.get(
            f"{self.base_url}/users/{user_id}/tickets",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_history(self, user_id: str, limit: int = 20) -> dict:
        """
        Get conversation history.

        Args:
            user_id: User identifier
            limit: Maximum messages

        Returns:
            History dictionary
        """
        response = await self.client.get(
            f"{self.base_url}/users/{user_id}/history",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    async def submit_feedback(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> dict:
        """
        Submit feedback.

        Args:
            user_id: User identifier
            session_id: Session identifier
            message_id: Message identifier
            rating: Rating (1-5)
            comment: Optional comment

        Returns:
            Response dictionary
        """
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message_id": message_id,
            "rating": rating
        }
        if comment:
            payload["comment"] = comment

        response = await self.client.post(f"{self.base_url}/feedback", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_health(self) -> dict:
        """
        Get system health.

        Returns:
            Health status dictionary
        """
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


async def main():
    """Run example usage."""
    print("=" * 60)
    print("REST API Client Example")
    print("=" * 60)

    async with SupportClient() as client:
        # Check health
        print("\n1. Health Check")
        print("-" * 40)
        try:
            health = await client.get_health()
            print(f"Status: {health['status']}")
            print(f"Version: {health['version']}")
            print(f"Components: {list(health['components'].keys())}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            print("Please make sure the server is running:")
            print("  uvicorn src.api.main:app --reload")
            return

        # Chat interaction
        print("\n2. Chat Interaction")
        print("-" * 40)
        session_id = "example_session_123"

        messages = [
            "Hello, I need help!",
            "How do I reset my password?",
            "Thank you!"
        ]

        for msg in messages:
            print(f"\nYou: {msg}")
            try:
                response = await client.chat(msg, session_id=session_id)
                print(f"Agent: {response['message']}")
                print(f"  [Intent: {response['intent']}, "
                      f"Sentiment: {response['sentiment']}, "
                      f"Escalated: {response['escalated']}]")
            except Exception as e:
                print(f"✗ Error: {e}")

        # Get tickets
        print("\n3. Get User Tickets")
        print("-" * 40)
        try:
            tickets = await client.get_tickets(session_id)
            print(f"User: {tickets['user_id']}")
            print(f"Total tickets: {tickets['count']}")
        except Exception as e:
            print(f"✗ Error: {e}")

        # Get history
        print("\n4. Get Conversation History")
        print("-" * 40)
        try:
            history = await client.get_history(session_id)
            print(f"User: {history['user_id']}")
            print(f"Message count: {history['count']}")
        except Exception as e:
            print(f"✗ Error: {e}")

        # Submit feedback
        print("\n5. Submit Feedback")
        print("-" * 40)
        try:
            feedback = await client.submit_feedback(
                user_id=session_id,
                session_id=session_id,
                message_id="msg_001",
                rating=5,
                comment="Great help!"
            )
            print(f"Feedback status: {feedback['status']}")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("✓ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
