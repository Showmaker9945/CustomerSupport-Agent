"""
Example WebSocket client for testing the support agent.

Run this to test real-time chat functionality.
"""

import asyncio
import json
import websockets
from typing import Optional


async def test_websocket_chat(
    user_id: str = "test_user_ws",
    uri: str = "ws://localhost:8000"
):
    """
    Test WebSocket chat with the support agent.

    Args:
        user_id: User identifier
        uri: WebSocket URI
    """
    ws_uri = f"{uri}/ws/chat/{user_id}"

    print(f"Connecting to {ws_uri}...")

    try:
        async with websockets.connect(ws_uri) as websocket:
            print("✓ Connected!")

            # Receive connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Server: {data.get('type')} - session: {data.get('session_id')}")

            # Send test messages
            test_messages = [
                "Hello! I need help with my account.",
                "How do I reset my password?",
                "Can you check my subscription status?",
                "Thank you for your help!"
            ]

            for msg in test_messages:
                print(f"\nYou: {msg}")

                # Send message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": msg
                }))

                # Receive response
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "typing":
                    print("Agent is typing...")

                elif data.get("type") == "response":
                    metadata = data.get("metadata", {})
                    print(f"Agent: {data.get('content')}")
                    print(f"  [Intent: {metadata.get('intent')}, "
                          f"Sentiment: {metadata.get('sentiment', {}).get('label')}, "
                          f"Escalated: {metadata.get('escalated')}]")

                elif data.get("type") == "error":
                    print(f"Error: {data.get('error')}")

                # Small delay between messages
                await asyncio.sleep(1)

            print("\n✓ Test completed!")

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_typing_indicator(user_id: str = "test_typing"):
    """Test typing indicator functionality."""
    ws_uri = f"ws://localhost:8000/ws/chat/{user_id}"

    print(f"Testing typing indicators...")

    try:
        async with websockets.connect(ws_uri) as websocket:
            # Connection message
            await websocket.recv()

            # Send a message
            await websocket.send(json.dumps({
                "type": "message",
                "content": "Tell me a long story about your product features"
            }))

            # Listen for typing indicator
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "typing":
                    if data.get("typing"):
                        print("✓ Agent is typing...")
                    else:
                        print("✓ Agent stopped typing")
                        break

                elif data.get("type") == "response":
                    print("✓ Received response")
                    break

    except Exception as e:
        print(f"✗ Error: {e}")


async def test_concurrent_sessions(
    user_id: str = "test_concurrent",
    num_sessions: int = 3
):
    """
    Test multiple concurrent WebSocket sessions.

    Args:
        user_id: User identifier
        num_sessions: Number of concurrent sessions
    """
    print(f"Testing {num_sessions} concurrent sessions...")

    async def session_task(session_num: int):
        """Individual session task."""
        ws_uri = f"ws://localhost:8000/ws/chat/{user_id}_{session_num}"

        try:
            async with websockets.connect(ws_uri) as websocket:
                await websocket.recv()  # Connection message

                # Send unique message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": f"Hello from session {session_num}"
                }))

                # Get response
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "response":
                    print(f"✓ Session {session_num}: Got response")

                await asyncio.sleep(0.5)

        except Exception as e:
            print(f"✗ Session {session_num} error: {e}")

    # Run concurrent sessions
    tasks = [session_task(i) for i in range(num_sessions)]
    await asyncio.gather(*tasks)

    print(f"✓ Completed {num_sessions} concurrent sessions")


async def test_ping_pong(user_id: str = "test_ping"):
    """Test ping/pong for connection keepalive."""
    ws_uri = f"ws://localhost:8000/ws/chat/{user_id}"

    print("Testing ping/pong...")

    try:
        async with websockets.connect(ws_uri) as websocket:
            await websocket.recv()  # Connection message

            # Send ping
            await websocket.send(json.dumps({"type": "ping"}))
            print("Sent: ping")

            # Receive pong
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "pong":
                print("✓ Received: pong")
            else:
                print(f"✗ Unexpected response: {data}")

    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("WebSocket Client Tests")
    print("=" * 60)

    # Check if server is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            if response.status_code != 200:
                print("✗ Server is not healthy. Please start the server first.")
                return
    except Exception:
        print("✗ Cannot connect to server. Please start the server first:")
        print("  uvicorn src.api.main:app --reload")
        return

    print("✓ Server is running\n")

    # Run tests
    await test_ping_pong()
    print("\n" + "-" * 60 + "\n")

    await test_websocket_chat()
    print("\n" + "-" * 60 + "\n")

    await test_typing_indicator()
    print("\n" + "-" * 60 + "\n")

    await test_concurrent_sessions(num_sessions=3)
    print("\n" + "-" * 60 + "\n")

    print("✓ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
