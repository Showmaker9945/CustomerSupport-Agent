# CustomerSupport-Agent Examples

This directory contains example clients for testing the CustomerSupport-Agent API.

## Prerequisites

Make sure the server is running:
```bash
cd CustomerSupport-Agent
uvicorn src.api.main:app --reload
```

## WebSocket Client

Test real-time chat functionality using WebSockets.

### Running
```bash
python examples/websocket_client.py
```

### What it tests:
- Connection establishment
- Message sending and receiving
- Typing indicators
- Concurrent sessions
- Ping/pong keepalive

### Custom usage:
```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws/chat/your_user_id"

    async with websockets.connect(uri) as websocket:
        # Wait for connection confirmation
        await websocket.recv()

        # Send message
        await websocket.send(json.dumps({
            "type": "message",
            "content": "Your message here"
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Agent: {data['content']}")

asyncio.run(chat())
```

## REST Client

Test the REST API endpoints.

### Running
```bash
python examples/rest_client.py
```

### What it tests:
- Health check endpoint
- Chat endpoint
- Ticket retrieval
- Conversation history
- Feedback submission

### Custom usage:
```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Send message
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "content": "Your message here",
                "session_id": "session_123"
            }
        )

        data = response.json()
        print(f"Agent: {data['message']}")

asyncio.run(main())
```

## API Endpoints

### WebSocket
- **URL**: `ws://localhost:8000/ws/chat/{user_id}`
- **Description**: Real-time bidirectional chat

### REST
- **POST /chat** - Send message and get response
- **GET /users/{user_id}/tickets** - Get user's tickets
- **GET /users/{user_id}/history** - Get conversation history
- **POST /feedback** - Submit feedback
- **GET /health** - Health check
- **GET /docs** - Interactive API documentation (Swagger UI)

## Message Formats

### WebSocket Message (Client → Server)
```json
{
  "type": "message",
  "content": "Your message here",
  "session_id": "optional-session-id"
}
```

### WebSocket Response (Server → Client)
```json
{
  "type": "response",
  "content": "Agent response",
  "metadata": {
    "intent": "question",
    "sentiment": {
      "label": "neutral",
      "polarity": 0.1,
      "frustration_score": 0.2
    },
    "sources": ["FAQ Knowledge Base"],
    "escalated": false,
    "ticket_created": null
  },
  "timestamp": "2024-01-31T12:00:00Z"
}
```

### Typing Indicator
```json
{
  "type": "typing",
  "typing": true,
  "timestamp": "2024-01-31T12:00:00Z"
}
```

## Testing Tips

1. **Start the server first:**
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

2. **Check health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **View API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Test with curl (REST):**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"content": "Hello!"}'
   ```

5. **Monitor logs:**
   The server logs all requests and responses, useful for debugging.

## Troubleshooting

### Connection refused
- Make sure the server is running
- Check the port (default: 8000)

### WebSocket errors
- Ensure WebSocket support is enabled
- Check firewall settings

### Slow responses
- First message may be slow (model loading)
- Subsequent messages will be faster

## More Examples

### JavaScript/Node.js WebSocket
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/ws/chat/user_123');

ws.on('open', () => {
  ws.send(JSON.stringify({
    type: 'message',
    content: 'Hello from Node.js!'
  }));
});

ws.on('message', (data) => {
  const response = JSON.parse(data);
  console.log('Agent:', response.content);
});
```

### Python Requests (REST)
```python
import requests

response = requests.post(
    'http://localhost:8000/chat',
    json={'content': 'Hello from REST!'}
)

data = response.json()
print(f"Agent: {data['message']}")
```
