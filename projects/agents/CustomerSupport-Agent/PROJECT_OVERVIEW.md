# CustomerSupport-Agent - Project Overview

A production-ready customer support AI agent with memory, knowledge base, and real-time chat capabilities.

## 🎯 Project Overview

This project demonstrates a complete customer support system featuring:
- **Conversational AI Agent** using LangGraph for conversation flow
- **Long-term Memory** per user with conversation summarization
- **Knowledge Base (RAG)** with ChromaDB vector search
- **Support Tools** for ticket and account management
- **Sentiment Analysis** for intelligent routing
- **Human Handoff** when escalation is needed
- **Real-time Chat** via WebSocket
- **REST API** for integrations

### V2 Architecture Upgrade

- Migrated to **LangChain/LangGraph v1** agent APIs (`create_agent` + middleware).
- Introduced **multi-agent orchestration** (`Supervisor`, `Knowledge`, `Action`, `Escalation`).
- Added **HITL run interruption + resume** (`POST /runs/{thread_id}/resume`).
- Upgraded retrieval to **hybrid RAG** (vector + BM25 + optional reranker).
- Added **SSE stream endpoint** (`GET /chat/stream`) and **knowledge reindex API**.

## 📁 Project Structure

```
CustomerSupport-Agent/
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI + WebSocket server
│   ├── conversation/
│   │   └── support_agent.py        # LangGraph conversation agent
│   ├── knowledge/
│   │   └── faq_store.py            # FAQ knowledge base (RAG)
│   ├── memory/
│   │   └── conversation_memory.py  # Conversation + user memory
│   ├── sentiment/
│   │   └── analyzer.py             # Sentiment & frustration analysis
│   ├── tools/
│   │   └── support_tools.py        # Ticket/account management tools
│   ├── utils/                      # Shared utilities
│   └── config.py                   # Configuration settings
│
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
│
├── data/
│   ├── knowledge_base/             # FAQ documents
│   ├── chroma_db/                  # Vector store
│   └── tickets.json                # Ticket storage
│
├── examples/
│   ├── websocket_client.py         # WebSocket test client
│   ├── rest_client.py              # REST API test client
│   └── README.md                   # Examples documentation
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data for sentiment analysis
python -m textblob.download_corpora
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Prepare Knowledge Base (Optional)

Place FAQ documents in `data/knowledge_base/`:
- Supported formats: .txt, .md, .pdf
- Sample FAQs are loaded automatically

### 4. Run the Server

```bash
# Development mode with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/ws/chat/{user_id}
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/unit/test_support_agent.py

# Run with verbose output
pytest -v
```

## 📡 API Usage

### WebSocket (Real-time Chat)

```python
import websockets
import json
import asyncio

async def chat():
    uri = "ws://localhost:8000/ws/chat/user_123"

    async with websockets.connect(uri) as websocket:
        # Wait for connection confirmation
        await websocket.recv()

        # Send message
        await websocket.send(json.dumps({
            "type": "message",
            "content": "How do I reset my password?"
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Agent: {data['content']}")

asyncio.run(chat())
```

### REST API

```python
import httpx

async def send_message():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat",
            json={
                "content": "I need help with my account",
                "session_id": "session_123"
            }
        )

        data = response.json()
        print(f"Agent: {data['message']}")
        print(f"Intent: {data['intent']}")
        print(f"Escalated: {data['escalated']}")

asyncio.run(send_message())
```

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# API Keys
OPENAI_API_KEY=your_key_here

# Model Settings
# Model used: gpt-4o-mini (default) or gpt-3.5-turbo

# Memory
MAX_CONVERSATION_HISTORY=20
SESSION_TIMEOUT_HOURS=24

# Sentiment & Escalation
SENTIMENT_THRESHOLD=0.3
HANDOFF_THRESHOLD=-0.5
FRUSTRATION_KEYWORDS=angry,frustrated,terrible

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_WS_CONNECTIONS_PER_USER=5
```

## 📊 Key Features

### 1. Conversation Memory
- Short-term: Current conversation context
- Long-term: User profiles, preferences, past interactions
- Automatic summarization when conversations get long
- Persistent storage (JSON-based, upgradable to database)

### 2. Knowledge Base (RAG)
- 20+ pre-loaded FAQs for SaaS product
- Vector-based semantic search
- Category filtering (billing, account, technical, etc.)
- Confidence scoring for results

### 3. Support Tools
- `search_faq`: Search knowledge base
- `create_ticket`: Create support tickets
- `get_ticket_status`: Check ticket status
- `update_ticket`: Update tickets
- `get_user_tickets`: List user's tickets
- `lookup_account`: Retrieve account info
- `escalate_to_human`: Handoff to human support

### 4. Sentiment Analysis
- Polarity detection (-1 to +1)
- Frustration scoring (0 to 1)
- Keyword-based frustration detection
- Conversation trend analysis (improving/stable/declining)
- Automatic escalation recommendations

### 5. Routing Logic
```
High frustration (≥0.8) → Immediate escalation
question → Search FAQ + answer
request → Use appropriate tool
complaint + frustration → Consider escalation
greeting/feedback → Direct response
```

## 🎨 Conversation Flow

```
1. User sends message
   ↓
2. Understand Intent (classify + sentiment)
   ↓
3. Check Escalation (frustration threshold)
   ↓
4a. Search Knowledge Base (if question)
4b. Use Tool (if request)
4c. Generate Response (if greeting/feedback)
   ↓
5. Generate Final Response
   - Uses LLM with context
   - Incorporates FAQ results
   - Adjusts tone based on sentiment
   - Adds empathy for frustrated users
   ↓
6. Return to user
```

## 📈 Test Coverage

Current test coverage by module:

| Module | Coverage | Tests |
|--------|----------|-------|
| Config | 96% | - |
| Memory | 84% | 15 tests |
| Knowledge Base | 81% | 19 tests |
| Support Tools | 78% | 26 tests |
| Sentiment | 91% | 34 tests |
| Support Agent | 63% | 28 tests |
| API | 50% | 16 tests |

**Total**: 138 tests passing

## 🔍 Monitoring & Logging

The application logs:
- All chat interactions
- Sentiment analysis results
- Tool executions
- Escalation decisions
- WebSocket connections
- Errors and exceptions

Log levels:
- `DEBUG`: Detailed information
- `INFO`: General operational info
- `WARNING`: Warnings (e.g., high frustration)
- `ERROR`: Errors that need attention

## 🚢 Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

- Set `DEBUG=false`
- Set `ENVIRONMENT=production`
- Use production-grade database instead of JSON storage
- Configure proper CORS origins
- Set up SSL/TLS for HTTPS
- Use rate limiting
- Configure logging to file/aggregator

### Scaling

- Run multiple workers: `--workers 4`
- Use a load balancer (nginx, AWS ALB)
- Deploy database for persistent storage
- Use Redis for session management
- Monitor with Prometheus/Grafana

## 📚 Further Enhancements

Potential improvements:
1. **Database**: Replace JSON with PostgreSQL/MongoDB
2. **Caching**: Add Redis for faster response
3. **Analytics**: Track metrics, customer satisfaction
4. **Multi-language**: Add internationalization
5. **Voice Support**: Integrate speech-to-text
6. **Chat Widget**: Frontend widget for websites
7. **Admin Dashboard**: UI for managing tickets/users
8. **A/B Testing**: Test different response strategies
9. **Fine-tuning**: Fine-tune LLM on support conversations
10. **Multi-channel**: Email, SMS, social media integration

## 🤝 Contributing

This is a demonstration project. Feel free to:
- Fork and modify for your use case
- Add new features
- Improve test coverage
- Fix bugs
- Share feedback

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with:
- LangChain & LangGraph
- FastAPI
- OpenAI GPT
- ChromaDB
- TextBlob
- Sentence Transformers
- Pydantic

---

**Project**: CustomerSupport-Agent
**Version**: 1.0.0
**Last Updated**: January 2025
