# CustomerSupport-Agent

> 一个中文优先的智能客服 Agent Demo，具备长期记忆、知识库检索与实时对话能力。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-red.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-138%20passing-brightgreen.svg)](tests/)

---

## 📋 目录

- [Overview](#overview)
- [Use Case](#use-case)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Example Conversations](#example-conversations)
- [Customization Guide](#customization-guide)
- [Deployment](#deployment)
- [Testing](#testing)

---

## 🎯 Overview

**CustomerSupport-Agent** 是一个偏工程化、可演示的客服 AI 系统，当前重点展示以下能力：

- **多轮对话**：支持上下文感知与连续问答
- **长期记忆**：支持跨会话保留用户事实与历史
- **知识库检索（RAG）**：用于 FAQ 与说明类问题回答
- **情绪识别**：辅助风险判断与升级策略
- **客服工具链**：支持工单、账户查询、人工升级
- **人工介入**：高风险动作触发 HITL 审批
- **实时通信**：支持 WebSocket、REST 与 SSE

项目基于 **LangGraph**、**LangChain**、**FastAPI** 与 **Qwen** 构建，适合作为实习、面试或作品集中的 Agent 工程 Demo。

---

## 🚀 V2 Upgrade Highlights

- **LangChain/LangGraph v1 API**：迁移到 `create_agent` + middleware 体系。
- **多 Agent 核心**：包含 `Supervisor`、`Knowledge`、`Action`、`Escalation`。
- **人机协作（HITL）**：高风险工具调用会中断并等待 `approve/edit/reject`。
- **混合检索 RAG**：向量检索 + BM25 融合 + 可选 reranker。
- **持久化记忆**：支持线程级 checkpoint 与用户级长期记忆。
- **新增接口**：
  - `POST /chat`：支持 `thread_id`，返回 `run_status`、`interrupts`、`citations`
  - `POST /runs/{thread_id}/resume`
  - `GET /chat/stream`：SSE 流式事件
  - `POST /knowledge/reindex`

---

## 💼 Use Case

### 要解决的问题

企业客服系统往往需要同时满足这些诉求：
- 7x24 小时即时响应
- 记住用户历史上下文
- 自动回答高频 FAQ
- 识别用户是否焦虑或不满
- 在合适时机升级给人工
- 跟踪工单与服务过程

### 解决方案

这是一个客服 Agent Demo，它能够：
- ✅ 用知识库快速回答常见问题
- ✅ 保留用户偏好和关键历史
- ✅ 识别情绪与挫败感
- ✅ 创建、查询和更新工单
- ✅ 将复杂或高风险场景升级给人工
- ✅ 通过 REST / WebSocket / SSE 对外提供能力

### 目标使用者

- **客服团队**：自动处理重复问题，把精力集中在复杂场景
- **最终用户**：更快获得答案和处理结果
- **业务团队**：降低客服成本，提升满意度

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Web App    │  │  Mobile App  │  │  Third-party │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                  │                  │
└─────────┼─────────────────┼──────────────────┼──────────────────┘
          │                 │                  │
          │ WebSocket       │                  │
          │ / HTTP          │                  │
┌─────────┼─────────────────┼──────────────────┼──────────────────┐
│         │                 │                  │                  │
│  ┌──────▼───────────────────────────────────────────────────┐  │
│  │              FastAPI + WebSocket Server                  │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │         Connection Manager (WebSocket)            │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │              Support Agent (LangGraph)                │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  Nodes:                                        │  │  │
│  │  │  1. Understand Intent  → Classify + Sentiment   │  │  │
│  │  │  2. Check Escalation  → High frustration?      │  │  │
│  │  │  3. Search Knowledge  → FAQ retrieval (RAG)     │  │  │
│  │  │  4. Use Tools         → Tickets, accounts, etc. │  │  │
│  │  │  5. Generate Response → LLM with context        │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼────────┐  ┌───▼──────┐  ┌──────▼─────────┐
│ Conversation  │  │ Knowledge│  │   Support     │
│ Memory        │  │ Base     │  │   Tools       │
│               │  │(ChromaDB)│  │               │
│ • Short-term  │  │ • FAQs   │  │ • Tickets     │
│ • Long-term   │  │ • Vector │  │ • Accounts    │
│ • Summarize   │  │   Search │  │ • Escalation  │
└───────────────┘  └──────────┘  └────────────────┘
       │
┌──────▼──────────┐
│ Sentiment       │
│ Analysis        │
│                 │
│ • Polarity      │
│ • Frustration   │
│ • Keywords      │
└─────────────────┘
```

### Data Flow

```
1. User Message
   ↓
2. WebSocket/REST receives message
   ↓
3. Agent adds to conversation memory
   ↓
4. Analyze sentiment + classify intent
   ↓
5. Route based on analysis:
   - High frustration → Check escalation
   - Question → Search knowledge base
   - Request → Use tools
   - Greeting → Direct response
   ↓
6. Generate response using:
   - Conversation context
   - FAQ results
   - Tool outputs
   - User profile
   ↓
7. Return response via WebSocket/REST
   ↓
8. Update memory and user profile
```

---

## ✨ Features

### 🧠 Intelligent Conversation

- **LangGraph Orchestration**: State-based conversation flow
- **Multi-turn Dialogues**: Maintains context across conversation
- **Intent Classification**: Understands user purpose (question, complaint, request, etc.)
- **Context-Aware Responses**: Uses conversation history and user profile

### 💾 Advanced Memory

- **Short-term Memory**: Current conversation with recent messages
- **Long-term Memory**: User profiles, preferences, past interactions
- **Automatic Summarization**: Condenses long conversations
- **Searchable History**: Find past conversations by keyword

### 📚 Knowledge Base (RAG)

- **20+ Pre-loaded FAQs**: Common SaaS support questions
- **Semantic Search**: Vector-based matching using ChromaDB
- **Category Filtering**: Billing, account, technical, security, etc.
- **Confidence Scoring**: Returns relevance scores

### 🎯 Sentiment Analysis

- **Polarity Detection**: Positive/negative/neutral classification (-1 to +1)
- **Frustration Scoring**: Detects customer frustration (0 to 1)
- **Keyword Detection**: 50+ frustration indicators
- **Trend Analysis**: Improving, stable, or declining sentiment

### 🔧 Support Tools

- **`search_faq`**: Search knowledge base
- **`create_ticket`**: Create support tickets
- **`get_ticket_status`**: Check ticket status
- **`update_ticket`**: Update tickets with notes
- **`get_user_tickets`**: List user's tickets
- **`lookup_account`**: Retrieve account information
- **`escalate_to_human`**: Handoff to human support

### 👥 Intelligent Routing

```
Frustration ≥ 0.8          → Immediate escalation
3+ frustrated messages      → Escalation
Declining + negative        → Escalation
Question                    → Search FAQ + answer
Request                     → Use appropriate tool
Greeting/Feedback           → Friendly direct response
```

### ⚡ Real-time Communication

- **WebSocket**: Bidirectional real-time chat
- **REST API**: Request-response alternative
- **Typing Indicators**: Show when agent is "typing"
- **Session Management**: Track multiple connections per user
- **Connection Limits**: Prevent abuse

### 🔒 Production Ready

- **Error Handling**: Graceful failure recovery
- **Logging**: Comprehensive request/response logging
- **Type Safety**: Pydantic validation throughout
- **Tests**: 138 passing tests with good coverage
- **Documentation**: Complete API docs and examples

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- pip package manager

### Installation

```bash
# Clone or navigate to project
cd CustomerSupport-Agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP data
python -m textblob.download_corpora
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required:**
```bash
OPENAI_API_KEY=sk-your-key-here
```

**Optional (with defaults):**
```bash
DEBUG=false
ENVIRONMENT=development
MODEL_NAME=gpt-4o-mini
HANDOFF_THRESHOLD=-0.5
MAX_WS_CONNECTIONS_PER_USER=5
```

### Run the Server

```bash
# Development mode (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","timestamp":"...","version":"1.0.0","components":{...}}
```

---

## 📡 API Documentation

### WebSocket Endpoint

**URL:** `ws://localhost:8000/ws/chat/{user_id}`

**Client → Server:**
```json
{
  "type": "message",
  "content": "How do I reset my password?",
  "session_id": "optional-session-id"
}
```

**Server → Client:**
```json
{
  "type": "response",
  "content": "You can reset your password by...",
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

### REST Endpoints

#### POST /chat

Send a message and get response.

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How do I reset my password?",
    "session_id": "session_123"
  }'
```

**Response:**
```json
{
  "message": "You can reset your password...",
  "intent": "question",
  "sentiment": "neutral",
  "sentiment_polarity": 0.1,
  "frustration_score": 0.2,
  "sources": ["FAQ Knowledge Base"],
  "escalated": false,
  "ticket_created": null,
  "timestamp": "2024-01-31T12:00:00Z"
}
```

#### GET /users/{user_id}/tickets

Get user's support tickets.

```bash
curl http://localhost:8000/users/user_001/tickets
```

#### GET /users/{user_id}/history

Get conversation history.

```bash
curl http://localhost:8000/users/user_001/history?limit=20
```

#### POST /feedback

Submit feedback on AI response.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "session_id": "session_123",
    "message_id": "msg_001",
    "rating": 5,
    "comment": "Very helpful!"
  }'
```

#### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 💬 Example Conversations

### Example 1: Simple Question (Happy Path)

```
User: How do I reset my password?

Agent: You can reset your password by going to Settings > Security > Password
       and clicking 'Reset Password'. You'll receive an email with a secure link
       to create a new password. The link expires in 24 hours.

       [Intent: question, Sentiment: neutral, Escalated: false]
```

### Example 2: Ticket Creation

```
User: I need to create a support ticket for billing issue

Agent: I'll help you create a support ticket for your billing issue.
       Could you please provide more details about the problem?

       [Ticket created: TKT-20250131-0001]

User: The amount charged is wrong

Agent: I've updated your ticket with the details about the incorrect charge.
       Our billing team will review it and get back to you within 24 hours.
       Your ticket ID is TKT-20250131-0001 for reference.

       [Intent: request, Ticket updated, Escalated: false]
```

### Example 3: Frustration Handling

```
User: I have a problem

Agent: I'm sorry to hear that. What seems to be the problem?

User: It's still not working! This is ridiculous!

Agent: I understand your frustration, and I apologize for the difficulty.
       Let me connect you with a human agent who can better assist you.

       I've created ticket TKT-20250131-0002 and escalated your case.
       Our team typically responds within 2-4 hours for Pro accounts.

       [Intent: complaint, Sentiment: negative, Escalated: true]
```

### Example 4: Multi-turn with Memory

```
User: My name is Alice
Agent: Hello Alice! How can I help you today?

User: What's my name?
Agent: Your name is Alice. How can I assist you?

User: I forgot my password
Agent: No problem, Alice. I can help you reset your password. Go to Settings...

       [Agent remembers user's name from earlier in conversation]
```

---

## 🎨 Customization Guide

### Adding Custom FAQs

Create a JSON file at `data/knowledge_base/custom_faqs.json`:

```json
[
  {
    "question": "What is your refund policy?",
    "answer": "We offer full refunds within 30 days...",
    "category": "billing",
    "keywords": ["refund", "money back", "return"]
  }
]
```

Load it:
```python
from src.knowledge.faq_store import create_faq_store

store = create_faq_store()
store.load_faqs_from_file("./data/knowledge_base/custom_faqs.json")
```

### Customizing Frustration Keywords

Edit `src/sentiment/analyzer.py`:

```python
FRUSTRATION_KEYWORDS = {
    "your_word": 0.7,
    "another_phrase": 0.9,
    # ... existing keywords
}
```

### Adjusting Escalation Threshold

In `.env`:
```bash
HANDOFF_THRESHOLD=-0.5  # More negative = more sensitive
```

Or in `src/config.py`:
```python
handoff_threshold: float = -0.3  # Less sensitive
```

### Adding New Tools

1. Create tool function in `src/tools/support_tools.py`:

```python
@tool
def your_custom_tool(param: str) -> str:
    """Your tool description."""
    # Implementation
    return "result"
```

2. Add to `ALL_TOOLS` list:

```python
ALL_TOOLS = [
    # ... existing tools
    your_custom_tool
]
```

3. Use in conversation agent routing logic in `_use_tool()` method.

### Customizing System Prompt

Edit `src/conversation/support_agent.py`:

```python
SYSTEM_PROMPT = """You are a helpful support agent for [YOUR COMPANY].

Your guidelines:
- Be friendly and professional
- [Your custom rules]
...
"""
```

### Adding Database Storage

Replace JSON storage with PostgreSQL:

```python
# src/tools/support_tools.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class TicketStore:
    def __init__(self):
        engine = create_engine(settings.database_url)
        self.Session = sessionmaker(bind=engine)

    def create_ticket(self, ...):
        session = self.Session()
        ticket = Ticket(...)
        session.add(ticket)
        session.commit()
```

---

## 🐳 Deployment

### Docker Deployment

**Dockerfile** (included):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP data
RUN python -m textblob.download_corpora

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t customer-support-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key customer-support-agent
```

### Docker Compose

**docker-compose.yml** (included):
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=false
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

**Run with Docker Compose:**
```bash
docker-compose up -d
```

### Cloud Deployment

**AWS EC2:**
```bash
# Launch EC2 instance
# SSH in
git clone <your-repo>
cd CustomerSupport-Agent
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Google Cloud Run:**
```bash
gcloud run deploy customer-support-agent \
  --source . \
  --platform python \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

**Azure Container Instances:**
```bash
az container create \
  --resource-group myResourceGroup \
  --name mySupportAgent \
  --image your-registry/customer-support-agent \
  --environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Verbose mode
pytest -v
```

### Test Coverage

| Module | Coverage | Tests |
|--------|----------|-------|
| Memory | 84% | 15 |
| Knowledge Base | 81% | 19 |
| Support Tools | 78% | 26 |
| Sentiment | 91% | 34 |
| Support Agent | 63% | 28 |
| API | 50% | 16 |

**Total**: 138 tests passing

### Integration Tests

```bash
# Run integration tests
pytest tests/test_support_agent.py
```

---

## 📚 Additional Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Detailed technical documentation
- **[examples/README.md](examples/README.md)** - Client examples and usage
- **[API Docs](http://localhost:8000/docs)** - Interactive Swagger UI

---

## 🛣️ Roadmap

- [ ] PostgreSQL integration for production storage
- [ ] Redis caching for faster response
- [ ] Multi-language support
- [ ] Voice interface integration
- [ ] Admin dashboard for viewing analytics
- [ ] Customer satisfaction analytics
- [ ] A/B testing framework
- [ ] Email/SMS channel integration
- [ ] Frontend chat widget
- [ ] Fine-tuned LLM on support conversations

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- **Examples**: See [examples/](examples/)

---

**Built with ❤️ using LangChain, LangGraph, FastAPI, and OpenAI**
