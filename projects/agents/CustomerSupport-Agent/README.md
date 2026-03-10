# CustomerSupport-Agent

一个中文优先的客服 Agent Demo，基于 LangGraph 条件路由、多 Agent 协作、HITL 审批、混合检索 RAG、长短期记忆与 FastAPI 构建，适合作为实习简历里的工程化 Agent 项目。

## 简历版摘要

一句话版本：

基于 LangGraph 和 FastAPI 实现中文客服 Agent 系统，支持条件路由、多 Agent 协作、HITL 审批、混合检索 RAG 与持久记忆，并完成可演示 API 与测试闭环。

三点版本：

- 使用 LangGraph 搭建中文客服多 Agent 图，按意图、风险和上下文在知识检索、业务执行、人工升级等路径间条件路由。
- 实现 middleware、HITL 审批、线程级状态恢复与用户级长期记忆，保证高风险动作可中断、可恢复、可追踪。
- 基于 FastAPI 提供 REST、SSE、WebSocket 与 Swagger Demo，并补齐单元测试，形成可直接展示的完整后端项目。

更多求职用表述见 [CAREER_GUIDE.md](./CAREER_GUIDE.md)。

## 项目亮点

- LangGraph 条件边编排：分析、路由、检索、执行、升级、校验、回复职责清晰
- LangChain v1 / LangGraph v1：对齐 `create_agent` 与 middleware 体系
- 中文优先：默认中文输出、中文情绪识别、中文 FAQ 与业务示例
- 轻量增强 RAG：混合检索、查询规范化、分类推断、子问题拆分、一次改写兜底
- Memory 治理：线程级短期记忆、用户级长期记忆、记忆抽取与召回
- HITL：高风险工具先中断审批，再恢复同一线程执行
- FastAPI 演示友好：REST、WebSocket、SSE、Swagger、回归测试

## 适合展示的能力

- 会搭 LangGraph，而不是只写单条 chain
- 会做 Agent 的工程化落地，而不是只调模型
- 会把 RAG、记忆、工具调用、审批流整合成完整后端系统
- 会补测试、补文档、补运行方式，让项目真的能演示

## 核心架构

主流程：

`用户请求 -> 分析节点 -> 条件路由 -> 知识检索 / 业务执行 / 人工升级 -> 校验节点 -> 中文回复`

关键模块：

- `src/conversation/support_agent/`：Support Agent 图、服务层、中间件、持久化适配
- `src/knowledge/faq_store.py`：FAQ 知识库、混合检索、融合与轻量增强 RAG
- `src/tools/support_tools.py`：账户、订阅、账单、工单、人工升级等业务工具
- `src/api/main.py`：FastAPI、Swagger、SSE、WebSocket、恢复接口
- `src/sentiment/analyzer.py`：中文优先情绪识别

## 当前技术栈

- Python 3.11
- uv
- FastAPI
- LangChain 1.x
- LangGraph 1.x
- ChromaDB + BM25
- Sentence Transformers
- PostgreSQL + pgvector（可选，用于持久化和长期记忆扩展）

## 目录结构

```text
CustomerSupport-Agent/
├─ src/
│  ├─ api/                    # FastAPI / SSE / WebSocket
│  ├─ conversation/           # SupportAgent 图与服务层
│  ├─ knowledge/              # FAQStore / RAG
│  ├─ sentiment/              # 中文优先情绪分析
│  ├─ tools/                  # 工单、账户、账单、升级等工具
│  └─ config.py               # 配置中心
├─ tests/unit/                # 单元与 API 回归测试
├─ data/                      # 本地 demo 数据
├─ examples/                  # 调试与体验说明
├─ pyproject.toml             # uv 项目配置
├─ uv.lock                    # uv 锁文件
└─ .env.example               # 环境变量模板
```

## 快速开始

### 1. 安装依赖

```bash
uv sync --all-groups
```

如果是首次进入项目，也可以显式创建环境：

```bash
uv venv
uv sync --all-groups
```

### 2. 初始化 NLP 依赖

```bash
uv run python -m textblob.download_corpora
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

至少需要配置：

```bash
LLM_API_KEY=你的千问 API Key
LLM_PROVIDER=qwen
LLM_MODEL=qwen-plus
LLM_HIGH_QUALITY_MODEL=qwen-max
```

说明：

- 主配置项是 `LLM_API_KEY`
- `OPENAI_API_KEY` 仅保留向后兼容，不再作为默认说明
- 项目默认使用千问兼容接口，不需要额外改 provider

### 4. 启动服务

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

启动后可访问：

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

## 测试命令

常用回归：

```bash
uv run pytest tests/unit/test_faq_store.py
uv run pytest tests/unit/test_support_agent.py tests/unit/test_api.py
uv run pytest tests/unit/test_sentiment_analyzer.py tests/unit/test_business_flows.py
```

完整单测：

```bash
uv run pytest
```

带覆盖率：

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

## 推荐演示路径

建议在 `/docs` 里按下面顺序验证：

1. FAQ 路径
   - 问题：`如何重置密码？`
   - 预期：`active_agent = knowledge`

2. 真实业务查询
   - 问题：`帮我查一下当前套餐和下次续费时间`
   - 预期：走 action 路径，返回订阅摘要

3. RAG 复合问题
   - 问题：`怎么取消套餐，取消后什么时候生效？`
   - 预期：命中知识检索；开启 `debug=true` 时可看到更丰富的检索策略信息

4. HITL 中断与恢复
   - 问题：`帮我创建一个账单异常工单`
   - 预期：返回 `run_status = interrupted`
   - 用响应里的 `thread_id` 调 `/runs/{thread_id}/resume`

5. 升级人工
   - 问题：`我要投诉，现在就转人工`
   - 预期：升级路径触发，必要时进入审批

更详细的验证脚本和场景见 [examples/README.md](./examples/README.md)。

## Docker

构建镜像：

```bash
docker build -t customer-support-agent .
```

启动完整 demo：

```bash
docker compose up --build
```

如果要启用 Postgres 持久化，请在 `.env` 中设置：

```bash
LANGGRAPH_USE_POSTGRES=true
LANGGRAPH_PERSISTENCE_BACKEND=postgres
```

## 项目定位

这个项目更强调：

- Agent 图设计清晰，而不是把所有逻辑塞进单个 Agent
- 中间件、审批流、持久化、RAG、业务工具能够协同工作
- API、测试、文档齐全，适合作为简历 Demo 和现场演示项目

## 相关文档

- [CAREER_GUIDE.md](./CAREER_GUIDE.md)
- [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)
- [examples/README.md](./examples/README.md)

## 备注

- `.venv/` 已被忽略，推荐始终通过 `uv` 管理环境
- `requirements.txt` 保留为兼容性导出文件，源配置以 `pyproject.toml` 和 `uv.lock` 为准
