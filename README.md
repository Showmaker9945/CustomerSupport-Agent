# CustomerSupport-Agent

一个中文优先的客服 Agent Demo，基于 LangGraph 条件路由、多 Agent 协作、HITL 审批、混合检索 RAG、长短期记忆与 FastAPI 构建，适合作为实习简历里的工程化 Agent 项目。

## 项目亮点

- LangGraph 条件边编排：分析、路由、检索、执行、升级、校验、回复职责清晰
- LangChain v1 / LangGraph v1：对齐 `create_agent` 与 middleware 体系
- 中文优先：默认中文输出、中文情绪识别、中文帮助中心知识库与业务示例
- 轻量增强 RAG：混合检索、查询规范化、分类推断、子问题拆分、一次改写兜底
- Memory 治理：线程级短期记忆、用户级长期记忆、记忆抽取与召回
- HITL：高风险工具先中断审批，再恢复同一线程执行；显式写操作支持确定性中断
- FastAPI 演示友好：REST、WebSocket、SSE、Swagger、回归测试
- Debug 轻量化：接口仅返回路径、耗时与 LangSmith 链接，详细链路放到 LangSmith 查看
- LangSmith Tracing：`/chat` 与 `/resume` 都可返回 `run_url`，便于排查图节点、工具调用与审批恢复链路

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
- `src/knowledge/document_store.py`：帮助中心文档知识库、结构化切分、混合检索与重排
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
│  ├─ knowledge/              # 文档知识库 / 结构化 chunk / RAG
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
LLM_HIGH_QUALITY_MODEL=qwen3-max
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

## LangSmith 可观测性

推荐在 `.env` 中打开 LangSmith，并保持当前项目默认的 OTel 兼容策略：

```bash
LANGSMITH_TRACING=true
LANGSMITH_OTEL_ENABLED=false
LANGSMITH_API_KEY=你的 LangSmith API Key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=customer-support-agent
LANGSMITH_WORKSPACE_ID=你的 workspace uuid
```

说明：

- 如果 API Key 只绑定一个 workspace，`LANGSMITH_WORKSPACE_ID` 可以留空
- 如果 `list_projects` 或 `/runs/multipart` 返回 `403`，通常是 workspace 选错或未设置
- 本项目现在会为 `/chat` 和 `/resume` 都返回 `debug.langsmith.run_url`
- 更详细的配置、排障与 UI 阅读方式见 [docs/LANGSMITH_TRACING.md](./docs/LANGSMITH_TRACING.md)

## 测试命令

常用回归：

```bash
uv run pytest tests/unit/test_document_store.py
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

1. 帮助中心检索路径
   - 问题：`如何重置密码？`
   - 预期：`active_agent = knowledge`
   - 预期引用：`帮助中心::Customer Support Help Center > 账户与登录 > 重置密码`

2. 真实业务查询
   - 问题：`帮我查一下当前套餐和下次续费时间`
   - 预期：走 action 路径，返回订阅摘要

3. 文档型 RAG 复合问题
   - 问题：`怎么取消套餐，取消后什么时候生效？`
   - 预期：命中帮助中心文档；本地只看轻量 `route_path/node_timings`，详细过程到 LangSmith 看

4. HITL 中断与恢复
   - 问题：`请创建一个账单异常工单`
   - 预期：返回 `run_status = interrupted`
   - `approval.tools[0]` 中会明确显示待审批工具名、原因和参数摘要
   - `debug.langsmith.run_url` 可打开首次 `/chat` 的 trace
   - 用响应最外层的 `thread_id` 调 `/runs/{thread_id}/resume`
   - `approve`：执行写操作
   - `edit`：修改参数后执行
   - `reject`：取消写操作并返回最终答复
   - `/resume` 成功后也会返回新的 `debug.langsmith.run_url`，可继续查看 `support.resume -> create_ticket_resume_tool -> validate -> respond`

5. 升级人工
   - 问题：`我要投诉，现在就转人工`
   - 预期：升级路径触发，必要时进入审批

更详细的验证脚本和场景见 [examples/README.md](./examples/README.md) 与 [docs/HELP_CENTER_VALIDATION.md](./docs/HELP_CENTER_VALIDATION.md)。

## 业务数据说明

- 结构化业务数据默认使用 PostgreSQL：`users`、`subscriptions`、`invoices`、`invoice_items`、`tickets`
- 帮助中心文档检索继续使用 Chroma
- `data/demo_seed/` 中的 JSON 只作为演示初始化数据，不再作为运行时主存储

### 初始化 PostgreSQL 演示数据

```bash
# 1. 启动 PostgreSQL
docker compose up -d postgres

# 2. 初始化业务表与 demo 数据
uv run python scripts/init_demo_db.py
```

初始化完成后，可通过 `docker exec -it support-postgres psql -U support_user -d support` 进入数据库，再用 `\dt` 查看表结构。

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

## 最近更新

- 修复显式高风险写操作在部分场景下无法正确触发 HITL 的问题
- 为 `create_ticket` 等审批动作补齐确定性 `interrupt -> resume` 链路
- `/chat` 与 `/resume` 的调试输出改为轻量摘要，详细节点分析统一通过 LangSmith 查看
- 优化审批响应结构，待审批工具名称和参数预览更清晰
- 接入 LangSmith tracing，并补齐 `support.resume`、resume 工具调用与后续 `validate/respond` 的可观测性

## 相关文档

- [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)
- [examples/README.md](./examples/README.md)
- [docs/HELP_CENTER_VALIDATION.md](./docs/HELP_CENTER_VALIDATION.md)
- [docs/LANGSMITH_TRACING.md](./docs/LANGSMITH_TRACING.md)

## 备注

- `.venv/` 已被忽略，推荐始终通过 `uv` 管理环境
- `requirements.txt` 保留为兼容性导出文件，源配置以 `pyproject.toml` 和 `uv.lock` 为准
