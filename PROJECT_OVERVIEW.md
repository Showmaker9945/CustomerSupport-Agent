# Project Overview

## 定位

CustomerSupport-Agent 是一个偏工程化的中文客服 Agent Demo，重点展示：

- LangGraph 图式编排能力
- 多 Agent 协作与条件边路由
- middleware、HITL、RAG、memory 的组合落地
- FastAPI 下可测试、可演示、可恢复、可解释的交互链路

## 当前主链

```text
用户输入
-> analyze
-> knowledge / action / escalation（按条件路由）
-> validate
-> respond
```

说明：

- `analyze` 负责意图、风险、情绪、执行步骤判断
- `knowledge` 负责 FAQ / RAG 检索
- `action` 负责真实业务工具调用
- `escalation` 负责人工作单与升级交接
- `validate` 负责最终输出校验
- `respond` 负责生成用户看到的最终答复

## 关键能力

### 1. LangGraph + Middleware

- 图结构保留条件边，而不是退化成简单 chain
- middleware 保留上下文注入、工具拦截、输出校验等能力
- 高风险工具通过中断机制接入 HITL

### 2. 轻量增强 RAG

当前 FAQ 检索包含：

- 向量检索
- BM25 词法检索
- 融合排序
- 查询规范化
- 分类推断
- 子问题拆分
- 一次改写兜底
- 检索 trace 调试输出

目标不是堆砌复杂 RAG 组件，而是提升 demo 中最常见客服问题的命中率和可解释性。

### 3. Memory 治理

- 线程级状态用于多轮恢复
- 用户级长期记忆用于跨轮偏好和历史事实注入
- 支持 memory debug 输出，便于展示“记住了什么、为什么注入”

### 4. 真实业务工具

当前工具覆盖：

- FAQ 检索
- 账户查询
- 订阅查询
- 最近账单查询
- 账单解释
- 工单创建 / 更新 / 查询
- 升级人工

## API

- `POST /chat`
- `POST /runs/{thread_id}/resume`
- `GET /chat/stream`
- `WebSocket /ws/chat/{user_id}`
- `POST /knowledge/reindex`
- `GET /health`

## 环境管理

项目已切换到 `uv`：

```bash
uv sync --all-groups
uv run uvicorn src.api.main:app --reload
uv run pytest
```

源配置以：

- `pyproject.toml`
- `uv.lock`

为准。

## 适合面试讲解的点

- 为什么需要 LangGraph 而不是简单 agent chain
- 为什么高风险工具必须走 HITL
- 为什么 RAG 选择轻量增强而不是堆复杂模块
- 如何把 debug / trace / validation 做成可展示的工程能力
- 如何在 FastAPI 下把 Agent 做到可恢复、可测试、可演示
