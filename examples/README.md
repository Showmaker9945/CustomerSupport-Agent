# Examples Guide

这个目录用于配合 Swagger Docs、SSE 和 WebSocket 做演示。

## 启动服务

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

打开：

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 推荐验证顺序

### 1. 帮助中心知识检索

`POST /chat`

```json
{
  "user_id": "demo_user",
  "content": "如何重置密码？",
  "debug": true
}
```

关注点：

- `active_agent` 应为 `knowledge`
- `run_status` 应为 `completed`
- `debug.route_path` 应包含 `analyze -> knowledge -> validate -> respond`
- `result.citations` 应包含 `帮助中心::Customer Support Help Center > 账户与登录 > 重置密码`

### 2. 真实业务查询

```json
{
  "user_id": "user_001",
  "content": "帮我查一下当前套餐和下次续费时间",
  "debug": true
}
```

关注点：

- 走 `action` 路径
- 返回订阅状态、续费时间、套餐权益

### 3. 轻量 RAG 增强

```json
{
  "user_id": "demo_user",
  "content": "怎么取消套餐，取消后什么时候生效？",
  "debug": true
}
```

关注点：

- 应返回取消订阅相关帮助中心内容
- `debug.route_path` 与 `debug.node_timings` 可以快速判断本地执行路径
- 更细的节点过程建议直接打开 `debug.langsmith.run_url`

### 4. HITL 中断

```json
{
  "user_id": "user_001",
  "content": "帮我创建一个账单异常工单",
  "debug": true
}
```

关注点：

- `run_status = interrupted`
- `thread_id` 非空
- `approval.count` 表示需要提交几条审批决定
- `approval.tools` 应明确列出待审批工具名称

### 5. Resume 恢复

`POST /runs/{thread_id}/resume`

```json
{
  "decisions": [
    { "type": "approve" }
  ],
  "debug": true
}
```

关注点：

- `thread_id` 使用上一步 `/chat` 响应中的顶层 `thread_id`
- 如果 `approval.count > 1`，就必须传入同样数量的 `decisions`
- 成功后应回到 `completed`

### 6. 升级人工

```json
{
  "user_id": "demo_user",
  "content": "我要投诉，现在就转人工",
  "debug": true
}
```

关注点：

- 走 `escalation` 路径
- 高风险工具会进入审批

## SSE 测试

```bash
curl "http://localhost:8000/chat/stream?user_id=demo_user&content=如何重置密码？"
```

常见事件：

- `node`
- `token`
- `interrupt`
- `done`

## WebSocket 测试

```bash
uv run python examples/websocket_client.py
```

如果你手动发消息，基础报文如下：

```json
{
  "type": "message",
  "content": "帮我创建一个账单异常工单",
  "thread_id": "ws_demo_001"
}
```

如果收到了中断结果，再发送：

```json
{
  "type": "resume",
  "thread_id": "ws_demo_001",
  "decisions": [
    { "type": "approve" }
  ]
}
```

## 关键字段说明

- `thread_id`: 本次线程的唯一标识，resume 时使用它
- `run_status`: `completed` 或 `interrupted`
- `active_agent`: 当前主要执行的 agent
- `approval.count`: 当前还需要几条人工审批决定
- `approval.tools`: 待审批工具摘要
- `debug.route_path`: 实际执行过的图路径
- `debug.node_timings`: 各节点轻量耗时摘要
- `debug.langsmith.run_url`: 打开完整 trace
