# Help Center Validation Flow

本文档用于验证当前项目已经从问答卡片式知识检索切换到“企业帮助中心文档知识库”模式，并指导你在 FastAPI `/docs` 和 LangSmith 中观察完整调用链路。

## 1. 准备知识库文档

当前默认测试文档位于：

- `data/knowledge_base/customer_support_handbook.md`

这份文档已经覆盖以下高频主题：

- 重置密码
- 双重身份验证（2FA）
- 账户所有权转移
- 支付方式
- 免费试用
- 取消订阅与退款规则
- 套餐升级、降级与席位变更
- 账单异常与扣费说明
- 教育与公益优惠
- 存储配额与超限策略
- 邀请团队成员
- 角色与权限说明
- 数据安全
- 数据驻留与部署区域
- API 访问
- 第三方集成
- 数据导出
- 支持的设备与平台
- 联系客服
- 工单响应时效（SLA）
- 人工审核与恢复流程
- 产品更新节奏

## 2. 启动前检查

确认 `.env` 中至少具备以下配置：

```bash
LLM_API_KEY=你的模型 API Key
LANGSMITH_TRACING=true
LANGSMITH_OTEL_ENABLED=false
LANGSMITH_API_KEY=你的 LangSmith API Key
LANGSMITH_PROJECT=customer-support-agent
COLLECTION_NAME=document_knowledge_base
```

如果你的 API Key 绑定了多个 workspace，再补充：

```bash
LANGSMITH_WORKSPACE_ID=你的 workspace uuid
```

然后准备业务数据库。推荐直接启动本地 Postgres：

```bash
docker compose up -d postgres
uv run python scripts/init_demo_db.py
```

如果你暂时不想启动 Postgres，也可以把 `.env` 里的 `DATABASE_URL` 临时改成：

```bash
DATABASE_URL=sqlite+aiosqlite:///./data/support.db
```

注意：如果数据库没有启动，而 `.env` 仍然指向 `localhost:5432`，`/knowledge/reindex` 和 `/chat` 可能会直接报连接超时。

## 3. 重建帮助中心索引

启动服务：

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

打开 Swagger：

- `http://localhost:8000/docs`

在 `/docs` 中执行：

- `POST /knowledge/reindex`

建议请求体：

```json
{
  "clear_existing": true
}
```

预期结果：

- `status = success`
- `message` 中包含 `帮助中心知识库索引已刷新`
- 返回文档数量、父块数量、子块数量和集合名

## 4. 验证帮助中心检索

执行 `POST /chat`：

```json
{
  "user_id": "demo_user",
  "content": "如何重置密码？",
  "debug": true
}
```

预期结果：

- `active_agent = knowledge`
- `run_status = completed`
- `debug.route_path` 包含 `analyze -> knowledge -> validate -> respond`
- `result.citations` 包含类似：
  `帮助中心::Customer Support Help Center > 账户与登录 > 重置密码`

在 LangSmith 中重点看：

- 顶层 run 是否为 `/chat`
- `analyze` 节点是否把请求路由到 `knowledge`
- `knowledge` 节点是否调用了 `search_knowledge_base`
- tool 输出里是否包含帮助中心 section path

## 5. 验证复合文档检索

执行 `POST /chat`：

```json
{
  "user_id": "demo_user",
  "content": "怎么取消套餐，取消后什么时候生效？",
  "debug": true
}
```

预期结果：

- 命中帮助中心中的“取消订阅与退款规则”
- 返回说明“当前计费周期结束后生效”
- `result.sources` 包含 `Hybrid RAG` 和 `Help Center Knowledge Base`

在 LangSmith 中重点看：

- `metadata.user_id`
- `metadata.thread_id`
- `knowledge` 节点耗时
- `search_knowledge_base` 的输入 query
- tool 输出中的引用来源是否清晰

## 6. 验证业务工具查询

执行 `POST /chat`：

```json
{
  "user_id": "user_001",
  "content": "帮我查一下当前套餐和下次续费时间",
  "debug": true
}
```

预期结果：

- `active_agent = action`
- 返回当前套餐、续费时间、自动续费状态
- `result.citations` 中可看到业务工具来源

在 LangSmith 中重点看：

- `analyze` 是否直接路由到 `action`
- `action` 节点是否调用了 `get_subscription_status`
- `respond` 是否融合了工具返回内容

## 7. 验证 HITL 中断

执行 `POST /chat`：

```json
{
  "user_id": "user_001",
  "content": "请帮我创建一个账单异常工单",
  "debug": true
}
```

预期结果：

- `run_status = interrupted`
- 顶层响应中有 `thread_id`
- `approval.tools[0].tool` 为 `create_ticket`
- `approval.tools[0].tool_label` 为 `创建工单`

在 LangSmith 中重点看：

- `analyze` 是否判定为高风险动作
- `action` 节点是否中断而不是直接写入
- trace 中是否能看到中断前的决策上下文

## 8. 验证 Resume 恢复

拿上一步 `/chat` 返回的顶层 `thread_id`，执行：

- `POST /runs/{thread_id}/resume`

请求体：

```json
{
  "decisions": [
    { "type": "approve" }
  ],
  "debug": true
}
```

预期结果：

- `run_status = completed`
- 返回新的最终答复
- `result.ticket_created` 为非空
- `debug.route_path` 最终应收敛为 `analyze -> action -> validate -> respond`

在 LangSmith 中重点看：

- `/resume` 是否形成新的 trace
- 顶层 run 名称是否是 `support.resume`
- resume trace 中是否出现：
  - `create_ticket_resume_dispatch`
  - `validate`
  - `respond`

## 9. 如何通过 LangSmith 读懂整条链路

推荐按下面顺序看：

1. 先看顶层 run metadata
   - 核对 `user_id`、`thread_id`、`correlation_id`、`role`、`intent`、`risk`

2. 再看节点顺序
   - 先确认走的是 `knowledge`、`action` 还是 `escalation`

3. 再看工具输入输出
   - 知识类关注 `search_knowledge_base`
   - 业务类关注 `get_subscription_status`、`get_latest_invoice`、`create_ticket`

4. 最后看恢复链路
   - 首次 `/chat` 看中断点
   - `/resume` 看写操作真正发生在哪个节点

## 10. 常见问题

### Q1. 为什么我改了文档，但回答没变化？

通常是因为还没有重新执行 `/knowledge/reindex`。只修改 `data/knowledge_base/` 目录中的文档，不会自动重建 Chroma 向量和数据库中的 chunk 记录。

### Q2. 为什么 LangSmith 没有 trace？

优先检查：

- `LANGSMITH_TRACING=true`
- `LANGSMITH_OTEL_ENABLED=false`
- `LANGSMITH_API_KEY` 是否正确
- `LANGSMITH_WORKSPACE_ID` 是否与当前 API Key 对应

### Q3. 为什么 resume 后看不到继续的 trace？

如果 `/resume` 能成功返回，但 LangSmith 中没有后续链路，通常要检查：

- `/resume` 是否真的返回了新的 `debug.langsmith.run_url`
- 运行时是否正确读取了 LangSmith 环境变量
- 是否存在 workspace 权限或项目写入问题
