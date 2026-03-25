# LangSmith Tracing Guide

## 目标

本项目已经为两类关键链路接入 LangSmith：

- `support.chat`：用户首次请求的主链路
- `support.resume`：HITL 审批后的恢复链路

在 LangSmith 中，你现在可以看到：

- `support.chat -> support_orchestration_graph -> analyze/action/...`
- `support.chat -> support.resume -> create_ticket_resume_tool -> validate -> respond`
- 嵌套的 agent / model / tool 子调用

## 推荐环境变量

请在 `.env` 中设置：

```env
LANGSMITH_TRACING=true
LANGSMITH_OTEL_ENABLED=false
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=customer-support-agent
LANGSMITH_WORKSPACE_ID=
```

说明：

- 当前项目推荐先保持 `LANGSMITH_OTEL_ENABLED=false`
- 如果 API Key 关联多个 workspace，请显式填写 `LANGSMITH_WORKSPACE_ID`
- 如果 `list_projects`、`/runs/query` 或 `/runs/multipart` 返回 `403`，优先检查 workspace 是否正确

## 验证前检查

启动服务前，建议先确认配置已生效：

```powershell
@'
from src.config import settings
print("tracing =", settings.langsmith_tracing)
print("otel =", settings.langsmith_otel_enabled)
print("project =", settings.langsmith_project)
print("workspace_id =", repr(settings.langsmith_workspace_id))
print("enabled =", settings.langsmith_enabled)
'@ | .\.venv\Scripts\python.exe -
```

如果想单独验证 LangSmith API 权限：

```powershell
@'
from src.config import settings
from langsmith import Client

client = Client(
    api_key=settings.langsmith_api_key,
    api_url=settings.langsmith_endpoint,
    workspace_id=settings.langsmith_workspace_id or None,
)
print(list(client.list_projects(limit=1)))
'@ | .\.venv\Scripts\python.exe -
```

## 在 `/docs` 中验证完整链路

### 1. 首次请求触发审批

打开 Swagger UI：

- [http://localhost:8000/docs](http://localhost:8000/docs)

在 `POST /chat` 中发送：

```json
{
  "user_id": "user_001",
  "content": "请创建一个账单异常工单",
  "debug": true
}
```

预期结果：

- `run_status = "interrupted"`
- `active_agent = "action"`
- `approval.tools[0].tool = "create_ticket"`
- `debug.route_path = ["analyze", "action"]`
- `debug.node_timings` 有值
- `debug.langsmith.run_url` 有值

此时说明系统只生成了待审批动作，还没有真正写入工单。

### 2. 恢复审批并继续执行

记下上一步返回的 `thread_id`，然后在 `POST /runs/{thread_id}/resume` 中提交：

```json
{
  "decisions": [
    {
      "type": "approve"
    }
  ],
  "debug": true
}
```

预期结果：

- `run_status = "completed"`
- `result.ticket_created = "TKT-..."`
- `debug.route_path = ["analyze", "action", "validate", "respond"]`
- `debug.node_timings` 有值
- `debug.langsmith.run_url` 有值

此时说明：

- 审批通过
- `create_ticket` 已真正执行
- 后续 `validate` 与 `respond` 也已继续运行

### 3. 验证工单真的落库

可以在 `GET /users/{user_id}/tickets` 中查询 `user_001`，确认新工单已出现。

## 如何阅读 LangSmith

### 第一次 `/chat`

打开 `/chat` 返回的 `debug.langsmith.run_url` 后，重点看：

1. `support.chat`
   - 整次用户请求的根 run
2. `support_orchestration_graph`
   - 首次请求的 LangGraph 编排层
3. `analyze`
   - 看 `Metadata` 里的 `intent`、`risk`、`thread_id`
   - 看 `Outputs` 里的 `selected_agent`
4. `action`
   - 看 `Outputs` 里的 `run_status`
   - 看是否包含 `interrupts` 与 `pending_approval_plan`

判断标准：

- 如果 `action` 的输出里是 `run_status = interrupted`，说明审批门生效了
- 即使 LangSmith 顶层 run 显示 `success`，也不代表业务已完成，要看输出里的业务状态

### 第二次 `/resume`

打开 `/resume` 返回的 `debug.langsmith.run_url` 后，重点看：

1. `support.resume`
   - 说明恢复链路已经接入 LangSmith
   - `Metadata.resumed = true`
2. `create_ticket_resume_dispatch`
   - resume 内部对工具执行的调度层
3. `create_ticket_resume_tool`
   - 真正的工具调用
   - 看 `Inputs` 中的 `user_id`、`subject`、`description`
   - 看 `Outputs` 是否包含工单号
4. `validate`
   - 看校验结果和修正要求
5. `respond`
   - 看最终生成回复是否带上工单号与下一步说明

### Metadata 怎么看

本项目最常用的 metadata 字段有：

- `thread_id`：同一会话线程
- `correlation_id`：同一业务链路的关联 ID
- `role`：当前 run 对应的角色
- `intent`：意图识别结果
- `risk`：风险等级
- `resumed`：是否来自 resume 链路

快速判断建议：

- 先看 `thread_id`，确认是不是这次测试的 run
- 再看 `correlation_id`，确认 `/chat` 和 `/resume` 是否属于同一条链路
- 最后看 `role`、`intent`、`risk`，确认路由是否符合预期

## 现在本地 `debug` 里还看什么

为了避免接口响应里重复塞一份“迷你 trace”，当前 `debug=true` 只保留几项本地摘要：

- `debug.trace_id`：本次请求的关联 ID
- `debug.route_path`：实际经过的图节点路径
- `debug.node_timings`：各节点耗时摘要
- `debug.total_duration_ms`：整次请求总耗时
- `debug.langsmith.run_url`：跳转到完整 trace

如果你想看更细的节点输入输出、metadata、tool args、resume 衔接关系，优先直接打开 LangSmith。

## 当前限制

- LangSmith trace 页面更适合看“实际执行路径树”
- 它不会自动把整个 `StateGraph` 静态渲染成一张完整 DAG 图
- 如果想看完整图结构，建议配合 LangGraph Studio / LangSmith Studio

## 相关文档

- [README.md](../README.md)
- [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)
