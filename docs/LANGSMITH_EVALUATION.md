# LangSmith Evaluation Guide

## 目标

本项目新增了一个离线评估脚本，用来把 LangSmith dataset 里的样本逐条发送到本地 `POST /chat` 接口，再把结果回传到 LangSmith，形成一个新的 experiment。

脚本路径：

- [scripts/run_langsmith_eval.py](../scripts/run_langsmith_eval.py)

它适合评估当前这个“完整客服后端”，而不只是单独测 prompt。也就是说，这次评估会真实经过：

- LangGraph 路由
- RAG 检索
- 工具调用
- LangSmith tracing
- 接口返回结构

## 运行前准备

### 1. 确认 LangSmith 环境变量

至少要保证下面几项有效：

```env
LANGSMITH_TRACING=true
LANGSMITH_OTEL_ENABLED=false
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=customer-support-agent
LANGSMITH_WORKSPACE_ID=
```

如果你在 UI 中能看到 dataset，但脚本提示 `Invalid token` 或 `Authentication failed`，优先检查：

- 本地 `.env` 里的 `LANGSMITH_API_KEY` 是否已经换成最新的
- `LANGSMITH_WORKSPACE_ID` 是否与当前 dataset 所在 workspace 一致

### 2. 启动本地服务

```powershell
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

建议先访问一次：

- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3. 确认 dataset 已导入

你当前已经在 LangSmith 中导入了 `eval_case`，并且已经绑定了 `Correctness` evaluator。

这意味着：

- 运行新 experiment 时，`Correctness` 会自动跟着跑
- 不需要在脚本里重复手写这个 LLM-as-a-judge evaluator

## 最常用命令

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case
```

这条命令会：

1. 检查本地 `http://127.0.0.1:8000/health`
2. 检查 LangSmith dataset 是否可读
3. 对 dataset 中每条样本调用一次 `/chat`
4. 上传 experiment 到 LangSmith
5. 自动触发你在 UI 里绑定好的 `Correctness`
6. 额外增加一个本地规则型 evaluator：`source_match`

## 脚本里做了什么

### 请求目标

脚本会把 dataset 的输入字段映射成：

- `question` -> `/chat.content`

并为每条样本生成新的 `user_id`，避免长期记忆串样本。

### 输出字段

脚本会把 `/chat` 返回值整理成以下结构上传给 LangSmith：

- `output`
- `answer`
- `citations`
- `sources`
- `run_status`
- `active_agent`
- `intent`
- `thread_id`
- `trace_url`

其中：

- `output` / `answer` 用来给 `Correctness` 之类的 evaluator 比对
- `trace_url` 方便你点回真实 trace
- `sources` / `citations` 用来做本地 `source_match`

### 本地 evaluator

脚本默认会附带一个额外 evaluator：

- `source_match`

规则很简单：

- 读取 reference output 里的 `reference_source`
- 再检查实际输出里的 `sources + citations`
- 如果包含预期来源，记 `1.0`
- 否则记 `0.0`

如果你只想跑 UI 里已经绑定的 evaluator，不想带这个本地规则，可以用：

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case --no-local-source-evaluator
```

## 常用参数

### 改服务地址

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case --base-url http://127.0.0.1:8001
```

### 改并发度

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case --max-concurrency 1
```

建议一开始先用 `1` 或 `2`，这样更稳，也更容易对照 trace。

### 用 dataset ID

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-id <你的_dataset_id>
```

### 不上传，只本地跑

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case --no-upload
```

## 结果怎么看

运行完成后，去：

1. `LangSmith`
2. `Datasets & Experiments`
3. 打开 `eval_case`
4. 点 `Experiments`
5. 打开最新的 experiment

建议重点看这几列：

- `Correctness`
- `source_match`
- `inputs`
- `reference outputs`
- `outputs.output`
- `outputs.trace_url`

### 如果某条分数低

建议按下面顺序排查：

1. 先看 `outputs.output` 是否答偏了
2. 再看 `outputs.sources` / `outputs.citations` 是否引用错文档
3. 最后打开 `trace_url`，去 LangSmith trace 里看：
   - 是否走了知识库检索
   - top result 命中了哪个 section
   - 是否误走工具路由

## 典型报错

### `Invalid token`

说明本地脚本用到的 `LANGSMITH_API_KEY` 不对，或者已经失效。

### `Authentication failed`

通常是：

- API Key 错了
- workspace 不匹配

### `无法连接到本地服务`

说明 FastAPI 还没启动，或者端口不是 `8000`。

### `未能从 dataset example 中解析问题字段`

说明导入的数据输入列不叫 `question`。这时可以显式指定：

```powershell
uv run python scripts/run_langsmith_eval.py --dataset-name eval_case --input-key input
```

## 推荐使用顺序

1. 先在 `/docs` 手动验证 2 到 3 个问题
2. 再运行 `run_langsmith_eval.py`
3. 在 `Experiments` 里看低分样本
4. 打开对应 trace，定位是：
   - 召回问题
   - 重排问题
   - 回答组织问题
   - 引用问题

这样会比单纯手工问答更适合持续调 RAG。
