"""支持客服角色 Agent 的中间件与构造逻辑。"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from ...config import settings
from .graph import AgentRuntimeContext, build_role_system_prompt

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_SPACE_PATTERN = re.compile(r"\s+")


def _message_content_to_text(content: Any) -> str:
    """将消息内容规整成纯文本，便于统一估算 token。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
                parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def _estimate_text_tokens(text: str) -> int:
    """轻量估算文本 token 数，避免引入额外 tokenizer 依赖。"""
    normalized = _SPACE_PATTERN.sub(" ", (text or "").strip())
    if not normalized:
        return 0

    chinese_tokens = len(_CJK_PATTERN.findall(normalized))
    english_tokens = sum(max(1, (len(word) + 3) // 4) for word in _WORD_PATTERN.findall(normalized))

    residual = _WORD_PATTERN.sub("", normalized)
    residual = _CJK_PATTERN.sub("", residual)
    residual = _SPACE_PATTERN.sub("", residual)
    residual_tokens = (len(residual) + 3) // 4 if residual else 0

    return chinese_tokens + english_tokens + residual_tokens


def estimate_message_tokens(message: Any) -> int:
    """估算单条消息占用的上下文 token。"""
    content = _message_content_to_text(getattr(message, "content", ""))
    content_tokens = _estimate_text_tokens(content)

    if isinstance(message, ToolMessage):
        overhead = 12
    elif isinstance(message, AIMessage):
        overhead = 10
    elif isinstance(message, HumanMessage):
        overhead = 8
    else:
        overhead = 6

    return max(overhead, content_tokens + overhead)


def estimate_history_tokens(messages: List[Any]) -> int:
    """估算一段历史消息总共占用的 token。"""
    return sum(estimate_message_tokens(message) for message in messages)


def build_history_trim_removals(
    messages: List[Any],
    *,
    max_keep: int,
    max_tokens: int,
    min_keep: int = 6,
) -> List[RemoveMessage]:
    """根据消息条数和 token 预算生成删除指令。"""
    if not messages:
        return []

    normalized_max_keep = max(1, int(max_keep))
    normalized_max_tokens = max(200, int(max_tokens))
    token_counts = [estimate_message_tokens(message) for message in messages]
    total_tokens = sum(token_counts)

    if len(messages) <= normalized_max_keep and total_tokens <= normalized_max_tokens:
        return []

    keep_start = max(len(messages) - normalized_max_keep, 0)
    kept_count = len(messages) - keep_start
    min_preserved = max(1, min(int(min_keep), kept_count))
    kept_tokens = sum(token_counts[keep_start:])

    # 在保留最近若干轮对话的前提下，继续压缩超预算的旧消息。
    while kept_tokens > normalized_max_tokens and (len(messages) - keep_start) > min_preserved:
        kept_tokens -= token_counts[keep_start]
        keep_start += 1

    return [
        RemoveMessage(id=message.id)
        for message in messages[:keep_start]
        if getattr(message, "id", None)
    ]


def create_role_agent(owner: Any, role: str, tools: List[Any], enable_hitl: bool) -> Any:
    """为指定客服角色创建 LangChain Agent。"""

    @dynamic_prompt
    def role_prompt(request: ModelRequest) -> str:
        context = getattr(request.runtime, "context", None)
        user_id = getattr(context, "user_id", "unknown_user")
        thread_id = getattr(context, "thread_id", None)
        latest_user = ""
        for message in reversed(request.state.get("messages", [])):
            if isinstance(message, HumanMessage):
                latest_user = str(message.content)
                break

        memory_items = owner._search_memory(
            user_id=user_id,
            query=latest_user,
            limit=4,
            thread_id=thread_id,
        )
        return build_role_system_prompt(role=role, memory_items=memory_items)

    @before_model
    def trim_history(state: Dict[str, Any], _runtime: Any) -> Dict[str, Any] | None:
        """在模型调用前裁剪历史消息，兼顾消息条数和 token 预算。"""
        messages = state.get("messages", [])
        max_keep = max(8, settings.max_conversation_history)
        max_tokens = max(600, settings.max_history_tokens)
        removals = build_history_trim_removals(
            messages,
            max_keep=max_keep,
            max_tokens=max_tokens,
            min_keep=min(6, max_keep),
        )
        if removals:
            return {"messages": removals}
        return None

    @wrap_model_call
    def dynamic_model_selector(request: ModelRequest, handler: Any) -> ModelResponse:
        """复杂场景自动切换到高质量模型，普通场景走基础模型。"""
        if owner.advanced_model is None:
            return handler(request)
        message_count = len(request.state.get("messages", []))
        context = getattr(request.runtime, "context", None)
        risk = getattr(context, "risk", "low")
        if message_count > 12 or risk == "high":
            return handler(request.override(model=owner.advanced_model))
        return handler(request)

    @wrap_tool_call
    def safe_tool_wrapper(request: ToolCallRequest, handler: Any) -> Any:
        """统一记录工具调用并兜底工具异常。"""
        tool_name = request.tool_call.get("name", "unknown_tool")
        args = request.tool_call.get("args", {})
        logger.info(f"[tool_call] role={role} tool={tool_name} args={args}")

        context = getattr(request.runtime, "context", None)
        user_id = getattr(context, "user_id", "unknown_user")
        thread_id = getattr(context, "thread_id", "unknown_thread")
        owner._save_memory_item(
            user_id=user_id,
            payload={
                "kind": "tool_call",
                "role": role,
                "thread_id": thread_id,
                "tool_name": tool_name,
                "args": args,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        try:
            return handler(request)
        except Exception as error:
            logger.error(f"Tool execution failed: {tool_name}, error={error}")
            return ToolMessage(
                content=f"工具 `{tool_name}` 执行失败：{str(error)}",
                tool_call_id=request.tool_call.get("id", "unknown_id"),
                name=tool_name,
                status="error",
            )

    @after_model
    def output_guard(state: Dict[str, Any], _runtime: Any) -> Dict[str, Any] | None:
        """对模型输出做基础安全与中文一致性校验。"""
        messages = state.get("messages", [])
        last_ai: Optional[AIMessage] = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if last_ai is None:
            return None

        raw_content = str(last_ai.content or "")
        guarded = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", raw_content)
        if not re.search(r"[\u4e00-\u9fff]", guarded):
            guarded = "以下为中文回复：\n" + guarded

        if guarded == raw_content:
            return None
        if getattr(last_ai, "id", None):
            return {"messages": [RemoveMessage(id=last_ai.id), AIMessage(content=guarded)]}
        return None

    middleware: List[Any] = [
        role_prompt,
        trim_history,
        dynamic_model_selector,
        safe_tool_wrapper,
        output_guard,
    ]

    if enable_hitl:
        interrupt_on = {tool: True for tool in settings.hitl_high_risk_tools}
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model=owner.basic_model,
        tools=tools,
        middleware=middleware,
        context_schema=AgentRuntimeContext,
        checkpointer=owner.persistence.checkpointer,
        store=owner.persistence.store,
        name=f"{role}_agent",
    )
