"""Role-agent construction with LangChain middleware."""

from __future__ import annotations

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


def create_role_agent(owner: Any, role: str, tools: List[Any], enable_hitl: bool) -> Any:
    """Create a LangChain agent for a given support role."""

    @dynamic_prompt
    def role_prompt(request: ModelRequest) -> str:
        context = getattr(request.runtime, "context", None)
        user_id = getattr(context, "user_id", "unknown_user")
        latest_user = ""
        for message in reversed(request.state.get("messages", [])):
            if isinstance(message, HumanMessage):
                latest_user = str(message.content)
                break

        memory_items = owner._search_memory(user_id=user_id, query=latest_user, limit=4)
        return build_role_system_prompt(role=role, memory_items=memory_items)

    @before_model
    def trim_history(state: Dict[str, Any], _runtime: Any) -> Dict[str, Any] | None:
        messages = state.get("messages", [])
        max_keep = max(8, settings.max_conversation_history)
        if len(messages) <= max_keep:
            return None
        removals = [
            RemoveMessage(id=msg.id) for msg in messages[:-max_keep] if getattr(msg, "id", None)
        ]
        if removals:
            return {"messages": removals}
        return None

    @wrap_model_call
    def dynamic_model_selector(request: ModelRequest, handler: Any) -> ModelResponse:
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
