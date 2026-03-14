"""针对支持 Agent 中间件辅助函数的单元测试。"""

from langchain_core.messages import AIMessage, HumanMessage

from src.conversation.support_agent.middleware import (
    build_history_trim_removals,
    estimate_history_tokens,
    estimate_message_tokens,
)


def test_estimate_message_tokens_supports_structured_content():
    message = AIMessage(
        id="ai-1",
        content=[
            {"type": "text", "text": "请帮我解释最近账单的扣费原因"},
            {"type": "meta", "value": {"source": "billing"}},
        ],
    )

    assert estimate_message_tokens(message) > 10


def test_build_history_trim_removals_respects_message_limit():
    messages = [
        HumanMessage(id=f"msg-{index}", content=f"第 {index} 条消息")
        for index in range(10)
    ]

    removals = build_history_trim_removals(
        messages,
        max_keep=8,
        max_tokens=10_000,
        min_keep=6,
    )

    assert [item.id for item in removals] == ["msg-0", "msg-1"]


def test_build_history_trim_removals_respects_token_budget():
    messages = [
        HumanMessage(id=f"msg-{index}", content="这是一个很长的上下文消息。" * 40)
        for index in range(8)
    ]
    single_message_tokens = estimate_message_tokens(messages[0])

    removals = build_history_trim_removals(
        messages,
        max_keep=8,
        max_tokens=single_message_tokens * 6,
        min_keep=6,
    )

    assert [item.id for item in removals] == ["msg-0", "msg-1"]
    assert estimate_history_tokens(messages[2:]) <= single_message_tokens * 6


def test_build_history_trim_removals_keeps_recent_context_when_under_budget():
    messages = [
        HumanMessage(id="msg-1", content="你好"),
        AIMessage(id="msg-2", content="你好，我来帮你"),
        HumanMessage(id="msg-3", content="请帮我查一下套餐"),
    ]

    removals = build_history_trim_removals(
        messages,
        max_keep=8,
        max_tokens=1_000,
        min_keep=3,
    )

    assert removals == []
