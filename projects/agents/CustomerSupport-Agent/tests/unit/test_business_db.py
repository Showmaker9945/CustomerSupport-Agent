"""Tests for structured business data bootstrapping."""

from src.db.repositories import (
    append_conversation_message,
    build_recent_context_window,
    get_conversation_thread,
    get_latest_invoice_record,
    get_subscription_record,
    get_user_record,
    list_thread_messages,
    list_user_conversation_messages,
    list_ticket_records,
)


def test_demo_seed_bootstraps_business_records(isolated_business_db):
    user = get_user_record("user_001")
    subscription = get_subscription_record("user_001")
    invoice = get_latest_invoice_record("user_001")
    tickets = list_ticket_records("user_001")

    assert user is not None
    assert user["email"] == "alice.johnson@example.com"
    assert subscription is not None
    assert subscription["plan_name"] == "Pro 团队版"
    assert invoice is not None
    assert invoice["invoice_id"] == "INV-202603-0001"
    assert len(invoice["line_items"]) == 2
    assert len(tickets) == 2


def test_transcript_history_persists_and_builds_context_window(isolated_business_db):
    thread_id = "thread-db-001"
    append_conversation_message(
        thread_id=thread_id,
        user_id="history_user",
        role="user",
        content="你好，我想确认一下当前套餐。",
        thread_status="active",
    )
    append_conversation_message(
        thread_id=thread_id,
        user_id="history_user",
        role="assistant",
        content="当然可以，我先帮你看看当前订阅信息。",
        active_agent="knowledge",
        run_status="completed",
        thread_status="completed",
    )
    append_conversation_message(
        thread_id=thread_id,
        user_id="history_user",
        role="user",
        content="另外也想知道最近账单金额。",
        thread_status="active",
    )

    history = list_user_conversation_messages("history_user", limit=10)
    assert len(history) == 3
    assert all(item["thread_id"] == thread_id for item in history)

    thread = get_conversation_thread(thread_id)
    assert thread is not None
    assert thread["message_count"] == 3
    assert thread["title"] == "你好，我想确认一下当前套餐。"

    thread_messages = list_thread_messages(thread_id, limit=10)
    assert len(thread_messages) == 3
    assert thread_messages[0]["role"] == "user"
    assert thread_messages[-1]["content"] == "另外也想知道最近账单金额。"

    context = build_recent_context_window(
        thread_id=thread_id,
        recent_turns=2,
        max_messages=4,
        max_tokens=400,
    )
    assert "当前套餐" in context["text"]
    assert "最近账单金额" in context["text"]
    assert len(context["messages"]) == 3
