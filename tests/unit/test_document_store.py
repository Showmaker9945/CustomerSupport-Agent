"""Tests for the document-backed knowledge base and semantic memory store."""

from pathlib import Path

from src.knowledge.document_store import create_document_store
from src.memory.semantic_store import SemanticMemoryStore


def test_document_store_reindexes_markdown_corpus(tmp_path, isolated_business_db):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "support.md").write_text(
        (
            "# Support Manual\n\n"
            "## Billing\n\n"
            "### 账单异常\n\n"
            "如果用户反馈异常扣费，应先查询最近账单，再根据需要创建账单异常核查工单。\n\n"
            "## Workspace\n\n"
            "### 邀请成员\n\n"
            "进入设置中的团队成员页面，点击邀请成员并分配角色。\n"
        ),
        encoding="utf-8",
    )

    store = create_document_store(
        knowledge_base_path=knowledge_dir,
        chroma_path=tmp_path / "chroma",
    )
    stats = store.reindex(clear_existing=True)
    results = store.search_hybrid("怎么邀请团队成员", top_k=3)

    assert stats["total_documents"] == 1
    assert stats["total_child_chunks"] > 0
    assert results
    assert "邀请" in results[0].question or "邀请" in results[0].answer
    assert results[0].metadata["document_title"] == "Support Manual"


def test_semantic_memory_store_persists_and_searches(tmp_path, isolated_business_db):
    store = SemanticMemoryStore(chroma_path=tmp_path / "chroma")
    store.upsert_memory(
        user_id="memory_user",
        memory_id="profile:name",
        payload={
            "memory_type": "profile",
            "field": "name",
            "value": "小王",
            "content": "用户姓名是小王",
            "importance": 0.95,
            "status": "active",
        },
    )
    store.upsert_memory(
        user_id="memory_user",
        memory_id="open_issue:billing_anomaly",
        payload={
            "memory_type": "open_issue",
            "issue_code": "billing_anomaly",
            "content": "用户存在账单异常，仍需继续跟进。",
            "summary": "账单异常待处理",
            "importance": 1.0,
            "status": "open",
        },
    )

    listed = store.list_memories(user_id="memory_user")
    searched = store.search_memories(user_id="memory_user", query="账单异常还没解决", limit=3)

    assert {item["memory_id"] for item in listed} == {"profile:name", "open_issue:billing_anomaly"}
    assert searched
    assert searched[0]["memory_id"] == "open_issue:billing_anomaly"
