"""Tests for the document-backed knowledge base and semantic memory store."""

from pathlib import Path

from src.db.repositories import list_knowledge_documents
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


def test_document_store_parent_child_chunking_creates_finer_grained_children(tmp_path, isolated_business_db):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    long_section = "\n\n".join(
        [
            "收到账单异常投诉后，客服要先核对账单周期、支付方式、套餐变更、席位变更和附加存储。"
            "如果用户无法提供账单号，应该先补充材料，而不是立刻创建工单。"
            "当用户已经说明是重复扣款、退款争议或历史账单更正时，再进入人工复核路径。"
            for _ in range(10)
        ]
    )
    (knowledge_dir / "billing.md").write_text(
        (
            "# Billing Playbook\n\n"
            "## Billing\n\n"
            "### 账单异常核查\n\n"
            f"{long_section}\n"
        ),
        encoding="utf-8",
    )

    store = create_document_store(
        knowledge_base_path=knowledge_dir,
        chroma_path=tmp_path / "chroma",
    )
    stats = store.reindex(clear_existing=True)
    results = store.search_hybrid("重复扣款时创建什么工单", top_k=3)

    assert stats["total_parent_chunks"] > 0
    assert stats["total_child_chunks"] > stats["total_parent_chunks"]
    assert results
    assert "账单异常" in results[0].question or "账单异常" in results[0].answer


def test_document_store_prefers_account_recovery_section_for_unavailable_email_query(tmp_path, isolated_business_db):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "account.md").write_text(
        (
            "# 账户访问与安全管理指南\n\n"
            "## 登录恢复\n\n"
            "### 注册邮箱不可用时的身份核验\n\n"
            "如果用户原邮箱不可用、原邮箱停用、邮箱失效，或离职后收不到重置邮件，"
            "不能直接改绑邮箱或直接发送新密码。客服应先收集身份核验材料，再进入人工审核。\n"
        ),
        encoding="utf-8",
    )
    (knowledge_dir / "billing.md").write_text(
        (
            "# 账单异常处理手册\n\n"
            "## 工单创建与人工审核\n\n"
            "### 创建账单异常核查工单\n\n"
            "如果用户认为账单金额异常，可以补充账单号、金额和支付方式，再进入人工审核流程。\n"
        ),
        encoding="utf-8",
    )

    store = create_document_store(
        knowledge_base_path=knowledge_dir,
        chroma_path=tmp_path / "chroma",
    )
    store.reindex(clear_existing=True)
    results = store.search_hybrid("邮箱不可用时还能怎么重置密码？", top_k=3)

    assert results
    assert results[0].metadata["document_title"] == "账户访问与安全管理指南"
    assert "注册邮箱不可用时的身份核验" in results[0].metadata["section_path"]
    assert "身份核验" in results[0].answer or "原邮箱不可用" in results[0].answer


def test_document_store_incremental_reindex_syncs_added_and_removed_documents(tmp_path, isolated_business_db):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = knowledge_dir / "alpha.md"
    alpha_path.write_text(
        "# Alpha Guide\n\n## Alpha\n\nalpha-only keyword lives here.\n",
        encoding="utf-8",
    )

    store = create_document_store(
        knowledge_base_path=knowledge_dir,
        chroma_path=tmp_path / "chroma",
    )
    initial_stats = store.reindex(clear_existing=True)

    assert initial_stats["total_documents"] == 1
    assert [item["source_path"] for item in list_knowledge_documents()] == ["alpha.md"]
    assert store.search_hybrid("alpha-only keyword", top_k=2)

    alpha_path.unlink()
    beta_path = knowledge_dir / "beta.md"
    beta_path.write_text(
        "# Beta Guide\n\n## Beta\n\nbeta-only keyword lives here.\n",
        encoding="utf-8",
    )

    synced_stats = store.reindex(clear_existing=False)
    persisted_paths = [item["source_path"] for item in list_knowledge_documents()]
    beta_results = store.search_hybrid("beta-only keyword", top_k=2)
    alpha_results = store.search_hybrid("alpha-only keyword", top_k=2)

    assert synced_stats["clear_existing"] is False
    assert synced_stats["removed_documents"] == 1
    assert synced_stats["total_documents"] == 1
    assert persisted_paths == ["beta.md"]
    assert beta_results
    assert beta_results[0].metadata["source_path"] == "beta.md"
    assert all(result.metadata["source_path"] != "alpha.md" for result in alpha_results)


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
