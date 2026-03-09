"""Unit tests for the FAQ store."""

import json
import tempfile
from pathlib import Path

import pytest

from src.knowledge.faq_store import FAQResult, FAQStore, create_faq_store


_FAKE_VOCAB = [
    "\u5bc6\u7801", "\u91cd\u7f6e", "\u652f\u4ed8", "\u8d26\u5355", "\u6263\u8d39", "\u53d6\u6d88", "\u8ba2\u9605",
    "\u5957\u9910", "\u56e2\u961f", "\u6210\u5458", "api", "\u5ba2\u670d", "\u5de5\u5355", "\u5b89\u5168",
    "\u52a0\u5bc6", "\u6570\u636e", "\u5b58\u50a8", "\u8d26\u6237", "\u767b\u5f55", "\u9080\u8bf7",
    "password", "reset", "payment", "invoice", "cancel", "subscription", "team", "member", "support",
]


class FakeEmbedding(list):
    def tolist(self):
        return list(self)


class FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str):
        lowered = (text or "").lower()
        vector = FakeEmbedding(float(lowered.count(token.lower())) for token in _FAKE_VOCAB)
        vector.append(float(len(lowered)))
        return vector


class FakeCollection:
    def __init__(self, name: str, metadata=None):
        self.name = name
        self.metadata = metadata or {}
        self._records = []

    def count(self):
        return len(self._records)

    def add(self, embeddings, documents, metadatas, ids):
        for embedding, document, metadata, record_id in zip(embeddings, documents, metadatas, ids):
            self._records.append({
                "embedding": list(embedding),
                "document": document,
                "metadata": dict(metadata),
                "id": record_id,
            })

    def get(self, include=None):
        include = include or []
        payload = {"ids": [record["id"] for record in self._records]}
        if "documents" in include:
            payload["documents"] = [record["document"] for record in self._records]
        if "metadatas" in include:
            payload["metadatas"] = [record["metadata"] for record in self._records]
        return payload

    def query(self, query_embeddings, n_results, where=None):
        target = query_embeddings[0]
        scored = []
        target_norm = sum(value * value for value in target) ** 0.5
        for record in self._records:
            if where and any(record["metadata"].get(key) != value for key, value in where.items()):
                continue
            embedding = record["embedding"]
            embedding_norm = sum(value * value for value in embedding) ** 0.5
            if target_norm == 0 or embedding_norm == 0:
                similarity = 0.0
            else:
                similarity = sum(a * b for a, b in zip(target, embedding)) / (target_norm * embedding_norm)
            distance = 1.0 - similarity
            scored.append((distance, record))
        scored.sort(key=lambda item: item[0])
        top = scored[:n_results]
        return {
            "documents": [[record["document"] for _, record in top]],
            "metadatas": [[record["metadata"] for _, record in top]],
            "distances": [[distance for distance, _ in top]],
        }

    def delete(self, ids):
        remove = set(ids)
        self._records = [record for record in self._records if record["id"] not in remove]


class FakePersistentClient:
    def __init__(self, path, settings=None):
        self.path = path
        self.settings = settings
        self._collections = {}

    def get_or_create_collection(self, name, metadata=None):
        if name not in self._collections:
            self._collections[name] = FakeCollection(name=name, metadata=metadata)
        return self._collections[name]

    def create_collection(self, name, metadata=None):
        collection = FakeCollection(name=name, metadata=metadata)
        self._collections[name] = collection
        return collection

    def delete_collection(self, name):
        self._collections.pop(name, None)


@pytest.fixture(autouse=True)
def fake_faq_dependencies(monkeypatch):
    import src.knowledge.faq_store as faq_store_module

    faq_store_module.FAQStore._EMBEDDING_MODEL_CACHE.clear()
    faq_store_module.FAQStore._RERANKER_MODEL_CACHE.clear()
    monkeypatch.setattr(faq_store_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(faq_store_module.chromadb, "PersistentClient", FakePersistentClient)
    yield


class TestFAQResult:
    """Test FAQResult dataclass."""

    def test_create_result(self):
        result = FAQResult(
            question="Test question?",
            answer="Test answer",
            category="test",
            confidence=0.95,
            metadata={"keywords": ["test"]},
        )

        assert result.question == "Test question?"
        assert result.answer == "Test answer"
        assert result.confidence == 0.95
        assert result.metadata["keywords"] == ["test"]

    def test_str_representation(self):
        result = FAQResult(
            question="如何重置密码？",
            answer="点击重置按钮",
            category="account",
            confidence=0.85,
        )

        str_result = str(result)
        assert "account" in str_result
        assert "85.00%" in str_result


class TestFAQStore:
    """Test FAQStore class."""

    @pytest.fixture
    def temp_faq_store(self, tmp_path):
        return FAQStore(chroma_path=tmp_path / "chroma", collection_name="test_faqs")

    def test_init_with_samples(self, temp_faq_store):
        temp_faq_store.clear_all()
        temp_faq_store._load_sample_faqs()

        stats = temp_faq_store.get_stats()
        assert stats["total_faqs"] > 0
        assert len(stats["categories"]) > 0

    def test_add_faq(self, temp_faq_store):
        temp_faq_store.clear_all()

        faq_id = temp_faq_store.add_faq(
            question="Test question?",
            answer="Test answer",
            category="test",
            metadata={"priority": "high"},
        )

        assert faq_id is not None
        assert temp_faq_store.get_stats()["total_faqs"] == 1

    def test_search(self, temp_faq_store):
        results = temp_faq_store.search("如何重置密码？", top_k=3)

        assert results
        assert isinstance(results[0], FAQResult)
        assert results[0].confidence > 0
        assert (
            "密码" in results[0].question
            or "密码" in results[0].answer
        )

    def test_search_with_category_filter(self, temp_faq_store):
        results = temp_faq_store.search(
            "取消订阅",
            category="billing",
            top_k=5,
        )

        assert results
        assert all(result.category == "billing" for result in results)

    def test_search_min_confidence(self, temp_faq_store):
        results = temp_faq_store.search(
            "gibberish xyz123",
            top_k=5,
            min_confidence=0.5,
        )

        assert all(result.confidence >= 0.5 for result in results)

    def test_get_categories(self, temp_faq_store):
        categories = temp_faq_store.get_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        for category in ["billing", "account", "support", "technical"]:
            assert category in categories

    def test_get_stats(self, temp_faq_store):
        stats = temp_faq_store.get_stats()

        assert "total_faqs" in stats
        assert "categories" in stats
        assert "category_count" in stats
        assert "embedding_model" in stats
        assert stats["total_faqs"] > 0

    def test_delete_faq(self, temp_faq_store):
        temp_faq_store.clear_all()
        temp_faq_store.add_faq(
            question="Unique question for deletion xyz123",
            answer="Answer to be deleted",
            category="test",
        )

        assert temp_faq_store.search("Unique question for deletion xyz123")
        assert temp_faq_store.delete_faq("Unique question for deletion xyz123") is True
        assert temp_faq_store.search("Unique question for deletion xyz123") == []

    def test_clear_all(self, temp_faq_store):
        temp_faq_store.clear_all()
        assert temp_faq_store.get_stats()["total_faqs"] == 0

    def test_load_faqs_from_json(self, temp_faq_store):
        test_faqs = [
            {
                "question": "JSON test question?",
                "answer": "JSON test answer",
                "category": "json_test",
                "metadata": {"keywords": ["json", "test"]},
            },
            {
                "question": "Another JSON question?",
                "answer": "Another JSON answer",
                "category": "json_test",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(test_faqs, handle)
            temp_file = handle.name

        try:
            temp_faq_store.clear_all()
            count = temp_faq_store.load_faqs_from_file(temp_file)
            assert count == 2
            assert temp_faq_store.search("JSON test")
        finally:
            Path(temp_file).unlink()

    def test_load_faqs_from_csv(self, temp_faq_store):
        csv_content = """question,answer,category,keywords
CSV test question?,CSV test answer,csv_test,"csv, test"
Another CSV?,Another CSV answer,csv_test,csv"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write(csv_content)
            temp_file = handle.name

        try:
            temp_faq_store.clear_all()
            count = temp_faq_store.load_faqs_from_file(temp_file)
            assert count == 2
            assert temp_faq_store.search("CSV test")
        finally:
            Path(temp_file).unlink()

    def test_load_faqs_invalid_file(self, temp_faq_store):
        with pytest.raises(FileNotFoundError):
            temp_faq_store.load_faqs_from_file("nonexistent.json")

    def test_load_faqs_unsupported_format(self, temp_faq_store):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("Not a valid format")
            temp_file = handle.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                temp_faq_store.load_faqs_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_search_relevance_ranking(self, temp_faq_store):
        results = temp_faq_store.search("支付方式", top_k=5)
        assert all(results[i].confidence >= results[i + 1].confidence for i in range(len(results) - 1))

    def test_various_queries(self, temp_faq_store):
        test_queries = [
            ("如何重置密码？", "密码"),
            ("支持哪些付款方式？", "付款"),
            ("如何取消订阅？", "订阅"),
            ("如何邀请团队成员？", "成员"),
            ("是否提供 API？", "API"),
            ("如何联系客服？", "客服"),
        ]

        for query, expected_keyword in test_queries:
            results = temp_faq_store.search(query, top_k=2)
            assert results, f"No results for query: {query}"
            assert any(
                expected_keyword in result.question or expected_keyword in result.answer
                for result in results
            )

    def test_search_returns_clean_answer_field(self, temp_faq_store):
        result = temp_faq_store.search("如何取消订阅？", top_k=1)[0]
        assert result.answer
        assert not result.answer.startswith("Question:")

    def test_normalize_query_supports_english_and_synonyms(self, temp_faq_store):
        normalized = temp_faq_store._normalize_query("cancel subscription plan")
        assert "取消" in normalized
        assert "订阅" in normalized
        assert "套餐" in normalized

    def test_infer_query_category_prefers_billing(self, temp_faq_store):
        category = temp_faq_store._infer_query_category("为什么扣费199？")
        assert category == "billing"

    def test_split_subqueries_for_compound_question(self, temp_faq_store):
        subqueries = temp_faq_store._split_subqueries(
            "怎么取消套餐，取消后什么时候生效？"
        )
        assert len(subqueries) == 2
        assert subqueries[0] == "如何取消订阅？"
        assert subqueries[1] == "取消订阅后什么时候生效？"

    def test_search_hybrid_records_trace(self, temp_faq_store):
        results = temp_faq_store.search_hybrid(
            "为什么扣费199？",
            top_k=2,
        )
        trace = temp_faq_store.get_last_query_trace()

        assert results
        assert trace["original_query"] == "为什么扣费199？"
        assert trace["effective_category"] == "billing"
        assert trace["retrieval_rounds"]
        assert trace["final_result_count"] == len(results)

    def test_search_hybrid_rewrite_fallback(self, temp_faq_store, monkeypatch):
        rewritten = "如何取消订阅？"

        def fake_vector(query, category=None, top_k=3, min_confidence=0.0):
            if query == rewritten:
                return [
                    FAQResult(
                        question=rewritten,
                        answer="在设置 > 账单 > 订阅管理中可以取消套餐。",
                        category="billing",
                        confidence=0.82,
                    )
                ]
            return []

        monkeypatch.setattr(temp_faq_store, "_vector_search", fake_vector)
        monkeypatch.setattr(temp_faq_store, "_keyword_search", lambda query, top_k=6, category=None: [])

        results = temp_faq_store.search_hybrid("怎么退", top_k=2)
        trace = temp_faq_store.get_last_query_trace()

        assert results
        assert results[0].question == rewritten
        assert trace["rewrite_used"] is True
        assert trace["retrieval_rounds"][-1]["round_type"] == "rewrite_retry"


class TestCreateFAQStore:
    """Test FAQStore factory function."""

    def test_create_with_samples(self, tmp_path):
        store = create_faq_store(chroma_path=tmp_path / "test1", load_samples=True)
        assert store.get_stats()["total_faqs"] > 0

    def test_create_without_samples(self, tmp_path):
        store = create_faq_store(chroma_path=tmp_path / "test2", load_samples=False)
        assert store.get_stats()["total_faqs"] == 0
