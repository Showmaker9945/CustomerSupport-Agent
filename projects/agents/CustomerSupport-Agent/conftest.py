from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_FAKE_VOCAB = [
    "密码", "重置", "支付", "账单", "扣费", "取消", "订阅",
    "套餐", "团队", "成员", "api", "客服", "工单", "安全",
    "加密", "数据", "存储", "账户", "登录", "邀请",
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
            self._records.append(
                {
                    "embedding": list(embedding),
                    "document": document,
                    "metadata": dict(metadata),
                    "id": record_id,
                }
            )

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


@pytest.fixture
def isolated_business_db(monkeypatch, tmp_path):
    from src.config import settings
    from src.db.repositories import reset_business_database
    from src.db.session import reset_database_connection

    db_path = (tmp_path / "business.db").resolve()
    monkeypatch.setattr(settings, "database_url", f"sqlite+pysqlite:///{db_path.as_posix()}", raising=False)
    reset_database_connection()
    reset_business_database(seed_demo=True)
    yield
    reset_database_connection()
