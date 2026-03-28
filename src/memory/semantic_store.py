"""Persistent structured memory backed by the business DB and Chroma."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..db.repositories import (
    delete_user_memory_record,
    get_user_memory_record,
    list_user_memory_records,
    upsert_user_memory_record,
)

logger = logging.getLogger(__name__)


class SemanticMemoryStore:
    """Semantic retrieval for structured long-term user memory."""

    _EMBEDDING_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        *,
        chroma_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.chroma_path = Path(chroma_path or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.memory_collection_name
        self.embedding_model_name = embedding_model or settings.embedding_model

        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = self._load_embedding_model(self.embedding_model_name)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def _load_embedding_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._EMBEDDING_MODEL_CACHE:
            cls._EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return cls._EMBEDDING_MODEL_CACHE[model_name]

    def _normalize_embedding(self, embedding: Any) -> List[float]:
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            return [value / norm for value in values]
        return values

    def _encode(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        except TypeError:
            embedding = self.embedding_model.encode(text)
        return self._normalize_embedding(embedding)

    def _memory_text(self, payload: Dict[str, Any]) -> str:
        parts = [
            payload.get("memory_type", ""),
            payload.get("field", ""),
            payload.get("value", ""),
            payload.get("category", ""),
            payload.get("issue_code", ""),
            payload.get("summary", ""),
            payload.get("content", ""),
            " ".join(payload.get("tags", []) or []),
        ]
        return "\n".join(str(part) for part in parts if part)

    def _memory_tokens(self, text: str) -> List[str]:
        lowered = str(text or "").lower()
        english_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", str(text or ""))
        chinese_bigrams = [
            "".join(chinese_chars[index : index + 2])
            for index in range(max(0, len(chinese_chars) - 1))
        ]
        ordered: List[str] = []
        for token in [*english_tokens, *chinese_chars, *chinese_bigrams]:
            cleaned = token.strip()
            if cleaned and cleaned not in ordered:
                ordered.append(cleaned)
        return ordered

    def _lexical_score(self, payload: Dict[str, Any], query: str) -> float:
        query_tokens = self._memory_tokens(query)
        if not query_tokens:
            return 0.0
        memory_text = self._memory_text(payload).lower()
        overlap = sum(1 for token in query_tokens if token and token in memory_text)
        return overlap / max(1, len(query_tokens))

    def _recency_score(self, payload: Dict[str, Any]) -> float:
        updated_at = str(payload.get("updated_at", "")).strip()
        if not updated_at:
            return 0.0
        try:
            when = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - when).total_seconds() / 86400.0)
        return max(0.0, 0.18 - min(age_days, 30.0) * 0.006)

    def _extra_bonus(self, payload: Dict[str, Any]) -> float:
        bonus = 0.0
        if payload.get("memory_type") == "open_issue":
            bonus += 0.08
        if payload.get("status") == "resolved":
            bonus -= 0.02
        return bonus

    def upsert_memory(self, *, user_id: str, memory_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        stored = upsert_user_memory_record(user_id=user_id, memory_id=memory_id, payload=payload)
        metadata = {
            "memory_id": memory_id,
            "user_id": user_id,
            "memory_type": stored.get("memory_type", "memory"),
            "status": stored.get("status", "active"),
            "category": stored.get("category") or "",
            "issue_code": stored.get("issue_code") or "",
        }
        with_memory_text = self._memory_text(stored)
        try:
            self.collection.delete(ids=[memory_id])
        except Exception:
            pass
        self.collection.add(
            ids=[memory_id],
            embeddings=[self._encode(with_memory_text)],
            documents=[with_memory_text],
            metadatas=[metadata],
        )
        return stored

    def delete_memory(self, *, user_id: str, memory_id: str) -> bool:
        deleted = delete_user_memory_record(user_id, memory_id)
        if deleted:
            try:
                self.collection.delete(ids=[memory_id])
            except Exception:
                pass
        return deleted

    def get_memory(self, *, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        return get_user_memory_record(user_id, memory_id)

    def list_memories(self, *, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return list_user_memory_records(user_id, limit=limit)

    def search_memories(self, *, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        items = list_user_memory_records(user_id, limit=settings.max_memory_items_per_user)
        if not items:
            return []

        vector_scores: Dict[str, float] = {}
        try:
            response = self.collection.query(
                query_embeddings=[self._encode(query)],
                n_results=max(8, limit * 4),
                where={"user_id": user_id},
            )
            metadatas = response.get("metadatas", [[]])[0] or []
            distances = response.get("distances", [[]])[0] or []
            for metadata, distance in zip(metadatas, distances):
                memory_id = str((metadata or {}).get("memory_id") or "").strip()
                if memory_id:
                    vector_scores[memory_id] = max(vector_scores.get(memory_id, 0.0), max(0.0, 1.0 - float(distance or 0.0)))
        except Exception as error:
            logger.warning(f"Semantic memory vector lookup failed: {error}")

        ranked = sorted(
            items,
            key=lambda payload: (
                0.55 * vector_scores.get(payload["memory_id"], 0.0)
                + 0.20 * self._lexical_score(payload, query)
                + 0.15 * float(payload.get("importance", 0.0) or 0.0)
                + 0.10 * self._recency_score(payload)
                + self._extra_bonus(payload),
                payload.get("updated_at", ""),
            ),
            reverse=True,
        )

        filtered: List[Dict[str, Any]] = []
        for payload in ranked:
            score = (
                0.55 * vector_scores.get(payload["memory_id"], 0.0)
                + 0.20 * self._lexical_score(payload, query)
                + 0.15 * float(payload.get("importance", 0.0) or 0.0)
                + 0.10 * self._recency_score(payload)
                + self._extra_bonus(payload)
            )
            if score <= 0.12:
                continue
            filtered.append(payload)
            if len(filtered) >= limit:
                break
        return filtered

    def close(self) -> None:
        """Compatibility hook for the owner service."""
        return None
