"""Document-backed knowledge base for the customer support agent."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..config import settings
from ..db.repositories import (
    estimate_text_tokens,
    list_knowledge_chunks,
    list_knowledge_documents,
    replace_knowledge_corpus,
)

logger = logging.getLogger(__name__)

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


@dataclass
class KnowledgeResult:
    """Search result returned to the support tools layer."""

    question: str
    answer: str
    category: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSection:
    title: str
    section_path: str
    category: str
    content: str
    order: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryAnalysis:
    original_query: str
    expanded_query: str
    query_tokens: List[str]
    focus_terms: List[str] = field(default_factory=list)
    matched_intents: List[str] = field(default_factory=list)
    inferred_category: Optional[str] = None


class DocumentStore:
    """Hybrid document retriever with structure-aware parent/child chunking."""

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".json"}
    _EMBEDDING_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
    _RERANKER_MODEL_CACHE: Dict[str, Any] = {}
    QUERY_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
        "重置密码": ("忘记密码", "找回密码", "密码恢复", "登录恢复"),
        "忘记密码": ("重置密码", "找回密码", "密码恢复", "登录恢复"),
        "邮箱不可用": ("注册邮箱不可用", "原邮箱不可用", "原邮箱停用", "邮箱失效", "无法访问注册邮箱"),
        "付款方式": ("支付方式", "结算方式", "扣费方式", "支付"),
        "支付方式": ("付款方式", "结算方式", "扣费方式", "支付"),
        "取消套餐": ("取消订阅", "关闭自动续费", "退订", "退款规则"),
        "取消订阅": ("取消套餐", "关闭自动续费", "退订", "退款规则"),
        "邀请成员": ("邀请团队成员", "团队成员", "新增成员", "成员邀请"),
        "api": ("开发者", "接口", "API Key", "集成"),
        "账单异常": ("扣费异常", "重复扣款", "异常收费", "财务复核"),
        "转人工": ("人工客服", "人工审核", "人工复核", "升级人工"),
    }
    QUERY_INTENTS: Tuple[Dict[str, Any], ...] = (
        {
            "name": "account_recovery",
            "triggers": (
                "重置密码",
                "忘记密码",
                "找回密码",
                "密码恢复",
                "登录失败",
                "无法登录",
                "登不上",
                "邮箱不可用",
                "注册邮箱不可用",
                "原邮箱不可用",
                "原邮箱停用",
                "邮箱失效",
                "收不到重置邮件",
            ),
            "expansions": (
                "密码重置",
                "找回密码",
                "登录恢复",
                "账号恢复",
                "无法访问注册邮箱",
                "身份核验",
                "恢复链接",
            ),
            "focus_terms": (
                "重置密码",
                "密码恢复",
                "登录恢复",
                "邮箱不可用",
                "注册邮箱不可用",
                "原邮箱不可用",
                "身份核验",
            ),
            "categories": ("登录恢复",),
        },
        {
            "name": "two_factor_recovery",
            "triggers": (
                "2fa",
                "双重身份验证",
                "验证码",
                "验证器",
                "短信验证",
                "恢复码",
                "谷歌验证",
            ),
            "expansions": (
                "验证码 App",
                "短信验证",
                "双重验证",
                "恢复码",
                "身份核验",
            ),
            "focus_terms": ("2FA", "双重身份验证", "验证码", "恢复码", "身份核验"),
            "categories": ("双重身份验证与恢复码",),
        },
        {
            "name": "permissions_and_transfer",
            "triggers": (
                "管理员",
                "编辑者",
                "访客",
                "角色",
                "提权",
                "所有权",
                "owner",
                "转移",
                "权限",
            ),
            "expansions": (
                "角色提升",
                "权限变更",
                "管理员权限",
                "账户所有权转移",
            ),
            "focus_terms": ("管理员", "角色提升", "权限变更", "所有权转移"),
            "categories": ("权限与角色变更",),
        },
        {
            "name": "billing_review",
            "triggers": (
                "账单",
                "扣费",
                "重复扣款",
                "退款",
                "续费",
                "发票",
                "金额异常",
                "多扣",
            ),
            "expansions": (
                "账单异常",
                "扣费异常",
                "费用构成",
                "人工复核",
                "账单核查",
            ),
            "focus_terms": ("账单异常", "扣费异常", "重复扣款", "退款", "人工复核"),
            "categories": ("账单异常分诊", "对账与金额解释", "退款与补偿判断"),
        },
        {
            "name": "ticket_review",
            "triggers": ("工单", "人工审核", "人工复核", "审批", "升级人工", "转人工"),
            "expansions": ("工单创建", "人工审核", "人工复核", "升级人工"),
            "focus_terms": ("工单创建", "人工审核", "人工复核", "升级人工"),
            "categories": ("工单创建与人工审核", "支持与服务"),
        },
    )
    QUERY_STOPWORDS = {
        "的",
        "了",
        "吗",
        "呢",
        "啊",
        "呀",
        "吧",
        "和",
        "与",
        "及",
        "并",
        "还",
        "时",
        "后",
        "前",
        "再",
        "又",
        "都",
        "将",
        "把",
        "被",
        "请问",
        "怎么",
        "如何",
        "一下",
        "一下子",
        "可以",
        "能够",
        "当前",
        "现在",
        "问题",
        "用户",
        "客服",
        "你们",
        "我们",
    }

    def __init__(
        self,
        *,
        knowledge_base_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.knowledge_base_path = Path(knowledge_base_path or settings.knowledge_base_path)
        self.chroma_path = Path(chroma_path or settings.chroma_persist_dir)
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.collection_name = collection_name or settings.collection_name
        self.parent_collection_name = f"{self.collection_name}_parent"
        self.child_collection_name = f"{self.collection_name}_child"
        self.enable_reranker = bool(settings.enable_reranker)

        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = self._load_embedding_model(self.embedding_model_name)
        self.reranker = self._load_reranker(settings.reranker_model) if self.enable_reranker else None

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.parent_collection = self.client.get_or_create_collection(
            name=self.parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.child_collection = self.client.get_or_create_collection(
            name=self.child_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._parent_chunks: Dict[str, Dict[str, Any]] = {}
        self._child_chunks: Dict[str, Dict[str, Any]] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_records: List[Dict[str, Any]] = []
        self._last_query_trace: Dict[str, Any] = {}

        self._load_metadata_cache()
        if not self._child_chunks or self.child_collection.count() == 0:
            try:
                self.reindex(clear_existing=True)
            except Exception as error:
                logger.warning(f"Initial document knowledge indexing skipped: {error}")
        else:
            self._refresh_keyword_index()

    @classmethod
    def _load_embedding_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._EMBEDDING_MODEL_CACHE:
            cls._EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        return cls._EMBEDDING_MODEL_CACHE[model_name]

    @classmethod
    def _load_reranker(cls, model_name: str) -> Any:
        if model_name not in cls._RERANKER_MODEL_CACHE:
            cls._RERANKER_MODEL_CACHE[model_name] = CrossEncoder(model_name, max_length=512)
        return cls._RERANKER_MODEL_CACHE[model_name]

    def _build_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=max(120, int(chunk_size)),
            chunk_overlap=max(0, int(chunk_overlap)),
            length_function=len,
            separators=[
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                "。", "！", "？", "；",
                ". ", "! ", "? ", "; ",
                "，", ", ",
                " ",
                "",
            ],
        )

    def _load_metadata_cache(self) -> None:
        self._parent_chunks = {}
        self._child_chunks = {}
        for chunk in list_knowledge_chunks():
            if chunk.get("chunk_level") == "parent":
                self._parent_chunks[chunk["chunk_id"]] = chunk
            else:
                self._child_chunks[chunk["chunk_id"]] = chunk
        self._refresh_keyword_index()

    def _normalize_embedding(self, embedding: Any) -> List[float]:
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            return [value / norm for value in values]
        return values

    def _encode_embedding_text(self, text: str) -> List[float]:
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        except TypeError:
            embedding = self.embedding_model.encode(text)
        return self._normalize_embedding(embedding)

    def _create_query_embedding(self, query: str) -> List[float]:
        instruction = str(settings.embedding_query_instruction or "").strip()
        query_text = f"{instruction}{query}" if instruction else query
        return self._encode_embedding_text(query_text)

    def _create_document_embedding(self, text: str) -> List[float]:
        return self._encode_embedding_text(text)

    def _normalize_rerank_score(self, score: float) -> float:
        clipped = max(min(float(score), 12.0), -12.0)
        return 1.0 / (1.0 + math.exp(-clipped))

    def _preview_text(self, text: str, limit: int = 280) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"

    def _unique_phrases(self, phrases: List[str]) -> List[str]:
        ordered: List[str] = []
        for phrase in phrases:
            cleaned = str(phrase or "").strip()
            if cleaned and cleaned not in ordered:
                ordered.append(cleaned)
        return ordered

    def _analyze_query(self, query: str) -> QueryAnalysis:
        normalized = str(query or "").strip()
        if not normalized:
            return QueryAnalysis(original_query="", expanded_query="", query_tokens=[])

        lowered = normalized.lower()
        expanded_terms: List[str] = [normalized]
        focus_terms: List[str] = []
        matched_intents: List[str] = []
        category_scores: Dict[str, int] = {}

        for needle, expansions in self.QUERY_EXPANSIONS.items():
            if needle.lower() not in lowered:
                continue
            expanded_terms.extend(list(expansions))
            focus_terms.append(needle)

        for intent in self.QUERY_INTENTS:
            triggers = [str(item or "").lower() for item in intent.get("triggers", ())]
            if not any(trigger in lowered for trigger in triggers):
                continue
            matched_intents.append(str(intent.get("name") or "unknown"))
            expanded_terms.extend([str(item) for item in intent.get("expansions", ())])
            focus_terms.extend([str(item) for item in intent.get("focus_terms", ())])
            for category in intent.get("categories", ()):
                normalized_category = str(category or "").strip()
                if normalized_category:
                    category_scores[normalized_category] = category_scores.get(normalized_category, 0) + 1

        query_tokens = self._tokenize_for_bm25(normalized)
        if not focus_terms:
            focus_terms = [token for token in query_tokens if len(token) >= 2]

        inferred_category = None
        if category_scores:
            inferred_category = max(
                category_scores.items(),
                key=lambda item: (item[1], len(item[0])),
            )[0]

        expanded_query = " ".join(self._unique_phrases(expanded_terms))
        return QueryAnalysis(
            original_query=normalized,
            expanded_query=expanded_query,
            query_tokens=query_tokens,
            focus_terms=self._unique_phrases([term for term in focus_terms if len(term) >= 2]),
            matched_intents=self._unique_phrases(matched_intents),
            inferred_category=inferred_category,
        )

    def _expand_query(self, query: str) -> str:
        return self._analyze_query(query).expanded_query

    def _focus_match_bonus(self, analysis: QueryAnalysis, payload: Dict[str, Any]) -> float:
        if not analysis.focus_terms:
            return 0.0
        section_text = " ".join(
            [
                str(payload.get("title", "") or ""),
                str(payload.get("section_path", "") or ""),
                str(payload.get("category", "") or ""),
            ]
        ).lower()
        content_text = str(payload.get("content", "") or "").lower()
        section_hits = 0
        content_hits = 0
        for term in analysis.focus_terms:
            lowered_term = term.lower()
            if lowered_term in section_text:
                section_hits += 1
                continue
            if lowered_term in content_text:
                content_hits += 1
        total_hits = section_hits + content_hits
        if total_hits == 0:
            return 0.0
        coverage = total_hits / max(1, len(analysis.focus_terms))
        section_coverage = section_hits / max(1, len(analysis.focus_terms))
        bonus = coverage * 0.18 + section_coverage * 0.16
        if section_hits and content_hits:
            bonus += 0.04
        if coverage >= 0.95 and len(analysis.focus_terms) > 1:
            bonus += 0.08
        return min(0.46, bonus)

    def _title_match_bonus(self, analysis: QueryAnalysis, payload: Dict[str, Any]) -> float:
        query_tokens = set(analysis.query_tokens)
        if not query_tokens:
            return 0.0
        title_text = " ".join(
            [
                str(payload.get("title", "") or ""),
                str(payload.get("section_path", "") or ""),
            ]
        ).strip()
        title_tokens = set(self._tokenize_for_bm25(title_text))
        overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        compact_query = re.sub(r"\s+", "", analysis.original_query)
        compact_title = re.sub(r"\s+", "", title_text)
        exact_bonus = 0.18 if compact_query and compact_query in compact_title else 0.0
        return min(0.36, overlap * 0.24 + exact_bonus)

    def _category_match_bonus(self, analysis: QueryAnalysis, payload: Dict[str, Any]) -> float:
        inferred_category = str(analysis.inferred_category or "").strip()
        payload_category = str(payload.get("category", "") or "").strip()
        if not inferred_category or not payload_category:
            return 0.0
        return 0.14 if inferred_category == payload_category else 0.0

    def _stable_doc_id(self, source_path: str) -> str:
        return hashlib.md5(source_path.encode("utf-8")).hexdigest()

    def _checksum(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _supported_documents(self) -> List[Path]:
        paths = [
            path
            for path in self.knowledge_base_path.rglob("*")
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES
        ]
        return sorted(paths, key=lambda item: item.as_posix())

    def _infer_category(self, title: str, headings: List[str], source_path: str) -> str:
        raw = headings[0] if headings else title or Path(source_path).stem
        normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", raw.strip().lower()).strip("_")
        return normalized or "general"

    def _parse_markdown_sections(self, *, source_path: str, title: str, text: str) -> List[ParsedSection]:
        lines = text.splitlines()
        sections: List[ParsedSection] = []
        heading_stack: List[str] = []
        buffer: List[str] = []
        order = 0

        def flush() -> None:
            nonlocal order
            content = "\n".join(buffer).strip()
            if not content:
                return
            filtered_headings = [item for item in heading_stack if item and item != title]
            leaf_title = filtered_headings[-1] if filtered_headings else title
            section_path = " > ".join([title, *filtered_headings]) if filtered_headings else title
            category = self._infer_category(leaf_title, filtered_headings, source_path)
            sections.append(
                ParsedSection(
                    title=leaf_title,
                    section_path=section_path,
                    category=category,
                    content=content,
                    order=order,
                    metadata={"source_path": source_path, "doc_title": title},
                )
            )
            order += 1

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
            if match:
                flush()
                buffer = []
                level = len(match.group(1))
                heading = match.group(2).strip()
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(heading)
                continue
            buffer.append(line)

        flush()
        if sections:
            return sections

        fallback = text.strip()
        if not fallback:
            return []
        return [
            ParsedSection(
                title=title,
                section_path=title,
                category=self._infer_category(title, [], source_path),
                content=fallback,
                order=0,
                metadata={"source_path": source_path, "doc_title": title},
            )
        ]

    def _flatten_json(self, data: Any, prefix: str = "") -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(self._flatten_json(value, path))
            return rows
        if isinstance(data, list):
            for index, value in enumerate(data, 1):
                path = f"{prefix}[{index}]"
                rows.extend(self._flatten_json(value, path))
            return rows
        rows.append((prefix or "value", str(data)))
        return rows

    def _parse_json_sections(self, *, source_path: str, title: str, text: str) -> List[ParsedSection]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        flattened = self._flatten_json(payload)
        sections: List[ParsedSection] = []
        for order, (path, value) in enumerate(flattened):
            sections.append(
                ParsedSection(
                    title=path.split(".")[-1],
                    section_path=f"{title} > {path}",
                    category=self._infer_category(path, [path], source_path),
                    content=f"{path}: {value}",
                    order=order,
                    metadata={"source_path": source_path, "doc_title": title},
                )
            )
        return sections

    def _parse_text_sections(self, *, source_path: str, title: str, text: str) -> List[ParsedSection]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        sections: List[ParsedSection] = []
        for order, block in enumerate(blocks):
            first_line = block.splitlines()[0].strip()
            leaf_title = first_line[:64] if first_line else f"{title} Part {order + 1}"
            sections.append(
                ParsedSection(
                    title=leaf_title,
                    section_path=f"{title} > Section {order + 1}",
                    category=self._infer_category(leaf_title, [leaf_title], source_path),
                    content=block,
                    order=order,
                    metadata={"source_path": source_path, "doc_title": title},
                )
            )
        return sections

    def _parse_document(self, path: Path) -> Tuple[Dict[str, Any], List[ParsedSection]]:
        relative_source = path.relative_to(self.knowledge_base_path).as_posix()
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        title = path.stem
        if path.suffix.lower() in {".md", ".markdown"}:
            heading_match = re.search(r"^\s*#\s+(.+?)\s*$", raw_text, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip()
            sections = self._parse_markdown_sections(source_path=relative_source, title=title, text=raw_text)
            doc_type = "markdown"
        elif path.suffix.lower() == ".json":
            sections = self._parse_json_sections(source_path=relative_source, title=title, text=raw_text)
            doc_type = "json"
        else:
            sections = self._parse_text_sections(source_path=relative_source, title=title, text=raw_text)
            doc_type = "text"

        document = {
            "doc_id": self._stable_doc_id(relative_source),
            "source_path": relative_source,
            "title": title,
            "doc_type": doc_type,
            "checksum": self._checksum(raw_text),
        }
        return document, sections

    def _chunk_document(
        self,
        *,
        document: Dict[str, Any],
        sections: List[ParsedSection],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        parent_splitter = self._build_splitter(settings.parent_chunk_size, settings.parent_chunk_overlap)
        child_splitter = self._build_splitter(settings.chunk_size, settings.chunk_overlap)
        parent_chunks: List[Dict[str, Any]] = []
        child_chunks: List[Dict[str, Any]] = []
        parent_index = 0

        for section in sections:
            parent_texts = parent_splitter.split_text(section.content) or [section.content]
            for local_parent_index, parent_text in enumerate(parent_texts):
                parent_chunk_id = f"{document['doc_id']}:parent:{parent_index:04d}"
                base_metadata = {
                    "doc_id": document["doc_id"],
                    "document_title": document["title"],
                    "source_path": document["source_path"],
                    "doc_type": document["doc_type"],
                    "section_path": section.section_path,
                    "title": section.title,
                    "category": section.category,
                }
                parent_chunks.append(
                    {
                        "chunk_id": parent_chunk_id,
                        "doc_id": document["doc_id"],
                        "parent_chunk_id": None,
                        "chunk_level": "parent",
                        "chunk_index": parent_index,
                        "child_index": None,
                        "section_path": section.section_path,
                        "title": section.title,
                        "category": section.category,
                        "content": parent_text.strip(),
                        "char_count": len(parent_text),
                        "token_count": estimate_text_tokens(parent_text),
                        "metadata": {
                            **base_metadata,
                            "section_order": section.order,
                            "parent_local_index": local_parent_index,
                        },
                    }
                )

                child_texts = child_splitter.split_text(parent_text) or [parent_text]
                for child_index, child_text in enumerate(child_texts):
                    child_chunks.append(
                        {
                            "chunk_id": f"{parent_chunk_id}:child:{child_index:02d}",
                            "doc_id": document["doc_id"],
                            "parent_chunk_id": parent_chunk_id,
                            "chunk_level": "child",
                            "chunk_index": parent_index,
                            "child_index": child_index,
                            "section_path": section.section_path,
                            "title": section.title,
                            "category": section.category,
                            "content": child_text.strip(),
                            "char_count": len(child_text),
                            "token_count": estimate_text_tokens(child_text),
                            "metadata": {
                                **base_metadata,
                                "parent_chunk_id": parent_chunk_id,
                                "child_chunk_id": f"{parent_chunk_id}:child:{child_index:02d}",
                                "section_order": section.order,
                                "parent_local_index": local_parent_index,
                                "child_local_index": child_index,
                            },
                        }
                    )
                parent_index += 1

        return parent_chunks, child_chunks

    def _collection_document_text(self, chunk: Dict[str, Any]) -> str:
        title = chunk.get("title", "")
        section_path = chunk.get("section_path", "")
        document_title = chunk.get("metadata", {}).get("document_title", "")
        category = chunk.get("category", "")
        content = chunk.get("content", "")
        parts = [part for part in [document_title, category, section_path, title, content] if part]
        return "\n".join(parts)

    def _refresh_keyword_index(self) -> None:
        child_records = list(self._child_chunks.values())
        self._bm25_records = child_records
        if not child_records:
            self._bm25 = None
            return
        tokens = [self._tokenize_for_bm25(self._collection_document_text(record)) for record in child_records]
        self._bm25 = BM25Okapi(tokens)

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        lowered = str(text or "").lower()
        english_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", str(text or ""))
        chinese_bigrams = [
            "".join(chinese_chars[index : index + 2])
            for index in range(max(0, len(chinese_chars) - 1))
        ]
        chinese_trigrams = [
            "".join(chinese_chars[index : index + 3])
            for index in range(max(0, len(chinese_chars) - 2))
        ]
        ordered: List[str] = []
        for token in [*english_tokens, *chinese_chars, *chinese_bigrams, *chinese_trigrams]:
            cleaned = token.strip()
            if cleaned and cleaned not in self.QUERY_STOPWORDS and cleaned not in ordered:
                ordered.append(cleaned)
        return ordered

    def _vector_search(self, query: str, *, category: Optional[str], top_n: int) -> List[Tuple[Dict[str, Any], float]]:
        if not self._child_chunks:
            return []
        query_embedding = self._create_query_embedding(query)
        where = {"category": category} if category else None
        response = self.child_collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_n),
            where=where,
        )
        metadatas = response.get("metadatas", [[]])[0] or []
        distances = response.get("distances", [[]])[0] or []
        results: List[Tuple[Dict[str, Any], float]] = []
        for metadata, distance in zip(metadatas, distances):
            chunk_id = str((metadata or {}).get("child_chunk_id") or (metadata or {}).get("chunk_id") or "").strip()
            if not chunk_id or chunk_id not in self._child_chunks:
                continue
            results.append((self._child_chunks[chunk_id], max(0.0, 1.0 - float(distance or 0.0))))
        return results

    def _keyword_search(self, query: str, *, category: Optional[str], top_n: int) -> List[Tuple[Dict[str, Any], float]]:
        if self._bm25 is None or not self._bm25_records:
            return []
        tokens = self._tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = list(self._bm25.get_scores(tokens))
        max_score = max(scores) if scores else 0.0
        ranked: List[Tuple[Dict[str, Any], float]] = []
        for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True):
            record = self._bm25_records[index]
            if category and record.get("category") != category:
                continue
            raw_score = float(scores[index])
            normalized = (raw_score / max_score) if max_score > 0 else 0.0
            if normalized <= 0:
                continue
            ranked.append((record, normalized))
            if len(ranked) >= top_n:
                break
        return ranked

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        if not candidates or self.reranker is None:
            return {}
        pairs = [
            (query, self._collection_document_text(candidate["payload"]))
            for candidate in candidates
        ]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as error:
            logger.warning(f"Document rerank failed, using fusion ranking only: {error}")
            return {}
        rerank_scores: Dict[str, float] = {}
        for candidate, score in zip(candidates, scores):
            rerank_scores[candidate["payload"]["chunk_id"]] = self._normalize_rerank_score(score)
        return rerank_scores

    def _group_to_parent_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[KnowledgeResult]:
        if not candidates:
            return []
        rerank_scores = self._rerank_candidates(query, candidates[:12])
        grouped: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            payload = candidate["payload"]
            chunk_id = payload["chunk_id"]
            parent_chunk_id = payload["parent_chunk_id"]
            parent_chunk = self._parent_chunks.get(parent_chunk_id)
            if parent_chunk is None:
                continue
            rerank_score = rerank_scores.get(chunk_id, 0.0)
            combined = (
                settings.rag_vector_weight * candidate["vector_score"]
                + settings.rag_keyword_weight * candidate["keyword_score"]
                + candidate["rrf_score"]
                + candidate.get("title_bonus", 0.0)
                + candidate.get("focus_bonus", 0.0)
                + candidate.get("category_bonus", 0.0)
                + 0.25 * rerank_score
            )
            group = grouped.setdefault(
                parent_chunk_id,
                {
                    "parent": parent_chunk,
                    "best_score": 0.0,
                    "matched_chunks": [],
                },
            )
            group["best_score"] = max(group["best_score"], combined)
            group["matched_chunks"].append(
                {
                    "chunk_id": chunk_id,
                    "score": combined,
                    "title": payload.get("title", ""),
                    "preview": self._preview_text(payload.get("content", ""), limit=120),
                }
            )

        rows = sorted(grouped.values(), key=lambda item: item["best_score"], reverse=True)
        if not rows:
            return []

        best_score = max(item["best_score"] for item in rows) or 1.0
        results: List[KnowledgeResult] = []
        for row in rows:
            parent = row["parent"]
            matched_chunks = sorted(row["matched_chunks"], key=lambda item: item["score"], reverse=True)
            best_match_preview = matched_chunks[0]["preview"] if matched_chunks else self._preview_text(parent.get("content", ""))
            leaf_title = parent.get("title") or parent.get("section_path") or parent.get("metadata", {}).get("document_title", "Knowledge")
            confidence = min(0.99, max(0.15, row["best_score"] / best_score))
            results.append(
                KnowledgeResult(
                    question=leaf_title,
                    answer=best_match_preview,
                    category=parent.get("category", "general"),
                    confidence=confidence,
                    metadata={
                        "document_title": parent.get("metadata", {}).get("document_title", ""),
                        "source_path": parent.get("metadata", {}).get("source_path", ""),
                        "section_path": parent.get("section_path", ""),
                        "parent_chunk_id": parent.get("chunk_id"),
                        "matched_chunks": matched_chunks,
                    },
                )
            )
        return results

    def clear_all(self) -> None:
        for name in (self.parent_collection_name, self.child_collection_name):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self.parent_collection = self.client.get_or_create_collection(
            name=self.parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.child_collection = self.client.get_or_create_collection(
            name=self.child_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._parent_chunks = {}
        self._child_chunks = {}
        self._bm25 = None
        self._bm25_records = []
        self._last_query_trace = {}

    def _chunk_ids_for_documents(
        self,
        chunk_cache: Dict[str, Dict[str, Any]],
        doc_ids: List[str],
    ) -> List[str]:
        if not doc_ids:
            return []
        doc_id_set = set(doc_ids)
        return [
            chunk_id
            for chunk_id, payload in chunk_cache.items()
            if payload.get("doc_id") in doc_id_set
        ]

    def _delete_collection_records(self, collection: Any, ids: List[str]) -> None:
        cleaned_ids = [str(item).strip() for item in ids if str(item).strip()]
        if not cleaned_ids:
            return
        try:
            collection.delete(ids=cleaned_ids)
        except Exception as error:
            logger.warning(f"Failed to delete stale Chroma records: {error}")

    def _add_collection_records(self, collection: Any, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        chunk_texts = [self._collection_document_text(chunk) for chunk in chunks]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in chunks],
            embeddings=[self._create_document_embedding(text) for text in chunk_texts],
            documents=chunk_texts,
            metadatas=[
                {
                    **chunk["metadata"],
                    "chunk_id": chunk["chunk_id"],
                }
                for chunk in chunks
            ],
        )

    def reindex(self, clear_existing: bool = False) -> Dict[str, Any]:
        documents: List[Dict[str, Any]] = []
        parent_chunks: List[Dict[str, Any]] = []
        child_chunks: List[Dict[str, Any]] = []
        supported_documents = self._supported_documents()

        timestamp = datetime.now(timezone.utc).isoformat()
        for path in supported_documents:
            document, sections = self._parse_document(path)
            if not sections:
                logger.warning(f"Skipped empty knowledge document: {path}")
                continue
            document["status"] = "active"
            document["version"] = 1
            document["metadata"] = {
                "source_path": document["source_path"],
                "section_count": len(sections),
            }
            document["ingested_at"] = timestamp
            document["updated_at"] = timestamp
            documents.append(document)
            doc_parent_chunks, doc_child_chunks = self._chunk_document(document=document, sections=sections)
            parent_chunks.extend(doc_parent_chunks)
            child_chunks.extend(doc_child_chunks)

        incoming_doc_ids = sorted({document["doc_id"] for document in documents})
        removed_doc_ids: List[str] = []

        if clear_existing:
            self.clear_all()
        else:
            existing_doc_ids = {
                str(document.get("doc_id", "")).strip()
                for document in list_knowledge_documents()
                if str(document.get("doc_id", "")).strip()
            }
            removed_doc_ids = sorted(existing_doc_ids - set(incoming_doc_ids))
            doc_ids_to_refresh = sorted(set(incoming_doc_ids) | set(removed_doc_ids))
            self._delete_collection_records(
                self.parent_collection,
                self._chunk_ids_for_documents(self._parent_chunks, doc_ids_to_refresh),
            )
            self._delete_collection_records(
                self.child_collection,
                self._chunk_ids_for_documents(self._child_chunks, doc_ids_to_refresh),
            )

        replace_knowledge_corpus(
            documents=documents,
            chunks=[*parent_chunks, *child_chunks],
            clear_existing=clear_existing,
            remove_doc_ids=removed_doc_ids,
        )

        self._add_collection_records(self.parent_collection, parent_chunks)
        self._add_collection_records(self.child_collection, child_chunks)

        self._load_metadata_cache()
        stats = self.get_stats()
        stats.update(
            {
                "indexed_documents": len(documents),
                "indexed_parent_chunks": len(parent_chunks),
                "indexed_child_chunks": len(child_chunks),
                "clear_existing": clear_existing,
                "removed_documents": len(removed_doc_ids),
            }
        )
        return stats

    def search_hybrid(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        top_k: int = 4,
    ) -> List[KnowledgeResult]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            self._last_query_trace = {
                "query": query,
                "effective_category": category,
                "reason": "empty_query",
                "results": 0,
            }
            return []

        analysis = self._analyze_query(normalized_query)
        search_plans: List[Dict[str, Any]] = []
        if category:
            search_plans.append({"label": "explicit", "category": category, "bonus": 0.18})
        else:
            inferred_category = str(analysis.inferred_category or "").strip()
            if inferred_category:
                search_plans.append({"label": "inferred", "category": inferred_category, "bonus": 0.14})
            search_plans.append({"label": "global", "category": None, "bonus": 0.0})

        candidate_map: Dict[str, Dict[str, Any]] = {}
        total_vector_hits = 0
        total_keyword_hits = 0
        plan_trace: List[Dict[str, Any]] = []

        for plan in search_plans:
            plan_category = plan["category"]
            plan_bonus = float(plan.get("bonus", 0.0))
            vector_hits = self._vector_search(
                analysis.expanded_query,
                category=plan_category,
                top_n=max(8, top_k * 4),
            )
            keyword_hits = self._keyword_search(
                analysis.expanded_query,
                category=plan_category,
                top_n=max(8, top_k * 4),
            )
            total_vector_hits += len(vector_hits)
            total_keyword_hits += len(keyword_hits)
            plan_trace.append(
                {
                    "label": plan["label"],
                    "category": plan_category,
                    "vector_hits": len(vector_hits),
                    "keyword_hits": len(keyword_hits),
                }
            )

            for rank, (payload, score) in enumerate(vector_hits, 1):
                row = candidate_map.setdefault(
                    payload["chunk_id"],
                    {
                        "payload": payload,
                        "vector_score": 0.0,
                        "keyword_score": 0.0,
                        "rrf_score": 0.0,
                        "title_bonus": self._title_match_bonus(analysis, payload),
                        "focus_bonus": self._focus_match_bonus(analysis, payload),
                        "category_bonus": self._category_match_bonus(analysis, payload),
                    },
                )
                row["vector_score"] = max(row["vector_score"], score)
                row["rrf_score"] += settings.rag_vector_weight * (1.0 / (settings.rag_fusion_k + rank))
                if plan_category and payload.get("category") == plan_category:
                    row["category_bonus"] = max(row["category_bonus"], plan_bonus)

            for rank, (payload, score) in enumerate(keyword_hits, 1):
                row = candidate_map.setdefault(
                    payload["chunk_id"],
                    {
                        "payload": payload,
                        "vector_score": 0.0,
                        "keyword_score": 0.0,
                        "rrf_score": 0.0,
                        "title_bonus": self._title_match_bonus(analysis, payload),
                        "focus_bonus": self._focus_match_bonus(analysis, payload),
                        "category_bonus": self._category_match_bonus(analysis, payload),
                    },
                )
                row["keyword_score"] = max(row["keyword_score"], score)
                row["rrf_score"] += settings.rag_keyword_weight * (1.0 / (settings.rag_fusion_k + rank))
                if plan_category and payload.get("category") == plan_category:
                    row["category_bonus"] = max(row["category_bonus"], plan_bonus)

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda item: (
                item["rrf_score"]
                + item["vector_score"]
                + item["keyword_score"]
                + item.get("title_bonus", 0.0)
                + item.get("focus_bonus", 0.0)
                + item.get("category_bonus", 0.0),
                item["payload"].get("chunk_id", ""),
            ),
            reverse=True,
        )
        results = self._group_to_parent_results(analysis.expanded_query, ranked_candidates)
        limited = results[: max(1, top_k)]
        self._last_query_trace = {
            "query": normalized_query,
            "expanded_query": analysis.expanded_query if analysis.expanded_query != normalized_query else "",
            "effective_category": category or analysis.inferred_category,
            "inferred_category": analysis.inferred_category or "",
            "matched_intents": analysis.matched_intents,
            "focus_terms": analysis.focus_terms,
            "vector_hits": total_vector_hits,
            "keyword_hits": total_keyword_hits,
            "candidate_count": len(ranked_candidates),
            "results": len(limited),
            "documents_indexed": len(list_knowledge_documents()),
            "search_plans": plan_trace,
        }
        return limited

    def search(self, query: str, *, category: Optional[str] = None, top_k: int = 4) -> List[KnowledgeResult]:
        return self.search_hybrid(query, category=category, top_k=top_k)

    def get_last_query_trace(self) -> Dict[str, Any]:
        return copy.deepcopy(self._last_query_trace)

    def get_stats(self) -> Dict[str, Any]:
        documents = list_knowledge_documents()
        categories = sorted({chunk.get("category", "general") for chunk in self._child_chunks.values()})
        doc_types = sorted({document.get("doc_type", "markdown") for document in documents})
        return {
            "collection_name": self.collection_name,
            "knowledge_base_path": str(self.knowledge_base_path),
            "total_documents": len(documents),
            "total_parent_chunks": len(self._parent_chunks),
            "total_child_chunks": len(self._child_chunks),
            "categories": categories,
            "category_count": len(categories),
            "doc_types": doc_types,
            "embedding_model": self.embedding_model_name,
            "reranker_model": settings.reranker_model if self.enable_reranker else "",
        }


def create_document_store(
    *,
    knowledge_base_path: Optional[Path] = None,
    chroma_path: Optional[Path] = None,
) -> DocumentStore:
    return DocumentStore(
        knowledge_base_path=knowledge_base_path,
        chroma_path=chroma_path,
    )
