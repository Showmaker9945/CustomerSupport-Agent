"""
客服 FAQ 知识库。

提供基于 ChromaDB 的语义检索、BM25 词法检索、结果融合与置信度评分，
用于中文优先的客服问答场景。
"""

import copy
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi

from ..config import settings

logger = logging.getLogger(__name__)

# 抑制旧版 Chroma / PostHog 组合产生的噪声日志。
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)


@dataclass
class FAQResult:
    """FAQ 检索结果。"""
    question: str
    answer: str
    category: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """转为便于展示的字符串。"""
        return (
            f"问题：{self.question}\n"
            f"答案：{self.answer}\n"
            f"分类：{self.category} | 置信度：{self.confidence:.2%}"
        )


class FAQStore:
    """
    客服 FAQ 知识库。

    使用 ChromaDB + sentence-transformers 构建语义检索，
    并结合 BM25 实现混合检索。
    """

    # 面向通用 SaaS 场景的中文样本 FAQ
    SAMPLE_FAQS = [
        {
            "question": "如何重置密码？",
            "answer": "进入“设置 > 安全中心 > 密码”，点击“重置密码”。系统会向你的注册邮箱发送重置链接，链接 24 小时内有效。",
            "category": "account",
            "keywords": ["密码", "重置密码", "登录", "忘记密码", "password", "reset", "login", "forgot"]
        },
        {
            "question": "支持哪些付款方式？",
            "answer": "我们支持主流信用卡、PayPal，以及年付方案的对公转账。企业客户还支持账期发票结算。",
            "category": "billing",
            "keywords": ["支付", "付款", "信用卡", "PayPal", "发票", "payment", "credit card", "paypal", "invoice"]
        },
        {
            "question": "如何取消订阅？",
            "answer": "你可以在“设置 > 账单 > 订阅管理”中随时取消套餐。取消后服务会持续到当前计费周期结束，已使用周期不支持按天退款。",
            "category": "billing",
            "keywords": ["取消订阅", "退订", "退款", "套餐", "cancel", "subscription", "refund", "delete"]
        },
        {
            "question": "可以升级或降级套餐吗？",
            "answer": "可以。升级会立即生效并按比例补差价；降级会在下一个计费周期开始时生效。",
            "category": "billing",
            "keywords": ["升级套餐", "降级套餐", "套餐变更", "plan", "upgrade", "downgrade", "change"]
        },
        {
            "question": "如何邀请团队成员？",
            "answer": "进入“设置 > 团队 > 成员”，点击“邀请成员”，填写邮箱并选择角色（管理员、编辑者或访客），系统会自动发送邀请链接。",
            "category": "workspace",
            "keywords": ["团队", "成员", "邀请", "协作", "team", "member", "invite", "collaborator"]
        },
        {
            "question": "访客、编辑者和管理员有什么区别？",
            "answer": "访客只能查看内容；编辑者可以创建和编辑内容，但不能修改系统设置；管理员拥有包括账单、成员管理和系统配置在内的全部权限。",
            "category": "workspace",
            "keywords": ["角色", "权限", "管理员", "编辑者", "访客", "role", "permission", "admin", "editor", "viewer"]
        },
        {
            "question": "我的数据安全吗？",
            "answer": "是的。我们对静态数据使用 AES-256 加密，对传输链路使用 TLS 1.3，并支持双重身份验证（2FA）。",
            "category": "security",
            "keywords": ["数据安全", "加密", "隐私", "双因子", "security", "encryption", "gdpr", "soc2", "privacy"]
        },
        {
            "question": "是否提供 API？",
            "answer": "提供。我们支持 REST API，并提供完整文档。你可以在“设置 > 开发者”中创建 API Key。Pro 和 Enterprise 套餐可用。",
            "category": "technical",
            "keywords": ["API", "接口", "开发者", "集成", "webhook", "integration", "developer"]
        },
        {
            "question": "如何导出数据？",
            "answer": "进入“设置 > 数据 > 导出”，选择 CSV、JSON 或 PDF 格式即可。大数据量导出完成后，系统会通过邮件通知你。",
            "category": "technical",
            "keywords": ["导出", "下载", "备份", "数据", "export", "download", "backup", "data"]
        },
        {
            "question": "系统支持哪些设备和环境？",
            "answer": "Web 端支持主流现代浏览器；移动端支持 iOS 14+ 与 Android 10+；桌面端支持 macOS 11+ 和 Windows 10+。",
            "category": "technical",
            "keywords": ["系统要求", "浏览器", "兼容环境", "requirements", "browser", "system", "compatible"]
        },
        {
            "question": "如何联系客服？",
            "answer": "你可以通过在线客服、帮助中心提交工单，或发送邮件联系我们。Pro 和 Enterprise 用户享受更高优先级支持。",
            "category": "support",
            "keywords": ["联系客服", "帮助中心", "邮件", "在线客服", "contact", "help", "phone", "email", "chat"]
        },
        {
            "question": "客服工单的响应时效是多久？",
            "answer": "免费版通常 48 小时内响应，Pro 版 24 小时内响应，Enterprise 版 4 小时内响应；紧急问题会进一步加速处理。",
            "category": "support",
            "keywords": ["响应时间", "时效", "SLA", "等待", "response", "time", "sla", "wait"]
        },
        {
            "question": "购买前可以试用吗？",
            "answer": "可以。我们提供 14 天 Pro 套餐免费试用，无需绑定信用卡。试用结束后你可以选择付费或继续使用免费版。",
            "category": "billing",
            "keywords": ["试用", "免费", "体验", "demo", "trial", "free", "test"]
        },
        {
            "question": "教育机构或公益组织有优惠吗？",
            "answer": "有。认证通过的教育机构和公益组织可申请专属折扣，联系销售团队并提交资质材料后即可获得优惠。",
            "category": "billing",
            "keywords": ["优惠", "教育", "公益", "学生", "discount", "nonprofit", "education", "student"]
        },
        {
            "question": "产品多久更新一次？",
            "answer": "我们通常每周发布小更新，每季度发布一次重要功能。绝大多数更新可无感完成，不影响正常使用。",
            "category": "product",
            "keywords": ["更新", "发布", "新功能", "路线图", "update", "release", "feature", "roadmap"]
        },
        {
            "question": "可以集成其他工具吗？",
            "answer": "可以。我们已支持 Slack、Zapier、Google Workspace、Microsoft 365、Salesforce 等常见工具，Enterprise 还支持定制集成。",
            "category": "technical",
            "keywords": ["集成", "连接", "同步", "Slack", "Zapier", "integration", "connect", "sync"]
        },
        {
            "question": "超出存储上限后会怎样？",
            "answer": "当存储使用达到 80% 和 95% 时，系统会提醒你。若超过上限，新文件上传会被暂停，但已有数据仍可访问；你可以升级套餐或购买额外容量。",
            "category": "billing",
            "keywords": ["存储", "容量上限", "配额", "升级", "storage", "limit", "quota", "upgrade"]
        },
        {
            "question": "如何开启双重身份验证？",
            "answer": "进入“设置 > 安全中心 > 双重身份验证”，选择验证码 App 或短信方式，按提示完成绑定即可。",
            "category": "security",
            "keywords": ["双重身份验证", "2FA", "安全验证", "authentication", "two-factor", "security"]
        },
        {
            "question": "可以把账户转移给其他负责人吗？",
            "answer": "可以。Enterprise 套餐支持账户转移。联系客服发起流程后，新旧负责人都需要确认转移。",
            "category": "account",
            "keywords": ["账户转移", "所有权", "负责人", "account", "transfer", "ownership"]
        },
        {
            "question": "服务器部署在哪些地区？",
            "answer": "我们在美国、欧洲和亚洲均设有数据中心。你可以在注册时选择区域，也可以后续在“设置 > 数据 > 区域”中调整。",
            "category": "technical",
            "keywords": ["服务器", "部署地区", "区域", "数据中心", "server", "location", "region", "data center"]
        }
    ]

    CATEGORY_HINTS = {
        "billing": (
            "账单", "扣费", "扣款", "收费", "订阅", "套餐", "续费", "付款", "支付", "发票",
            "退款", "试用", "优惠", "存储", "配额", "billing", "invoice", "plan", "subscription",
        ),
        "account": (
            "密码", "登录", "账号", "账户", "邮箱", "所有权", "转移",
            "password", "reset", "login", "account",
        ),
        "workspace": (
            "团队", "成员", "角色", "权限", "工作区",
            "team", "member", "role", "permission", "workspace",
        ),
        "security": (
            "安全", "加密", "隐私", "双重", "双因素", "认证", "2fa",
            "security", "encryption", "privacy", "authentication",
        ),
        "technical": (
            "api", "接口", "导出", "集成", "开发者", "数据", "服务器", "区域", "webhook",
            "integration", "export", "developer", "server",
        ),
        "support": (
            "客服", "工单", "人工", "联系", "帮助", "时效", "sla",
            "support", "ticket", "response", "human",
        ),
        "product": (
            "更新", "发布", "路线图", "功能", "roadmap", "feature", "release",
        ),
    }

    QUERY_NORMALIZATION_PATTERNS = (
        (r"\bsubscriptions?\b", "订阅"),
        (r"\bplans?\b", "套餐"),
        (r"\bcancel(?:ation)?\b", "取消"),
        (r"\brefunds?\b", "退款"),
        (r"\bbilling\b", "账单"),
        (r"\binvoices?\b", "账单"),
        (r"\btickets?\b", "工单"),
        (r"\bsupport\b", "客服"),
        (r"\bpassword\b", "密码"),
        (r"\breset\b", "重置"),
        (r"\blog(?:\s|-)?in\b", "登录"),
        (r"\bworkspace\b", "工作区"),
        (r"\bteams?\b", "团队"),
        (r"\bmembers?\b", "成员"),
        (r"\bpermissions?\b", "权限"),
    )

    QUERY_PHRASE_PATTERNS = (
        (r"扣款|扣费|收费", "账单扣费"),
        (r"退订|取消续费|停用套餐|关掉自动续费|关闭自动续费", "取消订阅"),
        (r"改套餐|换套餐|套餐变更", "升级或降级套餐"),
        (r"找回密码|忘记密码", "重置密码"),
        (r"人工客服|真人客服", "人工客服"),
    )

    QUERY_REWRITE_PATTERNS = (
        (r"^(怎么|如何)?退(订)?$", "如何取消订阅？"),
        (r"(怎么|如何).*(取消订阅|退订|取消套餐|停用套餐)", "如何取消订阅？"),
        (r"(取消订阅|退订).*(什么时候|何时|多久).*(生效|失效)", "取消订阅后什么时候生效？"),
        (r"取消.*(什么时候|何时|多久).*(生效|失效)", "取消订阅后什么时候生效？"),
        (r"(怎么|如何).*(升级或降级套餐|升级套餐|降级套餐)", "可以升级或降级套餐吗？"),
        (r"(为什么|为何).*(账单扣费|扣费|收费|扣款)", "账单扣费的原因是什么？"),
        (r"(怎么|如何).*(重置密码)", "如何重置密码？"),
        (r"(工单|人工客服).*(多久|多长时间|什么时候|时效).*(回复|响应|处理)?", "客服工单的响应时效是多久？"),
    )
    _EMBEDDING_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
    _RERANKER_MODEL_CACHE: Dict[str, Any] = {}


    def __init__(
        self,
        chroma_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "faq_knowledge_base"
    ):
        """
        初始化 FAQ 知识库。

        Args:
            chroma_path: Path to persist ChromaDB data
            embedding_model: Name of sentence-transformers model
            collection_name: Name of ChromaDB collection
        """
        self.chroma_path = chroma_path or settings.chroma_persist_dir
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.embedding_query_instruction = settings.embedding_query_instruction.strip()
        self.reranker_model_name = settings.reranker_model
        self.collection_name = collection_name
        self._last_query_trace: Dict[str, Any] = {}

        # Initialize embedding model
        try:
            if self.embedding_model_name not in self._EMBEDDING_MODEL_CACHE:
                self._EMBEDDING_MODEL_CACHE[self.embedding_model_name] = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            self.embedding_model = self._EMBEDDING_MODEL_CACHE[self.embedding_model_name]
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support FAQ knowledge base"}
            )

            logger.info(f"ChromaDB initialized at {self.chroma_path}")

            # Hybrid retrieval caches
            self._bm25_index: Optional[BM25Okapi] = None
            self._bm25_docs: List[str] = []
            self._bm25_metadatas: List[Dict[str, Any]] = []
            self._bm25_ids: List[str] = []
            self._bm25_dirty: bool = True

            # Optional reranker
            self.reranker = None
            if settings.enable_reranker:
                try:
                    if self.reranker_model_name not in self._RERANKER_MODEL_CACHE:
                        self._RERANKER_MODEL_CACHE[self.reranker_model_name] = CrossEncoder(
                            self.reranker_model_name,
                            max_length=512,
                        )
                        logger.info(f"Loaded reranker model: {self.reranker_model_name}")
                    self.reranker = self._RERANKER_MODEL_CACHE[self.reranker_model_name]
                except Exception as rerank_error:
                    self.reranker = None
                    logger.warning(f"Reranker unavailable, fallback to fusion-only search: {rerank_error}")

            # 若集合为空，则自动写入中文样本 FAQ
            if self.collection.count() == 0:
                logger.info("Initializing with sample FAQs...")
                self._load_sample_faqs()

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _encode_embedding_text(self, text: str) -> List[float]:
        """Encode one text chunk with normalized embeddings for cosine retrieval."""
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
        )
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def _create_query_embedding(self, text: str) -> List[float]:
        """Create a query embedding with the BGE retrieval instruction."""
        query_text = (text or "").strip()
        if self.embedding_query_instruction and query_text:
            query_text = f"{self.embedding_query_instruction}{query_text}"
        return self._encode_embedding_text(query_text)

    def _create_document_embedding(self, text: str) -> List[float]:
        """Create a document embedding without the query-side instruction prefix."""
        return self._encode_embedding_text(text or "")

    @staticmethod
    def _normalize_rerank_score(score: float) -> float:
        """Map reranker logits into a stable 0-1 range for display and thresholds."""
        clipped_score = max(min(float(score), 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-clipped_score))

    def _load_sample_faqs(self) -> None:
        """将样本 FAQ 写入知识库。"""
        for faq in self.SAMPLE_FAQS:
            self.add_faq(
                question=faq["question"],
                answer=faq["answer"],
                category=faq["category"],
                metadata={"keywords": faq.get("keywords", [])}
            )
        logger.info(f"Loaded {len(self.SAMPLE_FAQS)} sample FAQs")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Build BM25 tokens for mixed Chinese and English text."""
        if not text:
            return []
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_chars = [char for char in text if "一" <= char <= "鿿"]
        return english_tokens + chinese_chars

    def _extract_field_from_doc(self, doc: str, field_name: str) -> str:
        """Extract a structured field from the stored FAQ document."""
        if not doc:
            return ""
        prefix = f"{field_name}:"
        for line in doc.splitlines():
            if line.startswith(prefix):
                return line.replace(prefix, "", 1).strip()
        return ""

    def _normalize_query(self, query: str) -> str:
        """Normalize mixed-language queries before retrieval."""
        normalized = re.sub(r"\s+", " ", (query or "").strip()).lower()
        for pattern, replacement in self.QUERY_NORMALIZATION_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        for pattern, replacement in self.QUERY_PHRASE_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = normalized.replace("?", "？").replace("!", "！")
        normalized = re.sub(r"\s+", " ", normalized).strip(" ，,;；。！？!?\n\t")
        return normalized

    def _infer_query_category(self, query: str) -> Optional[str]:
        """Infer the most likely FAQ category from the normalized query."""
        normalized = self._normalize_query(query)
        if not normalized:
            return None

        scores: Dict[str, int] = {}
        for category, hints in self.CATEGORY_HINTS.items():
            score = 0
            for hint in hints:
                if hint.lower() in normalized:
                    score += max(1, len(hint) // 2)
            if score > 0:
                scores[category] = score

        if not scores:
            return None
        return max(scores.items(), key=lambda item: (item[1], item[0]))[0]

    def _rewrite_query(self, query: str) -> str:
        """Rewrite short or noisy queries into FAQ-friendly phrasing."""
        normalized = self._normalize_query(query)
        if not normalized:
            return ""
        for pattern, rewritten in self.QUERY_REWRITE_PATTERNS:
            if re.search(pattern, normalized):
                return rewritten
        return normalized

    def _split_subqueries(self, query: str) -> List[str]:
        """Split compound questions into at most two focused subqueries."""
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        if re.search(r"(取消订阅|退订|取消).*(什么时候|何时|多久).*(生效|失效)", normalized):
            return [
                "如何取消订阅？",
                "取消订阅后什么时候生效？",
            ]

        parts = [
            part.strip()
            for part in re.split(r"[，,;；。]|(?:以及|还有|并且|同时|然后)", normalized)
            if part and part.strip()
        ]
        if len(parts) <= 1:
            return []

        subqueries: List[str] = []
        for part in parts:
            if len(part) < 3:
                continue
            candidate = self._rewrite_query(part)
            if candidate and candidate not in subqueries:
                subqueries.append(candidate)
            if len(subqueries) >= 2:
                break
        return subqueries

    def _merge_candidate_maps(
        self,
        target: Dict[Tuple[str, str], Dict[str, Any]],
        incoming: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> None:
        """Merge candidate scores from multiple retrieval rounds."""
        for key, item in incoming.items():
            if key not in target:
                target[key] = {
                    "result": copy.deepcopy(item["result"]),
                    "score": float(item.get("score", 0.0)),
                    "queries": list(item.get("queries", [])),
                    "round_types": list(item.get("round_types", [])),
                }
                continue

            target[key]["score"] += float(item.get("score", 0.0))
            target[key]["result"].confidence = max(
                target[key]["result"].confidence,
                item["result"].confidence,
            )
            for source_query in item.get("queries", []):
                if source_query not in target[key]["queries"]:
                    target[key]["queries"].append(source_query)
            for round_type in item.get("round_types", []):
                if round_type not in target[key]["round_types"]:
                    target[key]["round_types"].append(round_type)

    def _run_retrieval_round(
        self,
        query: str,
        category: Optional[str],
        candidate_k: int,
        min_confidence: float,
        round_type: str,
        round_weight: float,
    ) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
        """Run one retrieval round and collect fusion signals."""
        vector_results = self._vector_search(
            query=query,
            category=category,
            top_k=candidate_k,
            min_confidence=min_confidence,
        )
        keyword_results = self._keyword_search(
            query=query,
            top_k=candidate_k,
            category=category,
        )

        vector_weight = settings.rag_vector_weight * round_weight
        keyword_weight = settings.rag_keyword_weight * round_weight
        fusion_k = max(1, settings.rag_fusion_k)

        candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for source_weight, results in (
            (vector_weight, vector_results),
            (keyword_weight, keyword_results),
        ):
            for rank, result in enumerate(results, start=1):
                key = (result.question, result.answer)
                if key not in candidates:
                    candidates[key] = {
                        "result": copy.deepcopy(result),
                        "score": 0.0,
                        "queries": [query],
                        "round_types": [round_type],
                    }
                candidates[key]["score"] += source_weight * (1.0 / (fusion_k + rank))
                candidates[key]["result"].confidence = max(
                    candidates[key]["result"].confidence,
                    result.confidence,
                )

        return candidates, {
            "query": query,
            "category": category,
            "round_type": round_type,
            "vector_hits": len(vector_results),
            "keyword_hits": len(keyword_results),
        }

    def _finalize_candidates(
        self,
        candidates: Dict[Tuple[str, str], Dict[str, Any]],
        reference_query: str,
        candidate_k: int,
        min_confidence: float,
    ) -> List[FAQResult]:
        """Convert accumulated retrieval signals into final ranked results."""
        if not candidates:
            return []

        fusion_k = max(1, settings.rag_fusion_k)
        merged_results: List[FAQResult] = []
        for item in sorted(candidates.values(), key=lambda row: row["score"], reverse=True)[:candidate_k]:
            result = copy.deepcopy(item["result"])
            result.confidence = max(
                result.confidence,
                min(1.0, float(item["score"] * fusion_k)),
            )
            result.metadata = {
                **result.metadata,
                "fusion_score": round(float(item["score"]), 6),
                "fusion_confidence": round(float(result.confidence), 6),
                "retrieval_queries": list(item.get("queries", [])),
                "retrieval_round_types": list(item.get("round_types", [])),
            }
            merged_results.append(result)

        if self.reranker and merged_results:
            try:
                pairs = [[reference_query, f"{result.question}\n{result.answer}"] for result in merged_results]
                rerank_scores = self.reranker.predict(pairs)
                reranked = sorted(
                    zip(merged_results, rerank_scores),
                    key=lambda row: float(row[1]),
                    reverse=True,
                )
                normalized_results: List[FAQResult] = []
                for result, score in reranked:
                    rerank_score = float(score)
                    rerank_confidence = self._normalize_rerank_score(rerank_score)
                    result.metadata = {
                        **result.metadata,
                        "rerank_score": round(rerank_score, 6),
                        "rerank_confidence": round(rerank_confidence, 6),
                    }
                    result.confidence = rerank_confidence
                    normalized_results.append(result)
                merged_results = normalized_results
            except Exception as rerank_error:
                logger.warning(f"Rerank failed, using fusion ranking only: {rerank_error}")

        filtered_results = [result for result in merged_results if result.confidence >= min_confidence]
        filtered_results.sort(key=lambda result: result.confidence, reverse=True)
        return filtered_results

    def get_last_query_trace(self) -> Dict[str, Any]:
        """Return the latest retrieval trace for debugging or demos."""
        return copy.deepcopy(self._last_query_trace)

    def _refresh_keyword_index(self) -> None:
        """构建或刷新 BM25 词法索引。"""
        if not self._bm25_dirty and self._bm25_index is not None:
            return

        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", []) or []
        metadatas = all_data.get("metadatas", []) or []
        ids = all_data.get("ids", []) or []

        tokenized = [self._tokenize_for_bm25(doc or "") for doc in documents]
        tokenized = [tokens for tokens in tokenized if tokens]

        if not documents or not tokenized:
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_metadatas = []
            self._bm25_ids = []
            self._bm25_dirty = False
            return

        # Keep aligned references
        aligned_docs: List[str] = []
        aligned_meta: List[Dict[str, Any]] = []
        aligned_ids: List[str] = []
        aligned_tokens: List[List[str]] = []

        for idx, doc in enumerate(documents):
            tokens = self._tokenize_for_bm25(doc or "")
            if not tokens:
                continue
            aligned_docs.append(doc)
            aligned_meta.append(metadatas[idx] if idx < len(metadatas) else {})
            aligned_ids.append(ids[idx] if idx < len(ids) else f"faq_{idx}")
            aligned_tokens.append(tokens)

        if not aligned_tokens:
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_metadatas = []
            self._bm25_ids = []
            self._bm25_dirty = False
            return

        self._bm25_index = BM25Okapi(aligned_tokens)
        self._bm25_docs = aligned_docs
        self._bm25_metadatas = aligned_meta
        self._bm25_ids = aligned_ids
        self._bm25_dirty = False

    def _keyword_search(
        self,
        query: str,
        top_k: int = 6,
        category: Optional[str] = None
    ) -> List[FAQResult]:
        """Run BM25 keyword retrieval over the FAQ corpus."""
        self._refresh_keyword_index()
        if self._bm25_index is None:
            return []

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        max_score = max(scores) if len(scores) > 0 else 0.0
        if max_score <= 0:
            return []

        results: List[FAQResult] = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break
            metadata = self._bm25_metadatas[idx]
            doc = self._bm25_docs[idx]
            result_category = metadata.get("category", "general")
            if category and result_category != category:
                continue

            question = metadata.get("question", "") or self._extract_field_from_doc(doc, "Question")
            answer = metadata.get("answer", "") or self._extract_field_from_doc(doc, "Answer") or doc
            confidence = float(scores[idx] / max_score)
            results.append(
                FAQResult(
                    question=question or "Unknown question",
                    answer=answer,
                    category=result_category,
                    confidence=max(0.0, min(1.0, confidence)),
                    metadata={k: v for k, v in metadata.items() if k not in {"question", "answer", "category"}},
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: Optional[int] = None,
        min_confidence: float = 0.0
    ) -> List[FAQResult]:
        """Run lightweight hybrid retrieval with normalization, splitting, and rewrite fallback."""
        final_top_k = top_k or settings.rag_top_k
        candidate_k = max(final_top_k * 4, 12)

        normalized_query = self._normalize_query(query)
        rewritten_query = self._rewrite_query(query)
        subqueries = self._split_subqueries(query)
        inferred_category = self._infer_query_category(normalized_query) if not category else None
        effective_category = category or inferred_category

        trace: Dict[str, Any] = {
            "original_query": query,
            "normalized_query": normalized_query,
            "rewritten_query": rewritten_query if rewritten_query != normalized_query else None,
            "requested_category": category,
            "inferred_category": inferred_category,
            "effective_category": effective_category,
            "sub_queries": subqueries,
            "rewrite_used": False,
            "retrieval_rounds": [],
            "final_result_count": 0,
            "top_result": None,
        }

        queries_to_run: List[Tuple[str, str, float]] = []
        if normalized_query:
            queries_to_run.append((normalized_query, "primary", 1.0))
        for index, subquery in enumerate(subqueries, start=1):
            if subquery != normalized_query:
                queries_to_run.append((subquery, f"subquery_{index}", max(0.8, 0.95 - (index * 0.05))))

        candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
        seen_queries = set()
        for round_query, round_type, round_weight in queries_to_run:
            if not round_query or round_query in seen_queries:
                continue
            seen_queries.add(round_query)
            round_candidates, round_trace = self._run_retrieval_round(
                query=round_query,
                category=effective_category,
                candidate_k=candidate_k,
                min_confidence=min_confidence,
                round_type=round_type,
                round_weight=round_weight,
            )
            self._merge_candidate_maps(candidates, round_candidates)
            trace["retrieval_rounds"].append(round_trace)

        merged_results = self._finalize_candidates(
            candidates=candidates,
            reference_query=normalized_query or query,
            candidate_k=candidate_k,
            min_confidence=min_confidence,
        )
        top_result_gate_confidence = (
            float(merged_results[0].metadata.get("fusion_confidence", merged_results[0].confidence))
            if merged_results else 0.0
        )

        should_retry_with_rewrite = (
            rewritten_query
            and rewritten_query not in seen_queries
            and (not merged_results or top_result_gate_confidence < 0.45)
        )
        if should_retry_with_rewrite:
            round_candidates, round_trace = self._run_retrieval_round(
                query=rewritten_query,
                category=effective_category,
                candidate_k=candidate_k,
                min_confidence=min_confidence,
                round_type="rewrite_retry",
                round_weight=0.78,
            )
            self._merge_candidate_maps(candidates, round_candidates)
            trace["retrieval_rounds"].append(round_trace)
            trace["rewrite_used"] = True
            merged_results = self._finalize_candidates(
                candidates=candidates,
                reference_query=normalized_query or query,
                candidate_k=candidate_k,
                min_confidence=min_confidence,
            )

        final_results = merged_results[:final_top_k]
        trace["final_result_count"] = len(final_results)
        if final_results:
            trace["top_result"] = {
                "question": final_results[0].question,
                "category": final_results[0].category,
                "confidence": round(final_results[0].confidence, 4),
            }
        self._last_query_trace = trace
        return final_results

    def add_faq(
        self,
        question: str,
        answer: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        向知识库新增一条 FAQ。

        Args:
            question: FAQ question
            answer: FAQ answer
            category: FAQ category
            metadata: Optional metadata (keywords, priority, etc.)

        Returns:
            FAQ ID
        """
        try:
            # 将问题、答案和关键词一起写入检索文档，增强中英混合召回效果
            keyword_text = ""
            if metadata and metadata.get("keywords"):
                raw_keywords = metadata.get("keywords", [])
                if isinstance(raw_keywords, list):
                    keyword_text = ", ".join(str(item) for item in raw_keywords)
                else:
                    keyword_text = str(raw_keywords)

            combined_text = f"Question: {question}\nAnswer: {answer}"
            if keyword_text:
                combined_text += f"\nKeywords: {keyword_text}"

            # Create embedding
            embedding = self._create_document_embedding(combined_text)

            # Prepare metadata (ChromaDB doesn't allow lists)
            faq_metadata = {
                "category": category,
                "question": question,
                "answer": answer,
            }

            # Add metadata but convert lists to strings
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, list):
                        faq_metadata[key] = ",".join(str(v) for v in value)
                    else:
                        faq_metadata[key] = value

            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[faq_metadata],
                ids=[f"faq_{hash(question + answer)}"]
            )
            self._bm25_dirty = True

            logger.debug(f"Added FAQ: {question[:50]}...")
            return f"faq_{hash(question + answer)}"

        except Exception as e:
            logger.error(f"Failed to add FAQ: {e}")
            raise

    def load_faqs_from_file(self, file_path: str) -> int:
        """
        从文件批量导入 FAQ。

        Args:
            file_path: Path to JSON or CSV file

        Returns:
            Number of FAQs loaded

        Expected JSON format:
        [
            {
                "question": "...",
                "answer": "...",
                "category": "...",
                "metadata": {...}
            },
            ...
        ]
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"FAQ file not found: {file_path}")

        try:
            if path.suffix == ".json":
                with open(path, 'r', encoding='utf-8') as f:
                    faqs = json.load(f)
            elif path.suffix == ".csv":
                import csv
                faqs = []
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        faqs.append({
                            "question": row.get("question", ""),
                            "answer": row.get("answer", ""),
                            "category": row.get("category", "general"),
                            "metadata": {"keywords": row.get("keywords", "").split(",")}
                                if row.get("keywords") else {}
                        })
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Add FAQs
            count = 0
            errors = []
            for faq in faqs:
                try:
                    self.add_faq(
                        question=faq["question"],
                        answer=faq["answer"],
                        category=faq.get("category", "general"),
                        metadata=faq.get("metadata")
                    )
                    count += 1
                except Exception as e:
                    # Log individual failures but continue loading remaining FAQs
                    logger.warning(f"Failed to load FAQ: {faq.get('question', 'unknown')}: {e}")
                    errors.append({"faq": faq.get('question', 'unknown'), "error": str(e)})

            if errors:
                logger.warning(f"Loaded {count}/{len(faqs)} FAQs successfully. {len(errors)} failed.")

            logger.info(f"Loaded {count} FAQs from {file_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to load FAQs from file: {e}")
            raise

    def _vector_search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        min_confidence: float = 0.0
    ) -> List[FAQResult]:
        """Run vector retrieval against the FAQ collection."""
        try:
            query_embedding = self._create_query_embedding(query)
            where_clause = {"category": category} if category else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,
                where=where_clause
            )

            faq_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    confidence = max(0, 1 - (distance / 2))
                    if confidence < min_confidence:
                        continue

                    question = metadata.get("question", "") or self._extract_field_from_doc(doc, "Question")
                    answer = metadata.get("answer", "") or self._extract_field_from_doc(doc, "Answer") or doc
                    faq_results.append(FAQResult(
                        question=question or "Unknown question",
                        answer=answer,
                        category=metadata.get("category", "general"),
                        confidence=confidence,
                        metadata={
                            k: v for k, v in metadata.items()
                            if k not in ["question", "answer", "category"]
                        }
                    ))

            faq_results.sort(key=lambda x: x.confidence, reverse=True)
            return faq_results[:top_k]

        except Exception as e:
            logger.error(f"FAQ search failed: {e}")
            return []

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        min_confidence: float = 0.0
    ) -> List[FAQResult]:
        """
        检索相关 FAQ。

        默认走混合检索，以提升中文场景下的召回与排序稳定性。
        """
        return self.search_hybrid(
            query=query,
            category=category,
            top_k=top_k,
            min_confidence=min_confidence,
        )

    def get_categories(self) -> List[str]:
        """
        Get all unique FAQ categories.

        Returns:
            List of category names
        """
        try:
            # Get all documents to extract categories
            results = self.collection.get(include=["metadatas"])

            categories = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "category" in metadata:
                        categories.add(metadata["category"])

            return sorted(list(categories))

        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        获取 FAQ 知识库统计信息。

        Returns:
            Dictionary with stats
        """
        try:
            total_count = self.collection.count()
            categories = self.get_categories()

            return {
                "total_faqs": total_count,
                "categories": categories,
                "category_count": len(categories),
                "embedding_model": self.embedding_model_name,
                "embedding_query_instruction": self.embedding_query_instruction,
                "reranker_model": self.reranker_model_name,
                "reranker_enabled": settings.enable_reranker,
                "reranker_loaded": self.reranker is not None,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def delete_faq(self, question: str) -> bool:
        """
        Delete a FAQ by question.

        Args:
            question: Question text to match

        Returns:
            True if deleted, False if not found
        """
        try:
            # Search for FAQs with matching question in metadata
            # Get all and filter client-side since ChromaDB's where clause
            # has limited exact match capabilities
            all_results = self.collection.get(include=["metadatas"])

            matching_ids = []
            for i, metadata in enumerate(all_results.get("metadatas", [])):
                if metadata.get("question") == question:
                    matching_ids.append(all_results["ids"][i])

            if matching_ids:
                self.collection.delete(ids=matching_ids)
                self._bm25_dirty = True
                logger.info(f"Deleted {len(matching_ids)} FAQ(s): {question}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete FAQ: {e}")
            return False

    def clear_all(self) -> None:
        """清空知识库中的全部 FAQ。"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Customer support FAQ knowledge base"}
            )
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_metadatas = []
            self._bm25_ids = []
            self._bm25_dirty = True
            self._last_query_trace = {}
            logger.info("Cleared all FAQs")
        except Exception as e:
            logger.error(f"Failed to clear FAQs: {e}")
            raise

    def reindex(self, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Rebuild retrieval indexes for the knowledge base.

        Args:
            clear_existing: 是否先清空再重新加载样本 FAQ。

        Returns:
            Updated statistics.
        """
        if clear_existing:
            self.clear_all()
            self._load_sample_faqs()
        self._bm25_dirty = True
        self._refresh_keyword_index()
        return self.get_stats()


def create_faq_store(
    chroma_path: Optional[Path] = None,
    load_samples: bool = True
) -> FAQStore:
    """
    创建并初始化 FAQStore 的工厂函数。

    Args:
        chroma_path: Optional custom path for ChromaDB
        load_samples: 是否加载样本 FAQ

    Returns:
        Initialized FAQStore instance
    """
    store = FAQStore(chroma_path=chroma_path)

    if not load_samples:
        store.clear_all()

    return store


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    """演示 FAQ 知识库的基础用法。"""
    print("=" * 60)
    print("FAQ 知识库演示")
    print("=" * 60)

    # Create FAQ store
    store = create_faq_store()

    # Get stats
    stats = store.get_stats()
    print(f"\nFAQ 知识库统计：")
    print(f"  FAQ 总数：{stats['total_faqs']}")
    print(f"  分类列表：{', '.join(stats['categories'])}")
    print(f"  Embedding 模型：{stats['embedding_model']}")

    # Search examples
    queries = [
        "重置密码",
        "付款方式",
        "取消订阅",
    ]

    print("\n检索结果：")
    for query in queries:
        print(f"\n  查询：'{query}'")
        results = store.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result.question}")
            print(f"       置信度：{result.confidence:.1%}")

    print("\n" + "=" * 60)
