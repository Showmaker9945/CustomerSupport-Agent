"""Support agent service with modularized graph, middleware and public APIs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langsmith import Client as LangSmithClient, trace, tracing_context
from langsmith.run_helpers import get_current_run_tree

from ...config import settings
from ...db.repositories import (
    append_conversation_message,
    build_recent_context_window,
    clear_pending_conversation_state,
    create_or_touch_conversation_thread,
    delete_user_conversations,
    get_conversation_thread,
    mark_conversation_thread_status,
    list_user_conversation_messages,
    list_thread_messages,
    save_pending_conversation_state,
)
from ...memory import SemanticMemoryStore
from ...sentiment.analyzer import SentimentResult, get_sentiment_analyzer
from ...tools.support_tools import (
    create_ticket,
    escalate_to_human,
    explain_invoice_charge,
    get_tool_by_name,
    get_latest_invoice,
    get_latest_invoice_record,
    get_subscription_status,
    get_ticket_status,
    get_user_tickets,
    lookup_account,
    reindex_knowledge_base,
    search_knowledge_base,
    search_knowledge_base_bundle,
    update_ticket,
)
from .graph import (
    ACCOUNT_HINTS,
    ESCALATION_HINTS,
    EvidenceItem,
    QUESTION_HINTS,
    REQUEST_HINTS,
    TICKET_HINTS,
    AgentRuntimeContext,
    OrchestrationState,
    SupportAgentOrchestrator,
    SupportResponse,
    as_bool,
    append_node_timing,
    build_citations_from_evidence_items,
    build_trace_event,
    build_execution_steps,
    build_neutral_sentiment,
    extract_json_payload,
    infer_intent,
    infer_risk,
    merge_evidence_items,
    merge_unique,
    normalize_intent,
    normalize_risk,
    normalize_sentiment_label,
    normalize_execution_steps,
    safe_float,
    strip_source_annotations,
    ticket_id_from_text,
    is_positive_escalation_text,
)
from .middleware import create_role_agent
from .persistence import LangGraphPersistence

logger = logging.getLogger(__name__)

TOOL_LABELS: Dict[str, str] = {
    "create_ticket": "创建工单",
    "update_ticket": "更新工单",
    "get_ticket_status": "查询工单状态",
    "get_user_tickets": "查询用户工单",
    "lookup_account": "查询账户信息",
    "get_subscription_status": "查询订阅状态",
    "get_latest_invoice": "查询最近账单",
    "explain_invoice_charge": "解释账单扣费",
    "search_knowledge_base": "检索帮助中心知识库",
    "escalate_to_human": "升级人工客服",
    "reindex_knowledge_base": "重建知识库索引",
}

MEMORY_IMPORTANCE: Dict[str, float] = {
    "profile": 0.95,
    "preference": 0.85,
    "open_issue": 1.0,
    "resolved_issue": 0.55,
}


class SupportAgent:
    """
    Multi-agent customer support orchestrator built on LangGraph.

    Main graph:
    analyze -> knowledge/action/escalation/validate
    knowledge -> action/validate
    action/escalation -> validate -> respond
    """

    QUESTION_HINTS = QUESTION_HINTS
    REQUEST_HINTS = REQUEST_HINTS
    ESCALATION_HINTS = ESCALATION_HINTS
    ACCOUNT_HINTS = ACCOUNT_HINTS
    TICKET_HINTS = TICKET_HINTS

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_memory: bool = True,
        enable_sentiment: bool = True,
    ) -> None:
        self.model_name = model_name or settings.llm_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.enable_memory = enable_memory
        self.enable_sentiment = enable_sentiment
        disable_llm = os.getenv("DISABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}
        self.llm_enabled = settings.has_valid_llm_api_key and not disable_llm
        self.persistence = LangGraphPersistence()
        self.structured_memory_store = SemanticMemoryStore() if enable_memory else None
        self.sentiment_analyzer = get_sentiment_analyzer() if enable_sentiment else None

        self._lock = threading.Lock()
        self._thread_user: Dict[str, str] = {}
        self._pending_role: Dict[str, str] = {}
        self._pending_state: Dict[str, OrchestrationState] = {}
        self._trace_by_thread: Dict[str, str] = {}
        self._memory_debug_by_thread: Dict[str, Dict[str, Any]] = {}

        self.basic_model: Optional[ChatOpenAI] = None
        self.advanced_model: Optional[ChatOpenAI] = None
        self.role_agents: Dict[str, Any] = {}
        self.langsmith_client = self._create_langsmith_client()

        if self.llm_enabled:
            self.basic_model = self._create_model(self.model_name, self.temperature)
            self.advanced_model = self._create_model(settings.llm_high_quality_model, 0.2)
            self._build_role_agents()
        else:
            logger.warning("LLM 未可用：将使用规则路由与模板回复。")

        self.orchestrator = SupportAgentOrchestrator(self)
        self.orchestration_graph = self.orchestrator.build(
            checkpointer=self.persistence.checkpointer,
            store=self.persistence.store,
        )

    def close(self) -> None:
        """Release backend resources."""
        self.persistence.close()
        if self.structured_memory_store is not None:
            self.structured_memory_store.close()

    def _create_model(self, model_name: str, temperature: float) -> ChatOpenAI:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "api_key": settings.resolved_llm_api_key,
            "request_timeout": 45.0,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        return ChatOpenAI(**kwargs)

    def _create_langsmith_client(self) -> Optional[LangSmithClient]:
        if not settings.langsmith_enabled:
            return None

        kwargs: Dict[str, Any] = {
            "api_key": settings.langsmith_api_key,
            "api_url": settings.langsmith_endpoint,
        }
        if settings.langsmith_workspace_id:
            kwargs["workspace_id"] = settings.langsmith_workspace_id
        if settings.langsmith_otel_enabled:
            kwargs["otel_enabled"] = True

        try:
            return LangSmithClient(**kwargs)
        except Exception as error:
            logger.warning(f"LangSmith client init failed, tracing disabled: {error}")
            return None

    def _langsmith_disabled_reason(self) -> str:
        if not settings.langsmith_tracing:
            return "LANGSMITH_TRACING is disabled."
        if not settings.has_valid_langsmith_api_key:
            return "LANGSMITH_API_KEY is not configured."
        if self.langsmith_client is None:
            return "LangSmith client initialization failed."
        return ""

    def _default_langsmith_debug(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": bool(settings.langsmith_enabled and self.langsmith_client is not None),
            "project": settings.langsmith_project,
            "endpoint": settings.langsmith_endpoint,
        }
        if settings.langsmith_workspace_id:
            payload["workspace_id"] = settings.langsmith_workspace_id

        reason = self._langsmith_disabled_reason()
        if reason:
            payload["reason"] = reason
        return payload

    def _langsmith_tags(self, *, entrypoint: str, role: Optional[str] = None) -> List[str]:
        tags = [
            "customer-support-agent",
            f"env:{settings.environment}",
            f"entry:{entrypoint}",
        ]
        if role:
            tags.append(f"role:{role}")
        return tags

    def _langsmith_metadata(
        self,
        *,
        user_id: str,
        thread_id: str,
        trace_id: str,
        role: Optional[str] = None,
        intent: Optional[str] = None,
        risk: Optional[str] = None,
        resumed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            "user_id": user_id,
            "thread_id": thread_id,
            "correlation_id": trace_id,
            "langgraph_persistence_backend": self.persistence.backend,
            "llm_enabled": self.llm_enabled,
        }
        if role:
            metadata["role"] = role
        if intent:
            metadata["intent"] = intent
        if risk:
            metadata["risk"] = risk
        if resumed is not None:
            metadata["resumed"] = resumed
        return metadata

    def _build_runnable_config(
        self,
        *,
        run_name: str,
        entrypoint: str,
        user_id: str,
        thread_id: str,
        trace_id: str,
        checkpoint_thread_id: Optional[str] = None,
        role: Optional[str] = None,
        intent: Optional[str] = None,
        risk: Optional[str] = None,
        resumed: Optional[bool] = None,
        extra_tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = self._langsmith_metadata(
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            role=role,
            intent=intent,
            risk=risk,
            resumed=resumed,
        )
        if extra_metadata:
            metadata.update(extra_metadata)

        tags: List[str] = []
        for tag in [*self._langsmith_tags(entrypoint=entrypoint, role=role), *(extra_tags or [])]:
            cleaned = str(tag).strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)

        config: Dict[str, Any] = {
            "run_name": run_name,
            "tags": tags,
            "metadata": metadata,
        }
        if checkpoint_thread_id:
            config["configurable"] = {"thread_id": checkpoint_thread_id}
        return config

    def _langsmith_payload_from_run(self, run_tree: Any) -> Dict[str, Any]:
        payload = self._default_langsmith_debug()
        if run_tree is None or self.langsmith_client is None or not settings.langsmith_enabled:
            return payload

        payload["enabled"] = True
        payload.pop("reason", None)

        run_id = getattr(run_tree, "id", None)
        trace_id = getattr(run_tree, "trace_id", None)
        if run_id is not None:
            payload["run_id"] = str(run_id)
        if trace_id is not None:
            payload["trace_id"] = str(trace_id)

        with suppress(Exception):
            payload["run_url"] = self.langsmith_client.get_run_url(
                run=run_tree,
                project_name=settings.langsmith_project,
            )
        return payload

    @contextmanager
    def _langsmith_trace(
        self,
        *,
        name: str,
        entrypoint: str,
        user_id: str,
        thread_id: str,
        trace_id: str,
        role: Optional[str] = None,
        intent: Optional[str] = None,
        risk: Optional[str] = None,
        resumed: Optional[bool] = None,
        parent: Optional[Any] = None,
    ):
        payload = self._default_langsmith_debug()
        if self.langsmith_client is None or not settings.langsmith_enabled:
            yield payload
            return

        tags = self._langsmith_tags(entrypoint=entrypoint, role=role)
        metadata = self._langsmith_metadata(
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            role=role,
            intent=intent,
            risk=risk,
            resumed=resumed,
        )
        with tracing_context(
            enabled=True,
            project_name=settings.langsmith_project,
            tags=tags,
            metadata=metadata,
            parent=parent,
            client=self.langsmith_client,
        ):
            with trace(
                name=name,
                run_type="chain",
                project_name=settings.langsmith_project,
                tags=tags,
                metadata=metadata,
                parent=parent,
                client=self.langsmith_client,
            ) as run_tree:
                payload.update(self._langsmith_payload_from_run(run_tree))
                try:
                    yield payload
                finally:
                    payload.update(self._langsmith_payload_from_run(run_tree))

    def _langsmith_current_parent_headers(self) -> Optional[Dict[str, str]]:
        if self.langsmith_client is None or not settings.langsmith_enabled:
            return None
        with suppress(Exception):
            run_tree = get_current_run_tree()
            if run_tree is None:
                return None
            return dict(run_tree.to_headers())
        return None

    def _trace_resume_finalize_node(
        self,
        state: OrchestrationState,
        *,
        node: str,
        agent: str,
        handler: Any,
        user_id: str,
        thread_id: str,
        trace_id: str,
        intent: str,
        risk: str,
    ) -> Dict[str, Any]:
        if self.langsmith_client is None or not settings.langsmith_enabled:
            return self.orchestrator._run_timed_node(
                state,
                node=node,
                agent=agent,
                handler=handler,
            )

        tags = self._langsmith_tags(entrypoint="resume-node", role=agent)
        metadata = self._langsmith_metadata(
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            role=agent,
            intent=intent,
            risk=risk,
            resumed=True,
        )
        with trace(
            name=node,
            run_type="chain",
            project_name=settings.langsmith_project,
            tags=tags,
            metadata=metadata,
            client=self.langsmith_client,
        ):
            return self.orchestrator._run_timed_node(
                state,
                node=node,
                agent=agent,
                handler=handler,
            )

    def _finalize_after_execution_with_langsmith(
        self,
        state: OrchestrationState,
        *,
        user_id: str,
        thread_id: str,
        trace_id: str,
    ) -> OrchestrationState:
        merged = dict(state)
        intent = merged.get("intent", "resume")
        risk = merged.get("risk", "medium")
        for node_name, agent_name, handler in (
            ("validate", "validator", self.orchestrator.node_validate),
            ("respond", "responder", self.orchestrator.node_respond),
        ):
            merged.update(
                self._trace_resume_finalize_node(
                    merged,
                    node=node_name,
                    agent=agent_name,
                    handler=handler,
                    user_id=user_id,
                    thread_id=thread_id,
                    trace_id=trace_id,
                    intent=intent,
                    risk=risk,
                )
            )
        return merged

    def _telemetry_memory_namespace(self, user_id: str) -> Tuple[str, str, str]:
        return (settings.long_term_memory_namespace, user_id, "telemetry")

    def _structured_memory_namespace(self, user_id: str) -> Tuple[str, str, str]:
        return (settings.long_term_memory_namespace, user_id, "structured")

    def _reset_memory_debug(self, thread_id: str) -> None:
        if not thread_id:
            return
        if not settings.debug:
            self._memory_debug_by_thread.pop(thread_id, None)
            return
        self._memory_debug_by_thread[thread_id] = {
            "reads": [],
            "writes": [],
            "skips": [],
        }

    def _memory_debug_snapshot(self, thread_id: Optional[str]) -> Dict[str, Any]:
        if not thread_id or not settings.debug:
            return {}
        snapshot = self._memory_debug_by_thread.get(thread_id, {})
        return {
            "reads": list(snapshot.get("reads", [])),
            "writes": list(snapshot.get("writes", [])),
            "skips": list(snapshot.get("skips", [])),
        }

    def _memory_debug_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        content = (
            payload.get("content")
            or payload.get("summary")
            or payload.get("fact")
            or payload.get("message")
            or payload.get("response")
            or ""
        )
        return {
            "memory_id": payload.get("memory_id"),
            "memory_type": payload.get("memory_type", payload.get("kind", "memory")),
            "status": payload.get("status"),
            "content": content[:120],
            "importance": payload.get("importance"),
        }

    def _record_memory_skip(self, thread_id: Optional[str], reason: str, content: str = "") -> None:
        if not thread_id or thread_id not in self._memory_debug_by_thread:
            return
        self._memory_debug_by_thread[thread_id]["skips"].append(
            {
                "reason": reason,
                "content": content[:120],
            }
        )

    def _record_memory_write(self, thread_id: Optional[str], action: str, payload: Dict[str, Any]) -> None:
        if not thread_id or thread_id not in self._memory_debug_by_thread:
            return
        preview = self._memory_debug_entry(payload)
        preview["action"] = action
        self._memory_debug_by_thread[thread_id]["writes"].append(preview)

    def _record_memory_hit(
        self,
        thread_id: Optional[str],
        query: str,
        payloads: List[Dict[str, Any]],
    ) -> None:
        if not thread_id or thread_id not in self._memory_debug_by_thread or not payloads:
            return
        self._memory_debug_by_thread[thread_id]["reads"].append(
            {
                "query": query[:120],
                "items": [self._memory_debug_entry(payload) for payload in payloads[:4]],
            }
        )

    def _save_memory_item(self, user_id: str, payload: Dict[str, Any]) -> None:
        if not self.enable_memory or self.persistence.store is None:
            return
        digest = hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.persistence.store.put(self._telemetry_memory_namespace(user_id), digest, payload)

    def _list_structured_memory(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.enable_memory or self.structured_memory_store is None:
            return []
        try:
            return self.structured_memory_store.list_memories(
                user_id=user_id,
                limit=limit or settings.max_memory_items_per_user,
            )
        except Exception as error:
            logger.warning(f"Memory list failed: {error}")
            return []

    def _upsert_structured_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        payload: Dict[str, Any],
        thread_id: Optional[str] = None,
    ) -> None:
        if not self.enable_memory or self.structured_memory_store is None:
            return
        clean_payload = dict(payload)
        clean_payload["memory_id"] = memory_id
        clean_payload.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        try:
            stored = self.structured_memory_store.upsert_memory(
                user_id=user_id,
                memory_id=memory_id,
                payload=clean_payload,
            )
            self._record_memory_write(thread_id, "upsert", stored)
        except Exception as error:
            logger.warning(f"Memory upsert failed: {error}")

    def _delete_structured_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        thread_id: Optional[str] = None,
        reason: str = "delete",
    ) -> None:
        if not self.enable_memory or self.structured_memory_store is None:
            return
        with suppress(Exception):
            self.structured_memory_store.delete_memory(user_id=user_id, memory_id=memory_id)
        self._record_memory_write(
            thread_id,
            reason,
            {"memory_id": memory_id, "memory_type": "open_issue", "content": memory_id},
        )

    def _load_structured_memory_item(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        if not self.enable_memory or self.structured_memory_store is None:
            return None
        try:
            return self.structured_memory_store.get_memory(user_id=user_id, memory_id=memory_id)
        except Exception as error:
            logger.warning(f"Memory get failed: {error}")
            return None

    def _memory_tokens(self, text: str) -> List[str]:
        lowered = str(text or "").lower()
        english_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{1,4}", str(text or ""))
        tokens: List[str] = []
        for token in english_tokens + chinese_tokens:
            cleaned = token.strip()
            if cleaned and cleaned not in tokens:
                tokens.append(cleaned)
        return tokens

    def _memory_text(self, payload: Dict[str, Any]) -> str:
        parts = [
            payload.get("content", ""),
            payload.get("summary", ""),
            payload.get("value", ""),
            payload.get("category", ""),
            payload.get("issue_code", ""),
            " ".join(payload.get("tags", []) or []),
        ]
        return " ".join(str(part) for part in parts if part).lower()

    def _memory_score(self, payload: Dict[str, Any], query: str) -> float:
        importance = MEMORY_IMPORTANCE.get(payload.get("memory_type", ""), 0.3)
        score = importance
        query_tokens = self._memory_tokens(query)
        memory_text = self._memory_text(payload)
        overlap = 0
        for token in query_tokens:
            if token and token.lower() in memory_text:
                overlap += 1
        if overlap:
            score += overlap * 0.22
        elif payload.get("memory_type") in {"profile", "preference"}:
            score += 0.05
        else:
            score -= 0.15

        if payload.get("memory_type") == "open_issue":
            score += 0.08
        return score

    def _search_memory(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enable_memory or self.structured_memory_store is None:
            return []
        items = self._list_structured_memory(user_id, limit=settings.max_memory_items_per_user)
        if not items:
            self._record_memory_skip(thread_id, "no_memory_available", query)
            return []
        try:
            filtered = self.structured_memory_store.search_memories(
                user_id=user_id,
                query=query,
                limit=limit,
            )
        except Exception as error:
            logger.warning(f"Memory search failed: {error}")
            filtered = []
        if not filtered:
            self._record_memory_skip(thread_id, "no_relevant_memory", query)
            return []
        self._record_memory_hit(thread_id, query, filtered)
        return filtered

    def _build_role_agents(self) -> None:
        self.role_agents = {
            "supervisor": create_role_agent(owner=self, role="supervisor", tools=[], enable_hitl=False),
            "knowledge": create_role_agent(
                owner=self,
                role="knowledge",
                tools=[search_knowledge_base],
                enable_hitl=False,
            ),
            "action": create_role_agent(
                owner=self,
                role="action",
                tools=[
                    search_knowledge_base,
                    create_ticket,
                    update_ticket,
                    get_ticket_status,
                    get_user_tickets,
                    lookup_account,
                    get_subscription_status,
                    get_latest_invoice,
                    explain_invoice_charge,
                    escalate_to_human,
                ],
                enable_hitl=True,
            ),
            "escalation": create_role_agent(
                owner=self,
                role="escalation",
                tools=[escalate_to_human, create_ticket, get_ticket_status],
                enable_hitl=True,
            ),
            "responder": create_role_agent(owner=self, role="responder", tools=[], enable_hitl=False),
        }

    def _infer_intent(self, message: str) -> str:
        return infer_intent(message)

    def _infer_risk(self, message: str, sentiment: Optional[SentimentResult]) -> str:
        return infer_risk(message, sentiment)

    def _normalize_intent(self, value: Any) -> str:
        return normalize_intent(value)

    def _normalize_risk(self, value: Any) -> str:
        return normalize_risk(value)

    def _normalize_sentiment_label(self, value: Any) -> str:
        return normalize_sentiment_label(value)

    def _as_bool(self, value: Any) -> bool:
        return as_bool(value)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        return extract_json_payload(text)

    def _contains_any(self, message: str, tokens: Tuple[str, ...]) -> bool:
        lowered = message.lower()
        return any(token in lowered for token in tokens)

    def _is_subscription_status_query(self, message: str) -> bool:
        subscription_tokens = ("套餐", "订阅", "续费", "plan", "subscription", "renewal")
        status_tokens = ("当前", "什么", "哪个", "哪种", "下次", "自动续费", "状态", "费用", "席位")
        lowered = message.lower()
        return any(token in lowered for token in subscription_tokens) and any(
            token in lowered for token in status_tokens
        )

    def _is_subscription_policy_query(self, message: str) -> bool:
        subscription_tokens = ("套餐", "订阅", "续费", "plan", "subscription")
        policy_tokens = ("取消", "退订", "关闭自动续费", "升级", "降级", "变更", "切换", "取消套餐")
        lowered = message.lower()
        return any(token in lowered for token in subscription_tokens) and any(
            token in lowered for token in policy_tokens
        )

    def _is_billing_ticket_request(self, message: str) -> bool:
        lowered = message.lower()
        billing_tokens = ("账单", "发票", "扣费", "扣款", "invoice", "billing", "charge")
        if not any(token in lowered for token in billing_tokens):
            return False
        if self._is_billing_ticket_process_query(message):
            return False
        has_ticket_context = any(token in lowered for token in ("工单", "ticket", "异常"))
        direct_request_tokens = (
            "帮我创建",
            "请创建",
            "给我创建",
            "帮我新建",
            "请帮我新建",
            "帮我提交",
            "请帮我提交",
            "帮我开",
            "开一个",
            "新建一个",
            "提交一个",
            "发起一个",
        )
        has_direct_request = any(token in lowered for token in direct_request_tokens)
        has_request_intent = any(token in lowered for token in ("帮我", "请", "现在", "立刻", "马上"))
        has_create_verb = any(token in lowered for token in ("创建", "新建", "提交", "发起", "申请", "开"))
        return has_ticket_context and (has_direct_request or (has_request_intent and has_create_verb))

    def _is_invoice_explanation_query(self, message: str) -> bool:
        lowered = message.lower()
        billing_tokens = ("账单", "发票", "扣费", "扣款", "金额", "invoice", "billing", "charge")
        explain_tokens = ("为什么", "说明", "解释", "扣了", "构成", "why", "explain")
        return any(token in lowered for token in billing_tokens) and any(
            token in lowered for token in explain_tokens
        )

    def _is_invoice_lookup_query(self, message: str) -> bool:
        lowered = message.lower()
        billing_tokens = ("账单", "发票", "invoice", "billing")
        lookup_tokens = ("查询", "最近", "最新", "查看", "给我", "看看", "多少", "金额", "记录")
        return any(token in lowered for token in billing_tokens) and any(
            token in lowered for token in lookup_tokens
        )

    def _is_billing_ticket_process_query(self, message: str) -> bool:
        lowered = message.lower()
        billing_tokens = ("账单", "发票", "扣费", "扣款", "invoice", "billing", "charge")
        if not any(token in lowered for token in billing_tokens):
            return False
        if not any(token in lowered for token in ("工单", "ticket")):
            return False
        process_tokens = (
            "创建后",
            "提交后",
            "通过后",
            "之后",
            "流程",
            "说明",
            "会发生什么",
            "会怎么样",
            "多久",
            "审核",
            "审批",
            "状态",
            "进度",
            "需要多久",
        )
        question_tokens = ("什么", "如何", "怎么", "为什么", "吗", "？", "?")
        return any(token in lowered for token in process_tokens) or any(
            token in message for token in question_tokens
        )

    def _run_knowledge_lookup_bundle(self, message: str) -> Dict[str, Any]:
        try:
            bundle = search_knowledge_base_bundle(query=message, category=None)
        except Exception as error:
            logger.error(f"Structured knowledge lookup failed: {error}")
            return {
                "text": f"知识检索失败：{error}",
                "evidence_items": [],
                "trace": {},
                "result_count": 0,
            }

        return {
            "text": strip_source_annotations(str(bundle.get("text", "") or "")),
            "evidence_items": list(bundle.get("evidence_items", [])),
            "trace": dict(bundle.get("trace", {}) or {}),
            "result_count": int(bundle.get("result_count", 0) or 0),
        }

    def _tool_source_label(self, tool_name: str) -> str:
        return f"Support Tools::{self._tool_label(tool_name)}"

    def _build_tool_evidence_item(
        self,
        *,
        tool_name: str,
        text: str,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Optional[EvidenceItem]:
        snippet = strip_source_annotations(str(text or "")).strip()
        if not snippet:
            return None
        return {
            "evidence_id": f"tool:{tool_name}:{tool_call_id or hashlib.md5(snippet.encode('utf-8')).hexdigest()[:12]}",
            "kind": "tool",
            "source_type": "support_tool",
            "source_label": self._tool_source_label(tool_name),
            "snippet": snippet[:220],
            "tool_name": tool_name,
            "tool_label": self._tool_label(tool_name),
            "metadata": {
                "tool_call_id": tool_call_id,
                "tool_args": dict(tool_args or {}),
            },
        }

    def _tool_bundle(
        self,
        *,
        tool_name: str,
        text: str,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_source: str = "Support Tools",
        ticket_id: Optional[str] = None,
        escalated: Optional[bool] = None,
    ) -> Dict[str, Any]:
        evidence_item = self._build_tool_evidence_item(
            tool_name=tool_name,
            text=text,
            tool_args=tool_args,
        )
        bundle: Dict[str, Any] = {
            "text": strip_source_annotations(text),
            "evidence_items": [evidence_item] if evidence_item else [],
            "tool_source": tool_source,
        }
        if ticket_id is not None:
            bundle["ticket_id"] = ticket_id
        if escalated is not None:
            bundle["escalated"] = escalated
        return bundle

    def _extract_tool_evidence_items(self, result: Dict[str, Any]) -> List[EvidenceItem]:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        evidence_items: List[EvidenceItem] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_name = str(getattr(message, "name", "") or "unknown_tool")
            evidence_item = self._build_tool_evidence_item(
                tool_name=tool_name,
                text=str(message.content or ""),
                tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
            )
            if evidence_item:
                evidence_items.append(evidence_item)
        return self._merge_evidence_items([], evidence_items)

    def _run_structured_business_action(
        self,
        *,
        user_id: str,
        message: str,
    ) -> Optional[Dict[str, Any]]:
        if self._is_subscription_status_query(message):
            text = get_subscription_status.invoke({"user_id": user_id})
            return self._tool_bundle(
                tool_name="get_subscription_status",
                text=text,
                tool_args={"user_id": user_id},
            )

        if self._is_billing_ticket_request(message):
            # 写操作必须走带 HITL 的 action agent，避免绕过审批中间件。
            if self.llm_enabled:
                return None
            tool_args = {
                "user_id": user_id,
                "subject": "账单异常核查",
                "description": message,
                "priority": "high",
                "category": "billing",
            }
            text = create_ticket.invoke(tool_args)
            return self._tool_bundle(
                tool_name="create_ticket",
                text=text,
                tool_args=tool_args,
                ticket_id=self._tool_result_ticket_id("create_ticket", text),
            )

        if self._is_invoice_explanation_query(message):
            latest_invoice = get_latest_invoice_record(user_id)
            if latest_invoice:
                tool_args = {"invoice_id": latest_invoice["invoice_id"]}
                text = explain_invoice_charge.invoke(tool_args)
                return self._tool_bundle(
                    tool_name="explain_invoice_charge",
                    text=text,
                    tool_args=tool_args,
                )
            tool_args = {"user_id": user_id}
            text = get_latest_invoice.invoke(tool_args)
            return self._tool_bundle(
                tool_name="get_latest_invoice",
                text=text,
                tool_args=tool_args,
            )

        if self._is_invoice_lookup_query(message):
            tool_args = {"user_id": user_id}
            text = get_latest_invoice.invoke(tool_args)
            return self._tool_bundle(
                tool_name="get_latest_invoice",
                text=text,
                tool_args=tool_args,
            )

        return None

    def _issue_code_from_text(self, text: str) -> Optional[str]:
        lowered = text.lower()
        issue_patterns = (
            ("billing_anomaly", ("账单", "发票", "扣费", "扣款", "billing", "invoice", "charge")),
            ("subscription_issue", ("套餐", "订阅", "续费", "subscription", "plan", "renewal")),
            ("login_issue", ("登录", "登陆", "login", "signin", "sign in")),
            ("password_issue", ("密码", "password", "reset")),
            ("ticket_followup", ("工单", "ticket", "催单", "进度", "状态")),
            ("technical_error", ("报错", "错误", "故障", "异常", "接口", "api", "error")),
            ("account_issue", ("账户", "账号", "account", "member")),
        )
        for issue_code, tokens in issue_patterns:
            if any(token in lowered for token in tokens):
                return issue_code
        return None

    def _issue_content(self, issue_code: str, text: str) -> str:
        content_map = {
            "billing_anomaly": "用户存在账单/扣费问题，仍需继续跟进。",
            "subscription_issue": "用户关注套餐、订阅或续费问题，仍需继续跟进。",
            "login_issue": "用户存在登录问题，仍需继续跟进。",
            "password_issue": "用户存在密码或重置密码问题，仍需继续跟进。",
            "ticket_followup": "用户正在跟进工单进度或状态。",
            "technical_error": "用户反馈产品报错或技术异常，仍需继续跟进。",
            "account_issue": "用户存在账户相关问题，仍需继续跟进。",
        }
        default = "用户存在待跟进问题。"
        snippet = text.strip().replace("\n", " ")
        snippet = snippet[:90]
        return f"{content_map.get(issue_code, default)} 用户原话：{snippet}"

    def _extract_profile_memories(self, user_message: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for pattern in (r"我叫([^\s，。,.!?？!]{1,12})", r"我的名字是([^\s，。,.!?？!]{1,12})"):
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                records.append(
                    {
                        "memory_id": "profile:name",
                        "memory_type": "profile",
                        "field": "name",
                        "value": name,
                        "content": f"用户姓名是 {name}",
                        "importance": MEMORY_IMPORTANCE["profile"],
                        "status": "active",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                break
        return records

    def _extract_preference_memories(self, user_message: str) -> List[Dict[str, Any]]:
        lowered = user_message.lower()
        records: List[Dict[str, Any]] = []
        if any(token in lowered for token in ("中文", "汉语", "chinese")):
            records.append(
                {
                    "memory_id": "preference:language",
                    "memory_type": "preference",
                    "field": "language",
                    "value": "zh-CN",
                    "content": "用户偏好中文沟通。",
                    "importance": MEMORY_IMPORTANCE["preference"],
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        elif any(token in lowered for token in ("英文", "英语", "english")):
            records.append(
                {
                    "memory_id": "preference:language",
                    "memory_type": "preference",
                    "field": "language",
                    "value": "en-US",
                    "content": "用户偏好英文沟通。",
                    "importance": MEMORY_IMPORTANCE["preference"],
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        if any(token in lowered for token in ("邮件通知", "邮箱通知", "email")):
            records.append(
                {
                    "memory_id": "preference:contact_channel",
                    "memory_type": "preference",
                    "field": "contact_channel",
                    "value": "email",
                    "content": "用户偏好通过邮件接收通知。",
                    "importance": MEMORY_IMPORTANCE["preference"],
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        elif any(token in lowered for token in ("电话联系", "电话通知", "phone")):
            records.append(
                {
                    "memory_id": "preference:contact_channel",
                    "memory_type": "preference",
                    "field": "contact_channel",
                    "value": "phone",
                    "content": "用户偏好电话联系。",
                    "importance": MEMORY_IMPORTANCE["preference"],
                    "status": "active",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return records

    def _should_track_open_issue(
        self,
        *,
        intent: str,
        active_agent: str,
        sentiment: Optional[SentimentResult],
        user_message: str,
        assistant_message: str,
    ) -> bool:
        lowered_user = user_message.lower()
        lowered_assistant = assistant_message.lower()
        unresolved_tokens = ("没解决", "还没", "异常", "问题", "报错", "投诉", "失败", "无法", "不行")
        if active_agent in {"action", "escalation"}:
            return True
        if intent in {"complaint", "request"} and any(token in lowered_user for token in unresolved_tokens):
            return True
        return bool(sentiment and sentiment.frustration_score >= 0.45) or any(
            token in lowered_assistant for token in ("创建工单", "升级到人工")
        )

    def _is_resolution_message(self, text: str) -> bool:
        lowered = text.lower()
        resolution_tokens = (
            "已经解决",
            "解决了",
            "可以了",
            "没事了",
            "搞定了",
            "resolved",
            "fixed",
            "works now",
        )
        return any(token in lowered for token in resolution_tokens)

    def _latest_open_issue(self, user_id: str) -> Optional[Dict[str, Any]]:
        open_items = [
            item
            for item in self._list_structured_memory(user_id, limit=settings.max_memory_items_per_user)
            if item.get("memory_type") == "open_issue"
        ]
        if not open_items:
            return None
        return sorted(open_items, key=lambda item: item.get("updated_at", ""), reverse=True)[0]

    def _write_structured_memories(
        self,
        *,
        user_id: str,
        thread_id: str,
        intent: str,
        active_agent: str,
        user_message: str,
        assistant_message: str,
        sentiment: Optional[SentimentResult],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        for record in self._extract_profile_memories(user_message):
            self._upsert_structured_memory(
                user_id=user_id,
                memory_id=record["memory_id"],
                payload=record,
                thread_id=thread_id,
            )

        for record in self._extract_preference_memories(user_message):
            self._upsert_structured_memory(
                user_id=user_id,
                memory_id=record["memory_id"],
                payload=record,
                thread_id=thread_id,
            )

        if self._is_resolution_message(user_message) or self._is_resolution_message(assistant_message):
            issue_code = self._issue_code_from_text(user_message) or self._issue_code_from_text(assistant_message)
            open_issue = (
                self._load_structured_memory_item(user_id, f"open_issue:{issue_code}")
                if issue_code
                else self._latest_open_issue(user_id)
            )
            if open_issue is not None:
                open_issue_id = str(open_issue.get("memory_id"))
                self._delete_structured_memory(
                    user_id=user_id,
                    memory_id=open_issue_id,
                    thread_id=thread_id,
                    reason="resolve_open_issue",
                )
                resolved_issue_code = issue_code or str(open_issue.get("issue_code", "general_issue"))
                resolved_record = {
                    "memory_type": "resolved_issue",
                    "issue_code": resolved_issue_code,
                    "content": f"问题已解决：{open_issue.get('content', '')}",
                    "summary": open_issue.get("summary", ""),
                    "importance": MEMORY_IMPORTANCE["resolved_issue"],
                    "status": "resolved",
                    "updated_at": now,
                }
                self._upsert_structured_memory(
                    user_id=user_id,
                    memory_id=f"resolved_issue:{resolved_issue_code}",
                    payload=resolved_record,
                    thread_id=thread_id,
                )
            else:
                self._record_memory_skip(thread_id, "resolution_without_open_issue", user_message)
            return

        issue_code = self._issue_code_from_text(user_message)
        if issue_code and self._should_track_open_issue(
            intent=intent,
            active_agent=active_agent,
            sentiment=sentiment,
            user_message=user_message,
            assistant_message=assistant_message,
        ):
            open_issue_record = {
                "memory_type": "open_issue",
                "issue_code": issue_code,
                "category": issue_code.split("_", 1)[0],
                "content": self._issue_content(issue_code, user_message),
                "summary": user_message[:120],
                "importance": MEMORY_IMPORTANCE["open_issue"],
                "status": "open",
                "updated_at": now,
            }
            self._upsert_structured_memory(
                user_id=user_id,
                memory_id=f"open_issue:{issue_code}",
                payload=open_issue_record,
                thread_id=thread_id,
            )
        else:
            self._record_memory_skip(thread_id, "no_structured_issue_written", user_message)

    def _analyze_and_route(
        self,
        message: str,
        baseline_intent: str,
        baseline_risk: str,
        baseline_sentiment: SentimentResult,
    ) -> Dict[str, Any]:
        billing_ticket_process_query = self._is_billing_ticket_process_query(message)
        needs_knowledge = baseline_intent in {"question", "complaint"}
        needs_action = baseline_intent == "request" or any(
            token in message.lower() for token in self.TICKET_HINTS + self.ACCOUNT_HINTS
        ) and not billing_ticket_process_query
        needs_escalation = baseline_risk == "high"

        intent = baseline_intent
        risk = baseline_risk
        sentiment_label = baseline_sentiment.label
        frustration_score = baseline_sentiment.frustration_score
        selected_agent = "supervisor"
        execution_steps: List[str] = []
        route_reason = "heuristic"
        needs_action_after_knowledge = False

        if self.llm_enabled and self.basic_model is not None:
            system_prompt = (
                "你是客服路由决策器。"
                "请严格返回 JSON，不要输出任何额外解释。"
                "字段："
                "intent(question|complaint|request|feedback|greeting|other),"
                "risk(low|medium|high),"
                "sentiment_label(positive|neutral|negative),"
                "frustration_score(0到1),"
                "needs_knowledge(boolean),"
                "needs_action(boolean),"
                "needs_escalation(boolean),"
                "selected_agent(supervisor|knowledge|action|escalation),"
                "execution_steps(数组，元素只能是 knowledge|action|escalation),"
                "needs_action_after_knowledge(boolean),"
                "reason(字符串)。"
                "要求：如果只是寒暄/感谢，可选 supervisor 且 execution_steps 为空。"
                "如果高风险或明确要求人工，selected_agent 必须为 escalation。"
            )
            user_prompt = (
                f"用户消息：{message}\n"
                f"基线意图：{baseline_intent}\n"
                f"基线风险：{baseline_risk}\n"
                f"基线情绪：{baseline_sentiment.label}\n"
                f"基线挫败分：{baseline_sentiment.frustration_score:.2f}"
            )

            try:
                response = self.basic_model.invoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                payload = self._extract_json_payload(str(response.content))
                if payload:
                    intent = self._normalize_intent(payload.get("intent"))
                    risk = self._normalize_risk(payload.get("risk"))
                    sentiment_label = self._normalize_sentiment_label(payload.get("sentiment_label"))
                    frustration_score = self._safe_float(
                        payload.get("frustration_score"),
                        default=baseline_sentiment.frustration_score,
                    )
                    needs_knowledge = self._as_bool(payload.get("needs_knowledge")) or needs_knowledge
                    needs_action = self._as_bool(payload.get("needs_action")) or needs_action
                    needs_escalation = self._as_bool(payload.get("needs_escalation")) or (risk == "high")
                    selected_agent = str(payload.get("selected_agent", "")).strip().lower()
                    if selected_agent not in {"supervisor", "knowledge", "action", "escalation"}:
                        selected_agent = "supervisor"
                    execution_steps = normalize_execution_steps(payload.get("execution_steps"))
                    needs_action_after_knowledge = self._as_bool(
                        payload.get("needs_action_after_knowledge")
                    )
                    route_reason = str(payload.get("reason", "")).strip() or "llm"
            except Exception as error:
                logger.warning(f"LLM analyze+route failed, fallback to heuristic routing: {error}")

        if self._is_subscription_status_query(message):
            needs_knowledge = False
            needs_action = True
            selected_agent = "action"
            execution_steps = []
            needs_action_after_knowledge = False
            route_reason = f"{route_reason}+subscription_status"

        if self._is_invoice_explanation_query(message) or self._is_invoice_lookup_query(message):
            needs_knowledge = False
            needs_action = True
            selected_agent = "action"
            execution_steps = []
            needs_action_after_knowledge = False
            route_reason = f"{route_reason}+billing_action"

        if billing_ticket_process_query:
            intent = "question"
            needs_knowledge = True
            needs_action = False
            selected_agent = "knowledge"
            execution_steps = []
            needs_action_after_knowledge = False
            route_reason = f"{route_reason}+billing_ticket_process"

        if self._is_billing_ticket_request(message):
            intent = "request"
            needs_knowledge = False
            needs_action = True
            selected_agent = "action"
            execution_steps = []
            needs_action_after_knowledge = False
            route_reason = f"{route_reason}+billing_ticket"

        if self._is_subscription_policy_query(message):
            needs_knowledge = True
            needs_action = False
            selected_agent = "knowledge"
            execution_steps = []
            needs_action_after_knowledge = False
            route_reason = f"{route_reason}+subscription_policy"

        if intent == "request" and "怎么" in message:
            needs_knowledge = True

        if risk == "high":
            needs_escalation = True

        if not execution_steps:
            selected_agent, needs_action_after_knowledge = self._plan_route(
                intent=intent,
                risk=risk,
                needs_knowledge=needs_knowledge,
                needs_action=needs_action,
                needs_escalation=needs_escalation,
            )
            execution_steps = build_execution_steps(
                selected_agent=selected_agent,
                needs_knowledge=needs_knowledge,
                needs_action=needs_action,
                needs_action_after_knowledge=needs_action_after_knowledge,
                needs_escalation=needs_escalation,
            )
        elif selected_agent == "supervisor":
            selected_agent, fallback_action_after_knowledge = self._plan_route(
                intent=intent,
                risk=risk,
                needs_knowledge=needs_knowledge,
                needs_action=needs_action,
                needs_escalation=needs_escalation,
            )
            if not execution_steps:
                needs_action_after_knowledge = fallback_action_after_knowledge

        if selected_agent == "escalation":
            execution_steps = ["escalation"]
        elif selected_agent == "supervisor" and execution_steps:
            selected_agent = execution_steps[0]

        step_summary = " -> ".join(execution_steps) if execution_steps else "direct_respond"
        decision_summary = (
            f"意图={intent}，风险={risk}，首选代理={selected_agent}，"
            f"执行路径={step_summary}，原因={route_reason or 'heuristic'}。"
        )

        return {
            "intent": intent,
            "risk": risk,
            "sentiment_label": sentiment_label,
            "frustration_score": frustration_score,
            "needs_knowledge": needs_knowledge,
            "needs_action": needs_action,
            "needs_escalation": needs_escalation,
            "selected_agent": selected_agent,
            "execution_steps": execution_steps,
            "needs_action_after_knowledge": needs_action_after_knowledge,
            "route_reason": route_reason,
            "decision_summary": decision_summary,
        }
    def _plan_route(
        self,
        intent: str,
        risk: str,
        needs_knowledge: bool,
        needs_action: bool,
        needs_escalation: bool,
    ) -> Tuple[str, bool]:
        if risk == "high":
            needs_escalation = True
        if needs_escalation:
            return "escalation", False
        if needs_knowledge:
            return "knowledge", needs_action
        if needs_action:
            return "action", False
        if intent in {"question", "complaint"}:
            return "knowledge", False
        if intent == "request":
            return "action", False
        return "supervisor", False

    def _merge_unique(self, left: List[str], right: List[str]) -> List[str]:
        return merge_unique(left, right)

    def _merge_evidence_items(
        self,
        left: List[EvidenceItem],
        right: List[EvidenceItem],
    ) -> List[EvidenceItem]:
        return merge_evidence_items(left, right)

    def _citations_from_state(self, state: OrchestrationState) -> List[str]:
        return build_citations_from_evidence_items(state.get("evidence_items", []))

    def _ticket_id_from_text(self, text: str) -> Optional[str]:
        return ticket_id_from_text(text)

    def _is_positive_escalation_text(self, text: str) -> bool:
        return is_positive_escalation_text(text)

    def _agent_thread_id(self, thread_id: str, role: str) -> str:
        return f"{settings.langgraph_thread_prefix}:{role}:{thread_id}"

    def _graph_thread_id(self, thread_id: str) -> str:
        return f"{settings.langgraph_thread_prefix}:graph:{thread_id}"

    def _tool_label(self, tool_name: str) -> str:
        return TOOL_LABELS.get(tool_name, tool_name or "未知动作")

    def _summarize_tool_args(self, args: Any, limit: int = 4) -> Dict[str, Any]:
        if not isinstance(args, dict):
            return {}

        preview: Dict[str, Any] = {}
        for index, (key, value) in enumerate(args.items()):
            if index >= limit:
                break
            if isinstance(value, (int, float, bool)) or value is None:
                preview[str(key)] = value
                continue
            if isinstance(value, str):
                preview[str(key)] = value if len(value) <= 80 else f"{value[:77]}..."
                continue
            serialized = json.dumps(value, ensure_ascii=False)
            preview[str(key)] = serialized if len(serialized) <= 80 else f"{serialized[:77]}..."
        return preview

    def _interrupt_reason(self, tool_name: str, description: Optional[str] = None) -> str:
        if description:
            return str(description).strip()
        return f"{self._tool_label(tool_name)} 属于高风险写操作，需要人工审批。"

    def _build_interrupt_entry(
        self,
        *,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        allowed_decisions: Optional[List[str]] = None,
        interrupt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload_args = dict(args or {})
        return {
            "id": interrupt_id or f"manual:{tool_name}",
            "interrupt_id": interrupt_id,
            "tool": tool_name,
            "tool_label": self._tool_label(tool_name),
            "reason": self._interrupt_reason(tool_name, reason),
            "args_preview": self._summarize_tool_args(payload_args),
            "allowed_decisions": allowed_decisions or ["approve", "edit", "reject"],
        }

    def _build_billing_ticket_interrupt(
        self,
        *,
        user_id: str,
        message: str,
    ) -> Dict[str, Any]:
        tool_args = {
            "user_id": user_id,
            "subject": "账单异常核查",
            "description": message,
            "priority": "high",
            "category": "billing",
        }
        interrupt = self._build_interrupt_entry(
            tool_name="create_ticket",
            args=tool_args,
            reason="创建账单异常工单属于高风险写操作，需要人工审批后才能执行。",
        )
        return {
            "interrupts": [interrupt],
            "pending_approval_plan": {
                "mode": "deterministic_tool_call",
                "tool": "create_ticket",
                "tool_label": interrupt["tool_label"],
                "args": tool_args,
                "reason": interrupt["reason"],
                "reject_message": "已取消创建账单异常工单，本轮不会执行写操作。若仍需处理，请补充信息后重新提交。",
            },
        }

    def _build_approval_payload(self, interrupts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not interrupts:
            return None

        tools: List[Dict[str, Any]] = []
        for item in interrupts:
            tool_name = str(item.get("tool", "unknown_tool"))
            tools.append(
                {
                    "id": item.get("id"),
                    "tool": tool_name,
                    "tool_label": item.get("tool_label") or self._tool_label(tool_name),
                    "reason": item.get("reason") or self._interrupt_reason(tool_name),
                    "args_preview": item.get("args_preview", {}),
                    "allowed_decisions": item.get("allowed_decisions", ["approve", "edit", "reject"]),
                }
            )

        count = len(tools)
        labels = "、".join(tool["tool_label"] for tool in tools)
        return {
            "count": count,
            "required_decisions": count,
            "tools": tools,
            "message": f"当前有 {count} 个待审批动作：{labels}。请在 /resume 中提交 {count} 条 decisions。",
        }

    def _build_next_action(
        self,
        *,
        run_status: str,
        thread_id: Optional[str],
        approval: Optional[Dict[str, Any]] = None,
        error_message: str = "",
    ) -> str:
        if run_status == "completed":
            return "本轮已完成，无需继续操作。"
        if run_status == "interrupted":
            count = int((approval or {}).get("required_decisions", 1))
            return f"请调用 /runs/{thread_id}/resume，并提交 {count} 条 decisions。"

        if "未找到待审批线程" in error_message:
            return "请确认使用的是上一次 /chat 返回的 thread_id。"
        if approval:
            count = int(approval.get("required_decisions", len(approval.get("tools", [])) or 1))
            return f"请检查 decisions 数量是否与待审批动作一致，然后重新调用 /runs/{thread_id}/resume。当前应提交 {count} 条 decisions。"
        return "请检查请求参数或服务日志后重试。"

    def _format_resume_error(self, error: Exception, pending_state: OrchestrationState) -> str:
        raw = str(error)
        approval = self._build_approval_payload(pending_state.get("interrupts", []))
        mismatch = re.search(r"human decisions \((\d+)\).*hanging tool calls \((\d+)\)", raw, re.IGNORECASE)
        if mismatch and approval:
            received = mismatch.group(1)
            expected = mismatch.group(2)
            tools = "、".join(tool["tool_label"] for tool in approval.get("tools", []))
            return (
                f"恢复会话失败：当前有 {expected} 个待审批动作"
                f"（{tools}），但只收到了 {received} 条 decisions。"
            )
        return f"恢复会话失败：{raw}"

    def _resume_decision_count_error(
        self,
        decisions: List[Dict[str, Any]],
        pending_state: OrchestrationState,
    ) -> str:
        expected = len(pending_state.get("interrupts", []))
        if expected and len(decisions) != expected:
            approval = self._build_approval_payload(pending_state.get("interrupts", [])) or {}
            tools = "、".join(tool["tool_label"] for tool in approval.get("tools", [])) or "待审批动作"
            return (
                f"恢复会话失败：当前有 {expected} 个待审批动作"
                f"（{tools}），但只收到了 {len(decisions)} 条 decisions。"
            )
        return ""

    def _resume_decision_type(self, decision: Dict[str, Any]) -> str:
        return str((decision or {}).get("type", "")).strip().lower()

    def _edited_action_args(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        edited_action = (decision or {}).get("edited_action")
        if not isinstance(edited_action, dict):
            return {}
        nested_args = edited_action.get("args")
        if isinstance(nested_args, dict):
            return dict(nested_args)
        return {
            str(key): value
            for key, value in edited_action.items()
            if key not in {"tool", "tool_name", "name"}
        }

    def _extract_interrupts(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
        payloads: List[Dict[str, Any]] = []
        for item in raw_interrupts:
            interrupt_id = getattr(item, "id", None)
            value = getattr(item, "value", item)

            if isinstance(value, dict) and isinstance(value.get("action_requests"), list):
                review_configs = value.get("review_configs", [])
                for index, action_request in enumerate(value.get("action_requests", [])):
                    if not isinstance(action_request, dict):
                        continue
                    review_config = (
                        review_configs[index]
                        if index < len(review_configs) and isinstance(review_configs[index], dict)
                        else {}
                    )
                    tool_name = str(action_request.get("name", "unknown_tool"))
                    payloads.append(
                        {
                            "id": f"{interrupt_id}:{index}" if interrupt_id is not None else f"interrupt:{index}",
                            "interrupt_id": interrupt_id,
                            "tool": tool_name,
                            "tool_label": self._tool_label(tool_name),
                            "reason": self._interrupt_reason(tool_name, action_request.get("description")),
                            "args_preview": self._summarize_tool_args(action_request.get("args", {})),
                            "allowed_decisions": review_config.get(
                                "allowed_decisions", ["approve", "edit", "reject"]
                            ),
                        }
                    )
                continue

            tool_name = "unknown_tool"
            args: Dict[str, Any] = {}
            description: Optional[str] = None
            if isinstance(value, dict):
                nested_action = value.get("action")
                if isinstance(nested_action, dict):
                    tool_name = str(
                        nested_action.get("name")
                        or value.get("tool")
                        or value.get("name")
                        or value.get("action_name")
                        or "unknown_tool"
                    )
                    args = nested_action.get("args", {})
                else:
                    tool_name = str(
                        value.get("tool") or value.get("name") or value.get("action_name") or "unknown_tool"
                    )
                    args = value.get("args", {})
                description = value.get("description")

            payloads.append(
                {
                    "id": str(interrupt_id) if interrupt_id is not None else None,
                    "interrupt_id": interrupt_id,
                    "tool": tool_name,
                    "tool_label": self._tool_label(tool_name),
                    "reason": self._interrupt_reason(tool_name, description),
                    "args_preview": self._summarize_tool_args(args),
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            )
        return payloads

    def _extract_ai_text(self, result: Dict[str, Any]) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return "抱歉，我暂时无法给出有效回复。"

    def _tool_result_ticket_id(self, tool_name: str, text: str) -> Optional[str]:
        lowered = str(text or "").lower()
        failure_markers = ("失败", "error", "未找到", "不存在", "无效")
        if tool_name in {"create_ticket", "update_ticket", "escalate_to_human"} and any(
            marker in lowered for marker in failure_markers
        ):
            return None
        return self._ticket_id_from_text(text)

    def _complete_resumed_turn(
        self,
        *,
        thread_id: str,
        user_id: str,
        trace_id: str,
        role: str,
        pending_state: OrchestrationState,
        neutral_sentiment: SentimentResult,
        resumed_text: str,
        role_elapsed_ms: float,
        started: float,
        trace_summary: str,
        trace_details: Optional[Dict[str, Any]] = None,
        tool_source: Optional[str] = None,
        explicit_ticket_id: Optional[str] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
        preserve_ticket_lookup: bool = True,
        langsmith: Optional[Dict[str, Any]] = None,
    ) -> SupportResponse:
        merged_state: OrchestrationState = dict(pending_state)
        merged_state.update(
            append_node_timing(
                merged_state,
                node=role,
                agent=role,
                duration_ms=role_elapsed_ms,
                status="completed",
            )
        )
        merged_state["active_agent"] = role
        merged_state["run_status"] = "completed"
        merged_state["interrupts"] = []
        merged_state["evidence_items"] = self._merge_evidence_items(
            merged_state.get("evidence_items", []),
            evidence_items or [],
        )
        merged_state["trace_events"] = [
            *merged_state.get("trace_events", []),
            build_trace_event(
                node=role,
                agent=role,
                summary=trace_summary,
                status="completed",
                details=trace_details,
            ),
        ]

        if role == "knowledge":
            merged_state["retrieval_text"] = strip_source_annotations(resumed_text)
        if role in {"action", "escalation"}:
            merged_state["tool_text"] = strip_source_annotations(resumed_text)
            if tool_source:
                merged_state["tool_source"] = tool_source
            if merged_state.get("ticket_id") is None:
                if preserve_ticket_lookup:
                    merged_state["ticket_id"] = explicit_ticket_id or self._ticket_id_from_text(resumed_text)
                else:
                    merged_state["ticket_id"] = explicit_ticket_id
            if role == "action" and self._is_positive_escalation_text(resumed_text):
                merged_state["escalated"] = True
        if role == "escalation":
            merged_state["escalated"] = True

        merged_state.pop("pending_approval_plan", None)
        merged_state = self._finalize_after_execution_with_langsmith(
            merged_state,
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
        )
        response_message = merged_state.get("final_message", resumed_text)
        citations = self._citations_from_state(merged_state)

        self._pending_role.pop(thread_id, None)
        self._pending_state.pop(thread_id, None)
        self._record_history(
            thread_id=thread_id,
            user_id=user_id,
            role="assistant",
            content=response_message,
            intent=merged_state.get("intent", "resume"),
            active_agent=merged_state.get("selected_agent", role),
            run_status="completed",
            metadata={
                "trace_id": trace_id,
                "citations": citations,
                "ticket_id": merged_state.get("ticket_id"),
                "resumed": True,
            },
            thread_status="completed",
            trace_id=trace_id,
        )
        clear_pending_conversation_state(
            thread_id,
            status="completed",
            last_active_agent=merged_state.get("selected_agent", role),
            trace_id=trace_id,
        )
        self._refresh_thread_summary_if_needed(thread_id)
        self._save_turn_memory(
            user_id=user_id,
            thread_id=thread_id,
            intent=merged_state.get("intent", "resume"),
            active_agent=merged_state.get("selected_agent", role),
            user_message="[resume]",
            assistant_message=response_message,
            sentiment=neutral_sentiment,
        )

        return SupportResponse(
            message=response_message,
            intent=merged_state.get("intent", "resume"),
            sentiment=neutral_sentiment,
            sources=self._resolve_sources(merged_state),
            escalated=bool(merged_state.get("escalated")),
            ticket_created=merged_state.get("ticket_id"),
            thread_id=thread_id,
            run_status="completed",
            citations=citations,
            active_agent=merged_state.get("selected_agent", role),
            trace_id=trace_id,
            route_path=merged_state.get("route_path", []),
            validation_notes=merged_state.get("validation_notes", []),
            trace_preview=self._trace_preview(merged_state),
            node_timings=merged_state.get("node_timings", []),
            decision_summary=merged_state.get("decision_summary", ""),
            approval=None,
            memory_debug=self._memory_debug_snapshot(thread_id),
            langsmith=langsmith or {},
            next_action=self._build_next_action(
                run_status="completed",
                thread_id=thread_id,
            ),
            total_duration_ms=round((perf_counter() - started) * 1000, 2),
        )

    def _fallback_response(
        self,
        role: str,
        user_id: str,
        message: str,
        sentiment: Optional[SentimentResult],
        retrieval_text: str = "",
        tool_text: str = "",
    ) -> str:
        intro = ""
        if sentiment and sentiment.frustration_score >= 0.45:
            intro = "我理解你现在比较着急，我们先快速定位问题。"

        lowered = message.lower()
        if role == "knowledge":
            try:
                bundle = self._run_knowledge_lookup_bundle(message)
                return intro + str(bundle.get("text", "")).strip()
            except Exception as error:
                return intro + f"知识检索失败：{error}"

        if role == "action":
            structured_action = self._run_structured_business_action(
                user_id=user_id,
                message=message,
            )
            if structured_action is not None:
                return intro + structured_action

            if "工单" in message and any(token in lowered for token in ("创建", "create", "open", "new")):
                return intro + create_ticket.invoke(
                    {
                        "user_id": user_id,
                        "subject": message[:80],
                        "description": message,
                        "priority": "medium",
                        "category": "general",
                    }
                )
            if "状态" in message or "status" in lowered:
                ticket_id = self._ticket_id_from_text(message)
                if ticket_id:
                    return intro + get_ticket_status.invoke({"ticket_id": ticket_id})
                return intro + get_user_tickets.invoke({"user_id": user_id, "status": None})
            if "账户" in message or "account" in lowered:
                return intro + lookup_account.invoke({"user_id": user_id})
            return intro + "请提供更具体的执行信息（例如工单号、账户ID、操作目标）。"

        if role == "escalation":
            return intro + escalate_to_human.invoke(
                {
                    "user_id": user_id,
                    "reason": "高风险或高情绪场景",
                    "conversation_summary": message[:400],
                }
            )

        if role == "responder":
            response_parts: List[str] = []
            if retrieval_text:
                response_parts.append(strip_source_annotations(retrieval_text))
            if tool_text:
                response_parts.append(strip_source_annotations(tool_text))
            if response_parts:
                return strip_source_annotations(intro + "\n\n".join(response_parts))
            return (
                intro
                + "我可以帮助你处理产品使用、账单账户、技术排障与工单问题。"
                "请告诉我你遇到的具体现象（操作步骤、报错信息、期望结果）。"
            )

        return (
            intro
            + "我可以帮助你处理产品使用、账单账户、技术排障与工单问题。"
            "请告诉我你遇到的具体现象（操作步骤、报错信息、期望结果）。"
        )

    def _touch_thread(
        self,
        *,
        thread_id: str,
        user_id: str,
        status: str = "active",
        title: Optional[str] = None,
        last_active_agent: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """确保线程元数据存在，便于 transcript 与 API 查询统一落库。"""
        create_or_touch_conversation_thread(
            thread_id=thread_id,
            user_id=user_id,
            status=status,
            title=title,
            last_active_agent=last_active_agent,
            trace_id=trace_id,
        )

    def _load_recent_history_context(self, thread_id: str) -> Dict[str, Any]:
        """从数据库读取短期 transcript 窗口，供主图节点提示词使用。"""
        return build_recent_context_window(
            thread_id=thread_id,
            recent_turns=settings.conversation_recent_turns,
            max_messages=settings.conversation_context_messages,
            max_tokens=settings.conversation_context_tokens,
        )

    def _heuristic_rolling_summary(
        self,
        messages: List[Dict[str, Any]],
        previous_summary: str = "",
    ) -> str:
        """在无 LLM 或 LLM 摘要失败时，用规则生成结构化中文摘要。"""

        def normalize_summary_text(raw: str, limit: int = 72) -> str:
            compact = re.sub(r"\s+", " ", raw.strip())
            if len(compact) <= limit:
                return compact
            return compact[: limit - 1].rstrip("，。；、:： ") + "…"

        user_points: List[str] = []
        resolved_points: List[str] = []
        pending_points: List[str] = []
        pending_markers = ("请", "需要", "建议", "等待", "稍后", "后续", "审批", "确认")

        for item in messages:
            content = normalize_summary_text(str(item.get("content", "")))
            if not content:
                continue

            role = item.get("role")
            if role == "user" and content not in user_points:
                user_points.append(content)
                continue

            if role in {"assistant", "interrupt"}:
                target = pending_points if role == "interrupt" or any(
                    marker in content for marker in pending_markers
                ) else resolved_points
                if content not in target:
                    target.append(content)

        lines: List[str] = []
        previous_summary = normalize_summary_text(previous_summary, limit=96)
        if previous_summary:
            lines.append(f"历史摘要：{previous_summary}")
        if user_points:
            lines.append("用户近期诉求：" + "；".join(user_points[-2:]))
        if resolved_points:
            lines.append("已处理/已确认：" + "；".join(resolved_points[-2:]))
        if pending_points:
            lines.append("待跟进事项：" + "；".join(pending_points[-2:]))
        elif user_points:
            lines.append(f"待跟进事项：继续围绕“{user_points[-1]}”跟进并给出下一步。")
        return "\n".join(lines)[:420]

    def _generate_rolling_summary(
        self,
        messages: List[Dict[str, Any]],
        previous_summary: str = "",
    ) -> str:
        """优先用模型总结旧消息，失败时回退到规则摘要。"""
        if not messages:
            return ""

        if self.llm_enabled and self.basic_model is not None:
            transcript = []
            for item in messages[-10:]:
                role = item.get("role", "message")
                content = str(item.get("content", "")).strip()
                if content:
                    transcript.append(f"{role}: {content}")
            if transcript:
                prompt = (
                    "你是客服会话摘要器，请把更早的对话压缩成简洁、可续写的中文摘要。"
                    "请最多输出 3 行，优先使用以下字段：用户近期诉求、已处理/已确认、待跟进事项。"
                    "不要编造，不要重复，总长度控制在 180 字以内。\n\n"
                    f"已有滚动摘要：{previous_summary or '无'}\n"
                    "新增历史：\n"
                    + "\n".join(transcript)
                )
                try:
                    response = self.basic_model.invoke(prompt)
                    summary = str(getattr(response, "content", "") or "").strip()
                    if summary:
                        return summary[:420]
                except Exception as error:
                    logger.debug(f"Rolling summary fallback to heuristic: {error}")

        return self._heuristic_rolling_summary(messages, previous_summary=previous_summary)

    def _refresh_thread_summary_if_needed(self, thread_id: str) -> None:
        """在线程足够长时刷新 rolling summary，降低长对话的上下文开销。"""
        thread = get_conversation_thread(thread_id)
        if not thread:
            return

        message_count = int(thread.get("message_count") or 0)
        trigger = max(6, settings.conversation_summary_trigger_messages)
        interval = max(2, settings.conversation_summary_refresh_interval)
        keep_tail = max(4, settings.conversation_context_messages)

        if message_count < trigger:
            return
        if thread.get("rolling_summary") and (message_count - trigger) % interval != 0:
            return

        messages = list_thread_messages(thread_id, visible_only=True)
        if len(messages) <= keep_tail:
            return

        older_messages = messages[:-keep_tail]
        summary = self._generate_rolling_summary(
            older_messages,
            previous_summary=thread.get("rolling_summary", ""),
        )
        if not summary:
            return

        mark_conversation_thread_status(
            thread_id,
            status=thread.get("status", "active"),
            last_active_agent=thread.get("last_active_agent"),
            rolling_summary=summary,
            pending_role=thread.get("pending_role") or "",
            pending_state=thread.get("pending_state") or {},
            trace_id=thread.get("trace_id"),
        )

    def _load_pending_context(self, thread_id: str) -> Tuple[str, str, Optional[str], OrchestrationState]:
        """优先从内存缓存读取待审批状态，不存在时回退到数据库。"""
        user_id = self._thread_user.get(thread_id, "unknown_user")
        trace_id = self._trace_by_thread.get(thread_id, "")
        role = self._pending_role.get(thread_id)
        pending_state = self._pending_state.get(thread_id, {})

        if role and pending_state:
            loaded_state = dict(pending_state)
            loaded_state.setdefault("graph_thread_id", self._graph_thread_id(thread_id))
            self._pending_state[thread_id] = loaded_state
            return user_id, trace_id or str(uuid.uuid4()), role, loaded_state

        thread = get_conversation_thread(thread_id)
        if not thread:
            return user_id, trace_id or str(uuid.uuid4()), None, {}

        persisted_role = str(thread.get("pending_role") or "").strip() or None
        persisted_state = thread.get("pending_state") or {}
        persisted_trace_id = thread.get("trace_id") or trace_id or str(uuid.uuid4())
        persisted_user_id = thread.get("user_id") or user_id

        if persisted_role and isinstance(persisted_state, dict) and persisted_state:
            loaded_state = dict(persisted_state)
            loaded_state.setdefault("graph_thread_id", self._graph_thread_id(thread_id))
            self._thread_user[thread_id] = persisted_user_id
            self._trace_by_thread[thread_id] = persisted_trace_id
            self._pending_role[thread_id] = persisted_role
            self._pending_state[thread_id] = loaded_state
            return persisted_user_id, persisted_trace_id, persisted_role, loaded_state

        return persisted_user_id, persisted_trace_id, None, {}

    def _record_history(
        self,
        *,
        thread_id: str,
        user_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        active_agent: Optional[str] = None,
        run_status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        thread_status: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """将用户可见对话历史持久化到业务数据库。"""
        append_conversation_message(
            thread_id=thread_id,
            user_id=user_id,
            role=role,
            content=content,
            visible=visible,
            intent=intent,
            active_agent=active_agent,
            run_status=run_status,
            metadata=metadata,
            thread_status=thread_status,
            trace_id=trace_id,
        )

    def _save_turn_memory(
        self,
        user_id: str,
        thread_id: str,
        intent: str,
        active_agent: str,
        user_message: str,
        assistant_message: str,
        sentiment: Optional[SentimentResult],
    ) -> None:
        if not self.enable_memory:
            return
        telemetry_payload = {
            "kind": "conversation_turn",
            "thread_id": thread_id,
            "intent": intent,
            "active_agent": active_agent,
            "message": user_message,
            "response": assistant_message,
            "sentiment": sentiment.label if sentiment else "neutral",
            "frustration_score": sentiment.frustration_score if sentiment else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_memory_item(user_id=user_id, payload=telemetry_payload)
        self._write_structured_memories(
            user_id=user_id,
            thread_id=thread_id,
            intent=intent,
            active_agent=active_agent,
            user_message=user_message,
            assistant_message=assistant_message,
            sentiment=sentiment,
        )

    def _call_role_agent(
        self,
        role: str,
        user_id: str,
        thread_id: str,
        trace_id: str,
        intent: str,
        risk: str,
        message: str,
    ) -> Dict[str, Any]:
        if not self.llm_enabled or role not in self.role_agents:
            fallback = self._fallback_response(role, user_id, message, None)
            return {"messages": [AIMessage(content=fallback)]}

        agent = self.role_agents[role]
        config = self._build_runnable_config(
            run_name=f"{role}_agent_invoke",
            entrypoint="agent",
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            checkpoint_thread_id=self._agent_thread_id(thread_id, role),
            role=role,
            intent=intent,
            risk=risk,
        )
        context = AgentRuntimeContext(
            user_id=user_id,
            thread_id=thread_id,
            active_agent=role,
            intent=intent,
            risk=risk,
        )
        return agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            context=context,
        )

    def _build_sentiment_result(
        self,
        baseline: SentimentResult,
        state: OrchestrationState,
    ) -> SentimentResult:
        return SentimentResult(
            polarity=baseline.polarity,
            subjectivity=baseline.subjectivity,
            label=self._normalize_sentiment_label(state.get("sentiment_label", baseline.label)),
            frustration_score=self._safe_float(
                state.get("frustration_score"),
                default=baseline.frustration_score,
            ),
            keywords=baseline.keywords,
        )

    def _resolve_sources(self, state: OrchestrationState) -> List[str]:
        sources: List[str] = []
        if state.get("retrieval_text"):
            sources.extend(["Hybrid RAG", "Help Center Knowledge Base"])
        if state.get("tool_text"):
            sources.append(state.get("tool_source") or "Support Tools")
        if state.get("selected_agent") == "escalation" or state.get("escalated"):
            sources.append("Human Handoff")
        if self.llm_enabled:
            sources.append("Responder Agent")
        else:
            sources.append("Fallback Policy")
        return self._merge_unique([], sources)

    def _trace_preview(self, state: OrchestrationState, limit: int = 8) -> List[Dict[str, Any]]:
        trace_events = state.get("trace_events", [])
        return trace_events[-limit:]

    def chat(
        self,
        user_id: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> SupportResponse:
        """Process a user turn through the LangGraph workflow."""
        started = perf_counter()
        thread = thread_id or str(uuid.uuid4())
        graph_thread_id = self._graph_thread_id(thread)
        trace_id = str(uuid.uuid4())
        self._thread_user[thread] = user_id
        self._trace_by_thread[thread] = trace_id
        self._reset_memory_debug(thread)

        baseline_sentiment = (
            self.sentiment_analyzer.analyze(message)
            if self.sentiment_analyzer
            else build_neutral_sentiment()
        )
        initial_intent = self._infer_intent(message)
        initial_risk = self._infer_risk(message, baseline_sentiment)

        self._touch_thread(
            thread_id=thread,
            user_id=user_id,
            status="active",
            title=message,
            last_active_agent="supervisor",
            trace_id=trace_id,
        )
        recent_context = self._load_recent_history_context(thread)
        self._record_history(
            thread_id=thread,
            user_id=user_id,
            role="user",
            content=message,
            intent=initial_intent,
            active_agent="supervisor",
            run_status="active",
            metadata={"source": "chat"},
            thread_status="active",
            trace_id=trace_id,
        )

        initial_state: OrchestrationState = {
            "user_id": user_id,
            "thread_id": thread,
            "graph_thread_id": graph_thread_id,
            "trace_id": trace_id,
            "current_message": message,
            "recent_history_text": recent_context.get("text", ""),
            "rolling_summary": recent_context.get("rolling_summary", ""),
            "intent": initial_intent,
            "risk": initial_risk,
            "sentiment_label": baseline_sentiment.label,
            "frustration_score": baseline_sentiment.frustration_score,
            "route_reason": "bootstrap",
            "selected_agent": "supervisor",
            "active_agent": "supervisor",
            "execution_steps": [],
            "route_path": [],
            "trace_events": [],
            "decision_summary": "",
            "needs_knowledge": False,
            "needs_action": False,
            "needs_action_after_knowledge": False,
            "needs_escalation": False,
            "retrieval_text": "",
            "tool_text": "",
            "tool_source": "",
            "evidence_items": [],
            "validation_notes": [],
            "validation_passed": True,
            "final_message": "",
            "citations": [],
            "pending_approval_plan": {},
            "run_status": "completed",
            "interrupts": [],
            "ticket_id": None,
            "escalated": False,
        }
        graph_config = self._build_runnable_config(
            run_name="support_orchestration_graph",
            entrypoint="graph",
            user_id=user_id,
            thread_id=thread,
            trace_id=trace_id,
            checkpoint_thread_id=graph_thread_id,
            role="supervisor",
            intent=initial_intent,
            risk=initial_risk,
        )
        with self._langsmith_trace(
            name="support.chat",
            entrypoint="chat",
            user_id=user_id,
            thread_id=thread,
            trace_id=trace_id,
            role="supervisor",
            intent=initial_intent,
            risk=initial_risk,
            resumed=False,
        ) as langsmith_debug:
            try:
                final_state = self.orchestration_graph.invoke(initial_state, config=graph_config)
            except Exception as error:
                logger.error(f"Graph invocation failed, fallback to template: {error}")
                fallback = self._fallback_response(
                    role="supervisor",
                    user_id=user_id,
                    message=message,
                    sentiment=baseline_sentiment,
                )
                self._record_history(
                    thread_id=thread,
                    user_id=user_id,
                    role="assistant",
                    content=fallback,
                    intent=initial_state["intent"],
                    active_agent=initial_state["selected_agent"],
                    run_status="completed",
                    metadata={"fallback": True, "trace_id": trace_id},
                    thread_status="completed",
                    trace_id=trace_id,
                )
                clear_pending_conversation_state(
                    thread,
                    status="completed",
                    last_active_agent=initial_state["selected_agent"],
                    trace_id=trace_id,
                )
                self._refresh_thread_summary_if_needed(thread)
                return SupportResponse(
                    message=fallback,
                    intent=initial_state["intent"],
                    sentiment=baseline_sentiment,
                    sources=["Fallback Policy"],
                    escalated=False,
                    ticket_created=self._ticket_id_from_text(fallback),
                    thread_id=thread,
                    run_status="completed",
                    interrupts=[],
                    citations=[],
                    active_agent=initial_state["selected_agent"],
                    trace_id=trace_id,
                    route_path=["fallback"],
                    validation_notes=[],
                    trace_preview=[
                        build_trace_event(
                            node="fallback",
                            agent="supervisor",
                            summary="图执行失败，已回退到模板回复。",
                            status="error",
                            details={"error": str(error)},
                        )
                    ],
                    node_timings=initial_state.get("node_timings", []),
                    decision_summary="图执行失败，已使用回退策略直接回复。",
                    approval=None,
                    memory_debug=self._memory_debug_snapshot(thread),
                    langsmith=langsmith_debug,
                    next_action=self._build_next_action(
                        run_status="completed",
                        thread_id=thread,
                    ),
                    total_duration_ms=round((perf_counter() - started) * 1000, 2),
                )

            if final_state.get("run_status") == "interrupted":
                pending_role = final_state.get("active_agent", final_state.get("selected_agent", "action"))
                parent_headers = self._langsmith_current_parent_headers()
                if parent_headers:
                    final_state["langsmith_parent_headers"] = parent_headers
                self._pending_role[thread] = pending_role
                self._pending_state[thread] = dict(final_state)
                save_pending_conversation_state(
                    thread_id=thread,
                    user_id=user_id,
                    pending_role=pending_role,
                    pending_state=dict(final_state),
                    trace_id=trace_id,
                    status="interrupted",
                    last_active_agent=pending_role,
                )
                wait_message = "检测到高风险动作，已暂停执行，等待人工审批（approve/edit/reject）。"
                self._record_history(
                    thread_id=thread,
                    user_id=user_id,
                    role="interrupt",
                    content=wait_message,
                    intent=final_state.get("intent", "other"),
                    active_agent=pending_role,
                    run_status="interrupted",
                    metadata={
                        "trace_id": trace_id,
                        "interrupt_count": len(final_state.get("interrupts", [])),
                    },
                    thread_status="interrupted",
                    trace_id=trace_id,
                )
                self._refresh_thread_summary_if_needed(thread)
                sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
                approval = self._build_approval_payload(final_state.get("interrupts", []))
                citations = self._citations_from_state(final_state)
                return SupportResponse(
                    message=wait_message,
                    intent=final_state.get("intent", "other"),
                    sentiment=sentiment,
                    sources=["HITL Middleware"],
                    escalated=pending_role == "escalation",
                    ticket_created=None,
                    thread_id=thread,
                    run_status="interrupted",
                    interrupts=final_state.get("interrupts", []),
                    citations=citations,
                    active_agent=pending_role,
                    trace_id=trace_id,
                    route_path=final_state.get("route_path", []),
                    validation_notes=final_state.get("validation_notes", []),
                    trace_preview=self._trace_preview(final_state),
                    node_timings=final_state.get("node_timings", []),
                    decision_summary=final_state.get("decision_summary", ""),
                    approval=approval,
                    memory_debug=self._memory_debug_snapshot(thread),
                    langsmith=langsmith_debug,
                    next_action=self._build_next_action(
                        run_status="interrupted",
                        thread_id=thread,
                        approval=approval,
                    ),
                    total_duration_ms=round((perf_counter() - started) * 1000, 2),
                )

            response_message = final_state.get("final_message") or final_state.get("tool_text") or final_state.get(
                "retrieval_text"
            ) or "抱歉，我暂时无法给出有效回复。"
            citations = self._citations_from_state(final_state)
            ticket_id = final_state.get("ticket_id") or self._ticket_id_from_text(response_message)
            active_agent = final_state.get("selected_agent", "supervisor")
            sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
            sources = self._resolve_sources(final_state)

            self._record_history(
                thread_id=thread,
                user_id=user_id,
                role="assistant",
                content=response_message,
                intent=final_state.get("intent", "other"),
                active_agent=active_agent,
                run_status="completed",
                metadata={
                    "trace_id": trace_id,
                    "citations": citations,
                    "ticket_id": ticket_id,
                    "escalated": bool(final_state.get("escalated")),
                },
                thread_status="completed",
                trace_id=trace_id,
            )
            clear_pending_conversation_state(
                thread,
                status="completed",
                last_active_agent=active_agent,
                trace_id=trace_id,
            )
            self._refresh_thread_summary_if_needed(thread)
            self._save_turn_memory(
                user_id=user_id,
                thread_id=thread,
                intent=final_state.get("intent", "other"),
                active_agent=active_agent,
                user_message=message,
                assistant_message=response_message,
                sentiment=sentiment,
            )

            return SupportResponse(
                message=response_message,
                intent=final_state.get("intent", "other"),
                sentiment=sentiment,
                sources=sources,
                escalated=bool(final_state.get("escalated")),
                ticket_created=ticket_id,
                thread_id=thread,
                run_status="completed",
                interrupts=[],
                citations=citations,
                active_agent=active_agent,
                trace_id=trace_id,
                route_path=final_state.get("route_path", []),
                validation_notes=final_state.get("validation_notes", []),
                trace_preview=self._trace_preview(final_state),
                node_timings=final_state.get("node_timings", []),
                decision_summary=final_state.get("decision_summary", ""),
                approval=None,
                memory_debug=self._memory_debug_snapshot(thread),
                langsmith=langsmith_debug,
                next_action=self._build_next_action(
                    run_status="completed",
                    thread_id=thread,
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

    def resume(
        self,
        thread_id: str,
        decisions: List[Dict[str, Any]],
    ) -> SupportResponse:
        """Resume a HITL-interrupted thread."""
        started = perf_counter()
        user_id, trace_id, role, pending_state = self._load_pending_context(thread_id)
        neutral_sentiment = build_neutral_sentiment()
        effective_user_id = user_id or self._thread_user.get(thread_id, "unknown_user")
        effective_trace_id = trace_id or str(uuid.uuid4())
        effective_role = role or "unknown"
        parent_headers = pending_state.get("langsmith_parent_headers") if pending_state else None

        if role is None:
            return SupportResponse(
                message="未找到待审批线程，无法恢复。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent="unknown",
                trace_id=trace_id,
                route_path=["resume"],
                validation_notes=[],
                trace_preview=[
                    build_trace_event(
                        node="resume",
                        agent="unknown",
                        summary="未找到待审批线程。",
                        status="error",
                    )
                ],
                node_timings=[],
                decision_summary="恢复失败：未找到待审批线程。",
                approval=None,
                memory_debug=self._memory_debug_snapshot(thread_id),
                next_action=self._build_next_action(
                    run_status="error",
                    thread_id=thread_id,
                    error_message="未找到待审批线程，无法恢复。",
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

        decision_count_error = self._resume_decision_count_error(decisions, pending_state)
        if decision_count_error:
            approval = self._build_approval_payload(pending_state.get("interrupts", []))
            return SupportResponse(
                message=decision_count_error,
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent=role,
                trace_id=trace_id,
                route_path=pending_state.get("route_path", []),
                validation_notes=pending_state.get("validation_notes", []),
                trace_preview=self._trace_preview(pending_state),
                node_timings=pending_state.get("node_timings", []),
                decision_summary=pending_state.get("decision_summary", ""),
                approval=approval,
                memory_debug=self._memory_debug_snapshot(thread_id),
                next_action=self._build_next_action(
                    run_status="error",
                    thread_id=thread_id,
                    approval=approval,
                    error_message=decision_count_error,
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

        deterministic_plan = pending_state.get("pending_approval_plan", {})
        if isinstance(deterministic_plan, dict) and deterministic_plan.get("mode") == "deterministic_tool_call":
            with self._langsmith_trace(
                name="support.resume",
                entrypoint="resume",
                user_id=effective_user_id,
                thread_id=thread_id,
                trace_id=effective_trace_id,
                role=effective_role,
                intent=pending_state.get("intent", "resume"),
                risk=pending_state.get("risk", "medium"),
                resumed=True,
                parent=parent_headers,
            ) as resume_langsmith:
                decision = decisions[0] if decisions else {}
                decision_type = self._resume_decision_type(decision)
                approval = self._build_approval_payload(pending_state.get("interrupts", []))

                if decision_type not in {"approve", "edit", "reject"}:
                    return SupportResponse(
                        message="恢复会话失败：decision.type 仅支持 approve、edit、reject。",
                        intent="resume",
                        sentiment=neutral_sentiment,
                        sources=["HITL Middleware"],
                        thread_id=thread_id,
                        run_status="error",
                        active_agent=role,
                        trace_id=effective_trace_id,
                        route_path=pending_state.get("route_path", []),
                        validation_notes=pending_state.get("validation_notes", []),
                        trace_preview=self._trace_preview(pending_state),
                        node_timings=pending_state.get("node_timings", []),
                        decision_summary=pending_state.get("decision_summary", ""),
                        approval=approval,
                        memory_debug=self._memory_debug_snapshot(thread_id),
                        langsmith=resume_langsmith,
                        next_action=self._build_next_action(
                            run_status="error",
                            thread_id=thread_id,
                            approval=approval,
                            error_message="decision.type 非法",
                        ),
                        total_duration_ms=round((perf_counter() - started) * 1000, 2),
                    )

                tool_name = str(deterministic_plan.get("tool", "")).strip()
                tool = get_tool_by_name(tool_name)
                if tool is None:
                    return SupportResponse(
                        message=f"恢复会话失败：未找到待执行工具 `{tool_name}`。",
                        intent="resume",
                        sentiment=neutral_sentiment,
                        sources=["HITL Middleware"],
                        thread_id=thread_id,
                        run_status="error",
                        active_agent=role,
                        trace_id=effective_trace_id,
                        route_path=pending_state.get("route_path", []),
                        validation_notes=pending_state.get("validation_notes", []),
                        trace_preview=self._trace_preview(pending_state),
                        node_timings=pending_state.get("node_timings", []),
                        decision_summary=pending_state.get("decision_summary", ""),
                        approval=approval,
                        memory_debug=self._memory_debug_snapshot(thread_id),
                        langsmith=resume_langsmith,
                        next_action=self._build_next_action(
                            run_status="error",
                            thread_id=thread_id,
                            approval=approval,
                            error_message=f"未找到待执行工具 {tool_name}",
                        ),
                        total_duration_ms=round((perf_counter() - started) * 1000, 2),
                    )

                tool_args = dict(deterministic_plan.get("args", {}))
                trace_summary = "审批完成，继续执行图流程。"
                trace_details: Dict[str, Any] = {
                    "tool": deterministic_plan.get("tool_label") or self._tool_label(tool_name),
                    "decision": decision_type,
                }
                tool_source = "Approval Decision"
                explicit_ticket_id: Optional[str] = None
                resumed_evidence_items: List[EvidenceItem] = []

                role_started = perf_counter()
                if decision_type == "reject":
                    resumed_text = str(
                        deterministic_plan.get("reject_message")
                        or f"已取消{self._tool_label(tool_name)}，本轮不会执行写操作。"
                    )
                    trace_summary = "审批已拒绝，本轮未执行写操作。"
                else:
                    if decision_type == "edit":
                        tool_args.update(self._edited_action_args(decision))
                        trace_summary = "审批通过并带修改，已执行调整后的动作。"
                        trace_details["edited_args"] = self._summarize_tool_args(tool_args)
                    else:
                        trace_summary = "审批通过，已执行待审批动作。"

                    self._save_memory_item(
                        user_id=user_id,
                        payload={
                            "kind": "tool_call",
                            "role": role,
                            "thread_id": thread_id,
                            "tool_name": tool_name,
                            "args": tool_args,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    tool_config = self._build_runnable_config(
                        run_name=f"{tool_name}_resume_tool",
                        entrypoint="resume-tool",
                        user_id=user_id,
                        thread_id=thread_id,
                        trace_id=effective_trace_id,
                        role=role,
                        intent=pending_state.get("intent", "resume"),
                        risk=pending_state.get("risk", "medium"),
                        resumed=True,
                    )
                    tool_metadata = self._langsmith_metadata(
                        user_id=user_id,
                        thread_id=thread_id,
                        trace_id=effective_trace_id,
                        role=role,
                        intent=pending_state.get("intent", "resume"),
                        risk=pending_state.get("risk", "medium"),
                        resumed=True,
                    )
                    tool_metadata["tool_name"] = tool_name
                    with trace(
                        name=f"{tool_name}_resume_dispatch",
                        run_type="chain",
                        project_name=settings.langsmith_project,
                        tags=self._langsmith_tags(entrypoint="resume-tool", role=role),
                        metadata=tool_metadata,
                        client=self.langsmith_client,
                    ):
                        resumed_text = str(tool.invoke(tool_args, config=tool_config))
                    tool_source = "Support Tools"
                    explicit_ticket_id = self._tool_result_ticket_id(tool_name, resumed_text)
                    evidence_item = self._build_tool_evidence_item(
                        tool_name=tool_name,
                        text=resumed_text,
                        tool_args=tool_args,
                    )
                    if evidence_item:
                        resumed_evidence_items.append(evidence_item)

                role_elapsed_ms = (perf_counter() - role_started) * 1000
                return self._complete_resumed_turn(
                    thread_id=thread_id,
                    user_id=user_id,
                    trace_id=effective_trace_id,
                    role=role,
                    pending_state=pending_state,
                    neutral_sentiment=neutral_sentiment,
                    resumed_text=resumed_text,
                    role_elapsed_ms=role_elapsed_ms,
                    started=started,
                    trace_summary=trace_summary,
                    trace_details=trace_details,
                    tool_source=tool_source,
                    explicit_ticket_id=explicit_ticket_id,
                    evidence_items=resumed_evidence_items,
                    preserve_ticket_lookup=False,
                    langsmith=resume_langsmith,
                )

        if not self.llm_enabled or role not in self.role_agents:
            self._pending_role.pop(thread_id, None)
            self._pending_state.pop(thread_id, None)
            clear_pending_conversation_state(
                thread_id,
                status="completed",
                last_active_agent=role,
                trace_id=trace_id,
            )
            return SupportResponse(
                message="当前处于无 LLM 模式，已跳过审批并结束流程。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["Fallback Policy"],
                thread_id=thread_id,
                run_status="completed",
                active_agent=role,
                trace_id=trace_id,
                route_path=pending_state.get("route_path", []),
                validation_notes=pending_state.get("validation_notes", []),
                trace_preview=self._trace_preview(pending_state),
                node_timings=pending_state.get("node_timings", []),
                decision_summary=pending_state.get("decision_summary", ""),
                approval=None,
                memory_debug=self._memory_debug_snapshot(thread_id),
                next_action=self._build_next_action(
                    run_status="completed",
                    thread_id=thread_id,
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

        agent = self.role_agents[role]
        config = self._build_runnable_config(
            run_name=f"{role}_agent_resume",
            entrypoint="resume-agent",
            user_id=user_id,
            thread_id=thread_id,
            trace_id=trace_id,
            checkpoint_thread_id=self._agent_thread_id(thread_id, role),
            role=role,
            intent=pending_state.get("intent", "resume"),
            risk=pending_state.get("risk", "medium"),
            resumed=True,
        )
        context = AgentRuntimeContext(
            user_id=user_id,
            thread_id=thread_id,
            active_agent=role,
            intent=pending_state.get("intent", "resume"),
            risk=pending_state.get("risk", "medium"),
        )

        try:
            role_started = perf_counter()
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config,
                context=context,
            )
            role_elapsed_ms = (perf_counter() - role_started) * 1000
        except Exception as error:
            logger.error(f"Resume failed: {error}")
            approval = self._build_approval_payload(pending_state.get("interrupts", []))
            return SupportResponse(
                message=self._format_resume_error(error, pending_state),
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent=role,
                trace_id=trace_id,
                route_path=pending_state.get("route_path", []),
                validation_notes=pending_state.get("validation_notes", []),
                trace_preview=[
                    *self._trace_preview(pending_state),
                    build_trace_event(
                        node=role,
                        agent=role,
                        summary="恢复执行失败。",
                        status="error",
                        details={"error": str(error)},
                    ),
                ][-8:],
                node_timings=pending_state.get("node_timings", []),
                decision_summary=pending_state.get("decision_summary", ""),
                approval=approval,
                memory_debug=self._memory_debug_snapshot(thread_id),
                next_action=self._build_next_action(
                    run_status="error",
                    thread_id=thread_id,
                    approval=approval,
                    error_message=str(error),
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

        interrupts = self._extract_interrupts(result)
        if interrupts:
            resumed_state = dict(pending_state)
            resumed_state.update(
                append_node_timing(
                    resumed_state,
                    node=role,
                    agent=role,
                    duration_ms=role_elapsed_ms,
                    status="interrupted",
                )
            )
            resumed_trace = [
                *resumed_state.get("trace_events", []),
                build_trace_event(
                    node=role,
                    agent=role,
                    summary="恢复后仍需继续人工审批。",
                    status="interrupted",
                    details={
                        "interrupt_count": len(interrupts),
                        "tools": [item.get("tool_label") or item.get("tool") for item in interrupts],
                    },
                ),
            ]
            updated_state = dict(resumed_state)
            updated_state["interrupts"] = interrupts
            updated_state["run_status"] = "interrupted"
            updated_state["active_agent"] = role
            updated_state["trace_events"] = resumed_trace
            self._pending_role[thread_id] = role
            self._pending_state[thread_id] = updated_state
            save_pending_conversation_state(
                thread_id=thread_id,
                user_id=user_id,
                pending_role=role,
                pending_state=updated_state,
                trace_id=trace_id,
                status="interrupted",
                last_active_agent=role,
            )
            approval = self._build_approval_payload(interrupts)
            return SupportResponse(
                message="仍有待审批动作，请继续提交决策。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="interrupted",
                interrupts=interrupts,
                active_agent=role,
                trace_id=trace_id,
                route_path=pending_state.get("route_path", []),
                validation_notes=pending_state.get("validation_notes", []),
                trace_preview=resumed_trace[-8:],
                node_timings=updated_state.get("node_timings", []),
                decision_summary=pending_state.get("decision_summary", ""),
                approval=approval,
                memory_debug=self._memory_debug_snapshot(thread_id),
                next_action=self._build_next_action(
                    run_status="interrupted",
                    thread_id=thread_id,
                    approval=approval,
                ),
                total_duration_ms=round((perf_counter() - started) * 1000, 2),
            )

        resumed_text = self._extract_ai_text(result)
        resumed_evidence_items = (
            self._extract_tool_evidence_items(result)
            if role in {"action", "escalation"}
            else []
        )
        return self._complete_resumed_turn(
            thread_id=thread_id,
            user_id=user_id,
            trace_id=trace_id,
            role=role,
            pending_state=pending_state,
            neutral_sentiment=neutral_sentiment,
            resumed_text=resumed_text,
            role_elapsed_ms=role_elapsed_ms,
            started=started,
            trace_summary="审批完成，继续执行图流程。",
            trace_details=None,
            tool_source="Support Tools" if role in {"action", "escalation"} else None,
            evidence_items=resumed_evidence_items,
        )

    def stream_chat(
        self,
        user_id: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Wrap a chat call as SSE-friendly events."""
        response = self.chat(user_id=user_id, message=message, thread_id=thread_id)
        yield {
            "type": "node",
            "thread_id": response.thread_id,
            "active_agent": response.active_agent,
            "trace_id": response.trace_id,
        }
        if response.run_status == "interrupted":
            yield {
                "type": "interrupt",
                "thread_id": response.thread_id,
                "approval": response.approval,
                "next_action": response.next_action,
            }
            yield {"type": "done", "payload": response.to_dict(include_debug=True)}
            return

        for token in response.message.split():
            yield {"type": "token", "content": token}
        yield {"type": "done", "payload": response.to_dict(include_debug=True)}

    def reindex_knowledge(self, clear_existing: bool = False) -> str:
        return reindex_knowledge_base.invoke({"clear_existing": clear_existing})

    def reset_conversation(self, user_id: str) -> None:
        with self._lock:
            to_remove = [thread for thread, owner in self._thread_user.items() if owner == user_id]
            for thread in to_remove:
                self._pending_role.pop(thread, None)
                self._pending_state.pop(thread, None)
                self._thread_user.pop(thread, None)
                self._trace_by_thread.pop(thread, None)
                self._memory_debug_by_thread.pop(thread, None)
        delete_user_conversations(user_id)

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        messages = list_user_conversation_messages(user_id=user_id, limit=max(1, limit))
        return messages


_support_agent: Optional[SupportAgent] = None
_agent_lock = threading.Lock()


def peek_support_agent() -> Optional[SupportAgent]:
    return _support_agent


def get_support_agent() -> SupportAgent:
    global _support_agent
    with _agent_lock:
        if _support_agent is None:
            _support_agent = SupportAgent()
        return _support_agent
