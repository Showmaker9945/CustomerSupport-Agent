"""Support agent service with modularized graph, middleware and public APIs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from ...config import settings
from ...sentiment.analyzer import SentimentResult, get_sentiment_analyzer
from ...tools.support_tools import (
    create_ticket,
    escalate_to_human,
    get_ticket_status,
    get_user_tickets,
    lookup_account,
    reindex_knowledge_base,
    search_faq,
    update_ticket,
)
from .graph import (
    ACCOUNT_HINTS,
    ESCALATION_HINTS,
    QUESTION_HINTS,
    REQUEST_HINTS,
    TICKET_HINTS,
    AgentRuntimeContext,
    OrchestrationState,
    SupportAgentOrchestrator,
    SupportResponse,
    as_bool,
    build_execution_steps,
    build_neutral_sentiment,
    extract_citations,
    extract_json_payload,
    infer_intent,
    infer_risk,
    merge_unique,
    normalize_intent,
    normalize_risk,
    normalize_sentiment_label,
    safe_float,
    ticket_id_from_text,
    is_positive_escalation_text,
)
from .middleware import create_role_agent
from .persistence import LangGraphPersistence

logger = logging.getLogger(__name__)


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
        self.sentiment_analyzer = get_sentiment_analyzer() if enable_sentiment else None

        self._lock = threading.Lock()
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._thread_user: Dict[str, str] = {}
        self._pending_role: Dict[str, str] = {}
        self._pending_state: Dict[str, OrchestrationState] = {}
        self._trace_by_thread: Dict[str, str] = {}

        self.basic_model: Optional[ChatOpenAI] = None
        self.advanced_model: Optional[ChatOpenAI] = None
        self.role_agents: Dict[str, Any] = {}

        if self.llm_enabled:
            self.basic_model = self._create_model(self.model_name, self.temperature)
            self.advanced_model = self._create_model(settings.llm_high_quality_model, 0.2)
            self._build_role_agents()
        else:
            logger.warning("LLM 未可用：将使用规则路由与模板回复。")

        self.orchestrator = SupportAgentOrchestrator(self)
        self.orchestration_graph = self.orchestrator.build()

    def close(self) -> None:
        """Release backend resources."""
        self.persistence.close()

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

    def _memory_namespace(self, user_id: str) -> Tuple[str, str]:
        return (settings.long_term_memory_namespace, user_id)

    def _save_memory_item(self, user_id: str, payload: Dict[str, Any]) -> None:
        if not self.enable_memory or self.persistence.store is None:
            return
        digest = hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.persistence.store.put(self._memory_namespace(user_id), digest, payload)

    def _search_memory(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enable_memory or self.persistence.store is None:
            return []
        try:
            items = self.persistence.store.search(
                self._memory_namespace(user_id),
                query=query,
                limit=limit,
            )
            return [item.value for item in items]
        except Exception as error:
            logger.warning(f"Memory search failed: {error}")
            return []

    def _build_role_agents(self) -> None:
        self.role_agents = {
            "supervisor": create_role_agent(owner=self, role="supervisor", tools=[], enable_hitl=False),
            "knowledge": create_role_agent(owner=self, role="knowledge", tools=[search_faq], enable_hitl=False),
            "action": create_role_agent(
                owner=self,
                role="action",
                tools=[
                    search_faq,
                    create_ticket,
                    update_ticket,
                    get_ticket_status,
                    get_user_tickets,
                    lookup_account,
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

    def _analyze_and_route(
        self,
        message: str,
        baseline_intent: str,
        baseline_risk: str,
        baseline_sentiment: SentimentResult,
    ) -> Dict[str, Any]:
        needs_knowledge = baseline_intent in {"question", "complaint"}
        needs_action = baseline_intent == "request" or any(
            token in message.lower() for token in self.TICKET_HINTS + self.ACCOUNT_HINTS
        )
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

    def _ticket_id_from_text(self, text: str) -> Optional[str]:
        return ticket_id_from_text(text)

    def _extract_citations(self, text: str) -> List[str]:
        return extract_citations(text)

    def _is_positive_escalation_text(self, text: str) -> bool:
        return is_positive_escalation_text(text)

    def _agent_thread_id(self, thread_id: str, role: str) -> str:
        return f"{settings.langgraph_thread_prefix}:{role}:{thread_id}"

    def _extract_interrupts(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
        payloads: List[Dict[str, Any]] = []
        for item in raw_interrupts:
            payloads.append(
                {
                    "id": getattr(item, "id", None),
                    "value": getattr(item, "value", item),
                }
            )
        return payloads

    def _extract_ai_text(self, result: Dict[str, Any]) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content)
        return "抱歉，我暂时无法给出有效回复。"

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
                return intro + search_faq.invoke({"query": message, "category": None})
            except Exception as error:
                return intro + f"知识检索失败：{error}"

        if role == "action":
            if "工单" in message and any(token in lowered for token in ("创建", "create", "open", "new")):
                return intro + create_ticket.invoke(
                    {
                        "user_id": user_id,
                        "subject": message[:80],
                        "description": message,
                        "priority": "medium",
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
                response_parts.append(retrieval_text)
            if tool_text:
                response_parts.append(tool_text)
            if response_parts:
                return intro + "\n\n".join(response_parts)
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

    def _record_history(self, user_id: str, role: str, content: str) -> None:
        with self._lock:
            self._history.setdefault(user_id, []).append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            max_items = max(settings.max_conversation_history * 4, 40)
            if len(self._history[user_id]) > max_items:
                self._history[user_id] = self._history[user_id][-max_items:]

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
        payload = {
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
        self._save_memory_item(user_id=user_id, payload=payload)

        for pattern in (r"我叫([^\s，。,.!?？!]+)", r"my name is ([a-zA-Z]+)"):
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                self._save_memory_item(
                    user_id=user_id,
                    payload={
                        "kind": "user_fact",
                        "fact": f"用户姓名可能是：{match.group(1)}",
                        "thread_id": thread_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

    def _call_role_agent(
        self,
        role: str,
        user_id: str,
        thread_id: str,
        intent: str,
        risk: str,
        message: str,
    ) -> Dict[str, Any]:
        if not self.llm_enabled or role not in self.role_agents:
            fallback = self._fallback_response(role, user_id, message, None)
            return {"messages": [AIMessage(content=fallback)]}

        agent = self.role_agents[role]
        config = {"configurable": {"thread_id": self._agent_thread_id(thread_id, role)}}
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
            sources.extend(["Hybrid RAG", "FAQ Knowledge Base"])
        if state.get("tool_text"):
            sources.append("Support Tools")
        if state.get("selected_agent") == "escalation" or state.get("escalated"):
            sources.append("Human Handoff")
        if self.llm_enabled:
            sources.append("Responder Agent")
        else:
            sources.append("Fallback Policy")
        return self._merge_unique([], sources)

    def chat(
        self,
        user_id: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> SupportResponse:
        """Process a user turn through the LangGraph workflow."""
        thread = thread_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        self._thread_user[thread] = user_id
        self._trace_by_thread[thread] = trace_id

        baseline_sentiment = (
            self.sentiment_analyzer.analyze(message)
            if self.sentiment_analyzer
            else build_neutral_sentiment()
        )

        self._record_history(user_id, "user", message)

        initial_state: OrchestrationState = {
            "user_id": user_id,
            "thread_id": thread,
            "current_message": message,
            "intent": self._infer_intent(message),
            "risk": self._infer_risk(message, baseline_sentiment),
            "sentiment_label": baseline_sentiment.label,
            "frustration_score": baseline_sentiment.frustration_score,
            "route_reason": "bootstrap",
            "selected_agent": "supervisor",
            "active_agent": "supervisor",
            "execution_steps": [],
            "needs_knowledge": False,
            "needs_action": False,
            "needs_action_after_knowledge": False,
            "needs_escalation": False,
            "retrieval_text": "",
            "tool_text": "",
            "validation_notes": [],
            "validation_passed": True,
            "final_message": "",
            "citations": [],
            "run_status": "completed",
            "interrupts": [],
            "ticket_id": None,
            "escalated": False,
        }

        try:
            final_state = self.orchestration_graph.invoke(initial_state)
        except Exception as error:
            logger.error(f"Graph invocation failed, fallback to template: {error}")
            fallback = self._fallback_response(
                role="supervisor",
                user_id=user_id,
                message=message,
                sentiment=baseline_sentiment,
            )
            self._record_history(user_id, "assistant", fallback)
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
                citations=self._extract_citations(fallback),
                active_agent=initial_state["selected_agent"],
                trace_id=trace_id,
            )

        if final_state.get("run_status") == "interrupted":
            pending_role = final_state.get("active_agent", final_state.get("selected_agent", "action"))
            self._pending_role[thread] = pending_role
            self._pending_state[thread] = dict(final_state)
            wait_message = "检测到高风险动作，已暂停执行，等待人工审批（approve/edit/reject）。"
            self._record_history(user_id, "assistant", wait_message)
            sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
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
                citations=final_state.get("citations", []),
                active_agent=pending_role,
                trace_id=trace_id,
            )

        response_message = final_state.get("final_message") or final_state.get("tool_text") or final_state.get(
            "retrieval_text"
        ) or "抱歉，我暂时无法给出有效回复。"
        citations = final_state.get("citations", [])
        ticket_id = final_state.get("ticket_id") or self._ticket_id_from_text(response_message)
        active_agent = final_state.get("selected_agent", "supervisor")
        sentiment = self._build_sentiment_result(baseline_sentiment, final_state)
        sources = self._resolve_sources(final_state)

        self._record_history(user_id, "assistant", response_message)
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
        )
    def resume(
        self,
        thread_id: str,
        decisions: List[Dict[str, Any]],
    ) -> SupportResponse:
        """Resume a HITL-interrupted thread."""
        user_id = self._thread_user.get(thread_id, "unknown_user")
        trace_id = self._trace_by_thread.get(thread_id, str(uuid.uuid4()))
        role = self._pending_role.get(thread_id)
        pending_state = self._pending_state.get(thread_id, {})
        neutral_sentiment = build_neutral_sentiment()

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
            )

        if not self.llm_enabled or role not in self.role_agents:
            self._pending_role.pop(thread_id, None)
            self._pending_state.pop(thread_id, None)
            return SupportResponse(
                message="当前处于无 LLM 模式，已跳过审批并结束流程。",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["Fallback Policy"],
                thread_id=thread_id,
                run_status="completed",
                active_agent=role,
                trace_id=trace_id,
            )

        agent = self.role_agents[role]
        config = {"configurable": {"thread_id": self._agent_thread_id(thread_id, role)}}
        context = AgentRuntimeContext(
            user_id=user_id,
            thread_id=thread_id,
            active_agent=role,
            intent=pending_state.get("intent", "resume"),
            risk=pending_state.get("risk", "medium"),
        )

        try:
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config,
                context=context,
            )
        except Exception as error:
            logger.error(f"Resume failed: {error}")
            return SupportResponse(
                message=f"恢复会话失败：{error}",
                intent="resume",
                sentiment=neutral_sentiment,
                sources=["HITL Middleware"],
                thread_id=thread_id,
                run_status="error",
                active_agent=role,
                trace_id=trace_id,
            )

        interrupts = self._extract_interrupts(result)
        if interrupts:
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
            )

        resumed_text = self._extract_ai_text(result)
        merged_state: OrchestrationState = dict(pending_state)
        merged_state["active_agent"] = role
        merged_state["run_status"] = "completed"
        merged_state["interrupts"] = []
        merged_state["citations"] = self._merge_unique(
            merged_state.get("citations", []),
            self._extract_citations(resumed_text),
        )

        if role == "knowledge":
            merged_state["retrieval_text"] = resumed_text
        if role in {"action", "escalation"}:
            merged_state["tool_text"] = resumed_text
            merged_state["ticket_id"] = merged_state.get("ticket_id") or self._ticket_id_from_text(resumed_text)
            if role == "action" and self._is_positive_escalation_text(resumed_text):
                merged_state["escalated"] = True
        if role == "escalation":
            merged_state["escalated"] = True

        merged_state = self.orchestrator.finalize_after_execution(merged_state)
        response_message = merged_state.get("final_message", resumed_text)

        self._pending_role.pop(thread_id, None)
        self._pending_state.pop(thread_id, None)
        self._record_history(user_id, "assistant", response_message)
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
            citations=merged_state.get("citations", []),
            active_agent=merged_state.get("selected_agent", role),
            trace_id=trace_id,
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
                "interrupts": response.interrupts,
            }
            yield {"type": "done", "payload": response.to_dict()}
            return

        for token in response.message.split():
            yield {"type": "token", "content": token}
        yield {"type": "done", "payload": response.to_dict()}

    def reindex_knowledge(self, clear_existing: bool = False) -> str:
        return reindex_knowledge_base.invoke({"clear_existing": clear_existing})

    def reset_conversation(self, user_id: str) -> None:
        with self._lock:
            self._history[user_id] = []
            to_remove = [thread for thread, owner in self._thread_user.items() if owner == user_id]
            for thread in to_remove:
                self._pending_role.pop(thread, None)
                self._pending_state.pop(thread, None)
                self._thread_user.pop(thread, None)
                self._trace_by_thread.pop(thread, None)

    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        with self._lock:
            history = self._history.get(user_id, [])
            return history[-max(1, limit):]


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
