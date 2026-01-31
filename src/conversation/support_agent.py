"""
Customer support conversation agent using LangGraph.

Integrates memory, knowledge base, tools, and sentiment analysis
for intelligent multi-turn conversations.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from ..config import settings
from ..knowledge.faq_store import FAQStore, create_faq_store
from ..memory.conversation_memory import ConversationMemory, UserMemoryStore
from ..sentiment.analyzer import SentimentAnalyzer, SentimentResult, get_sentiment_analyzer
from ..tools.support_tools import (
    ALL_TOOLS,
    create_ticket,
    get_ticket_status,
    update_ticket,
    get_user_tickets,
    lookup_account,
    escalate_to_human,
    search_faq,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class ConversationState(TypedDict):
    """State for conversation graph."""
    user_id: str
    messages: List[Dict[str, str]]
    current_message: str
    intent: Optional[str]
    sentiment: Optional[SentimentResult]
    faq_results: Optional[List[str]]
    tool_result: Optional[str]
    response: Optional[str]
    escalated: bool
    ticket_id: Optional[str]
    context: Optional[str]


@dataclass
class SupportResponse:
    """Response from the support agent."""
    message: str
    intent: str
    sentiment: SentimentResult
    sources: List[str] = field(default_factory=list)
    escalated: bool = False
    ticket_created: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "intent": self.intent,
            "sentiment": {
                "label": self.sentiment.label,
                "polarity": self.sentiment.polarity,
                "frustration_score": self.sentiment.frustration_score
            },
            "sources": self.sources,
            "escalated": self.escalated,
            "ticket_created": self.ticket_created
        }


# ============================================================================
# SUPPORT AGENT
# ============================================================================

class SupportAgent:
    """
    Customer support conversational agent.

    Uses LangGraph for conversation flow with sentiment-aware routing,
    knowledge base search, and tool execution.
    """

    # System prompt for the agent
    SYSTEM_PROMPT = """You are a helpful, empathetic customer support agent for a SaaS company.

Your role is to assist customers with:
- Answering questions about the product/service
- Resolving technical issues
- Handling billing and account inquiries
- Creating support tickets when needed
- Escalating to human agents when appropriate

Guidelines:
- Be friendly, professional, and empathetic
- Acknowledge the customer's feelings, especially if they're frustrated
- Use the customer's name when available
- Provide clear, actionable answers
- If you don't know something, admit it and offer to find out
- Always maintain a calm, helpful tone even with upset customers
- Create tickets for issues that need follow-up
- Escalate immediately for very angry or frustrated customers

When responding:
1. Address the customer's immediate concern
2. Provide relevant information from your knowledge base
3. Offer additional help
4. Be concise but thorough"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        enable_memory: bool = True,
        enable_sentiment: bool = True
    ):
        """
        Initialize the support agent.

        Args:
            model_name: OpenAI model to use
            temperature: Response randomness (0-1)
            enable_memory: Whether to use conversation memory
            enable_sentiment: Whether to use sentiment analysis
        """
        self.model_name = model_name
        self.temperature = temperature
        self.enable_memory = enable_memory
        self.enable_sentiment = enable_sentiment

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key,
            request_timeout=30.0
        )

        # Initialize components
        self.faq_store = create_faq_store()
        self.sentiment_analyzer = get_sentiment_analyzer() if enable_sentiment else None
        self.user_memory_store = UserMemoryStore()

        # State persistence path
        self.state_path = Path("./data/conversation_state")
        self.state_path.mkdir(parents=True, exist_ok=True)

        # Conversation memory per user (with thread-safe lock)
        self.memory: Dict[str, ConversationMemory] = {}
        self._memory_lock = threading.Lock()

        # Load previous state if enabled
        if enable_memory:
            self._load_state()

        # Build conversation graph
        self.graph = self._build_graph()

        logger.info(f"Initialized SupportAgent with model: {model_name}")

    def _get_user_state_path(self, user_id: str) -> Path:
        """Get the file path for a user's active conversation state."""
        return self.state_path / f"{user_id}_active.json"

    def _save_state(self) -> None:
        """Save active conversation states to disk."""
        try:
            state_data = {}
            for user_id, memory in self.memory.items():
                state_data[user_id] = {
                    "memory": memory.to_dict(),
                    "saved_at": datetime.now(timezone.utc).isoformat()
                }

            state_file = self.state_path / "active_conversations.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved state for {len(self.memory)} active users")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load active conversation states from disk."""
        state_file = self.state_path / "active_conversations.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            for user_id, data in state_data.items():
                memory_data = data.get("memory", {})
                if memory_data:
                    memory = ConversationMemory.from_dict(memory_data, llm=self.llm)
                    self.memory[user_id] = memory

            logger.info(f"Loaded state for {len(self.memory)} active users")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def _get_or_create_memory(self, user_id: str) -> ConversationMemory:
        """Get or create conversation memory for user (thread-safe)."""
        with self._memory_lock:
            if user_id not in self.memory:
                self.memory[user_id] = ConversationMemory(
                    user_id=user_id,
                    llm=self.llm
                )
            return self.memory[user_id]

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build the LangGraph conversation flow.

        Flow:
        1. understand_intent - Classify user's intent
        2. check_escalation - Check if immediate escalation needed
        3. search_knowledge - Search FAQ for questions
        4. use_tool - Execute relevant tools
        5. generate_response - Create final response
        """
        # Create graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("understand_intent", self._understand_intent)
        workflow.add_node("check_escalation", self._check_escalation)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("use_tool", self._use_tool)
        workflow.add_node("generate_response", self._generate_response)

        # Set entry point
        workflow.set_entry_point("understand_intent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "understand_intent",
            self._route_after_intent,
            {
                "escalate": "check_escalation",
                "search": "search_knowledge",
                "tool": "use_tool",
                "respond": "generate_response"
            }
        )

        workflow.add_conditional_edges(
            "check_escalation",
            self._route_after_escalation_check,
            {
                "escalate": "generate_response",
                "continue": "search_knowledge"
            }
        )

        # Knowledge and tool both lead to response
        workflow.add_edge("search_knowledge", "generate_response")
        workflow.add_edge("use_tool", "generate_response")

        # Response is the end
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _understand_intent(self, state: ConversationState) -> dict:
        """
        Classify user intent and sentiment.

        Returns:
            Updated state with intent and sentiment
        """
        message = state["current_message"]

        # Analyze sentiment
        sentiment = None
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.analyze(message)

        # Classify intent using LLM
        intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Classify the customer's intent into one of these categories:

- question: Asking for information, how-to, explanation
- complaint: Expressing dissatisfaction, reporting a problem
- request: Asking for action (create ticket, update account, etc.)
- feedback: Providing opinions or suggestions
- greeting: Saying hello, thanks, goodbye
- other: Doesn't fit other categories

Return only the category name."""),
            HumanMessage(content=message)
        ])

        try:
            intent_response = self.llm.invoke(intent_prompt.format_messages())
            intent = intent_response.content.strip().lower()
            # Ensure valid intent
            valid_intents = ["question", "complaint", "request", "feedback", "greeting", "other"]
            intent = intent if intent in valid_intents else "other"
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            intent = "other"

        # Get user context
        context = None
        if self.enable_memory:
            user_context = self.user_memory_store.get_user_context(state["user_id"])
            if user_context:
                memory = self._get_or_create_memory(state["user_id"])
                context = f"{user_context}\n\n{memory.get_context(max_tokens=1000)}"

        return {
            "intent": intent,
            "sentiment": sentiment,
            "context": context
        }

    def _check_escalation(self, state: ConversationState) -> dict:
        """
        Check if conversation should be escalated to human.

        Returns:
            Updated state with escalation flag
        """
        sentiment = state.get("sentiment")
        user_id = state["user_id"]

        escalated = False
        reason = None

        if sentiment and sentiment.frustration_score >= 0.8:
            escalated = True
            reason = f"High frustration detected (score: {sentiment.frustration_score:.2f})"

        # Check conversation history for escalation pattern
        if not escalated and self.enable_memory:
            memory = self._get_or_create_memory(user_id)
            recent_messages = memory.get_recent_messages(count=5)

            # Count negative sentiment messages
            if self.sentiment_analyzer and recent_messages:
                negative_count = 0
                for msg in recent_messages:
                    if msg["role"] == "user":
                        result = self.sentiment_analyzer.analyze(msg["content"])
                        if result.label == "negative" or result.frustration_score > 0.5:
                            negative_count += 1

                if negative_count >= 3:
                    escalated = True
                    reason = f"Multiple frustrated messages ({negative_count} out of {len(recent_messages)})"

        # Create escalation ticket if needed
        ticket_id = None
        if escalated:
            memory = self._get_or_create_memory(user_id)
            summary = memory.get_context(max_tokens=500)

            try:
                result = escalate_to_human.invoke({
                    "user_id": user_id,
                    "reason": reason or "High customer frustration",
                    "conversation_summary": summary
                })
                # Extract ticket ID from result
                if "TKT-" in result:
                    ticket_id = result.split("TKT-")[1].split("\n")[0].strip()
            except Exception as e:
                logger.error(f"Escalation error: {e}")

        return {
            "escalated": escalated,
            "ticket_id": ticket_id
        }

    def _search_knowledge(self, state: ConversationState) -> dict:
        """
        Search FAQ knowledge base.

        Returns:
            Updated state with FAQ results
        """
        message = state["current_message"]
        intent = state.get("intent", "other")

        # Search FAQ
        try:
            faq_result = search_faq.invoke({
                "query": message,
                "category": None
            })
        except Exception as e:
            logger.error(f"FAQ search error: {e}")
            faq_result = "No FAQ results available."

        return {
            "faq_results": [faq_result]
        }

    def _use_tool(self, state: ConversationState) -> dict:
        """
        Execute appropriate tool based on intent.

        Returns:
            Updated state with tool result
        """
        message = state["current_message"]
        intent = state.get("intent", "other")
        user_id = state["user_id"]

        tool_result = None
        ticket_id = None

        try:
            if intent == "request":
                # Determine what kind of request
                message_lower = message.lower()

                if "ticket" in message_lower and ("status" in message_lower or "check" in message_lower):
                    # Check ticket status
                    # Try to extract ticket ID
                    import re
                    ticket_match = re.search(r'TKT-\d+-\d+', message, re.IGNORECASE)
                    if ticket_match:
                        tool_result = get_ticket_status.invoke({"ticket_id": ticket_match.group()})
                    else:
                        tool_result = get_user_tickets.invoke({"user_id": user_id, "status": None})

                elif "create" in message_lower or "new" in message_lower or "open" in message_lower:
                    # Create new ticket
                    # Extract subject and description from message
                    subject = message[:100]
                    description = message
                    tool_result = create_ticket.invoke({
                        "user_id": user_id,
                        "subject": subject,
                        "description": description,
                        "priority": "medium"
                    })
                    # Extract ticket ID
                    if "TKT-" in tool_result:
                        ticket_id = tool_result.split("TKT-")[1].split("\n")[0].strip()

                elif "account" in message_lower or "profile" in message_lower:
                    # Lookup account
                    tool_result = lookup_account.invoke({"user_id": user_id})

                else:
                    # Generic response for other requests
                    tool_result = "I understand you need assistance. Could you provide more details about what you'd like me to help you with?"

            else:
                # For other intents, provide helpful response
                tool_result = "I'm here to help. Could you please provide more details about your inquiry?"

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            tool_result = "I apologize, but I encountered an error while trying to help you. Let me connect you with a human agent."

        return {
            "tool_result": tool_result,
            "ticket_id": ticket_id or state.get("ticket_id")
        }

    def _generate_response(self, state: ConversationState) -> dict:
        """
        Generate final response using LLM.

        Returns:
            Updated state with response message
        """
        user_id = state["user_id"]
        message = state["current_message"]
        intent = state.get("intent", "other")
        sentiment = state.get("sentiment")
        escalated = state.get("escalated", False)
        faq_results = state.get("faq_results", [])
        tool_result = state.get("tool_result")
        context = state.get("context")

        # Build response prompt
        response_parts = [self.SYSTEM_PROMPT]

        # Add user context
        if context:
            response_parts.append(f"\nCustomer Context:\n{context}")

        # Add conversation history
        if self.enable_memory:
            memory = self._get_or_create_memory(user_id)
            recent_context = memory.get_context(max_tokens=1500)
            if recent_context:
                response_parts.append(f"\nRecent Conversation:\n{recent_context}")

        # Add FAQ results if available
        if faq_results and faq_results[0]:
            response_parts.append(f"\nRelevant Information from Knowledge Base:\n{faq_results[0]}")

        # Add tool result if available
        if tool_result:
            response_parts.append(f"\nTool Results:\n{tool_result}")

        # Add sentiment info
        if sentiment:
            response_parts.append(f"\nCustomer Sentiment: {sentiment.label} (frustration: {sentiment.frustration_score:.2f})")

        # Add escalation info
        if escalated:
            response_parts.append("\nNOTE: This conversation has been escalated to a human agent.")

        # Add current message
        response_parts.append(f"\nCustomer's Message: {message}")

        # Create prompt
        response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="\n".join(response_parts)),
            HumanMessage(content="Please provide a helpful response to the customer.")
        ])

        try:
            # Generate response
            response = self.llm.invoke(response_prompt.format_messages())
            ai_message = response.content.strip()

            # If escalated, add escalation note
            if escalated:
                ticket_info = f"\n\n[Reference: {state.get('ticket_id', 'N/A')}]"
                ai_message += ticket_info

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            ai_message = "I apologize, but I'm having trouble generating a response. A human agent will be with you shortly."

        return {
            "response": ai_message
        }

    def _route_after_intent(self, state: ConversationState) -> str:
        """
        Determine next step after intent classification.

        Returns:
            Next node name
        """
        intent = state.get("intent", "other")
        sentiment = state.get("sentiment")

        # Immediate escalation for high frustration
        if sentiment and sentiment.frustration_score >= 0.8:
            return "escalate"

        # Route based on intent
        if intent == "question":
            return "search"
        elif intent == "request":
            return "tool"
        elif intent == "complaint":
            # Complaints might need FAQ search or escalation
            if sentiment and sentiment.frustration_score >= 0.5:
                return "escalate"
            return "search"
        elif intent in ["greeting", "feedback"]:
            return "respond"
        else:
            return "respond"

    def _route_after_escalation_check(self, state: ConversationState) -> str:
        """
        Determine next step after escalation check.

        Returns:
            Next node name
        """
        if state.get("escalated", False):
            return "escalate"
        return "continue"

    def chat(
        self,
        user_id: str,
        message: str
    ) -> SupportResponse:
        """
        Process a user message and return response.

        Args:
            user_id: User identifier
            message: User's message

        Returns:
            SupportResponse with message and metadata
        """
        logger.info(f"Processing message from user {user_id}: {message[:50]}...")

        # Update conversation memory
        if self.enable_memory:
            memory = self._get_or_create_memory(user_id)
            memory.add_message("user", message)

        # Prepare initial state
        initial_state: ConversationState = {
            "user_id": user_id,
            "messages": [],
            "current_message": message,
            "intent": None,
            "sentiment": None,
            "faq_results": None,
            "tool_result": None,
            "response": None,
            "escalated": False,
            "ticket_id": None,
            "context": None
        }

        # Run conversation graph
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Conversation graph error: {e}")
            # Fallback response
            return SupportResponse(
                message="I apologize, but I encountered an error. A human agent will assist you shortly.",
                intent="error",
                sentiment=SentimentResult(
                    polarity=0.0,
                    subjectivity=0.0,
                    label="neutral",
                    frustration_score=0.0,
                    keywords=[]
                ),
                escalated=True
            )

        # Extract results
        response_message = final_state.get("response", "I'm sorry, I couldn't generate a response.")
        intent = final_state.get("intent", "other")
        sentiment = final_state.get("sentiment") or SentimentResult(
            polarity=0.0,
            subjectivity=0.0,
            label="neutral",
            frustration_score=0.0,
            keywords=[]
        )
        escalated = final_state.get("escalated", False)
        ticket_id = final_state.get("ticket_id")

        # Gather sources
        sources = []
        if final_state.get("faq_results"):
            sources.append("FAQ Knowledge Base")
        if final_state.get("tool_result"):
            sources.append("Support Tools")

        # Update memory with AI response
        if self.enable_memory:
            memory = self._get_or_create_memory(user_id)
            memory.add_message("assistant", response_message)

            # Update user sentiment tracking
            if self.sentiment_analyzer:
                self.user_memory_store.update_sentiment_tracking(
                    user_id,
                    sentiment.label
                )

            # Periodically summarize if conversation is long
            if len(memory.messages) >= memory.max_messages:
                memory.summarize_if_needed()

        # Create response
        support_response = SupportResponse(
            message=response_message,
            intent=intent,
            sentiment=sentiment,
            sources=sources,
            escalated=escalated,
            ticket_created=ticket_id
        )

        logger.info(
            f"Response generated for user {user_id}: "
            f"intent={intent}, sentiment={sentiment.label}, "
            f"escalated={escalated}"
        )

        return support_response

    def reset_conversation(self, user_id: str) -> None:
        """
        Reset conversation for a user.

        Args:
            user_id: User identifier
        """
        with self._memory_lock:
            if user_id in self.memory:
                self.memory[user_id].clear()
                logger.info(f"Reset conversation for user {user_id}")

    def get_conversation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a user.

        Args:
            user_id: User identifier
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        memory = self._get_or_create_memory(user_id)
        return memory.get_recent_messages(count=limit)


# Global agent instance
_support_agent: SupportAgent = None
_agent_lock = threading.Lock()


def get_support_agent() -> SupportAgent:
    """Get or create global support agent instance."""
    global _support_agent
    with _agent_lock:
        if _support_agent is None:
            _support_agent = SupportAgent()
        return _support_agent
