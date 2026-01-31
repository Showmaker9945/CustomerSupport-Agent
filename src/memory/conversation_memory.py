"""
Conversation memory system for CustomerSupport-Agent.

Manages short-term conversation context and long-term user memory.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..config import settings


class ConversationMemory:
    """
    Manages conversation context for a user session.

    Handles short-term memory including messages, summaries,
    and context window management.
    """

    def __init__(
        self,
        user_id: str,
        max_messages: int = 20,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Initialize conversation memory.

        Args:
            user_id: Unique user identifier
            max_messages: Maximum messages before summarization
            llm: Optional LLM for summarization
        """
        self.user_id = user_id
        self.max_messages = max_messages
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=settings.openai_api_key,
            request_timeout=30.0
        )

        self.messages: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "turn_count": 0
        }
        self._lock = threading.Lock()

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (timestamp, sentiment, etc.)
        """
        with self._lock:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }

            self.messages.append(message)
            self.metadata["message_count"] = len(self.messages)

            if role in ["user", "assistant"]:
                self.metadata["turn_count"] += 1

            self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Check if we need to summarize
            if len(self.messages) >= self.max_messages:
                self.summarize_if_needed()

    def get_context(
        self,
        max_tokens: int = 2000,
        include_summary: bool = True
    ) -> str:
        """
        Get formatted conversation context for LLM.

        Args:
            max_tokens: Approximate token limit for context
            include_summary: Whether to include conversation summary

        Returns:
            Formatted context string
        """
        with self._lock:
            context_parts = []

            # Add summary if available
            if include_summary and self.summary:
                context_parts.append(f"[Previous Conversation Summary]\n{self.summary}\n")

            # Build context from messages
            current_tokens = 0
            estimated_tokens_per_char = 0.25  # Rough estimate

            # Collect messages in order (most recent first) without O(n²) insert
            messages_in_order = []
            for message in reversed(self.messages):
                msg_text = f"{message['role'].capitalize()}: {message['content']}\n"
                msg_tokens = len(msg_text) * estimated_tokens_per_char

                if current_tokens + msg_tokens > max_tokens:
                    break

                messages_in_order.append(msg_text)
                current_tokens += msg_tokens

            # Reverse to get correct order and append to context
            for msg_text in reversed(messages_in_order):
                context_parts.append(msg_text)

            return "\n".join(context_parts) if context_parts else "No conversation history yet."

    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """
        Get the most recent messages.

        Args:
            count: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
        with self._lock:
            return self.messages[-count:] if self.messages else []

    def summarize_if_needed(self, force: bool = False) -> Optional[str]:
        """
        Generate conversation summary if message count exceeds threshold.

        Args:
            force: Force summarization regardless of message count

        Returns:
            Generated summary or None if not needed
        """
        with self._lock:
            # Double-check inside lock to prevent race condition
            if not force and len(self.messages) < self.max_messages:
                return None

            try:
                # Build conversation text for summarization
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.messages
                ])

                # Generate summary
                prompt = f"""Summarize the following customer support conversation concisely.
Focus on:
1. The customer's issue or question
2. What has been discussed/tried
3. Current status or next steps
4. Any important details mentioned

Conversation:
{conversation_text}

Summary:"""

                response = self.llm.invoke(prompt)
                new_summary = response.content.strip()

                # Append to existing summary if we have one
                if self.summary:
                    self.summary = f"{self.summary}\n\n[Update]\n{new_summary}"
                else:
                    self.summary = new_summary

                # Keep only recent messages after summarization
                keep_count = self.max_messages // 4  # Keep last 25% of messages
                self.messages = self.messages[-keep_count:] if keep_count > 0 else []

                return self.summary

            except Exception as e:
                # If summarization fails, just truncate messages
                keep_count = self.max_messages // 2
                self.messages = self.messages[-keep_count:] if keep_count > 0 else []
                return None

    def clear(self) -> None:
        """Clear all messages and summary."""
        with self._lock:
            self.messages.clear()
            self.summary = None
            self.metadata["last_updated"] = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary for storage.

        Returns:
            Dictionary representation of memory
        """
        with self._lock:
            return {
                "user_id": self.user_id,
                "messages": self.messages.copy(),  # Copy to prevent external mutation
                "summary": self.summary,
                "metadata": self.metadata.copy()  # Copy to prevent external mutation
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm: Optional[ChatOpenAI] = None) -> "ConversationMemory":
        """
        Create ConversationMemory from dictionary.

        Args:
            data: Dictionary representation
            llm: Optional LLM instance

        Returns:
            ConversationMemory instance
        """
        memory = cls(
            user_id=data["user_id"],
            llm=llm
        )
        memory.messages = data.get("messages", [])
        memory.summary = data.get("summary")
        memory.metadata = data.get("metadata", memory.metadata)
        return memory


class UserMemoryStore:
    """
    Long-term memory storage for user profiles and conversation history.

    Provides persistent storage for user preferences, past conversations,
    and searchable history.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize user memory store.

        Args:
            storage_path: Path to store user data
        """
        self.storage_path = storage_path or Path("./data/user_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_user_path(self, user_id: str) -> Path:
        """Get file path for user data."""
        return self.storage_path / f"{user_id}.json"

    def _load_user_data(self, user_id: str) -> Dict[str, Any]:
        """Load user data from disk."""
        path = self._get_user_path(user_id)

        if not path.exists():
            return {
                "user_id": user_id,
                "profile": self._default_profile(),
                "conversations": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                "user_id": user_id,
                "profile": self._default_profile(),
                "conversations": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

    def _save_user_data(self, user_id: str, data: Dict[str, Any]) -> None:
        """Save user data to disk."""
        path = self._get_user_path(user_id)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _default_profile(self) -> Dict[str, Any]:
        """Create default user profile."""
        return {
            "name": None,
            "email": None,
            "preferences": {
                "language": "en",
                "timezone": "UTC"
            },
            "issues": [],  # Past issues for quick reference
            "sentiments": {  # Track sentiment over time
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "tags": [],  # Custom tags for categorization
            "notes": ""  # Support agent notes
        }

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's stored profile and preferences.

        Args:
            user_id: Unique user identifier

        Returns:
            User profile dictionary
        """
        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = self._load_user_data(user_id)

            return self._cache[user_id].get("profile", self._default_profile()).copy()

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> None:
        """
        Update user profile information.

        Args:
            user_id: Unique user identifier
            updates: Dictionary of profile fields to update
        """
        with self._lock:
            data = self._load_user_data(user_id)

            # Deep merge updates
            for key, value in updates.items():
                if isinstance(value, dict) and key in data["profile"]:
                    data["profile"][key].update(value)
                else:
                    data["profile"][key] = value

            self._save_user_data(user_id, data)
            self._cache[user_id] = data

    def save_conversation(
        self,
        user_id: str,
        conversation: ConversationMemory,
        title: Optional[str] = None
    ) -> str:
        """
        Save conversation summary to user history.

        Args:
            user_id: Unique user identifier
            conversation: ConversationMemory instance
            title: Optional title for the conversation

        Returns:
            Conversation ID (timestamp-based)
        """
        with self._lock:
            data = self._load_user_data(user_id)

            conv_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            conv_record = {
                "id": conv_id,
                "title": title or self._generate_title(conversation),
                "date": datetime.now(timezone.utc).isoformat(),
                "summary": conversation.summary,
                "metadata": conversation.metadata,
                "message_count": len(conversation.messages)
            }

            # Add to front of conversations list
            data["conversations"].insert(0, conv_record)

            # Keep only last 50 conversations
            data["conversations"] = data["conversations"][:50]

            self._save_user_data(user_id, data)
            self._cache[user_id] = data

            return conv_id

    def _generate_title(self, conversation: ConversationMemory) -> str:
        """Generate a title for the conversation based on first message."""
        if not conversation.messages:
            return "New Conversation"

        first_user_msg = next(
            (msg for msg in conversation.messages if msg["role"] == "user"),
            None
        )

        if first_user_msg:
            content = first_user_msg["content"]
            # Truncate to reasonable length
            return content[:50] + "..." if len(content) > 50 else content

        return "Conversation"

    def get_conversation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation summaries.

        Args:
            user_id: Unique user identifier
            limit: Maximum number of conversations to return

        Returns:
            List of conversation records
        """
        with self._lock:
            data = self._load_user_data(user_id)
            return data.get("conversations", [])[:limit]

    def search_history(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user's conversation history.

        Args:
            user_id: Unique user identifier
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching conversation records
        """
        with self._lock:
            data = self._load_user_data(user_id)
            conversations = data.get("conversations", [])

            query_lower = query.lower()
            results = []

            for conv in conversations:
                # Search in title and summary
                title_match = query_lower in conv.get("title", "").lower()
                summary_match = query_lower in (conv.get("summary") or "").lower()

                if title_match or summary_match:
                    results.append(conv)
                    if len(results) >= limit:
                        break

            return results

    def update_sentiment_tracking(
        self,
        user_id: str,
        sentiment: str
    ) -> None:
        """
        Update sentiment tracking for user.

        Args:
            user_id: Unique user identifier
            sentiment: Sentiment category (positive, neutral, negative)
        """
        with self._lock:
            data = self._load_user_data(user_id)
            sentiments = data["profile"]["sentiments"]

            if sentiment in sentiments:
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

            self._save_user_data(user_id, data)
            self._cache[user_id] = data

    def add_issue_to_history(
        self,
        user_id: str,
        issue: str,
        resolved: bool = False
    ) -> None:
        """
        Add an issue to user's issue history.

        Args:
            user_id: Unique user identifier
            issue: Issue description
            resolved: Whether the issue was resolved
        """
        with self._lock:
            data = self._load_user_data(user_id)
            issue_record = {
                "description": issue,
                "resolved": resolved,
                "date": datetime.now(timezone.utc).isoformat()
            }

            # Avoid duplicates
            existing = data["profile"]["issues"]
            if not any(i["description"] == issue for i in existing):
                existing.insert(0, issue_record)
                # Keep last 20 issues
                data["profile"]["issues"] = existing[:20]

            self._save_user_data(user_id, data)
            self._cache[user_id] = data

    def get_user_context(self, user_id: str) -> str:
        """
        Get formatted user context for LLM.

        Args:
            user_id: Unique user identifier

        Returns:
            Formatted context string
        """
        profile = self.get_user_profile(user_id)
        recent_convs = self.get_conversation_history(user_id, limit=3)

        context_parts = []

        # Add profile info
        if profile.get("name"):
            context_parts.append(f"Customer: {profile['name']}")

        if profile.get("email"):
            context_parts.append(f"Email: {profile['email']}")

        # Add recent issues
        if profile.get("issues"):
            context_parts.append("\nRecent Issues:")
            for issue in profile["issues"][:5]:
                status = "✓" if issue["resolved"] else "○"
                context_parts.append(f"  {status} {issue['description']}")

        # Add conversation titles
        if recent_convs:
            context_parts.append("\nPrevious Conversations:")
            for conv in recent_convs[:3]:
                context_parts.append(f"  - {conv['title']} ({conv['date'][:10]})")

        return "\n".join(context_parts) if context_parts else "New customer"
