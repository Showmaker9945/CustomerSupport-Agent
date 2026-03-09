"""
Configuration settings for CustomerSupport-Agent.

Loads settings from environment variables with sensible defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "CustomerSupport-Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001

    # LLM Configuration
    llm_provider: str = "qwen"
    llm_model: str = "qwen-plus"
    llm_high_quality_model: str = "qwen-max"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0.7

    # API Keys (keep OPENAI_API_KEY for backward compatibility)
    llm_api_key: str = Field(default="test-key")
    dashscope_api_key: str = Field(default="your_qwen_api_key_here")
    openai_api_key: str = Field(default="")

    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/support.db"
    postgres_uri: str = "postgresql://support_user:support_pass@localhost:5432/support?sslmode=disable"
    pgvector_enabled: bool = True

    # LangGraph Persistence / Store
    langgraph_persistence_backend: str = "memory"  # memory | postgres
    langgraph_use_postgres: bool = False
    langgraph_thread_prefix: str = "support"
    auto_setup_postgres: bool = True

    # Vector Store (ChromaDB)
    chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"))
    collection_name: str = "faq_knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    enable_reranker: bool = False

    # Knowledge Base
    knowledge_base_path: Path = Field(default=Path("./data/knowledge_base"))
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    rag_top_k: int = 4
    rag_fusion_k: int = 60
    rag_vector_weight: float = 0.6
    rag_keyword_weight: float = 0.4

    # Conversation Memory
    memory_type: str = "sqlite"  # Options: postgres, sqlite, memory
    max_conversation_history: int = 20
    session_timeout_hours: int = 24
    long_term_memory_namespace: str = "user_memory"
    max_memory_items_per_user: int = 100

    # Sentiment Analysis
    sentiment_threshold: float = 0.3
    frustration_keywords: List[str] = Field(
        default_factory=lambda: [
            "生气", "愤怒", "投诉", "太差", "离谱",
            "angry", "frustrated", "terrible", "awful",
            "hate", "stupid", "useless",
        ]
    )

    # Human Handoff
    handoff_threshold: float = -0.5
    max_ai_turns: int = 10
    human_handoff_message: str = (
        "我正在为你转接人工客服，请稍等。"
    )
    hitl_high_risk_tools: List[str] = Field(
        default_factory=lambda: ["create_ticket", "update_ticket", "escalate_to_human"]
    )

    # Rate Limiting
    max_requests_per_minute: int = 60
    max_ws_connections_per_user: int = 5

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Localization
    default_response_language: str = "zh-CN"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        """Support permissive DEBUG values from diverse environments."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "on", "debug", "dev", "development"}
            falsy = {"0", "false", "no", "off", "release", "prod", "production"}
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
        return bool(value)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"

    @property
    def use_postgres_langgraph(self) -> bool:
        """Whether LangGraph persistence should use Postgres."""
        if self.langgraph_use_postgres:
            return True
        return self.langgraph_persistence_backend.lower() == "postgres"

    @property
    def resolved_llm_api_key(self) -> str:
        """Resolve API key with backward-compatible fallbacks."""
        if self.llm_api_key and self.llm_api_key != "test-key":
            return self.llm_api_key
        if self.dashscope_api_key:
            return self.dashscope_api_key
        if self.openai_api_key:
            return self.openai_api_key
        return self.llm_api_key

    @property
    def has_valid_llm_api_key(self) -> bool:
        """Whether an actually usable LLM API key is configured."""
        key = (self.resolved_llm_api_key or "").strip()
        invalid_values = {
            "",
            "test-key",
            "your_qwen_api_key_here",
            "your_openai_api_key_here",
        }
        return key not in invalid_values


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
