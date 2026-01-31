"""
Configuration settings for CustomerSupport-Agent.

Loads settings from environment variables with sensible defaults.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
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

    # API Keys
    openai_api_key: str = Field(default="test-key")

    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/support.db"

    # Vector Store (ChromaDB)
    chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"))
    collection_name: str = "faq_knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Knowledge Base
    knowledge_base_path: Path = Field(default=Path("./data/knowledge_base"))
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3

    # Conversation Memory
    memory_type: str = "sqlite"  # Options: postgres, sqlite, memory
    max_conversation_history: int = 20
    session_timeout_hours: int = 24

    # Sentiment Analysis
    sentiment_threshold: float = 0.3
    frustration_keywords: List[str] = Field(
        default_factory=lambda: [
            "angry", "frustrated", "terrible", "awful",
            "hate", "stupid", "useless"
        ]
    )

    # Human Handoff
    handoff_threshold: float = -0.5
    max_ai_turns: int = 10
    human_handoff_message: str = (
        "I'm connecting you with a human agent who can better assist you."
    )

    # Rate Limiting
    max_requests_per_minute: int = 60
    max_ws_connections_per_user: int = 5

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
