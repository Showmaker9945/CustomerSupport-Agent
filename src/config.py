"""CustomerSupport-Agent 的统一配置。"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用运行时配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用基础信息
    app_name: str = "CustomerSupport-Agent"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001

    # LLM 配置
    llm_provider: str = "qwen"
    llm_model: str = "qwen-plus"
    llm_high_quality_model: str = "qwen3-max"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0.7

    # API Key（保留 OPENAI_API_KEY 兼容旧环境变量）
    llm_api_key: str = Field(default="test-key")
    dashscope_api_key: str = Field(default="your_qwen_api_key_here")
    openai_api_key: str = Field(default="")

    # 跨域配置
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )

    # 业务数据库
    database_url: str = "sqlite+aiosqlite:///./data/support.db"
    postgres_uri: str = "postgresql://support_user:support_pass@localhost:5432/support?sslmode=disable"
    pgvector_enabled: bool = True

    # LangGraph 持久化
    langgraph_persistence_backend: str = "memory"  # memory | postgres
    langgraph_use_postgres: bool = False
    langgraph_thread_prefix: str = "support"
    auto_setup_postgres: bool = True

    # 向量知识库（Chroma）
    chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"))
    collection_name: str = "document_knowledge_base"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranker: bool = True
    memory_collection_name: str = "user_memory_semantic"
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # 知识库检索参数
    knowledge_base_path: Path = Field(default=Path("./data/knowledge_base"))
    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 120
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    rag_top_k: int = 4
    rag_fusion_k: int = 60
    rag_vector_weight: float = 0.55
    rag_keyword_weight: float = 0.45

    # 短期对话上下文
    transcript_backend: str = "database"
    conversation_recent_turns: int = 6
    conversation_context_messages: int = 12
    conversation_context_tokens: int = 2000
    conversation_summary_trigger_messages: int = 16
    conversation_summary_refresh_interval: int = 6

    # Agent 内部消息裁剪与长期记忆
    max_conversation_history: int = 20
    max_history_tokens: int = 3000
    session_timeout_hours: int = 24
    long_term_memory_namespace: str = "user_memory"
    max_memory_items_per_user: int = 100

    # 情绪分析
    sentiment_threshold: float = 0.3
    frustration_keywords: List[str] = Field(
        default_factory=lambda: [
            "生气",
            "愤怒",
            "投诉",
            "太差",
            "离谱",
            "angry",
            "frustrated",
            "terrible",
            "awful",
            "hate",
            "stupid",
            "useless",
        ]
    )

    # 人工接管与审批
    handoff_threshold: float = -0.5
    max_ai_turns: int = 10
    human_handoff_message: str = "我正在为你转接人工客服，请稍等。"
    hitl_high_risk_tools: List[str] = Field(
        default_factory=lambda: ["create_ticket", "update_ticket", "escalate_to_human"]
    )

    # 限流
    max_requests_per_minute: int = 60
    max_ws_connections_per_user: int = 5

    # 日志
    log_level: str = "INFO"
    log_format: str = "json"

    # LangSmith 可观测性
    langsmith_tracing: bool = False
    langsmith_otel_enabled: bool = True
    langsmith_api_key: str = Field(default="")
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "customer-support-agent"
    langsmith_workspace_id: str = Field(default="")

    # 默认语言
    default_response_language: str = "zh-CN"

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        """兼容不同环境下的 DEBUG 写法。"""
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
        """当前是否为生产环境。"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """当前是否为开发环境。"""
        return self.environment.lower() == "development"

    @property
    def use_postgres_langgraph(self) -> bool:
        """LangGraph 持久化是否应切换到 Postgres。"""
        if self.langgraph_use_postgres:
            return True
        return self.langgraph_persistence_backend.lower() == "postgres"

    @property
    def resolved_llm_api_key(self) -> str:
        """按兼容顺序解析真实可用的 LLM API Key。"""
        if self.llm_api_key and self.llm_api_key != "test-key":
            return self.llm_api_key
        if self.dashscope_api_key:
            return self.dashscope_api_key
        if self.openai_api_key:
            return self.openai_api_key
        return self.llm_api_key

    @property
    def has_valid_llm_api_key(self) -> bool:
        """判断当前是否配置了有效的模型密钥。"""
        key = (self.resolved_llm_api_key or "").strip()
        invalid_values = {
            "",
            "test-key",
            "your_qwen_api_key_here",
            "your_openai_api_key_here",
        }
        return key not in invalid_values

    @property
    def has_valid_langsmith_api_key(self) -> bool:
        key = (self.langsmith_api_key or "").strip()
        invalid_values = {"", "your_langsmith_api_key_here"}
        return key not in invalid_values

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing and self.has_valid_langsmith_api_key


@lru_cache
def get_settings() -> Settings:
    """返回缓存后的配置实例。"""
    return Settings()


# 全局配置实例
settings = get_settings()
