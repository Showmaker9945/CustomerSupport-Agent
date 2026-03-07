"""Persistence backend bootstrap for LangGraph checkpointer and store."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from ...config import settings

logger = logging.getLogger(__name__)


class LangGraphPersistence:
    """Encapsulate store/checkpointer initialization and cleanup."""

    def __init__(self) -> None:
        self.checkpointer: Any = None
        self.store: Any = None
        self._checkpointer_cm: Any = None
        self._store_cm: Any = None
        self.backend = "memory"
        self._init_backend()

    def _init_backend(self) -> None:
        """Prefer Postgres and gracefully fall back to in-memory storage."""
        if settings.use_postgres_langgraph:
            try:
                self._store_cm = PostgresStore.from_conn_string(settings.postgres_uri)
                self._checkpointer_cm = PostgresSaver.from_conn_string(settings.postgres_uri)
                self.store = self._store_cm.__enter__()
                self.checkpointer = self._checkpointer_cm.__enter__()

                if settings.auto_setup_postgres:
                    with suppress(Exception):
                        self.store.setup()
                    with suppress(Exception):
                        self.checkpointer.setup()

                self.backend = "postgres"
                logger.info("LangGraph persistence backend: postgres")
                return
            except Exception as error:
                logger.warning(f"Postgres backend unavailable, fallback to memory: {error}")
                with suppress(Exception):
                    if self._store_cm:
                        self._store_cm.__exit__(None, None, None)
                with suppress(Exception):
                    if self._checkpointer_cm:
                        self._checkpointer_cm.__exit__(None, None, None)

        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.backend = "memory"
        logger.info("LangGraph persistence backend: memory")

    def close(self) -> None:
        """Release backend resources."""
        with suppress(Exception):
            if self._store_cm:
                self._store_cm.__exit__(None, None, None)
        with suppress(Exception):
            if self._checkpointer_cm:
                self._checkpointer_cm.__exit__(None, None, None)
