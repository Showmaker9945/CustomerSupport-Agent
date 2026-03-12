"""SQLAlchemy session helpers for business data."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_configured_url: str | None = None
_lock = Lock()


def normalize_database_url(url: str) -> str:
    """Normalize configured URLs to a sync SQLAlchemy driver."""
    normalized = (url or "").strip()
    if normalized.startswith("sqlite+aiosqlite://"):
        return normalized.replace("sqlite+aiosqlite://", "sqlite+pysqlite://", 1)
    if normalized.startswith("postgres://"):
        return normalized.replace("postgres://", "postgresql+psycopg://", 1)
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg://", 1)
    return normalized


def get_database_url() -> str:
    """Return the normalized business database URL."""
    return normalize_database_url(settings.database_url)


def _sqlite_path(url: str) -> Path | None:
    marker = "///"
    if not url.startswith("sqlite") or marker not in url:
        return None
    path_str = url.split(marker, 1)[1]
    if path_str in {":memory:", ""}:
        return None
    return Path(path_str)


def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for the active database URL."""
    global _engine, _session_factory, _configured_url

    database_url = get_database_url()
    with _lock:
        if _engine is not None and _configured_url == database_url:
            return _engine

        if _engine is not None:
            _engine.dispose()

        connect_args = {}
        sqlite_path = _sqlite_path(database_url)
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            connect_args["check_same_thread"] = False

        _engine = create_engine(
            database_url,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=not database_url.startswith("sqlite"),
        )
        _session_factory = sessionmaker(
            bind=_engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
        _configured_url = database_url
        return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a cached session factory."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session scope."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_database_connection() -> None:
    """Dispose the cached database engine/session factory."""
    global _engine, _session_factory, _configured_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
        _configured_url = None

