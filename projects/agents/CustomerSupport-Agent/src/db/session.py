"""业务数据层的 SQLAlchemy 会话与连接管理。"""

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


def _ensure_postgres_connect_timeout(url: str, timeout_seconds: int = 3) -> str:
    """为 Postgres 连接补上较短的 connect_timeout，避免接口长时间挂起。"""
    if "connect_timeout=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}connect_timeout={timeout_seconds}"


def normalize_database_url(url: str) -> str:
    """将配置中的数据库地址转换为同步 SQLAlchemy 驱动格式。"""
    normalized = (url or "").strip()
    if normalized.startswith("sqlite+aiosqlite://"):
        return normalized.replace("sqlite+aiosqlite://", "sqlite+pysqlite://", 1)
    if normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql+psycopg://", 1)
        return _ensure_postgres_connect_timeout(normalized)
    if normalized.startswith("postgresql://"):
        normalized = normalized.replace("postgresql://", "postgresql+psycopg://", 1)
        return _ensure_postgres_connect_timeout(normalized)
    if normalized.startswith("postgresql+psycopg://"):
        return _ensure_postgres_connect_timeout(normalized)
    return normalized


def get_database_url() -> str:
    """返回归一化后的业务数据库地址。"""
    return normalize_database_url(settings.database_url)


def _sqlite_path(url: str) -> Path | None:
    """从 SQLite URL 中提取本地文件路径。"""
    marker = "///"
    if not url.startswith("sqlite") or marker not in url:
        return None
    path_str = url.split(marker, 1)[1]
    if path_str in {":memory:", ""}:
        return None
    return Path(path_str)


def get_engine() -> Engine:
    """返回与当前配置匹配的缓存 Engine。"""
    global _engine, _session_factory, _configured_url

    database_url = get_database_url()
    with _lock:
        if _engine is not None and _configured_url == database_url:
            return _engine

        if _engine is not None:
            _engine.dispose()

        connect_args = {}
        if database_url.startswith("sqlite"):
            # 本地 SQLite 被其他进程占用时尽快失败，避免接口长时间挂起。
            connect_args["timeout"] = 3
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
    """返回缓存的 Session 工厂。"""
    global _session_factory
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """提供一个带提交/回滚语义的事务会话上下文。"""
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
    """重置缓存的数据库连接与会话工厂。"""
    global _engine, _session_factory, _configured_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
        _configured_url = None
