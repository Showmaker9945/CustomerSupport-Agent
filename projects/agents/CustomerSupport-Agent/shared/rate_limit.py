"""Rate limiting helpers for FastAPI.

This module degrades gracefully when ``slowapi`` is not installed so
application startup does not fail in local/dev environments.
"""

from fastapi.responses import JSONResponse

from src.config import settings

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.max_requests_per_minute}/minute"],
    )

    # Keep name expected by src.api.main
    rate_limit_exception_handler = _rate_limit_exceeded_handler
except ImportError:
    class RateLimitExceeded(Exception):
        """Fallback exception when slowapi is unavailable."""

    class _NoopLimiter:
        """No-op limiter fallback used when slowapi is not installed."""

        def limit(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

    limiter = _NoopLimiter()

    async def rate_limit_exception_handler(_request, _exc):
        """Fallback handler that returns a 429 payload."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": f"Max {settings.max_requests_per_minute} requests per minute",
            },
        )
