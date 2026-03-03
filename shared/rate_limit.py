"""Rate limiting helpers for FastAPI."""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import settings


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.max_requests_per_minute}/minute"],
)

# Keep name expected by src.api.main
rate_limit_exception_handler = _rate_limit_exceeded_handler

