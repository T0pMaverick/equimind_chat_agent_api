"""
Rate limiting middleware using SlowAPI
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from app.config import settings


def get_rate_limit_key(request: Request) -> str:
    """
    Generate rate limit key based on IP address.
    Can be extended to use user ID, API key, etc.
    """
    return get_remote_address(request)


# Initialize limiter
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[
        f"{settings.rate_limit_per_minute}/minute",
        f"{settings.rate_limit_per_hour}/hour"
    ],
    storage_uri="memory://",  # Use Redis for production: "redis://localhost:6379"
)


def get_limiter():
    """Get limiter instance"""
    return limiter