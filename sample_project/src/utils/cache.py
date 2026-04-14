"""Simple in-memory cache implementation."""

import time
from typing import Any


class Cache:
    """Basic TTL cache."""

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key, returning None if expired."""
        if key in self._store:
            value, expiry = self._store[key]
            if time.time() < expiry:
                return value
            # TODO(minor): Log cache misses for monitoring
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL override."""
        # TODO(important): Add max size limit to prevent unbounded memory growth
        expiry = time.time() + (ttl or self._default_ttl)
        self._store[key] = (value, expiry)

    def clear(self) -> None:
        """Remove all entries."""
        # TODO(minor): Add selective invalidation by prefix
        self._store.clear()
