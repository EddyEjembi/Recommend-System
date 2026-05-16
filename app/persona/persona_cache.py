"""In-memory persona cache.

Swap in a Redis-backed implementation later; the interface should remain the
same so callers do not need to change.
"""

from app.models.schemas import Persona


class PersonaCache:
    """Simple per-process persona cache keyed by user id."""

    def __init__(self) -> None:
        self._store: dict[str, Persona] = {}

    def get(self, user_id: str) -> Persona | None:
        """Return the cached persona for `user_id`, or `None` if missing."""
        return self._store.get(user_id)

    def set(self, user_id: str, persona: Persona) -> None:
        """Cache `persona` under `user_id`."""
        self._store[user_id] = persona

    def invalidate(self, user_id: str) -> None:
        """Drop the cached persona for `user_id` if present."""
        self._store.pop(user_id, None)

    def clear(self) -> None:
        """Remove every cached persona."""
        self._store.clear()
