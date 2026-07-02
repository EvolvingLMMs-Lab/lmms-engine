from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small name-to-factory registry for RL components."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str, value: Any | None = None):
        if value is not None:
            self._items[key] = value
            return value

        def decorator(obj):
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}") from exc

    def build(self, key: str, *args, **kwargs) -> T:
        factory = self.get(key)
        return factory(*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)
