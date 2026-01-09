import time
from typing import List


class CacheManager:
    """
    Керує життєвим циклом кешу.
    """
    def __init__(self, semantic_cache, ttl: int):
        self.cache = semantic_cache
        self.ttl = ttl

    def cleanup(self) -> None:
        """
        Видаляє протерміновані або невалідні записи.
        """
        now = time.time()

        self.cache._entries = [
            entry for entry in self.cache._entries
            if entry["valid"] and (now - entry["timestamp"] <= self.ttl)
        ]

    def invalidate_all(self) -> None:
        """
        Повністю очищає кеш.
        """
        self.cache._entries.clear()

    def invalidate_by_query(self, query: str) -> None:
        """
        Інвалідовує кеш для конкретного запиту.
        """
        for entry in self.cache._entries:
            if entry["query"] == query:
                entry["valid"] = False
