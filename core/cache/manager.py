import time


class CacheManager:
    def __init__(self, semantic_cache):
        self.cache = semantic_cache

    def invalidate_by_collection(self, collection_name: str):
        for entry in self.cache.entries.values():
            if collection_name in entry["collections"]:
                entry["is_valid"] = False
                entry["invalid_reason"] = "collection_updated"

    def cleanup_expired(self):
        now = time.time()

        for entry in self.cache.entries.values():
            if entry["ttl"] is None:
                continue

            if now - entry["created_at"] > entry["ttl"]:
                entry["is_valid"] = False
                entry["invalid_reason"] = "ttl_expired"
