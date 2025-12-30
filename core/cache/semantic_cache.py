import time
import uuid
from typing import Optional, List

import numpy as np


class SemanticCache:
    def __init__(self, embedder, similarity_threshold: float):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

        # тимчасово — in-memory
        self.entries: dict[str, dict] = {}

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, question: str) -> Optional[dict]:
        query_emb = np.array(self.embedder.embed(question))

        best_entry = None
        best_score = 0.0

        for entry in self.entries.values():
            if not entry["is_valid"]:
                continue

            sim = self._cosine_similarity(
                query_emb,
                np.array(entry["question_embedding"])
            )

            if sim >= self.similarity_threshold and sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry:
            best_entry["last_accessed_at"] = time.time()
            return best_entry

        return None

    def store(
        self,
        question: str,
        answer: str,
        used_chunk_ids: List[str],
        collections: List[str],
        evaluation: Optional[dict] = None,
        ttl: Optional[int] = None,
    ):
        cache_id = str(uuid.uuid4())

        self.entries[cache_id] = {
            "cache_id": cache_id,
            "question": question,
            "question_embedding": self.embedder.embed(question),
            "answer": answer,
            "used_chunk_ids": used_chunk_ids,
            "collections": collections,
            "relevance_score": evaluation.get("relevance") if evaluation else None,
            "groundedness_score": evaluation.get("groundedness") if evaluation else None,
            "created_at": time.time(),
            "last_accessed_at": time.time(),
            "ttl": ttl,
            "is_valid": True,
            "invalid_reason": None,
        }

    def invalidate(self, cache_id: str, reason: str):
        if cache_id in self.entries:
            self.entries[cache_id]["is_valid"] = False
            self.entries[cache_id]["invalid_reason"] = reason
