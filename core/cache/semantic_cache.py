import time
from typing import Dict, List, Optional
import numpy as np


class SemanticCache:
    """
    Семантичний кеш запитів.
    """

    def __init__(self, embedder, similarity_threshold: float):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

        self._entries: List[Dict] = []

    def lookup(self, query: str) -> Optional[Dict]:
        """
        Повертає кешований результат або None.
        """

        if not self._entries:
            return None

        query_emb = self.embedder.embed(query)

        best_match = None
        best_score = 0.0

        for entry in self._entries:
            score = self._cosine_similarity(query_emb, entry["embedding"])

            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.similarity_threshold:
            return best_match["result"]

        return None

    def store(self, query: str, result: Dict) -> None:
        """
        Зберігає новий результат у кеш.
        """

        self._entries.append({
            "query": query,
            "embedding": self.embedder.embed(query),
            "result": result,
            "timestamp": time.time(),
            "valid": True
        })

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
