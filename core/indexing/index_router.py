from typing import List, Dict
import numpy as np


class SemanticIndexRouter:
    """
    Semantic router між індексами.

    Визначає, які індекси релевантні запиту
    ДО запуску FAISS retrieval.
    """

    def __init__(
        self,
        embedder,
        index_registry,
        similarity_threshold: float = 0.35,
        top_k: int = 3
    ):
        self.embedder = embedder
        self.index_registry = index_registry
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    # -----------------------------------------------------

    def route(self, query: str) -> List[Dict]:
        """
        Повертає metadata індексів,
        які семантично релевантні запиту.
        """

        query_emb = np.array(self.embedder.embed(query))

        scored_indexes = []

        for meta in self.index_registry.list_indexes():
            rep_text = self._representative_text(meta)
            if not rep_text:
                continue

            index_emb = np.array(self.embedder.embed(rep_text))

            score = self._cosine_similarity(query_emb, index_emb)

            if score >= self.similarity_threshold:
                scored_indexes.append((score, meta))

        scored_indexes.sort(key=lambda x: x[0], reverse=True)

        return [meta for _, meta in scored_indexes[: self.top_k]]

    # -----------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.dot(a, b) /
            (np.linalg.norm(a) * np.linalg.norm(b))
        )

    @staticmethod
    def _representative_text(index_meta: Dict) -> str:
        """
        Представницький текст індексу.
        Мінімальна, але стабільна евристика.
        """

        # 1️⃣ Якщо буде summary — беремо його
        if "summary" in index_meta:
            return index_meta["summary"]

        # 2️⃣ fallback: назва документа або index_id
        if index_meta.get("document_ids"):
            return " ".join(index_meta["document_ids"])

        return index_meta.get("index_id", "")
