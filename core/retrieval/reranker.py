from typing import List, Dict
import numpy as np

from core.learning.state_maneger import StateManager


class Reranker:
    """
    Learning-to-Rank reranker (core).

    Роль:
    - фінальний скоринг кандидатів (чанків)
    - використовує ТІЛЬКИ числові сигнали
    """

    def __init__(
            self,
            embedder,
            state_manager: StateManager
    ):
        self.embedder = embedder
        self.state = state_manager

    def rerank(
            self,
            query: str,
            chunks: List[Dict],
            top_k: int
    ) -> List[Dict]:
        if not chunks:
            return []

        query_emb = np.array(self.embedder.embed(query))
        scored = []

        for chunk in chunks:
            chunk_emb = np.array(self.embedder.embed(chunk["content"]))

            # 1. base semantic similarity
            score = self._cosine(query_emb, chunk_emb)

            # 2. learned weights
            chunk_id = chunk.get("chunk_id")
            doc_id = chunk.get("document_id")
            index_ids = chunk.get("metadata", {}).get("index_ids", [])

            score *= self.state.get_chunk_weight(chunk_id)
            score *= self.state.get_document_weight(doc_id)

            # 3. index prior (AGGREGATED, not multiplied)
            if index_ids:
                priors = [
                    self.state.get_index_prior(index_id)
                    for index_id in index_ids
                ]
                index_prior = sum(priors) / len(priors)
                score *= index_prior

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    # --------------------------------------------------
    # Допоміжний метод для обчислення косинусної схожості
    # --------------------------------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
