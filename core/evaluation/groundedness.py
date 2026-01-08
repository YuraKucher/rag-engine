from typing import List, Dict
import numpy as np


class GroundednessEvaluator:
    """
    Перевіряє, наскільки відповідь опирається на наданий контекст,
    згідно з контрактом prompt.
    """

    FALLBACK_ANSWER = "I do not know."

    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, answer: str, chunks: List[Dict]) -> float:
        """
        Логіка:
        - fallback → 1.0 (коректна поведінка за prompt)
        - немає чанків → 0.0
        - інакше → max cosine similarity між відповіддю і чанками
        """

        if answer.strip() == self.FALLBACK_ANSWER:
            return 1.0

        if not chunks:
            return 0.0

        answer_emb = self.embedder.embed(answer)
        max_similarity = 0.0

        for chunk in chunks:
            chunk_emb = self.embedder.embed(chunk["content"])
            similarity = float(
                np.dot(answer_emb, chunk_emb) /
                (np.linalg.norm(answer_emb) * np.linalg.norm(chunk_emb))
            )
            max_similarity = max(max_similarity, similarity)

        return round(max_similarity, 3)
