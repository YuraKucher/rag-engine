from typing import Dict
import numpy as np


class RelevanceEvaluator:
    """
    Оцінює релевантність відповіді до питання
    з урахуванням контракту prompt.
    """

    FALLBACK_ANSWER = "I do not know."

    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, question: str, answer: str) -> float:
        """
        Логіка:
        - fallback → 1.0 (коректна відповідь на питання)
        - інакше → cosine similarity між question і answer
        """

        if answer.strip() == self.FALLBACK_ANSWER:
            return 1.0

        q_emb = self.embedder.embed(question)
        a_emb = self.embedder.embed(answer)

        similarity = float(
            np.dot(q_emb, a_emb) /
            (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        )

        return round(similarity, 3)
