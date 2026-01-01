from typing import Dict
import numpy as np


class RelevanceEvaluator:
    """
    Оцінює релевантність відповіді до питання.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, question: str, answer: str) -> float:
        """
        Cosine similarity між embedding питання і відповіді.
        """

        q_emb = self.embedder.embed(question)
        a_emb = self.embedder.embed(answer)

        similarity = float(
            np.dot(q_emb, a_emb) /
            (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        )

        return round(similarity, 3)
