from typing import List, Dict


class AnswerabilityEvaluator:
    """
    Чи достатньо контексту, щоб відповісти,
    з урахуванням контракту prompt.
    """

    FALLBACK_ANSWER = "I do not know."

    def score(self, answer: str, chunks: List[Dict]) -> float:
        """
        Логіка:
        - fallback → 0.0
        - немає чанків → 0.0
        - інакше → нормалізована кількість чанків
        """

        if answer.strip() == self.FALLBACK_ANSWER:
            return 0.0

        if not chunks:
            return 0.0

        return min(1.0, len(chunks) / 5)
