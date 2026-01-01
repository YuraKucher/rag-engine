from typing import List, Dict


class AnswerabilityEvaluator:
    """
    Чи достатньо контексту, щоб відповісти.
    """

    def score(self, chunks: List[Dict]) -> float:
        """
        Простий сигнал:
        - якщо немає чанків → 0
        - інакше нормалізована кількість
        """

        if not chunks:
            return 0.0

        return min(1.0, len(chunks) / 5)
