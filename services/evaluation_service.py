"""
Evaluation Service
==================

Сервіс для оцінювання відповідей RAG-системи.

Відповідальність:
- виклик Evaluator
- збереження evaluation
- повернення результатів для learning / indexing

НЕ:
- не генерує відповіді
- не приймає user feedback
"""

from typing import Dict, List

from core.evaluation.evaluator import Evaluator
from core.learning.evaluation_store import EvaluationStore


class EvaluationService:
    """
    Сервіс оцінювання відповідей.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        evaluation_store: EvaluationStore
    ):
        self.evaluator = evaluator
        self.evaluation_store = evaluation_store

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Оцінює відповідь і зберігає evaluation.
        """

        evaluation = self.evaluator.evaluate(
            question=question,
            answer=answer,
            chunks=chunks
        )

        evaluation_id = self.evaluation_store.save(evaluation)

        return {
            "evaluation_id": evaluation_id,
            "metrics": evaluation["metrics"]
        }
