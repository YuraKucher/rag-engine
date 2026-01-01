from typing import Dict, List
from datetime import datetime
import uuid

from .relevance import RelevanceEvaluator
from .groundedness import GroundednessEvaluator
from .answerability import AnswerabilityEvaluator


class Evaluator:
    """
    Оркестратор evaluation layer.

    ВАЖЛИВО:
    - строго відповідає evaluation.schema.json
    - НЕ приймає рішень
    - НЕ змінює поведінку системи
    """

    def __init__(self, embedder):
        self.relevance = RelevanceEvaluator(embedder)
        self.groundedness = GroundednessEvaluator()
        self.answerability = AnswerabilityEvaluator()

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Повертає evaluation result,
        який ПОВНІСТЮ відповідає evaluation.schema.json
        """

        return {
            "evaluation_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "metrics": {
                "relevance": self.relevance.score(question, answer),
                "groundedness": self.groundedness.score(answer, chunks),
                "answerability": self.answerability.score(chunks)
            },
            "created_at": datetime.utcnow().isoformat()
        }
