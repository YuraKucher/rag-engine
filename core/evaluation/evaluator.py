from typing import Dict
from datetime import datetime
import uuid
from typing import List
from .relevance import RelevanceEvaluator
from .groundedness import GroundednessEvaluator
from .answerability import AnswerabilityEvaluator


class Evaluator:
    """
    Оркестратор evaluation layer.

    ВАЖЛИВО:
    - відповідає evaluation.schema.json
    - узгоджений з prompt-контрактом
    - не приймає рішень
    """

    def __init__(self, embedder):
        self.relevance = RelevanceEvaluator(embedder)
        self.groundedness = GroundednessEvaluator(embedder)
        self.answerability = AnswerabilityEvaluator()

    def evaluate(
            self,
            question: str,
            answer: str,
            chunks: List[Dict],
            index_ids: List[str]
    ) -> Dict:
        """
        Повертає evaluation result,
        який ПОВНІСТЮ відповідає evaluation.schema.json
        """

        relevance_score = self.relevance.score(question, answer)
        groundedness_score = self.groundedness.score(answer, chunks)
        answerability_score = self.answerability.score(answer, chunks)

        return {
            "evaluation_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,
            "index_ids": index_ids,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "metrics": {
                "relevance": relevance_score,
                "groundedness": groundedness_score,
                "answerability": answerability_score
            },
            "created_at": datetime.utcnow().isoformat()
        }