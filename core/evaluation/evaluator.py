"""
Evaluator — оцінювання якості відповіді.
"""


class Evaluator:
    def evaluate(self, question: str, answer: str, context: list[dict]) -> dict:
        """
        Returns evaluation metrics according to evaluation.schema.json
        """
        raise NotImplementedError
