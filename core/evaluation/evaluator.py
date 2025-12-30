class EvaluationResult(dict):
    pass


class Evaluator:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: list[str],
    ) -> EvaluationResult:

        relevance = self._relevance(question, answer)
        groundedness = self._groundedness(answer, chunks)
        context_rel = self._context_quality(question, chunks)

        passed = (
            relevance >= self.thresholds["relevance"]
            and groundedness >= self.thresholds["groundedness"]
        )

        return EvaluationResult({
            "relevance": relevance,
            "groundedness": groundedness,
            "context_relevance": context_rel,
            "passed": passed,
            "warnings": []
        })

    def _relevance(self, q, a): ...
    def _groundedness(self, a, c): ...
    def _context_quality(self, q, c): ...
