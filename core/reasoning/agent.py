from typing import List, Dict

from .context_builder import ContextBuilder
from .strategies import ReasoningStrategy


class ReasoningAgent:
    """
    Агент reasoning.
    Готує вхідні дані для generation layer.
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE
    ):
        self.context_builder = ContextBuilder(strategy)

    def prepare(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Повертає структуру для LLM generation.
        """

        context = self.context_builder.build(chunks)

        return {
            "question": question,
            "context": context,
            "sources": [chunk["chunk_id"] for chunk in chunks]
        }
