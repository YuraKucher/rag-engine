from typing import List, Dict

from .context_builder import ContextBuilder
from .strategies import ReasoningStrategy


class ReasoningAgent:
    """
    Агент reasoning.
    Готує структурований payload для generation layer.
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE
    ):
        self.strategy = strategy
        self.context_builder = ContextBuilder(strategy)

    def prepare(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Повертає reasoning payload з явним контрактом.
        """

        context = self.context_builder.build(chunks)

        sources = [
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"]
            }
            for chunk in chunks
        ]

        return {
            "question": question,
            "context": context,
            "sources": sources,
            "strategy": self.strategy.value
        }
