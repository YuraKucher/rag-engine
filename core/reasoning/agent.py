from typing import List, Dict

from .context_builder import ContextBuilder
from .strategies import ReasoningStrategy


class ReasoningAgent:
    """
    Агент reasoning.

    Відповідальність:
    - підготувати структурований payload для generation layer
    - НЕ виконувати retrieval
    - НЕ змінювати чанки
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE
    ):
        self.strategy = strategy
        self.context_builder = ContextBuilder(strategy)

    # ======================================================
    # PUBLIC API
    # ======================================================

    def prepare(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Формує reasoning payload з явним контрактом.

        payload:
        {
            question: str
            context: str
            sources: List[{chunk_id, document_id}]
            strategy: str
        }
        """

        context = self.context_builder.build(chunks)

        sources = self._build_sources(chunks)

        return {
            "question": question,
            "context": context,
            "sources": sources,
            "strategy": self.strategy.value
        }

    # ======================================================
    # INTERNALS
    # ======================================================

    @staticmethod
    def _build_sources(chunks: List[Dict]) -> List[Dict]:
        """
        Витягує мінімальний набір metadata для grounding.
        """
        return [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id")
            }
            for chunk in chunks
            if chunk.get("chunk_id") and chunk.get("document_id")
        ]
