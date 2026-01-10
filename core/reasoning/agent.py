from typing import List, Dict
from .context_builder import ContextBuilder
from .strategies import ReasoningStrategy


class ReasoningAgent:
    """
    Агент reasoning.
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE
    ):
        self.strategy = strategy
        self.context_builder = ContextBuilder(strategy)

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def prepare(
        self,
        question: str,
        chunks: List[Dict]
    ) -> Dict:
        context = self.context_builder.build(chunks)
        sources = self._build_sources(chunks)

        return {
            "question": question,
            "context": context,
            "sources": sources,
            "strategy": self.strategy.value
        }

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    @staticmethod
    def _build_sources(chunks: List[Dict]) -> List[Dict]:
        """
        Metadata для explainability / debug / UI.
        """
        return [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "index_ids": chunk.get("metadata", {}).get("index_ids", [])
            }
            for chunk in chunks
            if chunk.get("chunk_id") and chunk.get("document_id")
        ]
