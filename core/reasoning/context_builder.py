from typing import List, Dict
from .strategies import ReasoningStrategy


class ContextBuilder:
    """
    Формує текстовий контекст для LLM.
    """

    def __init__(self, strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE):
        self.strategy = strategy

    def build(self, chunks: List[Dict]) -> str:
        """
        chunks: список chunk.schema.json
        return: context string
        """

        if self.strategy == ReasoningStrategy.SIMPLE:
            return self._simple_context(chunks)

        if self.strategy == ReasoningStrategy.QA:
            return self._qa_context(chunks)

        raise ValueError(f"Unsupported strategy: {self.strategy}")

    def _simple_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join(chunk["content"] for chunk in chunks)

    def _qa_context(self, chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[Source {i}]\n{chunk['content']}")
        return "\n\n".join(context_parts)
