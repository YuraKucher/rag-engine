from typing import List, Dict
from .strategies import ReasoningStrategy


class ContextBuilder:
    """
    Формує текстовий контекст для LLM
    на основі чанків і стратегії reasoning.
    """

    def __init__(self, strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE):
        self.strategy = strategy

    def build(self, chunks: List[Dict]) -> str:
        """
        chunks: список обʼєктів chunk.schema.json
        return: context string
        """

        if not chunks:
            return ""

        if self.strategy == ReasoningStrategy.SIMPLE:
            return self._simple_context(chunks)

        if self.strategy == ReasoningStrategy.QA:
            return self._qa_context(chunks)

        raise ValueError(f"Unsupported strategy: {self.strategy}")

    # ------------------------------------------------------------------

    def _simple_context(self, chunks: List[Dict]) -> str:
        """
        Простий контекст без форматування.
        Зберігає порядок чанків.
        """
        return "\n\n".join(chunk["content"] for chunk in chunks)

    def _qa_context(self, chunks: List[Dict]) -> str:
        """
        Контекст з явними джерелами.
        Корисно для grounded QA.
        """

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {i}] "
                f"(doc: {chunk['document_id']}, "
                f"chunk: {chunk['chunk_id']})"
            )

            context_parts.append(
                f"{header}\n{chunk['content']}"
            )

        return "\n\n".join(context_parts)
