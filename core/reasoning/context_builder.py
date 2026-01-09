from typing import List, Dict
from .strategies import ReasoningStrategy
from config.settings import settings


class ContextBuilder:
    """
    Формує текстовий контекст для LLM
    з урахуванням системних обмежень.
    """

    def __init__(self, strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE):
        self.strategy = strategy

        # system-level constraints
        self.max_chars: int = settings.system.get(
            "max_context_chars", 4000
        )

    def build(self, chunks: List[Dict]) -> str:
        """
        chunks: список chunk.schema.json
        return: context string (truncated)
        """

        if not chunks:
            return ""

        if self.strategy == ReasoningStrategy.SIMPLE:
            context = self._simple_context(chunks)

        elif self.strategy == ReasoningStrategy.QA:
            context = self._qa_context(chunks)

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return self._truncate(context)

    # ------------------------------------------------------------------

    def _simple_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join(chunk["content"] for chunk in chunks)

    def _qa_context(self, chunks: List[Dict]) -> str:
        parts = []

        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {i}] "
                f"(doc: {chunk['document_id']}, "
                f"chunk: {chunk['chunk_id']})"
            )
            parts.append(f"{header}\n{chunk['content']}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """
        Жорстке обрізання контексту
        для уникнення token overflow.
        """
        if len(text) <= self.max_chars:
            return text

        return text[: self.max_chars].rstrip() + "\n\n[Context truncated]"
