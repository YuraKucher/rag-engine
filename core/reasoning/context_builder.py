from typing import List, Dict
from .strategies import ReasoningStrategy
from config.settings import settings


class ContextBuilder:
    """
    Формує текстовий контекст для LLM
    з урахуванням:
    - стратегії reasoning
    - обмежень моделі
    - якості джерел
    """

    def __init__(self, strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE):
        self.strategy = strategy

        # system-level constraints
        self.max_chars: int = settings.system.get(
            "max_context_chars", 4000
        )

    # ======================================================
    # PUBLIC API
    # ======================================================

    def build(self, chunks: List[Dict]) -> str:
        """
        chunks: список chunk.schema.json (вже відранкований!)
        return: context string
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

    # ======================================================
    # STRATEGIES
    # ======================================================

    def _simple_context(self, chunks: List[Dict]) -> str:
        """
        Мінімалістичний контекст:
        - без metadata
        - тільки текст
        """
        return "\n\n".join(
            chunk["content"].strip()
            for chunk in chunks
            if chunk.get("content")
        )

    def _qa_context(self, chunks: List[Dict]) -> str:
        """
        QA-контекст:
        - явні джерела
        - стабільний формат
        - придатний для grounded answers
        """
        parts: List[str] = []

        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {i}] "
                f"(doc_id={chunk['document_id']}, "
                f"chunk_id={chunk['chunk_id']})"
            )

            body = chunk["content"].strip()

            parts.append(f"{header}\n{body}")

        return "\n\n".join(parts)

    # ======================================================
    # SAFETY
    # ======================================================

    def _truncate(self, text: str) -> str:
        """
        Жорстке обрізання контексту
        для уникнення token overflow.
        """

        if len(text) <= self.max_chars:
            return text

        truncated = text[: self.max_chars].rstrip()

        return (
            truncated
            + "\n\n[Context truncated due to system limits]"
        )
