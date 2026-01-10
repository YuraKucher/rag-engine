from typing import List, Dict
from .strategies import ReasoningStrategy
from config.settings import settings


class ContextBuilder:
    """
    Формує текстовий контекст для LLM
    з урахуванням якості чанків (LTR output).
    """

    def __init__(self, strategy: ReasoningStrategy = ReasoningStrategy.SIMPLE):
        self.strategy = strategy
        self.max_chars: int = settings.system.get("max_context_chars", 4000)

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def build(self, chunks: List[Dict]) -> str:
        if not chunks:
            return ""

        if self.strategy == ReasoningStrategy.SIMPLE:
            context = self._simple_context(chunks)

        elif self.strategy == ReasoningStrategy.QA:
            context = self._qa_context(chunks)

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return self._truncate(context)

    # --------------------------------------------------
    # STRATEGIES
    # --------------------------------------------------

    def _simple_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join(
            chunk["content"].strip()
            for chunk in chunks
            if chunk.get("content")
        )

    def _qa_context(self, chunks: List[Dict]) -> str:
        """
        QA-контекст з явним grounding signal.
        """
        parts: List[str] = []

        for i, chunk in enumerate(chunks, start=1):
            groundedness = chunk.get("groundedness")

            header = (
                f"[Source {i}] "
                f"(doc_id={chunk['document_id']}, "
                f"chunk_id={chunk['chunk_id']}"
            )

            if groundedness is not None:
                header += f", groundedness={groundedness:.2f}"

            header += ")"

            body = chunk["content"].strip()
            parts.append(f"{header}\n{body}")

        return "\n\n".join(parts)

    # --------------------------------------------------
    # SAFETY
    # --------------------------------------------------

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text

        return (
            text[: self.max_chars].rstrip()
            + "\n\n[Context truncated due to system limits]"
        )
