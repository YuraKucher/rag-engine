"""
RAG Service — orchestration layer.

Зʼєднує:
- Retriever
- Reasoning
- Generation
- Evaluation
"""


class RAGService:
    def answer(self, question: str) -> dict:
        """
        Main entry point for answering questions.
        """
        raise NotImplementedError
