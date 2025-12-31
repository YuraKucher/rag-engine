from typing import Dict


class PromptFactory:
    """
    Фабрика prompt-ів для generation layer.
    """

    @staticmethod
    def default_prompt(question: str, context: str) -> str:
        return f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    @staticmethod
    def qa_prompt(payload: Dict) -> str:
        """
        payload очікується з ReasoningAgent.prepare():
        {
            "question": str,
            "context": str,
            "sources": [...]
        }
        """
        return PromptFactory.default_prompt(
            question=payload["question"],
            context=payload["context"]
        )
