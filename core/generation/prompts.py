from typing import Dict, List


class PromptFactory:
    """
    Фабрика prompt-ів для generation layer.
    """

    @staticmethod
    def qa_prompt(payload: Dict) -> str:
        """
        payload очікується з ReasoningAgent.prepare():
        {
            "question": str,
            "context": str,
            "sources": List[Dict]
        }
        """

        question: str = payload["question"]
        context: str = payload["context"]
        sources: List[Dict] = payload.get("sources", [])

        sources_block = "\n".join(
            f"- {s.get('chunk_id', 'unknown')} (doc: {s.get('document_id', 'unknown')})"
            for s in sources
        )

        return f"""
You are a factual question-answering assistant.

Your task:
- Answer the question using ONLY the information from the provided context.
- Do NOT use any external knowledge.
- If the answer cannot be derived from the context, respond exactly with: "I do not know."

Context (extracted document fragments):
{context}

Sources (for grounding reference):
{sources_block}

Question:
{question}

Answer requirements:
- Be concise and factual.
- Do not speculate.
- Do not add information not present in the context.

Final Answer:
""".strip()
