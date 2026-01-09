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
            "sources": List[{chunk_id, document_id}],
            "strategy": str
        }
        """

        question: str = payload["question"]
        context: str = payload["context"]
        sources: List[Dict] = payload.get("sources", [])

        if sources:
            sources_block = "\n".join(
                f"- chunk_id={s['chunk_id']}, document_id={s['document_id']}"
                for s in sources
            )
        else:
            sources_block = "None"

        return f"""
You are a factual question-answering assistant.

Your task:
- Answer the question using ONLY the information from the provided context.
- Do NOT use any external knowledge.
- If the answer cannot be derived from the context, respond exactly with:
  "I do not know."

Context:
{context}

Grounding sources (for reference only):
{sources_block}

Question:
{question}

Answer guidelines:
- Provide a synthesized answer, not a quote.
- Be concise, clear, and factual.
- Do NOT copy sentences verbatim from the context.
- Do NOT mention sources or metadata in the answer.
- Do NOT speculate or add information not present in the context.

Final Answer:
""".strip()
