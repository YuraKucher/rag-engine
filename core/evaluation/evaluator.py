"""
Evaluator
=========

Єдиний компонент, який:
- оцінює відповідь LLM
- агрегує сигнали на рівні:
  - чанків
  - документів
  - індексів

НЕ:
- не оновлює state
- не зберігає evaluation
- не знає про learning
"""

import uuid
from datetime import datetime
from typing import Dict, List
import numpy as np


class Evaluator:
    """
    Online evaluator for RAG responses.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: List[Dict],
        index_ids: List[str]
    ) -> Dict:
        """
        Формує evaluation у форматі,
        сумісному з learning та мультиіндексацією.
        """

        # --------- Chunk-level ---------
        chunk_evals = self._evaluate_chunks(question, answer, chunks)

        # --------- Document-level ---------
        document_evals = self._aggregate_documents(chunk_evals)

        # --------- Index-level ---------
        index_evals = self._aggregate_indexes(chunk_evals, index_ids)

        return {
            "evaluation_id": str(uuid.uuid4()),
            "question": question,
            "answer": answer,

            "chunks": chunk_evals,
            "documents": document_evals,
            "indexes": index_evals,

            "created_at": datetime.utcnow().isoformat() + "Z"
        }

    # --------------------------------------------------
    # CHUNK LEVEL
    # --------------------------------------------------

    def _evaluate_chunks(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Оцінка чанків:
        - relevance: близькість чанка до питання
        - groundedness: близькість чанка до відповіді
        """

        q_emb = np.array(self.embedder.embed(question))
        a_emb = np.array(self.embedder.embed(answer))

        results = []

        for chunk in chunks:
            c_emb = np.array(self.embedder.embed(chunk["content"]))

            relevance = self._cosine(q_emb, c_emb)
            groundedness = self._cosine(a_emb, c_emb)

            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "index_ids": chunk.get("metadata", {}).get("index_ids", []),
                "relevance": float(relevance),
                "groundedness": float(groundedness)
            })

        return results

    # --------------------------------------------------
    # DOCUMENT LEVEL
    # --------------------------------------------------

    def _aggregate_documents(self, chunk_evals: List[Dict]) -> List[Dict]:
        """
        Агрегація чанків → документи.
        """

        docs: Dict[str, Dict] = {}

        for ch in chunk_evals:
            doc_id = ch["document_id"]
            doc = docs.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "relevance": [],
                    "groundedness": []
                }
            )

            doc["relevance"].append(ch["relevance"])
            doc["groundedness"].append(ch["groundedness"])

        results = []

        for doc in docs.values():
            results.append({
                "document_id": doc["document_id"],
                "relevance": float(np.mean(doc["relevance"])),
                "answerability": float(np.mean(doc["groundedness"]))
            })

        return results

    # --------------------------------------------------
    # INDEX LEVEL
    # --------------------------------------------------

    def _aggregate_indexes(
        self,
        chunk_evals: List[Dict],
        index_ids: List[str]
    ) -> List[Dict]:
        """
        Агрегація чанків → індекси.
        """

        indexes: Dict[str, Dict] = {}

        for ch in chunk_evals:
            for idx in ch.get("index_ids", []):
                index = indexes.setdefault(
                    idx,
                    {
                        "index_id": idx,
                        "used_chunks": [],
                        "relevance": [],
                        "groundedness": []
                    }
                )

                index["used_chunks"].append(ch["chunk_id"])
                index["relevance"].append(ch["relevance"])
                index["groundedness"].append(ch["groundedness"])

        results = []

        for idx in index_ids:
            if idx not in indexes:
                continue

            data = indexes[idx]
            results.append({
                "index_id": idx,
                "used_chunks": data["used_chunks"],
                "avg_relevance": float(np.mean(data["relevance"])),
                "avg_groundedness": float(np.mean(data["groundedness"]))
            })

        return results

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
