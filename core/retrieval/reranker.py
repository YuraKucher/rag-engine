from typing import List, Dict
import numpy as np


class Reranker:
    """
    Semantic reranker на cosine similarity.
    Підвищує precision після FAISS retrieval.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        chunks: список chunk.schema.json
        """

        if not chunks:
            return []

        # 1. Embed query один раз
        query_emb = np.array(self.embedder.embed(query))

        scored_chunks = []

        for chunk in chunks:
            chunk_emb = np.array(
                self.embedder.embed(chunk["content"])
            )

            score = float(
                np.dot(query_emb, chunk_emb) /
                (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
            )

            scored_chunks.append((score, chunk))

        # 2. Sort by similarity (desc)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # 3. Return top_k chunks
        return [chunk for _, chunk in scored_chunks[:top_k]]
