from typing import List, Dict
import numpy as np
import re


class Reranker:
    """
    Semantic reranker на cosine similarity + lightweight heuristics.

    Завдання:
    - підвищити precision після FAISS
    - відсіяти технічний шум (мануали)
    - підняти definition-style чанки
    """

    def __init__(
        self,
        embedder,
        max_words: int = 40,
        manual_penalty: float = 0.15,
        definition_bonus: float = 0.10
    ):
        self.embedder = embedder
        self.max_words = max_words
        self.manual_penalty = manual_penalty
        self.definition_bonus = definition_bonus

        # прості патерни технічних мануалів
        self._manual_patterns = re.compile(
            r"\b(install|installation|path|folder|directory|ini|config|svn|github|version)\b",
            re.IGNORECASE
        )

    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        if not chunks:
            return []

        # 1. Embed query ОДИН раз
        query_emb = np.array(self.embedder.embed(query))

        scored_chunks = []

        for chunk in chunks:
            rep_text = self._representative_text(chunk["content"])

            # 2. Embed representative text
            chunk_emb = np.array(self.embedder.embed(rep_text))

            # 3. Base cosine similarity
            score = self._cosine(query_emb, chunk_emb)

            # 4. Heuristics
            score += self._definition_bonus(query, rep_text)
            score -= self._manual_penalty(rep_text)

            scored_chunks.append((score, chunk))

        # 5. Sort by score DESC
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in scored_chunks[:top_k]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _representative_text(self, content: str) -> str:
        """
        Беремо перше речення або перші max_words слів.
        """
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        first_sentence = sentences[0]

        words = first_sentence.split()
        if len(words) > self.max_words:
            return " ".join(words[:self.max_words])

        return first_sentence

    def _definition_bonus(self, query: str, text: str) -> float:
        """
        Бонус для definition-запитів, якщо ключове слово
        присутнє у першому реченні.
        """
        q = query.lower()
        t = text.lower()

        if q.startswith(("what is", "who is", "define")):
            # беремо головний термін (останнє слово питання)
            term = q.replace("?", "").split()[-1]
            if term in t:
                return self.definition_bonus

        return 0.0

    def _manual_penalty(self, text: str) -> float:
        """
        Пенальті для технічних фрагментів.
        """
        if self._manual_patterns.search(text):
            return self.manual_penalty
        return 0.0

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
