from typing import List, Dict


class Reranker:
    """
    Переранжування чанків.
    """

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        chunks: список chunk.schema.json
        """

        # Базова реалізація — без ML
        # Просто обрізаємо список
        return chunks[:top_k]
