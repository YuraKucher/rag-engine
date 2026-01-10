from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    """
    Thin wrapper над embedding-моделлю.

    Відповідає ТІЛЬКИ за:
    - перетворення тексту → вектор
    - гарантію стабільної embedding dimension
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

        # 🔒 Фіксуємо dimension один раз
        test_vec = self._embeddings.embed_query("test")
        self.dimension = len(test_vec)

    def embed(self, text: str) -> List[float]:
        """
        Single embedding (query, evaluation).
        """
        return self._embeddings.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding (indexing).
        """
        return self._embeddings.embed_documents(texts)
