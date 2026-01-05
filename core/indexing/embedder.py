from typing import List
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings


class Embedder:
    """
    Обгортка над embedding-моделлю.
    Узгоджена з LangChain та index.schema.json
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embeddings: Embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    @property
    def embedding_dim(self) -> int:
        """
        Розмір embedding-вектора.
        Публічний контракт для індексів.
        """
        return self._embeddings.embedding_size

    def embed(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_text(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

