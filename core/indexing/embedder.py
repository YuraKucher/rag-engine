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

    def embed_text(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)
