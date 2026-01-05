from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )

    def embed_texts(self, texts: List[str]):
        """
        Batch embedding для індексації.
        """
        return self._embeddings.embed_documents(texts)

    def embed(self, text: str):
        """
        Single embedding (query, evaluation).
        """
        return self._embeddings.embed_query(text)

    def embed_text(self, text: str):
        """
        Alias для query embedding (очікується IndexManager).
        """
        return self.embed(text)

