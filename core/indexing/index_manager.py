from typing import List, Dict
from datetime import datetime
import uuid

from .embedder import Embedder
from .faiss_index import FaissIndex


class IndexManager:
    """
    Керує створенням та використанням індексу.
    Узгоджений з index.schema.json
    """

    def __init__(self, embedding_model: str):
        self.embedder = Embedder(embedding_model)
        self.embedding_model = embedding_model
        self.faiss_index = None
        self.chunk_ids: List[str] = []

    def build_index(self, chunks: List[Dict]) -> Dict:
        """
        chunks: список обʼєктів, що відповідають chunk.schema.json
        """

        texts = [chunk["content"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)

        dimension = len(embeddings[0])
        self.faiss_index = FaissIndex(dimension)
        self.faiss_index.add(embeddings)

        index_metadata = {
            "index_id": str(uuid.uuid4()),
            "index_type": "faiss",
            "embedding_model": self.embedding_model,
            "chunk_ids": self.chunk_ids,
            "created_at": datetime.utcnow().isoformat()
        }

        return index_metadata

    def query(self, query: str, k: int) -> List[str]:
        """
        Повертає chunk_ids найближчих чанків
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built")

        query_vector = self.embedder.embed_text(query)
        indices, _ = self.faiss_index.search(query_vector, k)

        return [self.chunk_ids[i] for i in indices if i < len(self.chunk_ids)]
