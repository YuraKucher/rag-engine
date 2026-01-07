from typing import List, Dict, Optional
from datetime import datetime
import uuid
from pathlib import Path
from .embedder import Embedder
from .faiss_index import FaissIndex


class IndexManager:
    """
    Керує створенням та використанням індексу.
    Узгоджений з index.schema.json
    """

    def __init__(self, embedding_model: str, indexes_path: str):
        self.embedder = Embedder(embedding_model)
        self.embedding_model = embedding_model
        self.indexes_path = Path(indexes_path)

        self.faiss_index: Optional[FaissIndex] = None
        self.chunk_ids: List[str] = []
        self.index_id: Optional[str] = None

    def build_index(self, chunks: List[Dict]) -> Dict:
        texts = [chunk["content"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)

        dimension = len(embeddings[0])
        self.faiss_index = FaissIndex(dimension)
        self.faiss_index.add(embeddings)

        self.index_id = str(uuid.uuid4())
        index_path = self.indexes_path / f"{self.index_id}.faiss"

        # КЛЮЧОВИЙ РЯДОК
        self.faiss_index.save(index_path)

        return {
            "index_id": self.index_id,
            "index_type": "faiss",
            "embedding_model": self.embedding_model,
            "chunk_ids": self.chunk_ids,
            "index_path": str(index_path),
            "created_at": datetime.utcnow().isoformat()
        }

    def query(self, query: str, k: int) -> List[str]:
        """
        Повертає chunk_ids найближчих чанків
        """
        if self.faiss_index is None:
            raise RuntimeError("Index not built")

        query_vector = self.embedder.embed_text(query)
        indices, _ = self.faiss_index.search(query_vector, k)

        return [self.chunk_ids[i] for i in indices if i < len(self.chunk_ids)]
