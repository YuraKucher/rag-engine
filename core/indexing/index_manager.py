from typing import List, Dict, Optional
from datetime import datetime
import uuid
from pathlib import Path

from .embedder import Embedder
from .faiss_index import FaissIndex


class IndexManager:
    """
    Керує FAISS-індексом.

    Відповідальність:
    - build index
    - save index
    - load index (явно)
    - query index

    НЕ:
    - cache
    - orchestration
    - policy
    """

    def __init__(self, embedding_model: str, indexes_path: str):
        self.embedder = Embedder(embedding_model)
        self.embedding_model = embedding_model

        self.indexes_path = Path(indexes_path)
        self.indexes_path.mkdir(parents=True, exist_ok=True)

        self.faiss_index: Optional[FaissIndex] = None
        self.chunk_ids: List[str] = []
        self.index_id: Optional[str] = None
        self.index_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # BUILD + SAVE
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> Dict:
        texts = [chunk["content"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)
        dimension = len(embeddings[0])

        self.faiss_index = FaissIndex(dimension)
        self.faiss_index.add(embeddings)

        self.index_id = str(uuid.uuid4())
        self.index_path = self.indexes_path / f"{self.index_id}.faiss"

        self.faiss_index.save(str(self.index_path))

        return {
            "index_id": self.index_id,
            "index_type": "faiss",
            "embedding_model": self.embedding_model,
            "chunk_ids": self.chunk_ids,
            "index_path": str(self.index_path),
            "created_at": datetime.utcnow().isoformat()
        }

    # ------------------------------------------------------------------
    # LOAD (ЯВНО)
    # ------------------------------------------------------------------

    def load_index(self, index_path: str, chunk_ids: List[str]) -> None:
        """
        Явно завантажує індекс.
        Викликається orchestration-шаром.
        """

        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")

        self.faiss_index = FaissIndex.load(str(path))
        self.chunk_ids = chunk_ids
        self.index_id = path.stem
        self.index_path = path

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(self, query: str, k: int) -> List[str]:
        if self.faiss_index is None:
            raise RuntimeError(
                "Index not loaded. "
                "Call load_index() before query()."
            )

        query_vector = self.embedder.embed_text(query)
        indices, _ = self.faiss_index.search(query_vector, k)

        return [
            self.chunk_ids[i]
            for i in indices
            if i < len(self.chunk_ids)
        ]
