from typing import List, Dict, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path

from .embedder import Embedder
from .faiss_index import FaissIndex


class IndexManager:
    """
    Керує FAISS-індексом.

    Відповідальність:
    - build index
    - save index + metadata
    - load index (з metadata)
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
        self.document_ids: List[str] = []

        self.index_id: Optional[str] = None
        self.index_path: Optional[Path] = None
        self.metadata_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # BUILD + SAVE
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> Dict:
        texts = [chunk["content"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.document_ids = sorted(
            {chunk["document_id"] for chunk in chunks}
        )

        embeddings = self.embedder.embed_texts(texts)
        dimension = len(embeddings[0])

        self.faiss_index = FaissIndex(dimension)
        self.faiss_index.add(embeddings)

        self.index_id = str(uuid.uuid4())
        self.index_path = self.indexes_path / f"{self.index_id}.faiss"
        self.metadata_path = self.indexes_path / f"{self.index_id}.index.json"

        self.faiss_index.save(str(self.index_path))

        metadata = {
            "index_id": self.index_id,
            "index_type": "faiss",
            "embedding_model": self.embedding_model,
            "index_path": str(self.index_path),
            "chunk_ids": self.chunk_ids,
            "document_ids": self.document_ids,
            "created_at": datetime.utcnow().isoformat()
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    # ------------------------------------------------------------------
    # LOAD (З METADATA)
    # ------------------------------------------------------------------

    def load_index(self, index_id: str) -> None:
        faiss_path = self.indexes_path / f"{index_id}.faiss"
        metadata_path = self.indexes_path / f"{index_id}.index.json"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Index metadata not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.faiss_index = FaissIndex.load(str(faiss_path))
        self.chunk_ids = metadata["chunk_ids"]
        self.document_ids = metadata["document_ids"]

        self.index_id = metadata["index_id"]
        self.index_path = faiss_path
        self.metadata_path = metadata_path

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

    # ---- буде реалізовано потім -------
    def list_indexes(self) -> List[str]:
        return [
            p.stem.replace(".index", "")
            for p in self.indexes_path.glob("*.index.json")
        ]
