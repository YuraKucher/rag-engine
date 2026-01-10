import json
import uuid
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from core.indexing.embedder import Embedder
from core.indexing.faiss_index import FaissIndex


class IndexManager:
    """
    Керує множиною FAISS-індексів.

    Відповідальність:
    - build index (+ metadata)
    - load index
    - query index
    - index discovery (role → index_id)

    НЕ:
    - routing
    - ranking
    - learning
    """

    def __init__(self, embedding_model: str, indexes_path: str):
        self.embedder = Embedder(embedding_model)
        self.embedding_model = embedding_model

        self.indexes_path = Path(indexes_path)
        self.indexes_path.mkdir(parents=True, exist_ok=True)

        # loaded indexes
        self._indexes: Dict[str, FaissIndex] = {}

        # metadata cache
        self._metadata: Dict[str, Dict] = {}

        # load metadata on startup
        self._load_all_metadata()

    # --------------------------------------------------
    # METADATA DISCOVERY
    # --------------------------------------------------

    def _load_all_metadata(self) -> None:
        """
        Читає всі *.index.json з диску.
        """
        for meta_path in self.indexes_path.glob("*.index.json"):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._metadata[meta["index_id"]] = meta

    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------

    def build_index(
        self,
        chunks: List[Dict],
        index_role: str = "general"
    ) -> Dict:
        texts = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        document_ids = sorted({chunk["document_id"] for chunk in chunks})

        embeddings = self.embedder.embed_batch(texts)
        dimension = len(embeddings[0])

        faiss_index = FaissIndex(dimension)
        faiss_index.add(embeddings)

        index_id = str(uuid.uuid4())
        index_path = self.indexes_path / f"{index_id}.faiss"
        metadata_path = self.indexes_path / f"{index_id}.index.json"

        faiss_index.save(str(index_path))

        metadata = {
            "index_id": index_id,
            "index_type": "faiss",
            "index_role": index_role,
            "embedding_model": self.embedding_model,
            "index_path": str(index_path),
            "chunk_ids": chunk_ids,
            "document_ids": document_ids,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._metadata[index_id] = metadata
        self._indexes[index_id] = faiss_index

        return metadata

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def load_index(self, index_id: str) -> None:
        if index_id in self._indexes:
            return

        meta = self._metadata.get(index_id)
        if not meta:
            raise KeyError(f"Index metadata not found: {index_id}")

        faiss_index = FaissIndex.load(meta["index_path"])
        self._indexes[index_id] = faiss_index

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def query(self, query: str, k: int, index_id: str) -> List[str]:
        self.load_index(index_id)

        faiss_index = self._indexes[index_id]
        meta = self._metadata[index_id]

        query_vector = self.embedder.embed(query)
        indices, _ = faiss_index.search(query_vector, k)

        chunk_ids = meta.get("chunk_ids", [])

        return [
            chunk_ids[i]
            for i in indices
            if i < len(chunk_ids)
        ]

    # --------------------------------------------------
    # DISCOVERY API (NEW)
    # --------------------------------------------------

    def list_indexes(self) -> List[str]:
        """
        Повертає всі index_id (UUID).
        """
        return list(self._metadata.keys())

    def get_indexes_by_role(self, role: str) -> List[str]:
        """
        Повертає всі index_id з заданою semantic role.
        """
        return [
            index_id
            for index_id, meta in self._metadata.items()
            if meta.get("index_role") == role
        ]
