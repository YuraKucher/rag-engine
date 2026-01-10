from pathlib import Path
from typing import Dict, List
import json
import uuid
from datetime import datetime

from core.indexing.embedder import Embedder
from core.indexing.faiss_index import FaissIndex
from config.settings import settings


class IndexManager:
    """
    Manages multiple FAISS indexes (config-driven).
    """

    def __init__(self, indexes_path: str):
        self.embedder = Embedder()

        self.indexes_path = Path(indexes_path)
        self.indexes_path.mkdir(parents=True, exist_ok=True)

        self._indexes: Dict[str, FaissIndex] = {}
        self._metadata: Dict[str, Dict] = {}

        self._load_all_metadata()

    # --------------------------------------------------
    # METADATA DISCOVERY
    # --------------------------------------------------

    def _load_all_metadata(self) -> None:
        for meta_path in self.indexes_path.glob("*.index.json"):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._metadata[meta["index_id"]] = meta

    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------

    def build_index(self, chunks: List[Dict], index_role: str = "general") -> Dict:
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
            "embedding_config": settings.models["embeddings"],
            "index_path": str(index_path),
            "chunk_ids": chunk_ids,
            "document_ids": document_ids,
            "created_at": datetime.utcnow().isoformat() + "Z",
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

        self._indexes[index_id] = FaissIndex.load(meta["index_path"])

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def query(self, query: str, k: int, index_id: str) -> List[str]:
        self.load_index(index_id)

        query_vector = self.embedder.embed(query)
        indices, _ = self._indexes[index_id].search(query_vector, k)

        chunk_ids = self._metadata[index_id]["chunk_ids"]
        return [chunk_ids[i] for i in indices if i < len(chunk_ids)]

    # --------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------

    def list_indexes(self) -> List[str]:
        return list(self._metadata.keys())

    def get_indexes_by_role(self, role: str) -> List[str]:
        return [
            index_id
            for index_id, meta in self._metadata.items()
            if meta.get("index_role") == role
        ]
