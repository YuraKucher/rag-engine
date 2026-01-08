import json
import os
from typing import Dict, List, Optional


class ChunkStore:
    """
    Сховище чанків.
    Source of truth для chunk-level knowledge.
    Працює з chunk.schema.json
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # --------------------------------------------------
    # BASIC IO
    # --------------------------------------------------

    def save(self, chunk: Dict) -> None:
        """
        Зберігає чанк як окремий JSON-файл.
        """
        chunk_id = chunk["chunk_id"]
        path = os.path.join(self.base_path, f"{chunk_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

    def load(self, chunk_id: str) -> Optional[Dict]:
        """
        Завантажує чанк за ID.
        """
        path = os.path.join(self.base_path, f"{chunk_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------
    # QUERY HELPERS (OFFLINE / ORCHESTRATION)
    # --------------------------------------------------

    def list_chunk_ids(self) -> List[str]:
        """
        Повертає список всіх chunk_id.
        """
        return [
            filename.replace(".json", "")
            for filename in os.listdir(self.base_path)
            if filename.endswith(".json")
        ]

    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        """
        Повертає всі чанки документа.
        """
        return self._filter_chunks(
            lambda c: c.get("document_id") == document_id
        )

    def get_chunks_by_index(self, index_id: str) -> List[Dict]:
        """
        Повертає всі чанки, що належать до індексу.
        """
        return self._filter_chunks(
            lambda c: c.get("metadata", {}).get("index_id") == index_id
        )

    # --------------------------------------------------
    # INTERNAL
    # --------------------------------------------------

    def _filter_chunks(self, predicate) -> List[Dict]:
        """
        Універсальний фільтр чанків (offline use).
        """
        results = []

        for filename in os.listdir(self.base_path):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(self.base_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                if predicate(chunk):
                    results.append(chunk)

        return results
