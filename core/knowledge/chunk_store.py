import json
import os
from typing import Dict, List


class ChunkStore:
    """
    Сховище чанків.
    Працює з chunk.schema.json
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save(self, chunk: Dict) -> None:
        chunk_id = chunk["chunk_id"]
        path = os.path.join(self.base_path, f"{chunk_id}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

    def load(self, chunk_id: str) -> Dict:
        path = os.path.join(self.base_path, f"{chunk_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        chunks = []

        for filename in os.listdir(self.base_path):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(self.base_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                chunk = json.load(f)
                if chunk["document_id"] == document_id:
                    chunks.append(chunk)

        return chunks
