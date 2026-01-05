import os
import json
from datetime import datetime
from typing import Dict, Optional
from typing import Dict, Type
from .base_loader import BaseLoader
from .pdf_loader import PDFLoader

REGISTRY_PATH = "data/state/ingestion_registry.json"


class IngestionRegistry:
    def __init__(self):
        os.makedirs("data/state", exist_ok=True)
        self._state = self._load()

    def _load(self) -> Dict:
        if not os.path.exists(REGISTRY_PATH):
            return {
                "version": 1,
                "documents": {}
            }
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)

    def exists(self, doc_hash: str) -> bool:
        return doc_hash in self._state["documents"]

    def get(self, doc_hash: str) -> Optional[Dict]:
        return self._state["documents"].get(doc_hash)

    def register(self, metadata: Dict):
        metadata["registered_at"] = datetime.utcnow().isoformat()
        self._state["documents"][metadata["hash"]] = metadata
        self._save()


class LoaderRegistry:
    """
    Реєстр loader-ів за типом файлу.
    """

    _loaders: Dict[str, Type[BaseLoader]] = {
        "pdf": PDFLoader
    }

    @classmethod
    def get_loader(cls, file_type: str) -> BaseLoader:
        if file_type not in cls._loaders:
            raise ValueError(f"No loader registered for type: {file_type}")
        return cls._loaders[file_type]()