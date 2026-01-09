from pathlib import Path
from typing import Dict, List, Optional
import json


class IndexRegistry:
    """
    Реєстр усіх індексів у системі.

    Відповідальність:
    - зчитувати index metadata з диску
    - тримати in-memory каталог індексів
    - надавати інформацію orchestration / retrieval layer

    НЕ:
    - не виконує retrieval
    - не працює з FAISS напряму
    """

    def __init__(self, indexes_path: str):
        self.indexes_path = Path(indexes_path)
        self.indexes_path.mkdir(parents=True, exist_ok=True)

        # index_id -> metadata
        self._indexes: Dict[str, Dict] = {}

        self._load_all_indexes()

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def _load_all_indexes(self) -> None:
        """
        Завантажує всі *.index.json з папки indexes.
        """

        self._indexes.clear()

        for meta_file in self.indexes_path.glob("*.index.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            index_id = metadata.get("index_id")
            if index_id:
                self._indexes[index_id] = metadata

    # ------------------------------------------------------------------
    # READ API
    # ------------------------------------------------------------------

    def list_indexes(self) -> List[Dict]:
        """
        Повертає metadata всіх індексів.
        """
        return list(self._indexes.values())

    def get_index(self, index_id: str) -> Optional[Dict]:
        """
        Повертає metadata конкретного індексу.
        """
        return self._indexes.get(index_id)

    def get_indexes_for_document(self, document_id: str) -> List[Dict]:
        """
        Повертає всі індекси, які містять даний документ.
        """
        return [
            meta for meta in self._indexes.values()
            if document_id in meta.get("document_ids", [])
        ]

    def get_all_index_ids(self) -> List[str]:
        return list(self._indexes.keys())

    # ------------------------------------------------------------------
    # MUTATION
    # ------------------------------------------------------------------

    def register_index(self, metadata: Dict) -> None:
        """
        Реєструє новий індекс у registry (in-memory).
        Викликається після build_index().
        """

        index_id = metadata["index_id"]
        self._indexes[index_id] = metadata

    def reload(self) -> None:
        """
        Повне перевантаження registry з диску.
        Корисно після рестарту.
        """
        self._load_all_indexes()
